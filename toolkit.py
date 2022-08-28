import csv
from enum import Enum
import os
import struct
import subprocess
import sys
from collections import namedtuple
import time
from typing import BinaryIO, Iterable, List, Optional, Tuple

"""
NOTE: if you change the number of tracks and load an existing savegame, don't forget to enter track list editing mode
(new tracks are disabled by default).
The game won't crash though, which I find amazing. The developers could have messed it up so easily and no one would know.
"""

Track = namedtuple("Track", "filename_base artist album title_line1 title_line2 have_video")


def decode_trackinfo(file: BinaryIO) -> List[Track]:
    n_of_entries = file.read(1)[0]
    data = file.read()
    def read_str_at(offset: int):
        sl = data[offset:]
        # Find null terminator (two 0 bytes)
        idx = next(x[0] for x in zip(range(0, len(sl), 2), sl[0::2], sl[1::2]) if x[1:] == (0, 0))
        sl = sl[:idx]
        return sl.decode("utf-16le")

    entry_size = 0x12
    entries = [data[x * entry_size:(x+1) * entry_size] for x in range(n_of_entries)]
    entries_decoded = []
    for entry in entries:
        unpacked = struct.unpack("<5B??B5H", entry)
        lengths = unpacked[:5]
        strings = tuple(map(read_str_at, unpacked[-5:]))
        assert tuple(map(len, strings)) == lengths
        non_strings = unpacked[5:-5]
        entries_decoded.append(Track(
            strings[0],
            strings[1],
            strings[2],
            strings[3],
            strings[4],
            non_strings[0],
        ))

    return entries_decoded


def store_csv(filename: str, tracks: Iterable[Track]):
    with open(filename, "w", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, Track._fields, dialect='excel', delimiter=';', lineterminator='\n')
        writer.writeheader()
        for track in tracks:
            writer.writerow(dict(zip(Track._fields, track)))


def load_csv(filename: str) -> List[Track]:
    tracks = []
    file = open(filename, "r", encoding="utf-8-sig")
    for entry in csv.DictReader(file, dialect='excel', delimiter=';'):
        if isinstance(entry['have_video'], str) and entry['have_video'].isdigit():
            entry['have_video'] = bool(int(entry['have_video']))
        elif isinstance(entry['have_video'], str) and entry['have_video'].lower() == 'false':
            entry['have_video'] = False
        else:
            entry['have_video'] = bool(entry['have_video'])

        tr = Track(*map(lambda x: entry[x], Track._fields))
        if not tr.filename_base:
            print(f"Ignoring {tr}, no filename", file=sys.stderr)
            continue

        tracks.append(tr)
    return tracks


def encode_trackinfo(file: BinaryIO, tracks: List[Track]) -> None:
    file.write(struct.pack('B', len(tracks)))
    before_stringstore_size = len(tracks) * struct.calcsize("<5B??B5H")
    stringstore = b''
    all_strings = {}
    for track in tracks:
        new_data = []
        for string in track[:5]:
            if string not in all_strings:
                pos = len(stringstore) + before_stringstore_size
                strlen = len(string)
                stringstore += string.encode("utf-16le") + b"\0\0"
                all_strings[string] = (strlen, pos)

            new_data.append(all_strings[string])

        new_data.extend(track[5:])
        tr = Track(*new_data)
        file.write(struct.pack("5B", *map(lambda x: x[0], tr[:5])))
        file.write(struct.pack("??B", tr.have_video, True, 1))
        file.write(struct.pack("5H", *map(lambda x: x[1], tr[:5])))

    assert file.tell() == before_stringstore_size + 1
    file.write(stringstore)


def make_short_filenames():
    lines = []
    while True:
        a = input()
        if not a:
            break

        num, art, _, title = a.split('\t')
        art = ''.join(filter(lambda x: x.isalpha(), ''.join(art.split(" ")[:3])))
        title = ''.join(map(lambda x: x[0], title.split(" ")))
        lines.append(f"u{int(num):02}{art}_{title}")

    for line in lines:
        print(line)


class FileFormat(Enum):
    PCM_WAV = 1,
    ATRAC3_OMA = 2,
    ATRAC3PLUS_OMA = 3,


def ffprobe_file(path: str, expected: FileFormat) -> Tuple[int, int]:
    process = subprocess.run(
        [
            'ffprobe',
            "-show_format",
            "-show_streams",
            path,
        ],
        capture_output=True,
    )

    process.check_returncode()
    text = process.stdout.decode("utf-8")
    n_streams = text.count("[STREAM]")
    if n_streams > 1:
        raise ValueError("File must have exactly 1 stream")

    f_stream = text[text.index("[STREAM]") + len("[STREAM]") : text.index("[/STREAM]")].strip().splitlines()
    f_format = text[text.index("[FORMAT]") + len("[FORMAT]") : text.index("[/FORMAT]")].strip().splitlines()
    f_stream = dict(map(lambda x: x.split("=", 1), f_stream))
    f_format = dict(map(lambda x: x.split("=", 1), f_format))
    if int(f_stream["sample_rate"]) != 44100:
        raise ValueError("File must have a sampling rate of 44100")

    if int(f_stream["channels"]) != 2:
        raise ValueError("File must be in stereo")

    correct_format = False
    if expected == FileFormat.PCM_WAV:
        correct_format = f_format["format_name"] == "wav" and f_stream["codec_name"].startswith("pcm_s16")
    elif expected == FileFormat.ATRAC3_OMA:
        correct_format = f_format["format_name"] == "oma" and f_stream["codec_name"] == "atrac3"
    elif expected == FileFormat.ATRAC3PLUS_OMA:
        correct_format = f_format["format_name"] == "oma" and f_stream["codec_name"] == "atrac3p"

    if not correct_format:
        raise ValueError(f"Unexpected audio format - expected {expected.name}")


    return (int(f_stream["duration_ts"]), int(f_stream["bit_rate"]))


def ffmpeg_convert_file(path_in: str, path_out: str):
    process = subprocess.run([
            'ffmpeg',
            "-i", path_in,
            "-ar", "44100",
            "-ac", "2",
            path_out,
    ])
    process.check_returncode()

EXAMPLE_FMT_CHUNK = bytes.fromhex('666D742034000000FEFF020044AC0000943E0000E80200002200000803000000BFAA23E958CB7144A119FFFA01E4CE620100285C0000000000000000')

def write_new_headers(file_out: BinaryIO, format_chunk: bytes, new_data_length: int, n_samples: int):
    # We need to replace 16 bytes with that bit of "magic" data
    # I don't know what it does, but it has to be there
    # Also, we need to add a 'fact' chunk with the number of samples
    # (otherwise the playback will either cut short or crash after the file is finished).
    format_chunk = (
        format_chunk[:0x20] +
        EXAMPLE_FMT_CHUNK[0x20:0x30] +
        format_chunk[0x30:] +
        b'fact' +
        struct.pack("<IQ", 8, n_samples)
    )

    calculated_riff_size = len(b'WAVE') + len(format_chunk) + len(b'data') + 4 + new_data_length
    file_out.write(b'RIFF')
    file_out.write(struct.pack("<I", calculated_riff_size))
    file_out.write(b'WAVE')
    file_out.write(format_chunk)
    file_out.write(b'data')
    file_out.write(struct.pack("<I", new_data_length))


def process_atx_to_aud(file_in: BinaryIO, file_out: BinaryIO, n_samples: int):
    if file_in.read(4) != b"RIFF":
        raise ValueError("Not a RIFF file")
    file_in.seek(4, 1)
    if file_in.read(4) != b"WAVE":
        raise ValueError("Not a WAVE file")

    format_chunk = None
    frame_size = None
    # This is the 8-byte thing before each sound block that we have to remove because it's screwing everything up.
    # Perhaps UMD Stream Composer uses it for muxing or it's a vestige of the video track that's not present.
    offending_bytes = None
    while True:
        chunk_type = file_in.read(4)
        chunk_size = struct.unpack("<I", file_in.read(4))[0]
        if chunk_type == b'fmt ':
            format_chunk = b'fmt ' + struct.pack("<I", chunk_size) + file_in.read(chunk_size)
            # Only the length of each block with sound data is important, but let's unpack it all just for fun
            interesting_facts = "<HHIIHH"
            unpacked = struct.unpack(interesting_facts, format_chunk[8 : 8 + struct.calcsize(interesting_facts)])
            audio_format, n_channels, sample_rate, byte_rate, frame_size, bits_sample = unpacked
            print(f"Debug info: format {audio_format}, channels {n_channels}, srate {sample_rate}, brate {byte_rate * 8}, block {frame_size}, bpsample {bits_sample},")
        elif chunk_type == b'data':
            assert frame_size is not None
            assert chunk_size % (frame_size + 8) == 0
            n_sound_blocks = chunk_size // (frame_size + 8)
            result_chunk_length = n_sound_blocks * frame_size
            write_new_headers(file_out, format_chunk, result_chunk_length, n_samples)
            for _ in range(n_sound_blocks):
                if offending_bytes is None:
                    offending_bytes = file_in.read(8)
                    # Make sure the thing we're gonna be removing is what I expect it to be
                    assert offending_bytes.startswith(b'\x0F\xD0\x28') and offending_bytes.endswith(b'\0\0\0\0')
                else:
                    # Make sure we're not removing some other thing
                    assert file_in.read(8) == offending_bytes

                file_out.write(file_in.read(frame_size))
            break
        else:
            print(f"Unknown chunk type: {chunk_type}")


# According to https://wiki.multimedia.cx/index.php/ATRAC3plus
AT3P_BITRATE_TO_FRAME_SIZE = {
    48: 280,
    64: 376,
    96: 560,
    128: 744,
    160: 936,
    192: 1120,
    256: 1488,
    320: 1864,
    352: 2048,
}


def process_oma_to_aud(file_in: BinaryIO, file_out: BinaryIO, n_samples: int, bitrate: int, file_in_size: int):
    header = file_in.read(6)
    if header.startswith(b'ea3\x03\x00'):
        # We have to skip the ID3v2 header
        size = 0
        for _ in range(4):
            size <<= 7
            b = file_in.read(1)[0]
            assert (b & 0x80) == 0
            size |= b & 0x7F

        file_in.seek(size, 1)
        header = file_in.read(6)

    assert header.startswith(b'EA3\x01\x00'), f"Unrecognized data at position {file_in.tell()}"
    header_remaining = header[-1] - 6
    header_bytes = file_in.read(header_remaining)
    frame_size_bytes = header_bytes[0x1D:0x1F]
    frame_size = (struct.unpack("<H", frame_size_bytes)[0] + 1) * 8
    if frame_size != AT3P_BITRATE_TO_FRAME_SIZE[bitrate // 1000]:
        print(f"Frame size of {frame_size} may be incorrect for file with bitrate {bitrate // 1000}")

    start_of_audio_data = file_in.tell()
    data_length = file_in_size - start_of_audio_data
    assert data_length % frame_size == 0
    # NOTE: this assumes that sampling rate is 44100 and there are 2 channels.
    fmt_chunk = (
        EXAMPLE_FMT_CHUNK[:0x10] +
        struct.pack("<IH", bitrate // 8, frame_size) +
        EXAMPLE_FMT_CHUNK[0x16:]
    )
    write_new_headers(file_out, fmt_chunk, data_length, n_samples)
    while True:
        a = file_in.read(frame_size)
        if len(a) != frame_size:
            break

        if a[0] != 0x3A and a[0] != 0x3F:
            print(f"Unrecognized start of frame ({a[0]}) at {file_in.tell() - len(a)}")

        file_out.write(a)


def extract_n_samples_from_name(base_name: str) -> Tuple[str, Optional[int]]:
    orig_base_name = base_name
    n_samples = None
    if base_name.startswith("!"):
        base_name = base_name[1:]
        if base_name.lower().endswith("_audio1"):
            base_name = base_name[:-len("_audio1")]

        if "_" in base_name:
            base_name, n_samples_str = base_name.rsplit("_", 1)
            if n_samples_str.isdigit():
                n_samples = int(n_samples_str)

    if n_samples is None:
        return (orig_base_name, None)
    else:
        return (base_name, n_samples)


def process_input_file(path: str):
    path = os.path.abspath(path)
    dir_and_name, extension = os.path.splitext(path)
    extension = extension.lower()
    dir_name, base_name = os.path.split(dir_and_name)
    if extension == '' and base_name == 'trackinfo':
        csv_path = os.path.join(dir_name, f"{base_name}_{round(time.time())}.csv")
        with open(path, "rb") as file:
            store_csv(csv_path, decode_trackinfo(file))

        print(f"trackinfo decoded into {csv_path}")
    elif extension == '.csv':
        new_trackinfo_path = os.path.join(dir_name, f"{base_name}")
        with open(new_trackinfo_path, "wb") as file:
            encode_trackinfo(file, load_csv(path))

        print(f"Saved new trackinfo file to {new_trackinfo_path}")
    elif extension == ".atx":
        base_name, n_samples = extract_n_samples_from_name(base_name)

        if n_samples is None:
            raise ValueError(".atx file must be named like !{base_name}_{n_of_samples}.atx or !{base_name}_{n_of_samples}_Audio1.atx")

        # IMPORTANT: seems like file names have to be lowercase, regardless of how they are in the trackinfo file
        # this might be a limitation of UMDGen or something in the game engine
        path_out = os.path.join(dir_name, f"{base_name.lower()}.aud")
        with open(path, "rb") as file_in:
            with open(path_out, "wb") as file_out:
                process_atx_to_aud(file_in, file_out, n_samples)

        print(f'atx file converted to {path_out}')
    elif extension == ".aud":
        visualization_dat = os.path.join(dir_name, "!!example_vis.dat")
        if not os.path.exists(visualization_dat):
            raise FileNotFoundError("Take some visualization file from the original game and rename it to \"!!example_vis.dat\"")
        destination = os.path.join(dir_name, f"{base_name}.dat")
        with open(visualization_dat, "rb") as file_in:
            with open(destination, "wb") as file_out:
                file_out.write(file_in.read())

        print(f'Visualization file "generated" into {destination}')
    elif extension == ".oma":
        base_name, _ = extract_n_samples_from_name(base_name)
        n_samples, bitrate = ffprobe_file(path, FileFormat.ATRAC3PLUS_OMA)
        path_out = os.path.join(dir_name, f"{base_name.lower()}.aud")
        with open(path, "rb") as file_in:
            with open(path_out, "wb") as file_out:
                process_oma_to_aud(file_in, file_out, n_samples, bitrate, os.path.getsize(path))

        print(f'oma file converted to {path_out}')
    else:
        if extension == ".wav":
            try:
                n_samples, _ = ffprobe_file(path, FileFormat.PCM_WAV)
                conversion_necessary = False
                temp_wave_path = path
                print("Renaming original file")
            except ValueError as e:
                print(f"Cannot use original WAV file: {e}. Will convert it to a compatible WAV file.")
                conversion_necessary = True
            except subprocess.CalledProcessError:
                raise ValueError("File incompatible with FFmpeg")
        else:
            conversion_necessary = True

        if conversion_necessary:
            print("Converting file")
            temp_wave_path = os.path.join(dir_name, f"{time.time()}.wav")
            ffmpeg_convert_file(path, temp_wave_path)
            n_samples, _ = ffprobe_file(temp_wave_path, FileFormat.PCM_WAV)

        final_wave_path = os.path.join(dir_name, f"!{base_name}_{n_samples}.wav")
        os.rename(temp_wave_path, final_wave_path)

        print(f'.wav file for UMD Stream Composer saved as {destination}')


if __name__ == "__main__":
    if len(sys.argv) == 1:
        input("""
NFS on PSP music toolkit.
Helps with adding/removing/editing music tracks for some Need for Speed games on the PSP.
You can drag and drop files onto the Python script file.
They will be processed automatically according to their type.
Tested with NFS Underground Rivals and NFS Carbon.
Should also work with NFS Most Wanted, NFS ProStreet and NFS Undercover.

Available functions:
* convert "trackinfo" file into a CSV file for editing
* convert a CSV file into "trackinfo" file
* convert any music file compatible with FFmpeg into .wav file for use with UMD Stream Composer
    * This also gives the file a special name that will be used later for converting .atx -> .aud
    * Result files can be used with SonicStage too. When converting with SonicStage the filename doesn't matter.
* convert .atx file generated by UMD Stream Composer into a .aud file that can be played by the game
* "generate" visualization data for an .aud file
    * for now this simply copies and renames a file from the original game
    * this is just to keep playback in game's track browser from glitching

Press enter to exit
        """.strip())
    else:
        for path in sys.argv[1:]:
            try:
                process_input_file(path)
            except Exception as e:
                print(f"Error processing {path}: {type(e).__name__}: {e}")
                input("Press enter to continue")
        input("All files processed, press enter")