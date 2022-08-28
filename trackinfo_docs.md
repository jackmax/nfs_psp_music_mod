# `trackinfo` file documentation
The file has the following structure:
* 1 `uint8_t` - total number of tracks (referred to from now on as N)
* (N * 18) bytes with descriptions of each track
* Rest of file is storage of string data

Each track description contains 5 strings, always in that order:
`Filename`, `Artist`, `Album`, `Title_line1`, `Title_line2`.
*   `Filename` is the base name for the files associated with the track. It is not case-sensitive.
    * `eatrax/{Filename}.aud` is the audio.
    * `eatrax/{Filename}.pmf` is the music video (if present).
    * `eavis/dat/{Filename}.dat` is the visualizer data.
*   Title can be split into two lines so long titles display nicely in the menu. Later games don't seem to use this though.

The structure of a track description is as follows:
* 5 bytes - lengths (in characters) of each string.
  * Seem to be unused, strings are null-terminated so not necessary either.
  * Giving the wrong length doesn't crash the game, full string still displayed. I suggest just give the right length,
  you're not trying to buffer overflow your own console :P
* 1 byte (boolean) - is whether the track has an associated music video. If file not found, playing the song in the menu crashes the game.
* 1 byte (boolean) - I don't know what it means. Seems to be always set to 1. Setting to 0 doesn't seem to change anything (tested on real console too).
* 1 byte (possibly boolean) - probably unused. NFS:UR always sets the corresponding field to 1 in software.
  Other games have 1 in this field in the file itself.
* 5 `uint16_t` values - "pointer" to each string. Offset into the file (minus one byte) where the string is stored.

In the string storage, every string is encoded as UTF-16 LE and terminated with two 0x00 bytes. They are simply placed there
one after another. It seems that the same string can be referenced multiple times without issues (saves space if you have multiple
tracks from the same artist or album).
