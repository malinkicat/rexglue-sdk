/**
 * ReXGlue runtime - the player's own music over a title's soundtrack.
 *
 * The music is mixed into the output frames. In place of the title's music
 * (user_music=on), the title's streamed music is silenced at the XMA decoder.
 * (A real console turned the title's background-music voice category down;
 * user_music_categories does that too, but in Amped 3 category 0 holds all of
 * its audio, so it's off by default.)
 */

#pragma once

#include <cstdint>
#include <string>

namespace rex::audio {

// Voice categories reported to the title as turned down (a bit each).
uint32_t GetMutedVoiceCategories();

// Mixes the player's music (if any is playing) into a guest output frame: 6
// channels of `samples` big-endian floats each, one after the other, 48 kHz.
void MixUserMusic(float* frame, uint32_t samples);

// "Artist - Title" of the track playing now and a counter that changes with
// each new track (for a now-playing popup); empty when none plays.
std::string UserMusicNowPlaying(uint32_t* track_counter = nullptr);

// Whether the player's music is on and has a track (playing or paused).
bool UserMusicActive();

// Paused by the title (it took playback control, or asked to pause): silent,
// and the title's music categories are no longer turned down.
void PauseUserMusic(bool paused);
bool UserMusicPaused();

// Whether the title's own music streams should be silent: the player's music
// plays in place of it (user_music=on, not paused).
bool UserMusicReplacingTitleMusic();

// Called (with UserMusicReplacingTitleMusic()) whenever that changes: the
// mode, the other player starting or stopping, a cutscene taking music over.
// XMP tells the title, which only asks again when told (Amped 3 otherwise
// keeps its soundtrack silent after the player's music stops).
void SetTitleMusicChangedCallback(void (*callback)(bool replacing));
void NotifyTitleMusicChanged();

// The title started a new music stream (--user_music=mix: ends its turn).
void TitleMusicStreamStarted();

// The title's own songs (the ones the player picks and skips: Amped 3's
// audiobin\Music.xwb, --title_music_bank): the kernel's file reads note where
// each read lands (physical addresses), so an XMA stream decoding from there
// is one of those songs - not event music or effects from other banks.
void NoteTitleFileRead(uint32_t physical_address, uint32_t length, const std::string& file_name);
bool IsTitleMusicBuffer(uint32_t physical_address, uint32_t length);
// The title's other background music (--title_background_music_banks, e.g. the
// loading screens'): silent while the player's music plays, but not songs.
bool IsTitleBackgroundMusicBuffer(uint32_t physical_address, uint32_t length);
// The title's event music (stingers, jingles: --title_event_music_banks):
// while a buffer of it is decoded, the player's music files hold (they don't
// play over it). NoteEventMusicChunk() per decoded chunk of such a buffer.
bool IsTitleEventMusicBuffer(uint32_t physical_address, uint32_t length);
void NoteEventMusicChunk();
// A chunk of one of the title's songs was decoded (muted or not): with
// --user_music_follow_soundtrack the player's music files play only while the
// title's own soundtrack would (not over its silences: cutscenes, event
// intros, the front end without a song).
void NoteTitleSongChunk();
// Reads from the song bank have been seen (else the song test falls back to
// the stream's shape).
bool TitleMusicReadsSeen();
bool MixMode();

// Skip to the next track.
void SkipUserMusicTrack(int32_t delta = 1);

}  // namespace rex::audio

// The same as plain C exports, for callers built against an SDK import library
// that predates them (they look these up in rexruntime.dll by name).
extern "C" {
// Copies "Artist - Title" (UTF-8, NUL-terminated, cut to fit) into `out`;
// returns the track counter (0 = nothing has played yet). An empty string
// when none plays.
uint32_t ReXUserMusicNowPlaying(char* out, uint32_t out_size);
// --user_music=mix: 1 once when the title's song is next (start it).
int32_t ReXUserMusicTakeGameTurn();
// Next track (delta 1), or previous (delta -1): the start of this track if
// it has played over 3 s, else the previous track.
void ReXUserMusicSkip(int32_t delta);
// Whether the player's music is playing (on, has a track, not paused; with
// user_music=system, whether the other player is).
int32_t ReXUserMusicPlaying();
// user_music=system: the host app reports another player (e.g. Spotify via
// the OS media session): playing or not, and "Artist - Title".
void ReXSetExternalMusic(int32_t playing, const char* now_playing);
// Whether the title owns music playback now (XMPSetPlaybackController: its
// cutscenes): user_music=system's host app pauses the other player meanwhile.
int32_t ReXTitleOwnsMusic();
// The host app: the title's own music plays without XMP being told (e.g. an
// in-engine cutscene); the player's music steps aside while on.
void ReXSetTitleMusicOverride(int32_t on);
}
