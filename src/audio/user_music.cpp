/**
 * ReXGlue runtime - the player's own music over a title's soundtrack.
 *
 * MP3 / FLAC / WAV (dr_libs, public domain or MIT-0) and Ogg Vorbis
 * (stb_vorbis, public domain or MIT) files from a folder, shuffled, decoded on
 * a thread into a ring of 48 kHz stereo, mixed into the front left and right
 * of each guest output frame. While a track plays, the title is told its
 * background-music voice categories are turned down (see user_music.h).
 */

#include <rex/audio/user_music.h>

#include <algorithm>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/logging.h>

#define DR_MP3_IMPLEMENTATION
#define DR_FLAC_IMPLEMENTATION
#define DR_WAV_IMPLEMENTATION
#include "../../thirdparty/dr_libs/dr_flac.h"
#include "../../thirdparty/dr_libs/dr_mp3.h"
#include "../../thirdparty/dr_libs/dr_wav.h"
#undef STB_VORBIS_HEADER_ONLY
#define STB_VORBIS_NO_PUSHDATA_API
#include "../../thirdparty/stb/stb_vorbis.c"
// stb_vorbis leaves these short macro names defined.
#undef L
#undef C
#undef R

REXCVAR_DEFINE_STRING(user_music, "off", "Audio",
                      "Your own music: off, on (instead of the game's music), mix (over it) or "
                      "system (another player, e.g. Spotify, in place of the game's music while "
                      "it plays; the host app reports it, see ReXSetExternalMusic)");
REXCVAR_DEFINE_STRING(user_music_dir, "", "Audio",
                      "Folder of your music (mp3, ogg, flac, wav; subfolders too); default: the "
                      "music folder next to the executable");
REXCVAR_DECLARE(bool, xma_stream_log);

REXCVAR_DEFINE_STRING(title_music_bank, "music.xwb", "Audio",
                      "The title's song file (its songs make way for the player's music; event music and effects don't)");
REXCVAR_DEFINE_STRING(title_background_music_banks, "music - loading.xwb", "Audio",
                      "The title's other background music files, ';' between them (silent while the player's music "
                      "plays, but not songs: a stream starting from one doesn't count as the title's next song)");
REXCVAR_DEFINE_BOOL(user_music_follow_soundtrack, true, "Audio",
                    "Your music files play only while the title's own soundtrack would (its song stream is "
                    "decoding, muted under yours): not over its cutscenes, event intros and silences");
REXCVAR_DEFINE_STRING(title_event_music_banks, "awesome - *;shellhud strm.xwb", "Audio",
                      "The title's event music files (stingers, jingles; ';' between them, a trailing * matches "
                      "any ending): the player's music holds while one plays");
REXCVAR_DEFINE_INT32(user_music_volume, 100, "Audio", "Volume of your own music, 0-100");
REXCVAR_DEFINE_INT32(user_music_categories, 0, "Audio",
                     "Voice categories (a bit each) holding the title's background music, "
                     "reported turned down while your music plays");
REXCVAR_DEFINE_INT32(audio_muted_categories, 0, "Audio",
                     "Debug: voice categories (a bit each) always reported turned down");

namespace rex::audio {
bool EventMusicPlaying();      // below
bool TitleSoundtrackPlaying();  // below

namespace {

constexpr uint32_t kRate = 48000;
constexpr uint32_t kRingFrames = kRate;  // 1 s of stereo

// A path as UTF-8 (std::filesystem::path::string() throws for names outside
// the Windows code page).
std::string U8(const std::filesystem::path& path) {
  auto u8 = path.u8string();
  return std::string(u8.begin(), u8.end());
}

struct Track {
  std::filesystem::path path;
  std::string title;  // read when it first plays
  bool failed = false;
};

// Streams a file as interleaved stereo float at its own rate.
class Decoder {
 public:
  ~Decoder() { Close(); }

  bool Open(const std::filesystem::path& path) {
    std::string ext = U8(path.extension());
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    const std::wstring file = path.wstring();
    if (ext == ".mp3" && drmp3_init_file_w(&mp3_, file.c_str(), nullptr)) {
      kind_ = Kind::kMp3;
      channels_ = mp3_.channels;
      rate_ = mp3_.sampleRate;
    } else if (ext == ".flac" && (flac_ = drflac_open_file_w(file.c_str(), nullptr))) {
      kind_ = Kind::kFlac;
      channels_ = flac_->channels;
      rate_ = flac_->sampleRate;
    } else if (ext == ".wav" && drwav_init_file_w(&wav_, file.c_str(), nullptr)) {
      kind_ = Kind::kWav;
      channels_ = wav_.channels;
      rate_ = wav_.sampleRate;
    } else if (ext == ".ogg" && (vorbis_file_ = _wfopen(file.c_str(), L"rb")) &&
               (vorbis_ = stb_vorbis_open_file(vorbis_file_, 1, nullptr, nullptr))) {
      kind_ = Kind::kVorbis;
      stb_vorbis_info info = stb_vorbis_get_info(vorbis_);
      channels_ = uint32_t(info.channels);
      rate_ = info.sample_rate;
    } else {
      // An .ogg that stb_vorbis couldn't open is already closed by it (it was
      // given the file to close); closing it again could crash.
      vorbis_file_ = nullptr;
      return false;
    }
    if (!channels_ || !rate_) {
      Close();
      return false;
    }
    return true;
  }

  uint32_t rate() const { return rate_; }

  // Up to `frames` stereo frames into `out`; 0 at the end.
  uint32_t Read(float* out, uint32_t frames) {
    scratch_.resize(size_t(frames) * channels_);
    uint64_t got = 0;
    switch (kind_) {
      case Kind::kMp3: got = drmp3_read_pcm_frames_f32(&mp3_, frames, scratch_.data()); break;
      case Kind::kFlac: got = drflac_read_pcm_frames_f32(flac_, frames, scratch_.data()); break;
      case Kind::kWav: got = drwav_read_pcm_frames_f32(&wav_, frames, scratch_.data()); break;
      case Kind::kVorbis:
        got = uint64_t(stb_vorbis_get_samples_float_interleaved(vorbis_, int(channels_), scratch_.data(),
                                                                int(frames * channels_)));
        break;
      default: return 0;
    }
    for (uint64_t i = 0; i < got; ++i) {
      out[2 * i] = scratch_[i * channels_];
      out[2 * i + 1] = scratch_[i * channels_ + (channels_ > 1 ? 1 : 0)];
    }
    return uint32_t(got);
  }

 private:
  enum class Kind { kNone, kMp3, kFlac, kWav, kVorbis };
  void Close() {
    switch (kind_) {
      case Kind::kMp3: drmp3_uninit(&mp3_); break;
      case Kind::kFlac: drflac_close(flac_); break;
      case Kind::kWav: drwav_uninit(&wav_); break;
      case Kind::kVorbis: stb_vorbis_close(vorbis_); break;  // closes vorbis_file_ too
      default: break;
    }
    kind_ = Kind::kNone;
  }
  Kind kind_ = Kind::kNone;
  drmp3 mp3_{};
  drflac* flac_ = nullptr;
  drwav wav_{};
  stb_vorbis* vorbis_ = nullptr;
  std::FILE* vorbis_file_ = nullptr;
  uint32_t channels_ = 0, rate_ = 0;
  std::vector<float> scratch_;
};

// "Artist - Title" from an ID3v2 tag (TPE1 / TIT2, ISO-8859-1 or UTF-8 text),
// else the file name.
// TITLE= / ARTIST= from Vorbis comments (Ogg Vorbis and FLAC).
void VorbisComment(const char* comment, size_t length, std::string& title, std::string& artist) {
  std::string c(comment, length);
  auto eq = c.find('=');
  if (eq == std::string::npos) return;
  std::string key = c.substr(0, eq);
  std::transform(key.begin(), key.end(), key.begin(), [](unsigned char ch) { return char(std::toupper(ch)); });
  if (key == "TITLE" && title.empty()) title = c.substr(eq + 1);
  if (key == "ARTIST" && artist.empty()) artist = c.substr(eq + 1);
}

std::string TrackTitle(const std::filesystem::path& path) {
  std::string title, artist;
  const std::string ext = U8(path.extension());
  if (ext == ".ogg" || ext == ".OGG") {
    if (std::FILE* f = _wfopen(path.wstring().c_str(), L"rb")) {
      if (stb_vorbis* v = stb_vorbis_open_file(f, 1, nullptr, nullptr)) {
        stb_vorbis_comment comments = stb_vorbis_get_comment(v);
        for (int i = 0; i < comments.comment_list_length; ++i) {
          VorbisComment(comments.comment_list[i], std::strlen(comments.comment_list[i]), title, artist);
        }
        stb_vorbis_close(v);
      }  // else stb_vorbis has closed f already (it was given it to close)
    }
  } else if (ext == ".flac" || ext == ".FLAC") {
    struct Tags {
      std::string title, artist;
    } tags;
    drflac* flac = drflac_open_file_with_metadata_w(
        path.wstring().c_str(),
        [](void* user, drflac_metadata* meta) {
          if (meta->type != DRFLAC_METADATA_BLOCK_TYPE_VORBIS_COMMENT) return;
          auto* t = static_cast<Tags*>(user);
          drflac_vorbis_comment_iterator it;
          drflac_init_vorbis_comment_iterator(&it, meta->data.vorbis_comment.commentCount,
                                              meta->data.vorbis_comment.pComments);
          drflac_uint32 length;
          while (const char* c = drflac_next_vorbis_comment(&it, &length)) {
            VorbisComment(c, length, t->title, t->artist);
          }
        },
        &tags, nullptr);
    if (flac) drflac_close(flac);
    title = tags.title;
    artist = tags.artist;
  }
  if (std::FILE* f = _wfopen(path.wstring().c_str(), L"rb")) {
    unsigned char h[10];
    if (std::fread(h, 1, 10, f) == 10 && !std::memcmp(h, "ID3", 3) && h[3] >= 3) {
      uint32_t size = (h[6] << 21) | (h[7] << 14) | (h[8] << 7) | h[9];
      std::vector<unsigned char> tag(std::min<uint32_t>(size, 1 << 20));
      tag.resize(std::fread(tag.data(), 1, tag.size(), f));
      for (size_t p = 0; p + 10 <= tag.size();) {
        const unsigned char* fr = &tag[p];
        if (!fr[0]) break;
        uint32_t fs = h[3] == 4 ? (fr[4] << 21) | (fr[5] << 14) | (fr[6] << 7) | fr[7]
                                : (fr[4] << 24) | (fr[5] << 16) | (fr[6] << 8) | fr[7];
        if (p + 10 + fs > tag.size()) break;
        if ((!std::memcmp(fr, "TIT2", 4) || !std::memcmp(fr, "TPE1", 4)) && fs > 1 &&
            (fr[10] == 0 || fr[10] == 3)) {
          std::string text(reinterpret_cast<const char*>(fr + 11), fs - 1);
          text.erase(std::find(text.begin(), text.end(), '\0'), text.end());
          (fr[1] == 'I' ? title : artist) = text;
        }
        p += 10 + fs;
      }
    }
    std::fclose(f);
  }
  if (title.empty()) return U8(path.stem());
  return artist.empty() ? title : artist + " - " + title;
}

class Player {
 public:
  // Starts the player's thread (once); it scans the folder itself, so the
  // callers (the audio thread, the title's audio and XMP calls) never wait.
  void EnsureStarted() {
    std::call_once(started_, [this] {
      ring_.resize(size_t(kRingFrames) * 2);
      std::thread([this] { Run(); }).detach();
    });
  }

  bool Playing() const { return playing_.load(std::memory_order_relaxed); }

  void SetPaused(bool paused) { paused_.store(paused, std::memory_order_relaxed); }
  bool Paused() const { return paused_.load(std::memory_order_relaxed); }

  void Mix(float* frame, uint32_t samples) {
    // (Holding for an event's music, or while the title's soundtrack is
    // silent: the ring keeps its place.)
    const int hold = Paused()                ? 1
                     : EventMusicPlaying()   ? 2
                     : (REXCVAR_GET(user_music_follow_soundtrack) && !TitleSoundtrackPlaying()) ? 3
                                             : 0;
    if (static std::atomic<int> logged{-1}; logged.exchange(hold) != hold && Playing()) {
      static const char* const kWhy[] = {"plays", "holds (the title owns the music: not in the game, or a cutscene)",
                                         "holds (an event's music)", "holds (the title's soundtrack is silent)"};
      REXLOG_INFO("user_music: your music {}", kWhy[hold]);
    }
    if (hold) return;
    const float volume = std::clamp(REXCVAR_GET(user_music_volume), 0, 100) / 100.0f;
    std::lock_guard<std::mutex> lock(mutex_);
    for (uint32_t i = 0; i < samples; ++i) {
      if (!filled_) break;
      for (uint32_t c = 0; c < 2; ++c) {
        uint32_t& word = reinterpret_cast<uint32_t*>(frame)[c * samples + i];
        uint32_t host = __builtin_bswap32(word);
        float value;
        std::memcpy(&value, &host, 4);
        value += ring_[read_ * 2 + c] * volume;
        std::memcpy(&host, &value, 4);
        word = __builtin_bswap32(host);
      }
      read_ = (read_ + 1) % kRingFrames;
      --filled_;
    }
  }

  std::string NowPlaying(uint32_t* counter) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (counter) *counter = track_counter_;
    return Playing() ? now_playing_ : std::string();
  }

  // --user_music=mix: the title's song between two of the player's tracks.
  // GameTurn(): the title's music plays; TakeGameTurnStart(): once per turn,
  // for the host app to start the title's next song.
  bool GameTurn() const { return game_turn_.load(std::memory_order_relaxed); }
  bool TakeGameTurnStart() { return game_turn_start_pending_.exchange(false); }
  // "Next" during the title's turn: the player's next track.
  void EndGameTurn() { game_turn_.store(false, std::memory_order_relaxed); }
  // The title started a new song: after the one it began the turn with (the
  // host's skip), it ends the title's turn.
  void TitleSongStarted() {
    if (GameTurn() && std::chrono::steady_clock::now() - game_turn_since_ > std::chrono::seconds(8)) {
      game_turn_.store(false, std::memory_order_relaxed);
    }
  }

  // delta 1: the next track; -1: the previous one.
  void Skip(int32_t delta) {
    skip_delta_.store(delta < 0 ? -1 : 1, std::memory_order_relaxed);
    skip_.store(true, std::memory_order_relaxed);
  }

 private:
  void Scan() {
    std::filesystem::path dir = REXCVAR_GET(user_music_dir).empty()
                                    ? rex::filesystem::GetExecutableFolder() / "music"
                                    : std::filesystem::path(REXCVAR_GET(user_music_dir));
    std::error_code ec;
    for (auto it = std::filesystem::recursive_directory_iterator(dir, ec);
         !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
      std::string ext = U8(it->path().extension());
      std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
      std::error_code file_ec;
      if (it->is_regular_file(file_ec) && (ext == ".mp3" || ext == ".ogg" || ext == ".flac" || ext == ".wav")) {
        tracks_.push_back({it->path(), {}});
      }
    }
    REXLOG_INFO("user_music: {} tracks in {}", tracks_.size(), U8(dir));
    track_count_.store(int(tracks_.size()));
    std::shuffle(tracks_.begin(), tracks_.end(), std::mt19937(std::random_device{}()));
  }

  void Run() {
    Scan();
    if (tracks_.empty()) return;
    std::vector<float> chunk;
    size_t failed_in_a_row = 0;
    for (size_t t = 0;; t = next_track_) {
      // The track after this one, unless a skip says otherwise.
      next_track_ = (t + 1) % tracks_.size();
      Decoder decoder;
      // The first 4096 frames now: a file that opens but holds no audio counts
      // as undecodable (else an all-empty folder would spin, "playing" each).
      std::vector<float> first(2 + 4096 * 2, 0.0f);
      uint32_t first_got = 0;
      const bool opened = decoder.Open(tracks_[t].path);
      if (opened) {
        first_got = decoder.Read(first.data() + 2, 4096);
        first.resize(2 + size_t(first_got) * 2);
      }
      if (!opened || !first_got) {
        if (!tracks_[t].failed) {
          tracks_[t].failed = true;
          REXLOG_WARN("user_music: can't decode {}", U8(tracks_[t].path));
        }
        if (++failed_in_a_row >= tracks_.size()) {
          // Nothing playable: stay quiet, and look again now and then.
          playing_.store(false, std::memory_order_relaxed);
          failed_in_a_row = 0;
          std::this_thread::sleep_for(std::chrono::seconds(10));
        }
        continue;
      }
      failed_in_a_row = 0;
      // Tags read as the track starts (not for the whole folder up front).
      if (tracks_[t].title.empty()) tracks_[t].title = TrackTitle(tracks_[t].path);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        now_playing_ = tracks_[t].title;
        ++track_counter_;
      }
      playing_.store(true, std::memory_order_relaxed);
      skip_.store(false, std::memory_order_relaxed);
      track_start_ = std::chrono::steady_clock::now();
      REXLOG_INFO("user_music: playing {}", tracks_[t].title);
      // Linear resampling to 48 kHz into the ring, a chunk of the file at a
      // time (`chunk` keeps one frame of the previous chunk to interpolate
      // across the boundary).
      const double step = double(decoder.rate()) / kRate;
      double pos = 0;  // in frames of `chunk`
      chunk.swap(first);  // a silent frame, then the start of the track
      bool more = true;
      while (!skip_.load(std::memory_order_relaxed)) {
        size_t frames = chunk.size() / 2;
        if (more && pos + 1 >= double(frames)) {
          // Keep the last frame, append the next 4096 of the file.
          std::vector<float> next(2 + 4096 * 2);
          next[0] = chunk[(frames - 1) * 2];
          next[1] = chunk[(frames - 1) * 2 + 1];
          uint32_t got = decoder.Read(next.data() + 2, 4096);
          next.resize(2 + size_t(got) * 2);
          pos -= double(frames - 1);
          chunk.swap(next);
          more = got > 0;
          continue;
        }
        if (!more && pos + 1 >= double(frames)) break;  // the end of the track
        bool full;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          while (filled_ < kRingFrames && pos + 1 < double(frames)) {
            size_t i = size_t(pos);
            float k = float(pos - double(i));
            uint32_t w = (read_ + filled_) % kRingFrames;
            for (uint32_t c = 0; c < 2; ++c) {
              ring_[w * 2 + c] = chunk[i * 2 + c] + (chunk[(i + 1) * 2 + c] - chunk[i * 2 + c]) * k;
            }
            ++filled_;
            pos += step;
          }
          full = filled_ >= kRingFrames;
        }
        if (full) std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      const bool skipped = skip_.load(std::memory_order_relaxed);
      if (MixMode() && !(skipped && skip_delta_.load(std::memory_order_relaxed) < 0)) {
        // The track ended (or "next"): the title's song next.
        {
          std::lock_guard<std::mutex> lock(mutex_);
          filled_ = 0;
        }
        game_turn_since_ = std::chrono::steady_clock::now();
        game_turn_.store(true, std::memory_order_relaxed);
        game_turn_start_pending_.store(true);
        playing_.store(false, std::memory_order_relaxed);
        REXLOG_INFO("user_music: the title's song next (mix)");
        NotifyTitleMusicChanged();
        // Until the title's next song starts, "next", or 4 minutes (a song;
        // the title can reuse a stream without the start being seen).
        while (GameTurn() && MixMode() &&
               std::chrono::steady_clock::now() - game_turn_since_ < std::chrono::minutes(4)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        game_turn_.store(false, std::memory_order_relaxed);
        continue;
      }
      if (skipped) {
        std::lock_guard<std::mutex> lock(mutex_);
        filled_ = 0;  // drop what was queued of the skipped track
        if (skip_delta_.load(std::memory_order_relaxed) < 0) {
          // Back to the start of this track after 3 s of it, else the one before.
          const double played = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                               track_start_).count();
          next_track_ = played > 3.0 ? t : (t + tracks_.size() - 1) % tracks_.size();
        }
      }
    }
  }

  std::once_flag started_;
  size_t next_track_ = 0;
  std::chrono::steady_clock::time_point track_start_;
  std::vector<Track> tracks_;
  std::mutex mutex_;
  std::vector<float> ring_;
  uint32_t read_ = 0, filled_ = 0;
  std::string now_playing_;
  uint32_t track_counter_ = 0;
  std::atomic<bool> playing_{false}, skip_{false}, paused_{false};
  std::atomic<int32_t> skip_delta_{1};
  std::atomic<bool> game_turn_{false}, game_turn_start_pending_{false};
  std::atomic<int> track_count_{-1};  // -1 until the folder is scanned

 public:
  int TrackCount() const { return track_count_.load(); }

 private:
  std::chrono::steady_clock::time_point game_turn_since_;
};

Player& GetPlayer() {
  // Never destroyed: its thread (detached) runs until the process ends, and
  // static destruction at exit would pull its tracks and buffers away from it.
  static Player* player = new Player;
  return *player;
}

// --user_music as a number, kept by a change callback: it's read on the audio,
// XMA and guest threads while the UI thread may change the string.
enum Mode : int { kOff, kOn, kMix, kSystem };
std::atomic<int> g_mode{-1};

int ParseMode(std::string_view v) {
  return v == "on" ? kOn : v == "mix" ? kMix : v == "system" ? kSystem : kOff;
}

int CurrentMode() {
  static std::once_flag registered;
  std::call_once(registered, [] {
    rex::cvar::RegisterChangeCallback("user_music", [](std::string_view, std::string_view value) {
      g_mode.store(ParseMode(value), std::memory_order_relaxed);
      NotifyTitleMusicChanged();
    });
    g_mode.store(ParseMode(REXCVAR_GET(user_music)), std::memory_order_relaxed);
  });
  return g_mode.load(std::memory_order_relaxed);
}

bool Enabled() {
  const int mode = CurrentMode();
  return mode == kOn || mode == kMix;
}

// --user_music=system: another player's state, reported by the host app.
// Also mix mode with nothing in the music folder: the other player takes the
// turns (the host app pauses it for the game's songs).
bool System() {
  const int mode = CurrentMode();
  if (mode == kSystem) return true;
  if (mode != kMix) return false;
  GetPlayer().EnsureStarted();
  return GetPlayer().TrackCount() == 0;
}
std::mutex external_mutex;
bool external_playing = false;
std::string external_text;
uint32_t external_counter = 0;

}  // namespace

bool MixMode() { return CurrentMode() == kMix; }

uint32_t GetMutedVoiceCategories() {
  uint32_t muted = uint32_t(REXCVAR_GET(audio_muted_categories));
  if (Enabled()) {
    GetPlayer().EnsureStarted();
    if (GetPlayer().Playing() && !GetPlayer().Paused()) {
      muted |= uint32_t(REXCVAR_GET(user_music_categories));
    }
  }
  // The player's own music starting or running out (checked each audio frame).
  if (Enabled()) NotifyTitleMusicChanged();
  return muted;
}

void MixUserMusic(float* frame, uint32_t samples) {
  if (!Enabled()) return;
  GetPlayer().EnsureStarted();
  GetPlayer().Mix(frame, samples);
}

std::string UserMusicNowPlaying(uint32_t* track_counter) {
  if (System()) {
    std::lock_guard<std::mutex> lock(external_mutex);
    if (track_counter) *track_counter = external_counter;
    return external_playing ? external_text : std::string();
  }
  if (!Enabled()) return {};
  return GetPlayer().NowPlaying(track_counter);
}

bool UserMusicActive() {
  if (System()) {
    std::lock_guard<std::mutex> lock(external_mutex);
    return external_playing;
  }
  if (!Enabled()) return false;
  GetPlayer().EnsureStarted();
  return GetPlayer().Playing();
}

// The title has taken music playback over (XMPSetPlaybackController: its
// cutscenes and scripted songs): the player's music steps aside.
// The host app adds the title's in-engine cutscenes (Amped 3 plays their music
// without telling XMP).
std::atomic<bool> title_owns_music{false}, xmp_override{false}, host_override{false};

void ApplyTitleMusic() {
  title_owns_music = xmp_override || host_override;
  GetPlayer().SetPaused(title_owns_music);
  NotifyTitleMusicChanged();
}

void PauseUserMusic(bool paused) {
  // Whatever the mode: a pause from the title (XMP) mustn't be lost by
  // switching music off and back on in between.
  xmp_override = paused;
  ApplyTitleMusic();
}

bool UserMusicPaused() { return Enabled() && GetPlayer().Paused(); }

bool UserMusicReplacingTitleMusic() {
  if (System()) {
    // Another player chosen: the title's songs are silent whenever the title
    // doesn't own music (in game, outside its cutscenes and event music),
    // whether or not that player is playing right now. Waiting for it to be
    // seen playing left both playing (a song already started, or a player
    // Windows didn't report). The host app pauses the player while the
    // title owns music (ReXTitleOwnsMusic).
    return !title_owns_music;
  }
  return Enabled() && GetPlayer().Playing() && !GetPlayer().Paused();
}

// Where reads from the title's song bank (kind 1) and its other background
// music (kind 2) landed: physical [start, end).
struct BankRead {
  uint32_t start, end;
  uint8_t kind;
};
std::mutex music_reads_mutex;
std::vector<BankRead> music_reads;
std::atomic<bool> music_reads_seen{false};

namespace {
std::string Lower(std::string s) {
  for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

uint8_t BankKind(const std::string& file_name) {
  const std::string name = Lower(file_name);
  const std::string bank = Lower(REXCVAR_GET(title_music_bank));
  if (!bank.empty() && name == bank) return 1;
  // One of a ';' list (a trailing * matches any ending).
  auto in_list = [&](const std::string& list) {
    for (size_t pos = 0; pos < list.size();) {
      size_t next = list.find(';', pos);
      if (next == std::string::npos) next = list.size();
      std::string item = list.substr(pos, next - pos);
      pos = next + 1;
      if (item.empty()) continue;
      if (item.back() == '*' ? name.compare(0, item.size() - 1, item, 0, item.size() - 1) == 0 : name == item) {
        return true;
      }
    }
    return false;
  };
  if (in_list(Lower(REXCVAR_GET(title_background_music_banks)))) return 2;
  if (in_list(Lower(REXCVAR_GET(title_event_music_banks)))) return 3;
  return 0;
}

// How much of [address, address + length) holds data of this kind.
uint64_t Covered(uint32_t address, uint32_t length, uint8_t kind) {
  const uint32_t end = address + length;
  std::vector<std::pair<uint32_t, uint32_t>> parts;
  {
    std::lock_guard<std::mutex> lock(music_reads_mutex);
    for (const auto& r : music_reads) {
      if (r.kind == kind && r.start < end && address < r.end) {
        parts.emplace_back(std::max(r.start, address), std::min(r.end, end));
      }
    }
  }
  std::sort(parts.begin(), parts.end());
  uint64_t covered = 0;
  uint32_t reach = address;
  for (const auto& [a, b] : parts) {
    const uint32_t from = std::max(a, reach);
    if (b > from) covered += b - from;
    reach = std::max(reach, b);
  }
  return covered;
}
}  // namespace

void NoteTitleFileRead(uint32_t physical_address, uint32_t length, const std::string& file_name) {
  if (REXCVAR_GET(xma_stream_log) && length >= 0x10000 && physical_address != UINT32_MAX) {
    REXLOG_INFO("user_music: read {} bytes of {} to {:08X}", length, file_name, physical_address);
  }
  if (!length || physical_address == UINT32_MAX) return;
  const uint8_t kind = BankKind(file_name);
  const uint32_t start = physical_address, end = physical_address + length;
  std::lock_guard<std::mutex> lock(music_reads_mutex);
  // Whatever overlaps this read holds this file's data now: the ranges cut.
  std::vector<BankRead> kept;
  kept.reserve(music_reads.size() + 1);
  for (const auto& r : music_reads) {
    if (r.end <= start || end <= r.start) {
      kept.push_back(r);
      continue;
    }
    if (r.start < start) kept.push_back({r.start, start, r.kind});
    if (end < r.end) kept.push_back({end, r.end, r.kind});
    if (kind != r.kind && REXCVAR_GET(xma_stream_log)) {
      REXLOG_INFO("user_music: {} read to {:08X}-{:08X} replaces {} data at {:08X}-{:08X}", file_name, start, end,
                  r.kind == 1 ? "song" : "background music", r.start, r.end);
    }
  }
  music_reads.swap(kept);
  if (!kind) return;
  // Joined to the read just before it (page by page reads of one buffer).
  if (!music_reads.empty() && music_reads.back().end == start && music_reads.back().kind == kind) {
    music_reads.back().end = end;
  } else {
    music_reads.push_back({start, end, kind});
    if (music_reads.size() > 4096) music_reads.erase(music_reads.begin());
  }
  if (kind == 1 && !music_reads_seen.exchange(true)) {
    REXLOG_INFO("user_music: the title's songs are read from {} (only they make way for the player's music)",
                file_name);
  }
}

// Most of the buffer must be song data: the title reads each song buffer from
// a few packets in (the start carried over from the previous one), and other
// files' reads cut the song ranges they overwrite but can leave slivers next
// to them (the intro story's narration, Grover.xwb, landing where songs were
// had every second buffer muted for touching one).
bool IsTitleMusicBuffer(uint32_t physical_address, uint32_t length) {
  return Covered(physical_address, length, 1) * 2 >= length;
}

bool IsTitleBackgroundMusicBuffer(uint32_t physical_address, uint32_t length) {
  return Covered(physical_address, length, 2) * 2 >= length;
}

bool TitleMusicReadsSeen() { return music_reads_seen.load(); }

bool IsTitleEventMusicBuffer(uint32_t physical_address, uint32_t length) {
  return Covered(physical_address, length, 3) * 2 >= length;
}

std::atomic<int64_t> last_event_chunk_ms{0};
int64_t NowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
void NoteEventMusicChunk() { last_event_chunk_ms.store(NowMs(), std::memory_order_relaxed); }
std::atomic<int64_t> last_song_chunk_ms{-100000};
void NoteTitleSongChunk() { last_song_chunk_ms.store(NowMs(), std::memory_order_relaxed); }
// The title's soundtrack was playing (decoding) in the last 2 s (the gap
// between two of its songs is shorter).
bool TitleSoundtrackPlaying() { return NowMs() - last_song_chunk_ms.load(std::memory_order_relaxed) < 2000; }
// An event's music was decoded in the last second.
bool EventMusicPlaying() { return NowMs() - last_event_chunk_ms.load(std::memory_order_relaxed) < 1000; }

std::atomic<uint32_t> title_song_starts{0};

void TitleMusicStreamStarted() {
  ++title_song_starts;
  if (MixMode()) GetPlayer().TitleSongStarted();
}

std::atomic<void (*)(bool)> title_music_callback{nullptr};
std::atomic<int> title_music_told{-1};

void SetTitleMusicChangedCallback(void (*callback)(bool replacing)) { title_music_callback = callback; }

void NotifyTitleMusicChanged() {
  const int replacing = UserMusicReplacingTitleMusic() ? 1 : 0;
  if (title_music_told.exchange(replacing) == replacing) return;
  REXLOG_INFO("user_music: title music {}", replacing ? "replaced by the player's" : "back to the title's");
  if (auto* callback = title_music_callback.load()) callback(replacing != 0);
}

void SkipUserMusicTrack(int32_t delta) {
  if (Enabled()) GetPlayer().Skip(delta);
}

}  // namespace rex::audio

extern "C" uint32_t ReXUserMusicNowPlaying(char* out, uint32_t out_size) {
  uint32_t counter = 0;
  const std::string text = rex::audio::UserMusicNowPlaying(&counter);
  if (out && out_size) {
    const size_t n = std::min<size_t>(text.size(), out_size - 1);
    std::memcpy(out, text.data(), n);
    out[n] = '\0';
  }
  return counter;
}

extern "C" void ReXUserMusicSkip(int32_t delta) { rex::audio::SkipUserMusicTrack(delta); }

extern "C" void ReXSetExternalMusic(int32_t playing, const char* now_playing) {
  {
    std::lock_guard<std::mutex> lock(rex::audio::external_mutex);
    const std::string text = now_playing ? now_playing : "";
    if (playing && (text != rex::audio::external_text || !rex::audio::external_playing)) {
      ++rex::audio::external_counter;  // a new track (or playing again): the popup
    }
    rex::audio::external_playing = playing != 0;
    rex::audio::external_text = text;
  }
  rex::audio::NotifyTitleMusicChanged();
}

extern "C" int32_t ReXUserMusicPlaying() {
  return rex::audio::UserMusicActive() && !rex::audio::UserMusicPaused();
}

namespace rex::audio {

}  // namespace rex::audio

// For the host app (--user_music=system): whether the title owns music
// playback now, so it pauses the other player and resumes it afterwards.
extern "C" int32_t ReXTitleOwnsMusic() { return rex::audio::title_owns_music ? 1 : 0; }
// The host app: the title's own music is playing although XMP wasn't told
// (e.g. an in-engine cutscene); the player's music steps aside meanwhile.
extern "C" void ReXSetTitleMusicOverride(int32_t on) {
  rex::audio::host_override = on != 0;
  rex::audio::ApplyTitleMusic();
}

// --user_music=mix: 1 once when the title's turn begins (the host app starts
// the title's next song then).
extern "C" int32_t ReXUserMusicTakeGameTurn() {
  return rex::audio::MixMode() && rex::audio::GetPlayer().TakeGameTurnStart() ? 1 : 0;
}

// --user_music=mix: whether the title's song plays now (its turn), and "next"
// during it (the player's next track).
extern "C" int32_t ReXUserMusicGameTurn() {
  return rex::audio::MixMode() && rex::audio::GetPlayer().GameTurn() ? 1 : 0;
}
extern "C" void ReXUserMusicEndGameTurn() { rex::audio::GetPlayer().EndGameTurn(); }

// How many of the title's songs have started (the host app's mix mode with
// another player: the game's turn ends when its next song starts).
extern "C" uint32_t ReXTitleSongStarts() { return rex::audio::title_song_starts.load(); }
// The music folder's track count (-1: not scanned yet).
extern "C" int32_t ReXUserMusicTrackCount() {
  rex::audio::GetPlayer().EnsureStarted();
  return rex::audio::GetPlayer().TrackCount();
}
