/**
******************************************************************************
* Xenia : Xbox 360 Emulator Research Project                                 *
******************************************************************************
* Copyright 2021 Ben Vanik. All rights reserved.                             *
* Released under the BSD license - see LICENSE in the root for more details. *
******************************************************************************
*
* @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
*/

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include <rex/audio/user_music.h>
#include <rex/audio/xma/context.h>
#include <rex/cvar.h>
#include <rex/logging.h>

REXCVAR_DEFINE_BOOL(xma_stream_log, false, "Audio",
                    "Debug: log each XMA stream's shape (stereo, packets, muted) when it changes, and song starts");
REXCVAR_DEFINE_BOOL(audio_levels, false, "Audio",
                    "Debug: log the RMS of decoded XMA chunks and mixed frames to audio_levels.csv");
REXCVAR_DEFINE_STRING(xma_mute, "", "Audio",
                      "Debug: silence the XMA context at this guest address (e.g. FFCA6040)");
REXCVAR_DEFINE_INT32(title_music_volume, 100, "Audio", "Volume of the title's own songs, 0-100");
REXCVAR_DEFINE_BOOL(xma_single_music, false, "Audio",
                    "Keep one music stream audible: after 2 s of two playing together, "
                    "silence the older (Amped 3 resumes paused menu music next to the next song)");

namespace rex::audio {
namespace {
// --xma_single_music state per context id.
struct MusicStream {
  std::chrono::steady_clock::time_point started{}, last_output{}, overlap_since{};
  bool superseded = false;
};
std::mutex music_mutex;
std::unordered_map<uint32_t, MusicStream> music_streams;

void ForgetMusicIds(uint32_t id);

void ForgetMusicStream(uint32_t id) {
  {
    std::lock_guard lock(music_mutex);
    music_streams.erase(id);
  }
  ForgetMusicIds(id);
}

// Contexts playing a streamed stereo track (a title's music, as opposed to
// sound effects and voices in small, mostly mono buffers). Amped 3's music
// streams in 80 or 160 packet buffers (fewer at a track's end); its effects
// are mono in 2-28. Sticks until the context starts a new stream.
std::mutex music_ids_mutex;
std::unordered_set<uint32_t> music_ids;

// The title's songs (the ones the player's music replaces), judged by the
// buffer each context decodes: a song-sized buffer read from the song bank
// makes it a song, which it stays through the song's shorter last buffers
// (also from the bank); a buffer from anywhere else (an event's music, an
// effect) ends it. Before reads from the bank are seen: song-sized buffers.
std::unordered_map<uint32_t, bool> title_song_ids;
// Contexts decoding the title's other background music (the loading screens'),
// judged the same way: silent while the player's music plays, never a song.
std::unordered_set<uint32_t> title_background_ids;

void ForgetMusicIds(uint32_t id) {
  std::lock_guard lock(music_ids_mutex);
  music_ids.erase(id);
  title_song_ids.erase(id);
  title_background_ids.erase(id);
}

bool IsMusicStream(uint32_t id, const XMA_CONTEXT_DATA& data) {
  // An event's music (stereo or not): the player's music holds meanwhile.
  if (data.IsCurrentInputBufferValid() &&
      IsTitleEventMusicBuffer(data.GetCurrentInputBufferAddress(),
                              data.GetCurrentInputBufferPacketCount() * XmaContext::kBytesPerPacket)) {
    NoteEventMusicChunk();
  }
  std::lock_guard lock(music_ids_mutex);
  if (!data.is_stereo) {
    music_ids.erase(id);
    title_song_ids.erase(id);
    title_background_ids.erase(id);
    return false;
  }
  const uint32_t packets = std::max(data.input_buffer_0_packet_count, data.input_buffer_1_packet_count);
  if (packets >= 64) music_ids.insert(id);
  if (data.IsCurrentInputBufferValid()) {
    // The buffer being decoded (the other one may still hold an earlier
    // stream's address).
    const uint32_t address = data.GetCurrentInputBufferAddress();
    const uint32_t current_packets = data.GetCurrentInputBufferPacketCount();
    // As songs: a music-sized buffer of that data starts it, a short one
    // (its tail) keeps it, anything else (an effect reusing the context) ends it.
    const bool background =
        IsTitleBackgroundMusicBuffer(address, current_packets * XmaContext::kBytesPerPacket);
    const bool changed = !background                ? title_background_ids.erase(id) != 0
                         : current_packets >= 64 ? title_background_ids.insert(id).second
                                                 : false;
    if (changed && REXCVAR_GET(xma_stream_log)) {
      REXAPU_INFO("XmaContext {}: background music {} ({} packets at {:08X})", id,
                  background ? "starts" : "ends", current_packets, address);
    }
    bool& song = title_song_ids[id];
    const bool was_song = song;
    if (!TitleMusicReadsSeen()) {
      if (current_packets >= 64) song = current_packets >= 128;
    } else if (!IsTitleMusicBuffer(address, current_packets * XmaContext::kBytesPerPacket)) {
      song = false;
    } else if (current_packets >= 128) {
      song = true;
    }
    if (song && !was_song) {
      if (MixMode() || REXCVAR_GET(xma_stream_log)) REXAPU_INFO("XmaContext {}: title song starts ({} packets)", id, current_packets);
      TitleMusicStreamStarted();
    } else if (was_song && !song && REXCVAR_GET(xma_stream_log)) {
      REXAPU_INFO("XmaContext {}: title song ends ({} packets at {:08X})", id, current_packets, address);
    } else if (!song && current_packets >= 128 && REXCVAR_GET(xma_stream_log)) {
      static std::unordered_map<uint32_t, uint32_t> logged;
      if (logged[id] != address) {
        logged[id] = address;
        REXAPU_INFO("XmaContext {}: {} packets at {:08X}, not from the song bank", id, current_packets, address);
      }
    }
  }
  return music_ids.count(id) != 0;
}

// One of the title's songs (the ones the player's music replaces), as judged
// by IsMusicStream.
bool TitleSongStream(uint32_t id, bool music_stream) {
  if (!music_stream) return false;
  std::lock_guard lock(music_ids_mutex);
  const auto it = title_song_ids.find(id);
  return it != title_song_ids.end() && it->second;
}

// The title's other background music (IsMusicStream).
bool TitleBackgroundStream(uint32_t id, bool music_stream) {
  if (!music_stream) return false;
  std::lock_guard lock(music_ids_mutex);
  return title_background_ids.count(id) != 0;
}

// Called for each decoded chunk; true when this context's output should be silent.
bool SupersededMusic(uint32_t id, const XMA_CONTEXT_DATA& data) {
  const bool is_music =
      data.is_stereo &&
      std::max(data.input_buffer_0_packet_count, data.input_buffer_1_packet_count) >= 128;
  std::lock_guard lock(music_mutex);
  if (!is_music) {
    music_streams.erase(id);
    return false;
  }
  const auto now = std::chrono::steady_clock::now();
  auto [it, inserted] = music_streams.try_emplace(id);
  MusicStream& self = it->second;
  if (inserted) {
    self.started = now;
  }
  self.last_output = now;
  if (self.superseded) {
    return true;
  }
  bool overlapping = false, older = false;
  for (const auto& [other_id, other] : music_streams) {
    if (other_id != id && now - other.last_output < std::chrono::milliseconds(300)) {
      overlapping = true;
      older |= self.started < other.started;
    }
  }
  if (!overlapping) {
    self.overlap_since = {};
    return false;
  }
  if (self.overlap_since == std::chrono::steady_clock::time_point{}) {
    self.overlap_since = now;
  }
  if (older && now - self.overlap_since > std::chrono::seconds(2)) {
    self.superseded = true;
    REXAPU_INFO("XmaContext {:08X}: music superseded by a newer stream; silenced", id);
    return true;
  }
  return false;
}
}  // namespace

// Debug (--audio_levels): one line per decoded chunk or mixed frame.
void AudioLevelLog(const char* source, double rms) {
  static std::mutex mutex;
  static std::FILE* file = nullptr;
  static const auto start = std::chrono::steady_clock::now();
  std::lock_guard lock(mutex);
  if (!file && !(file = std::fopen("audio_levels.csv", "w"))) {
    return;
  }
  std::fprintf(file, "%.4f,%s,%.6f\n",
               std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
               source, rms);
}
}  // namespace rex::audio
#include <rex/audio/xma/decoder.h>
#include <rex/audio/xma/helpers.h>
#include <rex/dbg.h>
#include <rex/logging.h>
#include <rex/memory/ring_buffer.h>
#include <rex/platform.h>
#include <rex/stream.h>

extern "C" {
#if REX_COMPILER_MSVC
#pragma warning(push)
#pragma warning(disable : 4101 4244 5033)
#endif
#include "libavcodec/avcodec.h"
#include "libavutil/error.h"
#if REX_COMPILER_MSVC
#pragma warning(pop)
#endif
}  // extern "C"

// Credits for most of this code goes to:
// https://github.com/koolkdev/libertyv/blob/master/libav_wrapper/xma2dec.c

namespace rex::audio {

using stream::BitStream;

const uint32_t XmaContext::kBitsPerPacketHeader;
const uint32_t XmaContext::kOutputMaxSizeBytes;

XmaContext::XmaContext()
    : work_completion_event_(rex::thread::Event::CreateAutoResetEvent(false)) {}

XmaContext::~XmaContext() {
  if (av_context_) {
    avcodec_free_context(&av_context_);
  }
  if (av_frame_) {
    av_frame_free(&av_frame_);
  }
}

int XmaContext::Setup(uint32_t id, memory::Memory* memory, uint32_t guest_ptr) {
  id_ = id;
  memory_ = memory;
  guest_ptr_ = guest_ptr;

  // Allocate ffmpeg stuff:
  av_packet_ = av_packet_alloc();
  assert_not_null(av_packet_);
  av_packet_->buf = av_buffer_alloc(128 * 1024);

  // find the XMA2 audio decoder
  av_codec_ = avcodec_find_decoder(AV_CODEC_ID_XMAFRAMES);
  if (!av_codec_) {
    REXAPU_ERROR("XmaContext {}: Codec not found", id);
    return 1;
  }

  av_context_ = avcodec_alloc_context3(av_codec_);
  if (!av_context_) {
    REXAPU_ERROR("XmaContext {}: Couldn't allocate context", id);
    return 1;
  }

  // Initialize these to 0. They'll actually be set later.
  av_context_->channels = 0;
  av_context_->sample_rate = 0;

  av_frame_ = av_frame_alloc();
  if (!av_frame_) {
    REXAPU_ERROR("XmaContext {}: Couldn't allocate frame", id);
    return 1;
  }

  // FYI: We're purposely not opening the codec here. That is done later.
  return 0;
}

bool XmaContext::Work() {
  if (!is_allocated() || !is_enabled()) {
    return false;
  }

  std::lock_guard<std::mutex> lock(lock_);
  set_is_enabled(false);

  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  XMA_CONTEXT_DATA data(context_ptr);
  const XMA_CONTEXT_DATA initial_data = data;

  if (!data.output_buffer_valid) {
    return true;
  }

  memory::RingBuffer output_rb = PrepareOutputRingBuffer(&data);

  // Consume-only context: no input, just drain remaining subframes.
  if (data.IsConsumeOnlyContext()) {
    if (current_frame_remaining_subframes_ == 0) {
      return true;
    }
    Consume(&output_rb, &data);
    data.output_buffer_write_offset = output_rb.write_offset() / kOutputBytesPerBlock;
    StoreContextMerged(data, initial_data, context_ptr);
    return true;
  }

  // Minimum free blocks needed before attempting a decode.
  // Use subframe_decode_count (clamped to 1) instead of full frame size.
  const uint32_t effective_sdc = std::max(static_cast<uint32_t>(1), data.subframe_decode_count);
  const int32_t minimum_subframe_decode_count =
      static_cast<int32_t>(effective_sdc) + data.output_buffer_padding;

  if (minimum_subframe_decode_count > remaining_subframe_blocks_in_output_buffer_) {
    StoreContextMerged(data, initial_data, context_ptr);
    return true;
  }

  while (remaining_subframe_blocks_in_output_buffer_ >= minimum_subframe_decode_count) {
    Decode(&data);
    Consume(&output_rb, &data);

    if (!data.IsAnyInputBufferValid() || data.error_status == 4) {
      break;
    }
  }

  data.output_buffer_write_offset = output_rb.write_offset() / kOutputBytesPerBlock;

  if (output_rb.empty()) {
    data.output_buffer_valid = 0;
  }

  StoreContextMerged(data, initial_data, context_ptr);
  return true;
}

void XmaContext::Enable() {
  std::lock_guard<std::mutex> lock(lock_);
  set_is_enabled(true);
}

bool XmaContext::Block(bool poll) {
  if (!lock_.try_lock()) {
    if (poll) {
      return false;
    }
    lock_.lock();
  }
  lock_.unlock();
  return true;
}

void XmaContext::Clear() {
  ForgetMusicStream(id());  // --xma_single_music: a new stream starts here.
  std::lock_guard<std::mutex> lock(lock_);
  REXAPU_NOISY_DEBUG("XmaContext: reset context {}", id());

  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  XMA_CONTEXT_DATA data(context_ptr);
  ClearLocked(&data);
  data.Store(context_ptr);
}

void XmaContext::ClearLocked(XMA_CONTEXT_DATA* data) {
  data->input_buffer_0_valid = 0;
  data->input_buffer_1_valid = 0;
  data->output_buffer_valid = 0;

  data->input_buffer_read_offset = kBitsPerPacketHeader;
  data->output_buffer_read_offset = 0;
  data->output_buffer_write_offset = 0;

  ResetDecoderState();
}

void XmaContext::ResetDecoderState() {
  // A freed or re-initialized context is a new logical stream, so the previous
  // wave's MDCT overlap-add tail must not survive into frame 0 of the next one.
  // avcodec_flush_buffers() cannot drop it: ff_xmaframes_decoder declares no
  // flush callback, so the call never reaches the code clearing channel[].out.
  // Invalidating the cached format makes PrepareDecoder reopen the codec on the
  // next decode, which does discard the history.
  if (av_context_) {
    av_context_->sample_rate = 0;
    av_context_->channels = 0;
  }
  raw_frame_.fill(0);
  decoded_frame_.fill(0);
  carry_frame_.fill(0);
  carry_valid_ = false;
  pending_output_limit_ = 0;
  pending_start_skip_ = 0;
  current_frame_remaining_subframes_ = 0;
  loop_frame_output_limit_ = 0;
  loop_start_skip_pending_ = false;
}

void XmaContext::Disable() {
  std::lock_guard<std::mutex> lock(lock_);
  set_is_enabled(false);
}

void XmaContext::Release() {
  std::lock_guard<std::mutex> lock(lock_);
  assert_true(is_allocated());

  set_is_allocated(false);
  ResetDecoderState();
  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  std::memset(context_ptr, 0, sizeof(XMA_CONTEXT_DATA));
}

void XmaContext::SwapInputBuffer(XMA_CONTEXT_DATA* data) {
  if (data->current_buffer == 0) {
    data->input_buffer_0_valid = 0;
  } else {
    data->input_buffer_1_valid = 0;
  }
  data->current_buffer ^= 1;
  data->input_buffer_read_offset = kBitsPerPacketHeader;
}

void XmaContext::UpdateLoopStatus(XMA_CONTEXT_DATA* data) {
  if (data->loop_count == 0) {
    return;
  }

  const uint32_t loop_start = std::max(kBitsPerPacketHeader, data->loop_start);
  const uint32_t loop_end = std::max(kBitsPerPacketHeader, data->loop_end);

  if (data->input_buffer_read_offset != loop_end) {
    return;
  }

  data->input_buffer_read_offset = loop_start;
  loop_start_skip_pending_ = true;

  if (data->loop_count < 255) {
    data->loop_count--;
  }
}

int XmaContext::GetSampleRate(int id) {
  return kIdToSampleRate[std::min(id, 3)];
}

int16_t XmaContext::GetPacketNumber(size_t size, size_t bit_offset) {
  if (bit_offset < kBitsPerPacketHeader) {
    assert_always();
    return -1;
  }
  if (bit_offset >= (size << 3)) {
    assert_always();
    return -1;
  }
  size_t byte_offset = bit_offset >> 3;
  size_t packet_number = byte_offset / kBytesPerPacket;
  return static_cast<int16_t>(packet_number);
}

uint32_t XmaContext::GetCurrentInputBufferSize(XMA_CONTEXT_DATA* data) {
  return data->GetCurrentInputBufferPacketCount() * kBytesPerPacket;
}

uint8_t* XmaContext::GetCurrentInputBuffer(XMA_CONTEXT_DATA* data) {
  return memory()->TranslatePhysical(data->GetCurrentInputBufferAddress());
}

uint32_t XmaContext::GetAmountOfBitsToRead(uint32_t remaining_stream_bits, uint32_t frame_size) {
  return std::min(remaining_stream_bits, frame_size);
}

const uint8_t* XmaContext::GetNextPacket(XMA_CONTEXT_DATA* data, uint32_t next_packet_index,
                                         uint32_t current_input_packet_count) {
  if (next_packet_index < current_input_packet_count) {
    return memory()->TranslatePhysical(data->GetCurrentInputBufferAddress()) +
           next_packet_index * kBytesPerPacket;
  }

  const uint8_t next_buffer_index = data->current_buffer ^ 1;
  if (!data->IsInputBufferValid(next_buffer_index)) {
    return nullptr;
  }

  const uint32_t next_buffer_address = data->GetInputBufferAddress(next_buffer_index);
  if (!next_buffer_address) {
    REXAPU_ERROR("XmaContext {}: Buffer marked valid but has null pointer!", id());
    return nullptr;
  }

  return memory()->TranslatePhysical(next_buffer_address);
}

uint32_t XmaContext::GetNextPacketReadOffset(uint8_t* buffer, uint32_t next_packet_index,
                                             uint32_t current_input_packet_count) {
  while (next_packet_index < current_input_packet_count) {
    uint8_t* next_packet = buffer + (next_packet_index * kBytesPerPacket);
    const uint32_t packet_frame_offset = xma::GetPacketFrameOffset(next_packet);

    if (packet_frame_offset <= kMaxFrameSizeinBits) {
      return (next_packet_index * kBitsPerPacket) + packet_frame_offset;
    }
    next_packet_index++;
  }

  return kBitsPerPacketHeader;
}

memory::RingBuffer XmaContext::PrepareOutputRingBuffer(XMA_CONTEXT_DATA* data) {
  const uint32_t output_capacity = data->output_buffer_block_count * kOutputBytesPerBlock;
  const uint32_t output_read_offset = data->output_buffer_read_offset * kOutputBytesPerBlock;
  const uint32_t output_write_offset = data->output_buffer_write_offset * kOutputBytesPerBlock;

  if (output_capacity > kOutputMaxSizeBytes) {
    REXAPU_WARN(
        "XmaContext {}: Output buffer exceeds expected size! "
        "(Actual: {} Max: {})",
        id(), output_capacity, kOutputMaxSizeBytes);
  }

  uint8_t* output_buffer = memory()->TranslatePhysical(data->output_buffer_ptr);

  memory::RingBuffer output_rb(output_buffer, output_capacity);
  output_rb.set_read_offset(output_read_offset);
  output_rb.set_write_offset(output_write_offset);
  remaining_subframe_blocks_in_output_buffer_ =
      static_cast<int32_t>(output_rb.write_count()) / kOutputBytesPerBlock;

  return output_rb;
}

kPacketInfo XmaContext::GetPacketInfo(uint8_t* packet, uint32_t frame_offset) {
  kPacketInfo packet_info = {};

  const uint32_t first_frame_offset = xma::GetPacketFrameOffset(packet);
  BitStream stream(packet, kBitsPerPacket);
  stream.SetOffset(first_frame_offset);

  if (frame_offset < first_frame_offset) {
    packet_info.current_frame_ = 0;
    packet_info.current_frame_size_ = first_frame_offset - frame_offset;
  }

  while (true) {
    if (stream.BitsRemaining() < kBitsPerFrameHeader) {
      break;
    }

    const uint64_t frame_size = stream.Peek(kBitsPerFrameHeader);
    if (frame_size == 0 || frame_size == xma::kMaxFrameLength) {
      break;
    }

    if (stream.offset_bits() == frame_offset) {
      packet_info.current_frame_ = packet_info.frame_count_;
      packet_info.current_frame_size_ = static_cast<uint32_t>(frame_size);
    }

    packet_info.frame_count_++;

    if (frame_size > stream.BitsRemaining()) {
      break;
    }

    stream.Advance(frame_size - 1);

    if (stream.Read(1) == 0) {
      break;
    }
  }

  if (xma::IsPacketXma2Type(packet)) {
    const uint8_t xma2_frame_count = xma::GetPacketFrameCount(packet);
    if (xma2_frame_count > packet_info.frame_count_) {
      if (packet_info.current_frame_size_ == 0) {
        packet_info.current_frame_ = packet_info.frame_count_;
      }
      packet_info.frame_count_ = xma2_frame_count;
    }
  }
  return packet_info;
}

void XmaContext::StoreContextMerged(const XMA_CONTEXT_DATA& data,
                                    const XMA_CONTEXT_DATA& initial_data, uint8_t* context_ptr) {
  XMA_CONTEXT_DATA fresh(context_ptr);

  fresh.loop_count = data.loop_count;
  fresh.output_buffer_write_offset = data.output_buffer_write_offset;
  if (initial_data.input_buffer_0_valid && !data.input_buffer_0_valid) {
    fresh.input_buffer_0_valid = 0;
  }
  if (initial_data.input_buffer_1_valid && !data.input_buffer_1_valid) {
    fresh.input_buffer_1_valid = 0;
  }

  if (initial_data.output_buffer_valid && !data.output_buffer_valid) {
    fresh.output_buffer_valid = 0;
  }

  fresh.input_buffer_read_offset = data.input_buffer_read_offset;
  fresh.error_status = data.error_status;
  fresh.current_buffer = data.current_buffer;
  fresh.output_buffer_read_offset = data.output_buffer_read_offset;

  fresh.Store(context_ptr);
}

void XmaContext::Consume(memory::RingBuffer* output_rb, const XMA_CONTEXT_DATA* data) {
  if (!current_frame_remaining_subframes_) {
    return;
  }

  if (loop_frame_output_limit_ > 0) {
    const uint8_t total_subframes = (kBytesPerFrameChannel / kOutputBytesPerBlock)
                                    << data->is_stereo;
    const uint8_t consumed = total_subframes - current_frame_remaining_subframes_;
    if (consumed >= loop_frame_output_limit_) {
      remaining_subframe_blocks_in_output_buffer_ -= data->output_buffer_padding;
      current_frame_remaining_subframes_ = 0;
      loop_frame_output_limit_ = 0;
      return;
    }
  }

  const uint8_t effective_sdc = std::max(static_cast<uint32_t>(1), data->subframe_decode_count);
  int8_t subframes_to_write = std::min(static_cast<int8_t>(current_frame_remaining_subframes_),
                                       static_cast<int8_t>(effective_sdc));

  if (loop_frame_output_limit_ > 0) {
    const uint8_t total_subframes = (kBytesPerFrameChannel / kOutputBytesPerBlock)
                                    << data->is_stereo;
    const uint8_t consumed = total_subframes - current_frame_remaining_subframes_;
    const int8_t remaining_until_limit = static_cast<int8_t>(loop_frame_output_limit_ - consumed);
    if (subframes_to_write > remaining_until_limit) {
      subframes_to_write = remaining_until_limit;
    }
  }

  const int8_t raw_frame_read_offset =
      ((kBytesPerFrameChannel / kOutputBytesPerBlock) << data->is_stereo) -
      current_frame_remaining_subframes_;

  static const std::array<uint8_t, kBytesPerFrameChannel * 2> silence{};
  char context_name[16];
  std::snprintf(context_name, sizeof(context_name), "%08X", guest_ptr());
  // (Checked always: mix mode sees the title's songs start, and the music
  // volume applies to them.)
  const bool music_stream = IsMusicStream(id(), *data);
  if (TitleSongStream(id(), music_stream)) NoteTitleSongChunk();
  const bool muted = (!REXCVAR_GET(xma_mute).empty() && REXCVAR_GET(xma_mute) == context_name) ||
                     (REXCVAR_GET(xma_single_music) && SupersededMusic(id(), *data)) ||
                     // The player's own music in place of the title's songs only: the
                     // streams decoding from its song bank (TitleMusicReadsSeen), else
                     // its 160-packet streams, and its other background music (the
                     // loading screens'). Event music and effects keep playing.
                     (UserMusicReplacingTitleMusic() &&
                      (TitleSongStream(id(), music_stream) || TitleBackgroundStream(id(), music_stream)));
  const uint8_t* out = (muted ? silence.data() : raw_frame_.data()) + kOutputBytesPerBlock * raw_frame_read_offset;
  const int title_volume = std::clamp(REXCVAR_GET(title_music_volume), 0, 100);
  if (!muted && music_stream && title_volume < 100) {
    // The title's songs at --title_music_volume (16-bit big-endian PCM).
    std::array<uint8_t, kBytesPerFrameChannel * 2> scaled;
    const size_t bytes = std::min(scaled.size(), size_t(subframes_to_write) * kOutputBytesPerBlock);
    for (size_t i = 0; i + 1 < bytes; i += 2) {
      const int v = int16_t((out[i] << 8) | out[i + 1]) * title_volume / 100;
      scaled[i] = uint8_t(uint16_t(v) >> 8);
      scaled[i + 1] = uint8_t(v);
    }
    output_rb->Write(scaled.data(), bytes);
  } else {
    output_rb->Write(out, subframes_to_write * kOutputBytesPerBlock);
  }
  if (REXCVAR_GET(audio_levels) || REXCVAR_GET(xma_stream_log)) {
    static std::mutex described_mutex;
    static std::unordered_map<uint32_t, uint32_t> described;
    std::lock_guard lock(described_mutex);
    const uint32_t key = (uint32_t(data->is_stereo) << 31) |
                         (std::max(data->input_buffer_0_packet_count, data->input_buffer_1_packet_count) << 1) |
                         uint32_t(muted);
    auto [it, inserted] = described.try_emplace(guest_ptr(), ~0u);
    if (it->second != key) {
      it->second = key;
      REXAPU_INFO("XmaContext {:08X}: stereo {} packets {}/{} muted {} title song {}", guest_ptr(),
                  bool(data->is_stereo), uint32_t(data->input_buffer_0_packet_count),
                  uint32_t(data->input_buffer_1_packet_count), muted,
                  TitleSongStream(id(), music_stream));
    }
  }
  if (REXCVAR_GET(audio_levels)) {
    const uint8_t* pcm = raw_frame_.data() + (kOutputBytesPerBlock * raw_frame_read_offset);
    const size_t samples = subframes_to_write * kOutputBytesPerBlock / 2;
    double sum = 0.0;
    for (size_t i = 0; i < samples; ++i) {
      const double s = int16_t((pcm[2 * i] << 8) | pcm[2 * i + 1]) / 32768.0;
      sum += s * s;
    }
    char source[16];
    std::snprintf(source, sizeof(source), "%08X", guest_ptr());
    AudioLevelLog(source, samples ? std::sqrt(sum / samples) : 0.0);
  }

  const int8_t headroom = (current_frame_remaining_subframes_ - subframes_to_write == 0)
                              ? data->output_buffer_padding
                              : 0;

  remaining_subframe_blocks_in_output_buffer_ -= subframes_to_write + headroom;
  current_frame_remaining_subframes_ -= subframes_to_write;
}

int XmaContext::PrepareDecoder(int sample_rate, bool is_two_channel) {
  sample_rate = GetSampleRate(sample_rate);

  uint32_t channels = is_two_channel ? 2 : 1;
  if (av_context_->sample_rate != sample_rate ||
      av_context_->channels != static_cast<int>(channels)) {
    REXAPU_NOISY_DEBUG("XmaContext {}: Codec reinit: rate {} -> {}, channels {} -> {}", id(),
                       av_context_->sample_rate, sample_rate, av_context_->channels, channels);
    avcodec_free_context(&av_context_);
    av_context_ = avcodec_alloc_context3(av_codec_);

    av_context_->sample_rate = sample_rate;
    av_context_->channels = channels;
    av_context_->flags2 |= AV_CODEC_FLAG2_SKIP_MANUAL;

    if (avcodec_open2(av_context_, av_codec_, NULL) < 0) {
      REXAPU_ERROR("XmaContext: Failed to reopen FFmpeg context");
      return -1;
    }
    return 1;
  }
  return 0;
}

void XmaContext::PreparePacket(uint32_t frame_size, uint32_t frame_padding) {
  av_packet_->data = xma_frame_.data();
  av_packet_->size = static_cast<int>(1 + ((frame_padding + frame_size) / 8) +
                                      (((frame_padding + frame_size) % 8) ? 1 : 0));

  auto padding_end = av_packet_->size * 8 - (8 + frame_padding + frame_size);
  assert_true(padding_end < 8);
  xma_frame_[0] = ((frame_padding & 7) << 5) | ((padding_end & 7) << 2);
}

bool XmaContext::DecodePacket(AVCodecContext* av_context, const AVPacket* av_packet,
                              AVFrame* av_frame) {
  auto ret = avcodec_send_packet(av_context, av_packet);
  if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, errbuf, sizeof(errbuf));
    REXAPU_ERROR("XmaContext {}: Error sending packet for decoding: {} ({})", id(), errbuf, ret);
    return false;
  }
  ret = avcodec_receive_frame(av_context, av_frame);

  if (ret == AVERROR(EAGAIN)) {
    return false;
  }
  if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, errbuf, sizeof(errbuf));
    REXAPU_ERROR("XmaContext {}: Error during decoding: {} ({})", id(), errbuf, ret);
    return false;
  }
  return true;
}

void XmaContext::Decode(XMA_CONTEXT_DATA* data) {
  SCOPE_profile_cpu_f("apu");

  if (!data->IsAnyInputBufferValid()) {
    return;
  }

  if (current_frame_remaining_subframes_ > 0) {
    return;
  }

  if (!data->IsCurrentInputBufferValid()) {
    SwapInputBuffer(data);
    if (!data->IsCurrentInputBufferValid()) {
      return;
    }
  }

  uint8_t* current_input_buffer = GetCurrentInputBuffer(data);

  input_buffer_.fill(0);

  // Loop-end frame: decode it here (output limited to loop_subframe_end),
  // jump to loop_start afterwards in the next-offset step.
  bool is_loop_end_frame = false;
  if (data->loop_count > 0) {
    const uint32_t loop_end = std::max(kBitsPerPacketHeader, data->loop_end);
    is_loop_end_frame = (data->input_buffer_read_offset == loop_end);
  }

  if (!data->output_buffer_block_count) {
    REXAPU_ERROR("XmaContext {}: Error - Received 0 for output_buffer_block_count!", id());
    return;
  }

  if (data->input_buffer_read_offset < kBitsPerPacketHeader) {
    data->input_buffer_read_offset = kBitsPerPacketHeader;
  }

  const uint32_t current_input_size = GetCurrentInputBufferSize(data);
  const uint32_t current_input_packet_count = current_input_size / kBytesPerPacket;

  const int16_t packet_index = GetPacketNumber(current_input_size, data->input_buffer_read_offset);

  if (packet_index == -1) {
    REXAPU_ERROR("XmaContext {}: Invalid packet index. Input read offset: {}", id(),
                 static_cast<uint32_t>(data->input_buffer_read_offset));
    return;
  }

  uint8_t* packet = current_input_buffer + (packet_index * kBytesPerPacket);
  const uint32_t packet_first_frame_offset = xma::GetPacketFrameOffset(packet);
  uint32_t relative_offset = data->input_buffer_read_offset % kBitsPerPacket;

  if (relative_offset < packet_first_frame_offset) {
    data->input_buffer_read_offset = (packet_index * kBitsPerPacket) + packet_first_frame_offset;
    relative_offset = packet_first_frame_offset;
  }

  const uint8_t skip_count = xma::GetPacketSkipCount(packet);

  // Full packet skip (0xFF) -- no new frames begin in this packet.
  if (skip_count == 0xFF) {
    uint32_t next_input_offset =
        GetNextPacketReadOffset(current_input_buffer, packet_index + 1, current_input_packet_count);
    if (next_input_offset == kBitsPerPacketHeader) {
      SwapInputBuffer(data);
    }
    data->input_buffer_read_offset = next_input_offset;
    return;
  }

  kPacketInfo packet_info = GetPacketInfo(packet, relative_offset);
  const uint32_t packet_to_skip = skip_count + 1;
  const uint32_t next_packet_index = packet_index + packet_to_skip;

  // Frame header split across packet boundary.
  if (packet_info.current_frame_size_ == 0) {
    const uint8_t* next_packet = GetNextPacket(data, next_packet_index, current_input_packet_count);
    if (!next_packet) {
      SwapInputBuffer(data);
      return;
    }
    std::memcpy(input_buffer_.data(), packet + kBytesPerPacketHeader, kBytesPerPacketData);
    std::memcpy(input_buffer_.data() + kBytesPerPacketData, next_packet + kBytesPerPacketHeader,
                kBytesPerPacketData);

    BitStream combined(input_buffer_.data(), (kBitsPerPacket - kBitsPerPacketHeader) * 2);
    combined.SetOffset(relative_offset - kBitsPerPacketHeader);

    uint64_t frame_size = combined.Peek(kBitsPerFrameHeader);
    if (frame_size == xma::kMaxFrameLength) {
      // The end-of-frames marker: no more frames start in this packet. On to
      // the next packet with a frame, or the next buffer. (Returning here
      // without moving on left the stream stuck on this spot for good: the
      // game's voice looped its last output, its song never ended, and
      // leaving the mountain waited for it forever.)
      static std::atomic<int> logged{0};
      if (logged.fetch_add(1) < 20) {
        REXAPU_INFO("XmaContext {}: end-of-frames marker across packets at {}; moving on", id(),
                    static_cast<uint32_t>(data->input_buffer_read_offset));
      }
      const uint32_t next_input_offset =
          GetNextPacketReadOffset(current_input_buffer, next_packet_index, current_input_packet_count);
      if (next_input_offset == kBitsPerPacketHeader) {
        SwapInputBuffer(data);
      } else {
        data->input_buffer_read_offset = next_input_offset;
      }
      return;
    }
    packet_info.current_frame_size_ = static_cast<uint32_t>(frame_size);
  }

  BitStream stream(current_input_buffer, (packet_index + 1) * kBitsPerPacket);
  stream.SetOffset(data->input_buffer_read_offset);

  const uint64_t bits_to_copy = GetAmountOfBitsToRead(static_cast<uint32_t>(stream.BitsRemaining()),
                                                      packet_info.current_frame_size_);

  if (bits_to_copy == 0) {
    REXAPU_ERROR("XmaContext {}: There are no bits to copy!", id());
    SwapInputBuffer(data);
    return;
  }

  if (packet_info.isLastFrameInPacket()) {
    if (stream.BitsRemaining() < packet_info.current_frame_size_) {
      const uint8_t* next_packet =
          GetNextPacket(data, next_packet_index, current_input_packet_count);
      if (!next_packet) {
        // Its frame goes on in the next buffer, not queued yet: wait for it,
        // as the hardware does. The end of a stream never gets one (and the
        // title may wait for this buffer to finish before queuing more): after
        // a quarter of a second, this buffer is done.
        const auto now = std::chrono::steady_clock::now();
        if (input_stall_since_ == std::chrono::steady_clock::time_point{}) input_stall_since_ = now;
        if (now - input_stall_since_ < std::chrono::milliseconds(250)) {
          data->error_status = 4;
          return;
        }
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1) < 20) {
          REXAPU_INFO("XmaContext {}: no next buffer for 250 ms at the end of this one ({} packets); finishing it",
                      id(), current_input_packet_count);
        }
        input_stall_since_ = {};
        data->error_status = 0;
        SwapInputBuffer(data);
        return;
      }
      input_stall_since_ = {};
      std::memcpy(input_buffer_.data() + kBytesPerPacketData, next_packet + kBytesPerPacketHeader,
                  kBytesPerPacketData);
    }
  }

  std::memcpy(input_buffer_.data(), packet + kBytesPerPacketHeader, kBytesPerPacketData);

  stream = BitStream(input_buffer_.data(), (kBitsPerPacket - kBitsPerPacketHeader) * 2);
  stream.SetOffset(relative_offset - kBitsPerPacketHeader);

  xma_frame_.fill(0);

  const uint32_t padding_start =
      static_cast<uint8_t>(stream.Copy(xma_frame_.data() + 1, packet_info.current_frame_size_));

  decoded_frame_.fill(0);

  if (PrepareDecoder(data->sample_rate, bool(data->is_stereo)) == 1) {
    // Reopening the codec restarts its output; the carried tail belongs to the
    // old stream.
    carry_valid_ = false;
  }
  PreparePacket(packet_info.current_frame_size_, padding_start);
  const bool decoded = DecodePacket(av_context_, av_packet_, av_frame_);
  if (decoded) {
    ConvertFrame(reinterpret_cast<const uint8_t**>(&av_frame_->data), bool(data->is_stereo),
                 decoded_frame_.data());

    // Realign the decoder's output onto the bitstream's sample numbering: the
    // samples of the frame just decoded run from kDecoderStartPadding into this
    // block and finish in the head of the next one.
    const size_t pad_bytes = kDecoderStartPadding * kBytesPerSample << data->is_stereo;
    const size_t carry_bytes = (kBytesPerFrameChannel << data->is_stereo) - pad_bytes;

    // Loop end: limit output to subframes 0..loop_subframe_end.
    const uint8_t decoded_output_limit =
        is_loop_end_frame ? static_cast<uint8_t>((data->loop_subframe_end + 1) << data->is_stereo)
                          : 0;
    // Loop start: skip leading subframes per loop_subframe_skip. skip == 4
    // means the whole frame is a warm-up frame (frame-aligned loop start):
    // decode seeds the codec state, output is fully discarded.
    const uint8_t decoded_start_skip =
        loop_start_skip_pending_ ? static_cast<uint8_t>(data->loop_subframe_skip << data->is_stereo)
                                 : 0;
    loop_start_skip_pending_ = false;

    if (carry_valid_) {
      std::memcpy(raw_frame_.data(), carry_frame_.data(), carry_bytes);
      std::memcpy(raw_frame_.data() + carry_bytes, decoded_frame_.data(), pad_bytes);

      current_frame_remaining_subframes_ = 4 << data->is_stereo;
      loop_frame_output_limit_ = pending_output_limit_;
      current_frame_remaining_subframes_ -=
          std::min(pending_start_skip_, current_frame_remaining_subframes_);
    }

    std::memcpy(carry_frame_.data(), decoded_frame_.data() + pad_bytes, carry_bytes);
    carry_valid_ = true;
    pending_output_limit_ = decoded_output_limit;
    pending_start_skip_ = decoded_start_skip;
  } else {
    // A dropped frame breaks the carry's adjacency; re-prime rather than splice
    // two blocks that are not neighbors.
    carry_valid_ = false;
  }

  // Compute where to go next.
  if (is_loop_end_frame) {
    UpdateLoopStatus(data);
    return;
  }

  if (!packet_info.isLastFrameInPacket()) {
    const uint32_t next_frame_offset =
        (data->input_buffer_read_offset + bits_to_copy) % kBitsPerPacket;
    data->input_buffer_read_offset = (packet_index * kBitsPerPacket) + next_frame_offset;
    return;
  }

  uint32_t next_input_offset =
      GetNextPacketReadOffset(current_input_buffer, next_packet_index, current_input_packet_count);

  if (next_input_offset == kBitsPerPacketHeader) {
    SwapInputBuffer(data);
    if (data->IsAnyInputBufferValid()) {
      next_input_offset = xma::GetPacketFrameOffset(
          memory()->TranslatePhysical(data->GetCurrentInputBufferAddress()));

      if (next_input_offset > kMaxFrameSizeinBits) {
        SwapInputBuffer(data);
        return;
      }
    }
  }
  data->input_buffer_read_offset = next_input_offset;
}

void XmaContext::ConvertFrame(const uint8_t** samples, bool is_two_channel,
                              uint8_t* output_buffer) {
  // Loop through every sample, convert and drop it into the output array.
  // If more than one channel, we need to interleave the samples from each
  // channel next to each other. Always saturate because FFmpeg output is
  // not limited to [-1, 1] (for example 1.095 as seen in 5454082B).
  constexpr float scale = (1 << 15) - 1;
  auto out = reinterpret_cast<int16_t*>(output_buffer);

  // For testing of vectorized versions, stereo audio is common in 4D5307E6,
  // since the first menu frame; the intro cutscene also has more than 2
  // channels.
#if REX_ARCH_AMD64
  static_assert(kSamplesPerFrame % 8 == 0);
  const auto in_channel_0 = reinterpret_cast<const float*>(samples[0]);
  const __m128 scale_mm = _mm_set1_ps(scale);
  if (is_two_channel) {
    const auto in_channel_1 = reinterpret_cast<const float*>(samples[1]);
    const __m128i shufmask = _mm_set_epi8(14, 15, 6, 7, 12, 13, 4, 5, 10, 11, 2, 3, 8, 9, 0, 1);
    for (uint32_t i = 0; i < kSamplesPerFrame; i += 4) {
      // Load 8 samples, 4 for each channel.
      __m128 in_mm0 = _mm_loadu_ps(&in_channel_0[i]);
      __m128 in_mm1 = _mm_loadu_ps(&in_channel_1[i]);
      // Rescale.
      in_mm0 = _mm_mul_ps(in_mm0, scale_mm);
      in_mm1 = _mm_mul_ps(in_mm1, scale_mm);
      // Cast to int32.
      __m128i out_mm0 = _mm_cvtps_epi32(in_mm0);
      __m128i out_mm1 = _mm_cvtps_epi32(in_mm1);
      // Saturated cast and pack to int16.
      __m128i out_mm = _mm_packs_epi32(out_mm0, out_mm1);
      // Interleave channels and byte swap.
      out_mm = _mm_shuffle_epi8(out_mm, shufmask);
      // Store, as [out + i * 4] movdqu.
      _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[i * 2]), out_mm);
    }
  } else {
    const __m128i shufmask = _mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    for (uint32_t i = 0; i < kSamplesPerFrame; i += 8) {
      // Load 8 samples, as [in_channel_0 + i * 4] and
      // [in_channel_0 + i * 4 + 16] movups.
      __m128 in_mm0 = _mm_loadu_ps(&in_channel_0[i]);
      __m128 in_mm1 = _mm_loadu_ps(&in_channel_0[i + 4]);
      // Rescale.
      in_mm0 = _mm_mul_ps(in_mm0, scale_mm);
      in_mm1 = _mm_mul_ps(in_mm1, scale_mm);
      // Cast to int32.
      __m128i out_mm0 = _mm_cvtps_epi32(in_mm0);
      __m128i out_mm1 = _mm_cvtps_epi32(in_mm1);
      // Saturated cast and pack to int16.
      __m128i out_mm = _mm_packs_epi32(out_mm0, out_mm1);
      // Byte swap.
      out_mm = _mm_shuffle_epi8(out_mm, shufmask);
      // Store, as [out + i * 2] movdqu.
      _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[i]), out_mm);
    }
  }
#else
  uint32_t o = 0;
  for (uint32_t i = 0; i < kSamplesPerFrame; i++) {
    for (uint32_t j = 0; j <= uint32_t(is_two_channel); j++) {
      // Select the appropriate array based on the current channel.
      auto in = reinterpret_cast<const float*>(samples[j]);

      // Raw samples sometimes aren't within [-1, 1]
      float scaled_sample = rex::clamp_float(in[i], -1.0f, 1.0f) * scale;

      // Convert the sample and output it in big endian.
      auto sample = static_cast<int16_t>(scaled_sample);
      out[o++] = rex::byte_swap(sample);
    }
  }
#endif
}

}  // namespace rex::audio
