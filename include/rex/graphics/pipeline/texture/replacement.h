/**
 * @file        rex/graphics/pipeline/texture/replacement.h
 *
 * @brief       Texture dump and replacement pipeline.
 *
 *              Textures are identified by a stable content hash (XXH3 over the
 *              raw guest bytes as seen in physical memory — tiled and
 *              big-endian, exactly as the Xbox GPU would read them).  This
 *              makes the hash address-independent and run-independent.
 *
 *              Dump layout  (relative to the configured textures folder):
 *                  dump/<hash16>_<w>x<h>_<fmt_name>.dds
 *
 *              Replacement layout (scanned once at init, hot-reloaded on demand via Rescan()):
 *                  replace/<hash16>.dds   (RGBA8/BGRA8 or BC1/BC2/BC3)
 *                  replace/<hash16>.png   (RGBA8; loaded via stb_image)
 *              Images are used with straight alpha, like the game's own
 *              textures; a pack exported with premultiplied alpha is found
 *              out and converted (texture_replace_alpha). Images bigger than
 *              8192 per side are used halved until they fit.
 *
 *              Dump DDS format:
 *                - Compressed formats (DXT1/DXT3/DXT5/DXN/CTX1/DXT3A/DXT5A):
 *                    FOURCC DDS with raw BC blocks — lossless, directly usable
 *                    as a base for replacement authoring.
 *                - Uncompressed formats: RGBA8 unorm after untiling + endian
 *                    swap + channel expansion to 8bpc.
 *
 */
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <rex/cvar.h>
#include <rex/graphics/xenos.h>

namespace rex::thread {
class Thread;
}  // namespace rex::thread

// CVARs controlling the dump/replace pipeline (defined in cache.cpp).
REXCVAR_DECLARE(bool, texture_dump_enabled);
REXCVAR_DECLARE(bool, texture_replace_enabled);
REXCVAR_DECLARE(bool, texture_replace_preload);
REXCVAR_DECLARE(int32_t, texture_replace_ram_mb);
REXCVAR_DECLARE(std::string, texture_replace_alpha);
REXCVAR_DECLARE(std::string, texture_folder);

namespace rex::graphics {

// The original texture of a replacement, as the guest stores its base level -
// copied when the replacement is looked up, so the loading thread can compare
// the replacement with it (texture_replace_alpha).
struct TextureReplacementGuest {
  std::vector<uint8_t> bytes;
  uint32_t width = 0;
  uint32_t height = 0;
  // The base level's row pitch in bytes (texture_util::TextureGuestLayout).
  uint32_t row_pitch_bytes = 0;
  bool tiled = false;
  xenos::TextureFormat format = xenos::TextureFormat::k_8_8_8_8;
  xenos::Endian endianness = xenos::Endian::kNone;
  // The fetch swizzle it's shown with (replacement images hold the colours
  // the game shows).
  uint32_t swizzle = 0x688;  // XYZW
};

// ---------------------------------------------------------------------------
// Replacement descriptor returned to the injection path
// ---------------------------------------------------------------------------
struct TextureReplacementData {
  // Decoded pixels: RGBA8 unorm, row-major, tightly packed (no row padding).
  std::vector<uint8_t> pixels;
  uint32_t width = 0;
  uint32_t height = 0;
  // Number of mip levels present in the replacement file (>= 1).
  uint32_t mip_levels = 1;
  // Smaller levels made from `pixels` (level 1 first, down to 1x1), RGBA8 like
  // it.
  std::vector<std::vector<uint8_t>> mips;
  // The image already in the renderer's upload memory, made on the loading
  // thread (TextureCache::PrepareReplacementUploadImpl), so the frame that
  // first uses it only records copies; `pixels` and `mips` are dropped then,
  // and `prepared_bytes` is what it takes.
  std::shared_ptr<void> prepared_upload;
  uint64_t prepared_bytes = 0;
};

// ---------------------------------------------------------------------------
// TextureReplacement
// ---------------------------------------------------------------------------
class TextureReplacement {
 public:
  explicit TextureReplacement(std::filesystem::path textures_dir);
  ~TextureReplacement();

  TextureReplacement(const TextureReplacement&) = delete;
  TextureReplacement& operator=(const TextureReplacement&) = delete;

  // Rescans textures/replace/ and rebuilds the hash→path index.
  void Rescan();

  // How many replacement files are indexed.
  size_t GetReplacementCount() const;

  // ---------------------------------------------------------------------------
  // Dump path
  // ---------------------------------------------------------------------------
  // Untiles, endian-swaps, and writes the guest texture to a DDS file.
  //   guest_bytes  : raw physical memory at base_page << 12
  //   guest_size   : GetGuestBaseSize() bytes
  //   width/height : texel dimensions
  //   pitch_blocks : TextureKey::pitch (pitch in units of 32 texels)
  //   tiled        : TextureKey::tiled
  //   format       : TextureKey::format
  //   endianness   : TextureKey::endianness
  //   guest_swizzle: the fetch constant swizzle the texture was bound with -
  //                  when it only reorders the colour components (e.g. ARGB
  //                  data read with ZYXW), the dump holds the colours the
  //                  game shows, like replacement files do
  void DumpTexture(uint64_t content_hash, uint32_t width, uint32_t height, uint32_t pitch_blocks,
                   bool tiled, xenos::TextureFormat format, xenos::Endian endianness,
                   const uint8_t* guest_bytes, uint32_t guest_size,
                   uint32_t guest_swizzle = 0x688 /* XYZW */) const;

  // Replacement files hold the colours the game shows, not the guest's
  // component order. When a fetch swizzle only reorders the colour components
  // (its R, G and B read X, Y and Z in some order), sets channel_of[c] to the
  // RGBA channel of the shown image that guest component c ends up in (W ->
  // alpha) and returns true; otherwise (constants, a component used twice, W
  // used as a colour) returns false and the swizzle is applied as it is.
  static bool GetSwizzleColorOrder(uint32_t guest_swizzle, uint32_t channel_of[4]);

  // ---------------------------------------------------------------------------
  // Injection path
  // ---------------------------------------------------------------------------
  // Returns a pointer into the internal cache, or nullptr if not found.
  // The pointer is valid until the next call to Rescan(). The first call
  // starts decoding all the files in the background (texture_replace_preload),
  // ones not decoded yet are loaded here.
  [[nodiscard]] const TextureReplacementData* FindReplacement(uint64_t content_hash) const;
  // Whether there's a replacement for the hash, and its size - from the file's
  // header, without decoding it. The same as FindReplacement's width and
  // height when that succeeds.
  // The replacement is a JPEG (opaque): wrong for a texture with alpha.
  bool IsJpeg(uint64_t content_hash) const;
  bool FindReplacementSize(uint64_t content_hash, uint32_t& width, uint32_t& height) const;
  // A replacement that couldn't be used (e.g. a format the GPU upload doesn't
  // take): it isn't offered again, so the original texture is used.
  void MarkFailed(uint64_t content_hash) const;

  // Asynchronous loading (texture_replace_async): whether FindReplacement
  // would answer at once - the image is decoded in RAM (or is one of the
  // program's own), or it can't be used anyway.
  bool IsResident(uint64_t content_hash) const;
  // Decodes the replacement on a background thread, the most recently asked
  // first; the ready callback runs (on that thread) once it's in RAM. Does
  // nothing when it's already there or on its way.
  void RequestAsync(uint64_t content_hash) const;
  void SetReadyCallback(std::function<void()> callback) { ready_callback_ = std::move(callback); }
  // Puts a decoded image into the renderer's upload memory on the loading
  // threads (see TextureReplacementData::prepared_upload); null = not done.
  void SetPrepareCallback(
      std::function<std::shared_ptr<void>(const TextureReplacementData&)> callback) {
    prepare_callback_ = std::move(callback);
  }

  // Premultiplied alpha (texture_replace_alpha): whether the loading of this
  // replacement would use its original texture - the check is on, the file
  // isn't decoded yet and no original was given for it.
  bool WantsGuestReference(uint64_t content_hash) const;
  // The original for the next decoding of the replacement (kept until then).
  void SetGuestReference(uint64_t content_hash,
                         std::shared_ptr<const TextureReplacementGuest> guest) const;
  // Guest formats the alpha check can read (with alpha worth comparing).
  static bool CanDecodeGuestFormat(xenos::TextureFormat format);

  // The smaller mips of an RGBA8 image down to 1x1: each texel is the average
  // of 2x2 of the level above, weighted by alpha so the color of transparent
  // texels doesn't bleed into the edges of cutouts.
  static std::vector<std::vector<uint8_t>> BuildMips(const uint8_t* pixels, uint32_t width,
                                                     uint32_t height);

  // The biggest replacement side the texture cache takes (a texture key holds
  // up to 8192). Bigger images are downscaled to fit (area-averaged, the
  // shape kept).
  static constexpr uint32_t kMaxReplacementSize = xenos::kTexture2DCubeMaxWidthHeight;
  static void FitReplacementSize(uint32_t& width, uint32_t& height);

  // Premultiplied alpha (texture_replace_alpha). The game blends its textures
  // with straight alpha: a half-transparent texel holds its full colour. Some
  // upscaling tools export premultiplied alpha instead (colour x alpha, black
  // under transparent texels), which the game would darken a second time -
  // soft edges, glows, clouds, decals and alpha-masked layers come out dark or
  // black. Such images are converted to straight alpha when they're loaded.
  //
  // Each replacement is compared with the game's own texture it replaces
  // (given with the lookup - SetGuestReference): premultiplied when its colour
  // matches the original's colour x alpha, and then the original's colour
  // fills its (nearly) transparent texels - see replacement_alpha.h. Without
  // the original, its own texels and a survey of the pack decide:
  //
  // A premultiplied image's colour is never brighter than its alpha; in a
  // straight one most half-transparent texels are. Counts of a sample of an
  // image's texels:
  struct AlphaStats {
    uint64_t soft = 0;      // half-transparent (8 <= alpha <= 247)
    uint64_t brighter = 0;  // of those, max(R, G, B) > alpha + 8
    uint64_t sampled = 0;
    bool transparent = false;  // any texel with alpha below 255
  };
  static AlphaStats GetAlphaStats(const uint8_t* pixels, uint32_t width, uint32_t height);

  // ---------------------------------------------------------------------------
  // Hash
  // ---------------------------------------------------------------------------
  static uint64_t HashGuestData(const uint8_t* data, size_t size);
  // Cheap check of whether the content behind a cached hash is still the same
  // (samples of it, or all of a small one).
  static uint64_t FingerprintGuestData(const uint8_t* data, size_t size);

  std::filesystem::path dump_dir() const { return textures_dir_ / "dump"; }
  std::filesystem::path replace_dir() const { return textures_dir_ / "replace"; }

 private:
  enum class LoadResult {
    kLoaded,
    kFailed,
    // Its alpha has to be checked against its original, which wasn't given
    // yet (the preload, before the game asked for the texture): not cached,
    // loaded again when the game asks for it.
    kDeferred,
  };
  // Decodes a replacement file and makes its mips (straight alpha, at most
  // kMaxReplacementSize per side).
  LoadResult LoadReplacement(uint64_t hash, const std::filesystem::path& path,
                             TextureReplacementData& out, bool allow_defer) const;
  // Decodes an image file as it is (RGBA8).
  static bool DecodeFile(const std::filesystem::path& path, TextureReplacementData& out);
  // Premultiplied alpha to straight (texture_replace_alpha): compared with its
  // original when one was given (SetGuestReference), else decided by its own
  // texels and the pack's survey. False = deferred (see LoadResult).
  bool FixAlpha(uint64_t hash, const std::filesystem::path& path, TextureReplacementData& data,
                bool allow_defer) const;
  // The original's base level as RGBA8, in the colours the game shows.
  static bool DecodeGuest(const TextureReplacementGuest& guest, std::vector<uint8_t>& rgba);

  // Whether the pack's images have premultiplied alpha: texture_replace_alpha,
  // or with "auto" a look at a sample of them on background threads (started
  // with the first lookup), waited for by the loads.
  bool IsPackPremultiplied() const;
  void StartAlphaSurvey() const;  // alpha_mutex_ held
  void StopAlphaSurvey();
  void AlphaSurveyWorker() const;
  // Decides from the votes and wakes the waiting loads (alpha_mutex_ held).
  void FinishAlphaSurvey() const;
  enum class PackAlpha { kUnknown, kStraight, kPremultiplied };
  mutable std::mutex alpha_mutex_;
  mutable std::condition_variable alpha_decided_;
  mutable PackAlpha pack_alpha_ = PackAlpha::kUnknown;
  mutable bool alpha_survey_started_ = false;
  // The sample (files that can have alpha, small ones first), the next to
  // look at, the votes, and the threads.
  mutable std::vector<std::filesystem::path> alpha_survey_files_;
  mutable std::atomic<size_t> alpha_survey_next_{0};
  mutable std::atomic<uint32_t> alpha_survey_active_{0};
  mutable std::atomic<uint32_t> alpha_votes_premultiplied_{0};
  mutable std::atomic<uint32_t> alpha_votes_straight_{0};
  mutable std::atomic<uint32_t> alpha_votes_looked_{0};
  mutable std::atomic<bool> alpha_survey_stop_{false};
  mutable std::chrono::steady_clock::time_point alpha_survey_start_time_;
  mutable std::vector<std::unique_ptr<rex::thread::Thread>> alpha_survey_threads_;
  // RAM an image file takes decoded (RGBA8 with its mips), from its header; 0
  // if it can't be read.
  static uint64_t DecodedSize(const std::filesystem::path& path);
  // An image file's size from its header (PNG or DDS); false if unreadable.
  static bool ReadImageSize(const std::filesystem::path& path, uint32_t& width, uint32_t& height);
  // Bytes of a decoded texture in pixel_cache_.
  static uint64_t CachedSize(const TextureReplacementData& data);
  // Adds a decoded texture to pixel_cache_ (cache_mutex_ held).
  TextureReplacementData* AddToPixelCache(uint64_t hash, TextureReplacementData&& data) const;
  // Drops decoded textures, other than `keep`, while pixel_cache_ is over
  // texture_replace_ram_mb (cache_mutex_ held): the ones already handed out
  // first, least recently used first. Only on the thread asking for textures
  // (FindReplacement, RequestAsync): the pointers FindReplacement returned
  // before are used before it's called again.
  void TrimPixelCache(uint64_t keep) const;
  // Decodes the indexed files on background threads.
  void StartPreload() const;
  void StopPreload();
  void PreloadWorker() const;
  // The RequestAsync threads.
  void AsyncWorker() const;
  void StopAsync();

  std::filesystem::path textures_dir_;
  std::unordered_map<uint64_t, std::filesystem::path> replacements_;

  // Textures that have been loaded from disk are cached here so that
  // FindReplacement doesn't touch the filesystem again while they're in RAM.
  // Elements are never moved, so pointers to them stay valid while others are
  // added; the least recently used ones are dropped past texture_replace_ram_mb
  // (TrimPixelCache).
  mutable std::unordered_map<uint64_t, TextureReplacementData> pixel_cache_;
  // The program's in-memory replacements (SetTextureMemoryReplacement), with
  // their mips, once asked for. Never dropped (their keys keep their images).
  mutable std::unordered_map<uint64_t, TextureReplacementData> memory_cache_;
  // Per cached texture: its size and when it was last asked for.
  struct PixelCacheUse {
    uint64_t bytes = 0;
    uint64_t last_use = 0;
    // FindReplacement has handed it out (it's been uploaded): dropped before
    // the ones decoded in the background and not used yet.
    bool used = false;
  };
  mutable std::unordered_map<uint64_t, PixelCacheUse> pixel_cache_use_;
  mutable uint64_t pixel_cache_bytes_ = 0;
  mutable uint64_t pixel_cache_clock_ = 0;
  // TrimPixelCache's log, at most every few seconds.
  mutable std::chrono::steady_clock::time_point trim_log_time_{};
  mutable size_t trim_log_dropped_ = 0;
  // The originals given for replacements not decoded yet (SetGuestReference).
  mutable std::unordered_map<uint64_t, std::shared_ptr<const TextureReplacementGuest>> guest_refs_;
  // Textures compared with their originals: premultiplied / straight (for the
  // log), and how many were converted.
  mutable std::atomic<uint32_t> premultiplied_votes_{0};
  mutable std::atomic<uint32_t> straight_votes_{0};
  mutable std::atomic<uint32_t> alpha_fixed_count_{0};
  mutable std::atomic<bool> pack_verdict_logged_{false};
  // texture_replace_ram_mb in bytes, or the whole preloaded pack if that's
  // bigger (it fit the budget when the preload started).
  mutable uint64_t pixel_cache_budget_ = 0;
  // Hashes that failed to load are remembered so we don't retry every frame.
  mutable std::unordered_set<uint64_t> failed_cache_;
  // Sizes of the replacement files read so far (FindReplacementSize).
  mutable std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> size_cache_;
  // Guards the caches, shared with the preload threads.
  mutable std::mutex cache_mutex_;
  // Hashes being decoded right now (by the preload or FindReplacement), and
  // the signal for when one of them is done.
  mutable std::unordered_set<uint64_t> preload_in_progress_;
  mutable std::condition_variable preload_done_;

  mutable bool preload_started_ = false;
  mutable std::vector<std::pair<uint64_t, std::filesystem::path>> preload_queue_;
  mutable std::atomic<size_t> preload_next_{0};
  mutable std::atomic<size_t> preload_remaining_{0};
  mutable std::atomic<bool> preload_stop_{false};
  mutable std::chrono::steady_clock::time_point preload_start_time_;
  mutable std::vector<std::unique_ptr<rex::thread::Thread>> preload_threads_;

  // RequestAsync: the hashes waiting (newest at the back, taken first), the
  // threads (started with the first request) and their wake-up.
  mutable std::vector<uint64_t> async_queue_;
  mutable std::vector<std::unique_ptr<rex::thread::Thread>> async_threads_;
  mutable std::condition_variable async_wake_;
  mutable bool async_stop_ = false;
  std::function<void()> ready_callback_;
  std::function<std::shared_ptr<void>(const TextureReplacementData&)> prepare_callback_;
  // Makes the upload data of an image just decoded on a loading thread.
  void Prepare(TextureReplacementData& data) const;

  static bool WriteDDS_RGBA8(const std::filesystem::path& path, uint32_t width, uint32_t height,
                             const uint8_t* rgba8_rows, uint32_t row_pitch_bytes);

  static bool WriteDDS_BC(const std::filesystem::path& path, uint32_t width, uint32_t height,
                          const uint8_t* bc_blocks, uint32_t bytes_per_block, uint32_t fourcc);

  static bool ReadDDS(const std::filesystem::path& path, TextureReplacementData& out);

  static bool ReadPNG(const std::filesystem::path& path, TextureReplacementData& out);

  // Halves an RGBA8 image (2x2 texels each, colour weighted by alpha).
  static void Downsample2x(const uint8_t* source, uint32_t source_width, uint32_t source_height,
                           uint8_t* dest, uint32_t dest_width, uint32_t dest_height);
};

// Replacement images the program supplies from memory, used like the files of
// a texture pack (any size, mipmapped when loaded) whether or not
// texture_replace_enabled is on, and before the pack's files. `key` is the
// texture pack key of the guest texture they replace (see
// TiledTexture2DReplacementKey). A key keeps its image once set. Thread-safe.
struct TextureMemoryReplacement {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<uint8_t> rgba;  // R8G8B8A8, straight alpha, width * height * 4
};
void SetTextureMemoryReplacement(uint64_t key, uint32_t width, uint32_t height,
                                 std::vector<uint8_t> rgba);
std::shared_ptr<const TextureMemoryReplacement> FindTextureMemoryReplacement(uint64_t key);
bool HasTextureMemoryReplacements();

// The texture pack key of a 2D texture with one level, tiled as the GPU reads
// it, whose blocks (4x4-texel blocks of `bytes_per_block` - 8 or 16 - or
// single texels of 1, 2, 4 or 8 bytes with `block_texels` 1) are `blocks`,
// row by row, each as the guest stores it: XXH3 of the tiled base level with
// the padding taken as zero, like the texture cache (and the files' names).
uint64_t TiledTexture2DReplacementKey(const uint8_t* blocks, uint32_t width, uint32_t height,
                                      uint32_t bytes_per_block, uint32_t block_texels = 4);

}  // namespace rex::graphics
