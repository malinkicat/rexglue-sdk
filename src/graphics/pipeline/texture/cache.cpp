/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <rex/assert.h>
#include <rex/chrono/clock.h>
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/filesystem.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/pipeline/texture/cache.h>
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/pipeline/texture/replacement.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/xenos.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/perf/frame_rate.h>

// d3d12/command_processor.cpp: also logs the texture loads.
REXCVAR_DECLARE(int32_t, gpu_debug_log_draws);

REXCVAR_DEFINE_BOOL(texture_replace_stats, false, "GPU/Texture Replacement",
                    "Log texture replacement lookups and hashing per second");
REXCVAR_DEFINE_INT32(texture_cache_memory_limit_render_to_texture, 24, "GPU",
                     "Texture cache memory limit for render-to-texture (MB)")
    .range(1, 256)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_INT32(texture_cache_memory_limit_soft, 384, "GPU",
                     "Soft texture cache memory limit (MB)")
    .range(64, 4096)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_INT32(texture_cache_memory_limit_hard, 768, "GPU",
                     "Hard texture cache memory limit (MB)")
    .range(128, 8192)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_BOOL(texture_cache_pack_limits, true, "GPU",
                    "While a texture pack is used, raise the texture cache memory limits to fit "
                    "it (from the video memory: up to a quarter / half of it, at most 4096 / 8192 "
                    "MB, never below texture_cache_memory_limit_soft / _hard)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload)
    .debug_only();

REXCVAR_DEFINE_INT32(texture_cache_memory_limit_soft_lifetime, 30, "GPU",
                     "Soft texture cache memory limit lifetime (seconds)")
    .range(1, 3600);

REXCVAR_DEFINE_BOOL(gpu_3d_to_2d_texture, true, "GPU",
                    "Sample problematic 3D textures through 2D-compatible wrappers")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(anisotropic_override, 5, "GPU",
                     "Forces anisotropic filtering for eligible textures.\n"
                     "Higher values keep textures sharper at oblique angles, but increase texture "
                     "sampling cost.\n"
                     " -1 = No override\n"
                     "  0 = Disable anisotropic filtering\n"
                     "  1 = Force 1x anisotropic filtering\n"
                     "  2 = Force 2x anisotropic filtering\n"
                     "  3 = Force 4x anisotropic filtering\n"
                     "  4 = Force 8x anisotropic filtering\n"
                     "  5 = Force 16x anisotropic filtering")
    .range(-1, 5)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(draw_resolution_scale_x, 1, "GPU", "Draw resolution scale X (1 = no scaling)")
    .range(1, 8)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_INT32(draw_resolution_scale_y, 1, "GPU", "Draw resolution scale Y (1 = no scaling)")
    .range(1, 8)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_INT32(resolution_scale, 1, "GPU",
                     "Draw resolution scale for both X and Y axes (same as setting "
                     "draw_resolution_scale_x and draw_resolution_scale_y)")
    .range(1, 8)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_BOOL(pre_mask_resolve_l2_block, true, "GPU",
                    "Pre-mask scaled resolve L2 blocks to the write range before iterating");

REXCVAR_DEFINE_BOOL(texture_dump_enabled, false, "GPU/Texture Replacement",
                    "Dump loaded textures to DDS files for replacement authoring")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(texture_replace_enabled, false, "GPU/Texture Replacement",
                    "Inject replacement textures from disk when available")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(texture_replace_preload, true, "GPU/Texture Replacement",
                    "Decode all the replacement textures in the background at startup, so they "
                    "don't stutter the game the first time they're used (keeps all of them in "
                    "RAM)")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_BOOL(texture_replace_async, true, "GPU/Texture Replacement",
                    "Never wait for a replacement to decode: the original texture is shown until "
                    "the replacement is ready (a moment later), so new textures don't stutter the "
                    "game");

REXCVAR_DEFINE_INT32(texture_replace_ram_mb, 3072, "GPU/Texture Replacement",
                     "RAM for decoded replacement textures (MB). The least recently used ones are "
                     "dropped past it (and decoded again if needed), and a pack bigger than this "
                     "isn't preloaded")
    .range(256, 65536)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_STRING(texture_replace_alpha, "auto", "GPU/Texture Replacement",
                      "How the pack's images store transparency: auto (each image compared with "
                      "the game's own texture it replaces - or, without one, a sample of the "
                      "pack decides), straight (colour not multiplied by alpha, like the game's own "
                      "textures) or premultiplied (colour multiplied by alpha, as some upscaling "
                      "tools export it - converted to straight alpha when loaded; drawn as it is, "
                      "soft edges, glows, clouds and decals come out dark)")
    .allowed({"auto", "straight", "premultiplied"})
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_STRING(texture_folder, "", "GPU/Texture Replacement",
                      "Texture dump folder (empty = <executable folder>/textures)")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

// DEFINE_int32(
//     draw_resolution_scale_x, 1,
//     "Integer pixel width scale used for scaling the rendering resolution "
//     "opaquely to the game.\n"
//     "1, 2 and 3 may be supported, but support of anything above 1 depends on "
//     "the device properties, such as whether it supports sparse binding / tiled "
//     "resources, the number of virtual address bits per resource, and other "
//     "factors.\n"
//     "Various effects and parts of game rendering pipelines may work "
//     "incorrectly as pixels become ambiguous from the game's perspective and "
//     "because half-pixel offset (which normally doesn't affect coverage when "
//     "MSAA isn't used) becomes full-pixel.",
//     "GPU");
// DEFINE_int32(
//     draw_resolution_scale_y, 1,
//     "Integer pixel width scale used for scaling the rendering resolution "
//     "opaquely to the game.\n"
//     "See draw_resolution_scale_x for more information.",
//     "GPU");
// DEFINE_uint32(
//     texture_cache_memory_limit_soft, 384,
//     "Maximum host texture memory usage (in megabytes) above which old textures "
//     "will be destroyed.",
//     "GPU");
// DEFINE_uint32(
//     texture_cache_memory_limit_soft_lifetime, 30,
//     "Seconds a texture should be unused to be considered old enough to be "
//     "deleted if texture memory usage exceeds texture_cache_memory_limit_soft.",
//     "GPU");
// DEFINE_uint32(
//     texture_cache_memory_limit_hard, 768,
//     "Maximum host texture memory usage (in megabytes) above which textures "
//     "will be destroyed as soon as possible.",
//     "GPU");
// DEFINE_uint32(
//     texture_cache_memory_limit_render_to_texture, 24,
//     "Part of the host texture memory budget (in megabytes) that will be scaled "
//     "by the current drawing resolution scale.\n"
//     "If texture_cache_memory_limit_soft, for instance, is 384, and this is 24, "
//     "it will be assumed that the game will be using roughly 24 MB of "
//     "render-to-texture (resolve) targets and 384 - 24 = 360 MB of regular "
//     "textures - so with 2x2 resolution scaling, the soft limit will be 360 + "
//     "96 MB, and with 3x3, it will be 360 + 216 MB.",
//     "GPU");

namespace rex::graphics {

static std::filesystem::path GetTextureDumpFolder() {
  const std::string configured_texture_folder = REXCVAR_GET(texture_folder);
  if (!configured_texture_folder.empty()) {
    return rex::to_path(configured_texture_folder);
  }
  return rex::filesystem::GetExecutableFolder() / "textures";
}

const TextureCache::LoadShaderInfo TextureCache::load_shader_info_[kLoadShaderCount] = {
    // k8bpb
    {3, 4, 1, 4},
    // k16bpb
    {4, 4, 2, 4},
    // k32bpb
    {4, 4, 4, 3},
    // k64bpb
    {4, 4, 8, 2},
    // k128bpb
    {4, 4, 16, 1},
    // kR5G5B5A1ToB5G5R5A1
    {4, 4, 2, 4},
    // kR5G6B5ToB5G6R5
    {4, 4, 2, 4},
    // kR5G6B5ToRGBA8
    {4, 4, 4, 4},
    // kR5G5B6ToB5G6R5WithRBGASwizzle
    {4, 4, 2, 4},
    // kRGBA4ToBGRA4
    {4, 4, 2, 4},
    // kRGBA4ToARGB4
    {4, 4, 2, 4},
    // kRGBA4ToRGBA8
    {4, 4, 4, 4},
    // kGBGR8ToGRGB8
    {4, 4, 4, 3},
    // kGBGR8ToRGB8
    {4, 4, 8, 3},
    // kBGRG8ToRGBG8
    {4, 4, 4, 3},
    // kBGRG8ToRGB8
    {4, 4, 8, 3},
    // kR10G11B11ToRGBA16
    {4, 4, 8, 3},
    // kR10G11B11ToRGBA16SNorm
    {4, 4, 8, 3},
    // kR11G11B10ToRGBA16
    {4, 4, 8, 3},
    // kR11G11B10ToRGBA16SNorm
    {4, 4, 8, 3},
    // kR16UNormToFloat
    {4, 4, 2, 4},
    // kR16SNormToFloat
    {4, 4, 2, 4},
    // kRG16UNormToFloat
    {4, 4, 4, 3},
    // kRG16SNormToFloat
    {4, 4, 4, 3},
    // kRGBA16UNormToFloat
    {4, 4, 8, 2},
    // kRGBA16SNormToFloat
    {4, 4, 8, 2},
    // kDXT1ToRGBA8
    {4, 4, 4, 2},
    // kDXT3ToRGBA8
    {4, 4, 4, 1},
    // kDXT5ToRGBA8
    {4, 4, 4, 1},
    // kDXNToRG8
    {4, 4, 2, 1},
    // kDXT3A
    {4, 4, 1, 2},
    // kDXT3AAs1111ToBGRA4
    {4, 4, 2, 2},
    // kDXT3AAs1111ToARGB4
    {4, 4, 2, 2},
    // kDXT5AToR8
    {4, 4, 1, 2},
    // kCTX1
    {4, 4, 2, 2},
    // kDepthUnorm
    {4, 4, 4, 3},
    // kDepthFloat
    {4, 4, 4, 3},
};

TextureCache::TextureCache(const RegisterFile& register_file, SharedMemory& shared_memory,
                           uint32_t draw_resolution_scale_x, uint32_t draw_resolution_scale_y)
    : register_file_(register_file),
      shared_memory_(shared_memory),
      draw_resolution_scale_x_(draw_resolution_scale_x),
      draw_resolution_scale_y_(draw_resolution_scale_y) {
  assert_true(draw_resolution_scale_x >= 1);
  assert_true(draw_resolution_scale_x <= kMaxDrawResolutionScaleAlongAxis);
  assert_true(draw_resolution_scale_y >= 1);
  assert_true(draw_resolution_scale_y <= kMaxDrawResolutionScaleAlongAxis);

  // The pages last written by resolves: scaled textures are made from them
  // above 1x, and texture replacement skips them (render-to-texture data
  // changes all the time - finding a replacement for it would hash the whole
  // texture again each time it's used). At 1x they're tracked only once
  // texture dumping or replacement is on (EnsureTextureReplacement), so with
  // both off nothing changes.
  if (draw_resolution_scale_x > 1 || draw_resolution_scale_y > 1) {
    EnableResolvedPageTracking();
  }

  EnsureTextureReplacement();
}

void TextureCache::EnableResolvedPageTracking() {
  if (scaled_resolve_pages_) {
    return;
  }
  constexpr uint32_t kScaledResolvePageDwordCount = SharedMemory::kBufferSize / 4096 / 32;
  scaled_resolve_pages_ = std::unique_ptr<uint32_t[]>(new uint32_t[kScaledResolvePageDwordCount]);
  std::memset(scaled_resolve_pages_.get(), 0, kScaledResolvePageDwordCount * sizeof(uint32_t));
  std::memset(scaled_resolve_pages_l2_, 0, sizeof(scaled_resolve_pages_l2_));
  scaled_resolve_global_watch_handle_ =
      shared_memory().RegisterGlobalWatch(ScaledResolveGlobalWatchCallbackThunk, this);
}

void TextureCache::EnsureTextureReplacement() {
  // Nothing (no folder scan, no threads, no hashing) until texture_dump_enabled
  // or texture_replace_enabled is turned on; then made once.
  if (replacement_ ||
      (!REXCVAR_GET(texture_dump_enabled) && !REXCVAR_GET(texture_replace_enabled))) {
    return;
  }
  EnableResolvedPageTracking();
  InitTextureReplacement(GetTextureDumpFolder());
}

void TextureCache::InitTextureReplacement(const std::filesystem::path& textures_dir) {
  replacement_ = std::make_unique<TextureReplacement>(textures_dir);
  // A replacement decoded in the background: look the bound textures up again,
  // so it's used from the next draws.
  replacement_->SetReadyCallback(
      [this]() { texture_became_outdated_.store(true, std::memory_order_release); });
  // Its upload data made on the loading thread too, so the frame only copies.
  replacement_->SetPrepareCallback(
      [this](const TextureReplacementData& data) { return PrepareReplacementUploadImpl(data); });
}

TextureCache::~TextureCache() {
  DestroyAllTextures(true);

  if (scaled_resolve_global_watch_handle_) {
    shared_memory().UnregisterGlobalWatch(scaled_resolve_global_watch_handle_);
  }
}

namespace {

// The replacement key of a texture: XXH3 of its base level as the guest
// stores it, with the bytes the texture doesn't use - the padding of tiles
// and rows around its texels - taken as zero. Games leave whatever was in
// memory there, which would give the same texture a different key from run
// to run; zero is what the tools building a pack from the game files see.
// Textures without padding (or with a layout this doesn't handle - 3D,
// arrays, a base in the packed mip tail) hash their bytes as they are.
uint64_t HashReplacementBase(const texture_util::TextureGuestLayout& layout,
                             xenos::DataDimension dimension, xenos::TextureFormat format,
                             bool tiled, const uint8_t* guest_bytes, uint32_t guest_size) {
  const texture_util::TextureGuestLayout::Level& base = layout.base;
  const FormatInfo* format_info = FormatInfo::Get(format);
  const uint32_t bytes_per_block = format_info->bytes_per_block();
  if (dimension != xenos::DataDimension::k2DOrStacked || layout.array_size != 1 ||
      layout.packed_level == 0 || !bytes_per_block ||
      (bytes_per_block & (bytes_per_block - 1)) || !base.row_pitch_bytes) {
    return TextureReplacement::HashGuestData(guest_bytes, guest_size);
  }
  const uint32_t x_blocks = base.x_extent_blocks;
  const uint32_t y_blocks = base.y_extent_blocks;
  const uint32_t row_bytes = x_blocks * bytes_per_block;
  if (uint64_t(row_bytes) * y_blocks >= guest_size) {
    return TextureReplacement::HashGuestData(guest_bytes, guest_size);
  }
  std::vector<uint8_t> used(guest_size, 0);
  if (tiled) {
    const uint32_t bytes_per_block_log2 = rex::log2_floor(bytes_per_block);
    const uint32_t pitch_blocks = base.row_pitch_bytes / bytes_per_block;
    for (uint32_t y = 0; y < y_blocks; ++y) {
      for (uint32_t x = 0; x < x_blocks; ++x) {
        const uint32_t offset = uint32_t(
            texture_util::GetTiledOffset2D(int32_t(x), int32_t(y), pitch_blocks,
                                           bytes_per_block_log2));
        if (uint64_t(offset) + bytes_per_block <= guest_size) {
          std::memcpy(used.data() + offset, guest_bytes + offset, bytes_per_block);
        }
      }
    }
  } else {
    for (uint32_t y = 0; y < y_blocks; ++y) {
      const uint64_t offset = uint64_t(y) * base.row_pitch_bytes;
      if (offset + row_bytes > guest_size) {
        break;
      }
      std::memcpy(used.data() + offset, guest_bytes + offset, row_bytes);
    }
  }
  return TextureReplacement::HashGuestData(used.data(), used.size());
}

// What HashReplacementBase depends on besides the bytes.
uint64_t ReplacementLayoutSignature(const texture_util::TextureGuestLayout& layout,
                                    xenos::DataDimension dimension, xenos::TextureFormat format,
                                    bool tiled) {
  const uint32_t values[] = {uint32_t(dimension),
                             uint32_t(format),
                             uint32_t(tiled),
                             layout.array_size,
                             layout.packed_level,
                             layout.base.row_pitch_bytes,
                             layout.base.x_extent_blocks,
                             layout.base.y_extent_blocks};
  return TextureReplacement::HashGuestData(reinterpret_cast<const uint8_t*>(values),
                                           sizeof(values));
}

}  // namespace

bool TextureCache::GetConfigDrawResolutionScale(uint32_t& x_out, uint32_t& y_out) {
  uint32_t shared_scale = uint32_t(std::max(INT32_C(1), REXCVAR_GET(resolution_scale)));
  bool use_shared_scale = rex::cvar::HasNonDefaultValue("resolution_scale");
  uint32_t config_x = use_shared_scale && !rex::cvar::HasNonDefaultValue("draw_resolution_scale_x")
                          ? shared_scale
                          : uint32_t(std::max(INT32_C(1), REXCVAR_GET(draw_resolution_scale_x)));
  uint32_t config_y = use_shared_scale && !rex::cvar::HasNonDefaultValue("draw_resolution_scale_y")
                          ? shared_scale
                          : uint32_t(std::max(INT32_C(1), REXCVAR_GET(draw_resolution_scale_y)));
  uint32_t clamped_x = std::min(kMaxDrawResolutionScaleAlongAxis, config_x);
  uint32_t clamped_y = std::min(kMaxDrawResolutionScaleAlongAxis, config_y);
  x_out = clamped_x;
  y_out = clamped_y;
  return clamped_x == config_x && clamped_y == config_y;
}

void TextureCache::ClearCache() {
  DestroyAllTextures();
}

void TextureCache::GetMemoryLimits(uint32_t& soft_mb_out, uint32_t& hard_mb_out) {
  uint32_t soft_mb = uint32_t(std::max(REXCVAR_GET(texture_cache_memory_limit_soft), INT32_C(1)));
  uint32_t hard_mb = uint32_t(std::max(REXCVAR_GET(texture_cache_memory_limit_hard), INT32_C(1)));
  // A texture pack's images are several times the size of the originals (and
  // have full mip chains), so the defaults - made for the game's own textures
  // - can't hold the ones a scene uses, and textures would be destroyed and
  // made again from the decoded images all the time. The pack's limits come
  // from the video memory, and don't depend on the draw resolution scale (the
  // render-to-texture part below is added on top as usual).
  if (REXCVAR_GET(texture_cache_pack_limits) && REXCVAR_GET(texture_replace_enabled) &&
      replacement_ && replacement_->GetReplacementCount()) {
    if (!pack_limit_hard_mb_) {
      uint64_t budget = 0, usage = 0;
      if (!QueryLocalVideoMemory(budget, usage)) {
        budget = 0;
      }
      const uint64_t budget_mb = budget >> 20;
      if (budget_mb) {
        pack_limit_soft_mb_ = uint32_t(std::min<uint64_t>(budget_mb / 4, 4096));
        pack_limit_hard_mb_ = uint32_t(std::min<uint64_t>(budget_mb / 2, 8192));
      } else {
        pack_limit_soft_mb_ = 1024;
        pack_limit_hard_mb_ = 2048;
      }
      const uint32_t log_soft = std::max(soft_mb, pack_limit_soft_mb_);
      const uint32_t log_hard = std::max(hard_mb, pack_limit_hard_mb_);
      // Once per value (the cache is made again when the scale changes).
      static uint64_t logged_limits = 0;
      const uint64_t limits = (uint64_t(log_soft) << 32) | log_hard;
      if (logged_limits != limits) {
        logged_limits = limits;
        REXLOG_INFO(
            "Texture cache: a texture pack is in use ({} replacements) - memory limits soft {} "
            "MB / hard {} MB (video memory budget {} MB{}; texture_cache_memory_limit_soft / "
            "_hard {} / {} MB)",
            replacement_->GetReplacementCount(), log_soft, log_hard, budget_mb,
            budget_mb ? "" : " unknown", soft_mb, hard_mb);
      }
    }
    soft_mb = std::max(soft_mb, pack_limit_soft_mb_);
    hard_mb = std::max(hard_mb, pack_limit_hard_mb_);
  }
  // texture_cache_memory_limit_render_to_texture is assumed to be included in
  // texture_cache_memory_limit_soft and texture_cache_memory_limit_hard, at 1x,
  // so subtracting 1 from the scale.
  const uint32_t limit_scaled_resolve_add_mb =
      REXCVAR_GET(texture_cache_memory_limit_render_to_texture) *
      (draw_resolution_scale_x() * draw_resolution_scale_y() - 1);
  soft_mb_out = soft_mb + limit_scaled_resolve_add_mb;
  hard_mb_out = hard_mb + limit_scaled_resolve_add_mb;
}

void TextureCache::CompletedSubmissionUpdated(uint64_t completed_submission_index) {
  // If memory usage is too high, destroy unused textures.
  uint64_t current_time = rex::chrono::Clock::QueryHostUptimeMillis();
  uint32_t limit_soft_mb, limit_hard_mb;
  GetMemoryLimits(limit_soft_mb, limit_hard_mb);
  {
    rex::perf::TextureCacheStats& stats = rex::perf::GetTextureCacheStats();
    stats.memory_bytes.store(textures_total_host_memory_usage_, std::memory_order_relaxed);
    stats.limit_soft_mb.store(limit_soft_mb, std::memory_order_relaxed);
    stats.limit_hard_mb.store(limit_hard_mb, std::memory_order_relaxed);
    // The process's video memory, for perf_report - once a second.
    static uint64_t last_vram_query_time = 0;
    if (current_time - last_vram_query_time >= 1000) {
      last_vram_query_time = current_time;
      uint64_t budget = 0, usage = 0;
      if (QueryLocalVideoMemory(budget, usage)) {
        stats.vram_budget_bytes.store(budget, std::memory_order_relaxed);
        stats.vram_usage_bytes.store(usage, std::memory_order_relaxed);
      }
    }
  }
  uint32_t limit_soft_lifetime = REXCVAR_GET(texture_cache_memory_limit_soft_lifetime) * 1000;
  bool destroyed_any = false;
  while (texture_used_first_ != nullptr) {
    uint64_t total_host_memory_usage_mb =
        (textures_total_host_memory_usage_ + ((UINT32_C(1) << 20) - 1)) >> 20;
    bool limit_hard_exceeded = total_host_memory_usage_mb > limit_hard_mb;
    if (total_host_memory_usage_mb <= limit_soft_mb && !limit_hard_exceeded) {
      break;
    }
    Texture* texture = texture_used_first_;
    if (texture->last_usage_submission_index() > completed_submission_index) {
      break;
    }
    if (!limit_hard_exceeded && (texture->last_usage_time() + limit_soft_lifetime) > current_time) {
      break;
    }
    if (!destroyed_any) {
      destroyed_any = true;
      // The texture being destroyed might have been bound in the previous
      // submissions, and nothing has overwritten the binding yet, so completion
      // of the submission where the texture was last actually used on the GPU
      // doesn't imply that it's not bound currently. Reset bindings if
      // any texture has been destroyed.
      ResetTextureBindings();
    }
    // Remove the texture from the map and destroy it via its unique_ptr.
    auto found_texture_it = textures_.find(texture->key());
    assert_true(found_texture_it != textures_.end());
    if (found_texture_it != textures_.end()) {
      assert_true(found_texture_it->second.get() == texture);
      textures_.erase(found_texture_it);
      // `texture` is invalid now.
      rex::perf::GetTextureCacheStats().destroyed.fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (destroyed_any) {
    COUNT_profile_set("gpu/texture_cache/textures", textures_.size());
  }
}

void TextureCache::BeginSubmission(uint64_t new_submission_index) {
  assert_true(new_submission_index > current_submission_index_);
  current_submission_index_ = new_submission_index;
  current_submission_time_ = rex::chrono::Clock::QueryHostUptimeMillis();
}

void TextureCache::BeginFrame() {
  // In case there was a failure to create something in the previous frame, make
  // sure bindings are reset so a new attempt will surely be made if the texture
  // is requested again.
  ResetTextureBindings();
}

void TextureCache::MarkRangeAsResolved(uint32_t start_unscaled, uint32_t length_unscaled) {
  if (length_unscaled == 0) {
    return;
  }
  start_unscaled &= 0x1FFFFFFF;
  length_unscaled = std::min(length_unscaled, 0x20000000 - start_unscaled);

  // (Tracked above 1x, or with texture dumping / replacement on.)
  if (scaled_resolve_pages_) {
    uint32_t page_first = start_unscaled >> 12;
    uint32_t page_last = (start_unscaled + length_unscaled - 1) >> 12;
    uint32_t block_first = page_first >> 5;
    uint32_t block_last = page_last >> 5;
    auto global_lock = global_critical_region_.Acquire();
    for (uint32_t i = block_first; i <= block_last; ++i) {
      uint32_t add_bits = UINT32_MAX;
      if (i == block_first) {
        add_bits &= ~((UINT32_C(1) << (page_first & 31)) - 1);
      }
      if (i == block_last && (page_last & 31) != 31) {
        add_bits &= (UINT32_C(1) << ((page_last & 31) + 1)) - 1;
      }
      scaled_resolve_pages_[i] |= add_bits;
      scaled_resolve_pages_l2_[i >> 6] |= UINT64_C(1) << (i & 63);
    }
  }

  // Invalidate textures. Toggling individual textures between scaled and
  // unscaled also relies on invalidation through shared memory.
  shared_memory().RangeWrittenByGpu(start_unscaled, length_unscaled);
}

uint32_t TextureCache::GuestToHostSwizzle(uint32_t guest_swizzle, uint32_t host_format_swizzle) {
  uint32_t host_swizzle = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t guest_swizzle_component = (guest_swizzle >> (3 * i)) & 0b111;
    uint32_t host_swizzle_component;
    if (guest_swizzle_component >= xenos::XE_GPU_TEXTURE_SWIZZLE_0) {
      // Get rid of 6 and 7 values (to prevent host GPU errors if the game has
      // something broken) the simple way - by changing them to 4 (0) and 5 (1).
      host_swizzle_component = guest_swizzle_component & 0b101;
    } else {
      host_swizzle_component = (host_format_swizzle >> (3 * guest_swizzle_component)) & 0b111;
    }
    host_swizzle |= host_swizzle_component << (3 * i);
  }
  return host_swizzle;
}

uint32_t TextureCache::ReplacementHostSwizzle(uint32_t guest_swizzle,
                                              uint32_t host_format_swizzle) {
  // Only formats whose host texture keeps the guest components in place
  // (RGBA-like: 8_8_8_8, DXT1-5, ...); single- and two-channel formats
  // replicate components and keep the plain mapping.
  uint32_t channel_of[4];
  if (host_format_swizzle != xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA ||
      !TextureReplacement::GetSwizzleColorOrder(guest_swizzle, channel_of)) {
    return GuestToHostSwizzle(guest_swizzle, host_format_swizzle);
  }
  uint32_t host_swizzle = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t c = (guest_swizzle >> (3 * i)) & 0b111;
    const uint32_t host_c = c >= xenos::XE_GPU_TEXTURE_SWIZZLE_0 ? (c & 0b101) : channel_of[c];
    host_swizzle |= host_c << (3 * i);
  }
  return host_swizzle;
}

bool TextureCache::PrepareTextureLoad(Texture& texture, PendingTextureLoad& pending_load_out,
                                      PendingSharedMemoryRange* pending_ranges_out,
                                      size_t& pending_range_count_out) {
  uint32_t outdated_mask = texture.outdated_mask();
  if (!outdated_mask) {
    return false;
  }

  bool base_outdated = false;
  bool mips_outdated = false;
  {
    auto global_lock = global_critical_region_.Acquire();
    if (outdated_mask & Texture::kOutdatedBitBase) {
      base_outdated = texture.base_outdated(global_lock);
    }
    if (outdated_mask & Texture::kOutdatedBitMips) {
      mips_outdated = texture.mips_outdated(global_lock);
    }
  }
  if (!base_outdated && !mips_outdated) {
    return false;
  }

  pending_load_out.texture = &texture;
  pending_load_out.load_base = base_outdated;
  pending_load_out.load_mips = mips_outdated;
  pending_range_count_out = 0;

  TextureKey texture_key = texture.key();
  if (base_outdated) {
    PendingSharedMemoryRange pending_range;
    pending_range.start = texture_key.base_page << 12;
    pending_range.length =
        static_cast<uint32_t>(rex::align(texture.GetGuestBaseSize(), UINT32_C(16)));
    pending_ranges_out[pending_range_count_out++] = pending_range;
  }
  if (mips_outdated) {
    PendingSharedMemoryRange pending_range;
    pending_range.start = texture_key.mip_page << 12;
    pending_range.length =
        static_cast<uint32_t>(rex::align(texture.GetGuestMipsSize(), UINT32_C(16)));
    pending_ranges_out[pending_range_count_out++] = pending_range;
  }

  return true;
}

bool TextureCache::CommitPreparedTextureLoad(const PendingTextureLoad& pending_load) {
  if (!pending_load.texture || (!pending_load.load_base && !pending_load.load_mips)) {
    return true;
  }

  Texture& texture = *pending_load.texture;
  TextureKey texture_key = texture.key();
  if (texture_key.scaled_resolve) {
    // Make sure all the scaled resolve memory is resident and accessible from
    // the shader, including any possible padding that hasn't yet been touched
    // by an actual resolve, but is still included in the texture size, so the
    // GPU won't be trying to access unmapped memory.
    if (pending_load.load_base && !EnsureScaledResolveMemoryCommitted(
                                      texture_key.base_page << 12, texture.GetGuestBaseSize(), 4)) {
      return false;
    }
    if (pending_load.load_mips && !EnsureScaledResolveMemoryCommitted(
                                      texture_key.mip_page << 12, texture.GetGuestMipsSize(), 4)) {
      return false;
    }
  }

  bool uploaded_replacement = false;
  if (texture.replacement_content_hash_ != 0 && replacement_) {
    const auto upload_start = std::chrono::steady_clock::now();
    const TextureReplacementData* replacement =
        replacement_->FindReplacement(texture.replacement_content_hash_);
    const auto found = std::chrono::steady_clock::now();
    if (replacement) {
      uploaded_replacement = LoadTextureDataFromReplacementImpl(texture, *replacement);
    }
    const auto upload_end = std::chrono::steady_clock::now();
    const double upload_ms =
        std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
    if (upload_ms > 8.0 && replacement) {
      REXLOG_INFO("TextureReplacement: {}x{} took {:.1f} ms (find {:.1f} ms, upload {:.1f} ms)",
                  replacement->width, replacement->height, upload_ms,
                  std::chrono::duration<double, std::milli>(found - upload_start).count(),
                  std::chrono::duration<double, std::milli>(upload_end - found).count());
    }
    // The texture has the replacement's size and format: the guest data can't
    // go in. The file is marked failed and the bindings looked up again, so
    // the original texture is made instead.
    if (!uploaded_replacement) {
      replacement_->MarkFailed(texture.replacement_content_hash_);
      replacement_memo_.clear();  // lookups that resolved to it go through again
      texture_became_outdated_.store(true, std::memory_order_release);
      return false;
    }
  }

  if (!uploaded_replacement &&
      !LoadTextureDataFromResidentMemoryImpl(texture, pending_load.load_base,
                                             pending_load.load_mips)) {
    return false;
  }

  // Dump the originals (a replaced texture's key has the replacement's size
  // and format, and its original is in the pack already).
  if (replacement_ && pending_load.load_base && REXCVAR_GET(texture_dump_enabled) &&
      !texture_key.scaled_resolve && !texture.replacement_content_hash_) {
    const uint32_t guest_addr = texture_key.base_page << 12;
    const uint32_t guest_size = texture.GetGuestBaseSize();
    if (guest_size > 0 && !IsRangeResolved(guest_addr, guest_size)) {
      const uint8_t* guest_bytes = shared_memory().TranslatePhysical(guest_addr);
      const uint64_t content_hash =
          HashReplacementBase(texture.guest_layout(), texture_key.dimension, texture_key.format,
                              texture_key.tiled != 0, guest_bytes, guest_size);
      replacement_->DumpTexture(content_hash, texture_key.GetWidth(), texture_key.GetHeight(),
                                texture_key.pitch, texture_key.tiled != 0, texture_key.format,
                                texture_key.endianness, guest_bytes, guest_size,
                                texture.last_guest_swizzle_);
    }
  }

  // Mark the ranges as uploaded and watch them. This is needed for scaled
  // resolves as well to detect when the CPU wants to reuse the memory for a
  // regular texture or a vertex buffer, and thus the scaled resolve version is
  // not up to date anymore.
  texture.MakeUpToDateAndWatch(global_critical_region_.Acquire());
  texture.LogAction("Loaded");
  {
    rex::perf::TextureCacheStats& stats = rex::perf::GetTextureCacheStats();
    stats.loaded.fetch_add(1, std::memory_order_relaxed);
    if (uploaded_replacement) {
      stats.replacement_uploads.fetch_add(1, std::memory_order_relaxed);
    }
    if (rex::perf::IsGpuProfiling()) {
      rex::perf::GpuProfileStats& gpu_stats = rex::perf::GetGpuProfileStats();
      gpu_stats.texture_loads.fetch_add(1, std::memory_order_relaxed);
      gpu_stats.texture_load_bytes.fetch_add(texture.GetHostMemoryUsage(),
                                             std::memory_order_relaxed);
    }
    if (REXCVAR_GET(gpu_debug_log_draws) > 0) {
      REXGPU_INFO("TEXLOAD {:08X} {}x{} fmt {} scaled {} mips {} host {} KB{}",
                  texture_key.base_page << 12, texture_key.GetWidth(), texture_key.GetHeight(),
                  uint32_t(texture_key.format), uint32_t(texture_key.scaled_resolve),
                  uint32_t(texture_key.mip_max_level), texture.GetHostMemoryUsage() >> 10,
                  uploaded_replacement ? " replacement" : "");
    }
  }

  return true;
}

void TextureCache::RequestTextures(uint32_t used_texture_mask) {
  const auto& regs = register_file();

  // Texture dumping / replacement turned on while running (hot reload).
  if (!replacement_) {
    EnsureTextureReplacement();
  }

  // A plain load first: the exchange (a locked instruction) on every draw is
  // costly, and the flag is rarely set.
  if (texture_became_outdated_.load(std::memory_order_relaxed) &&
      texture_became_outdated_.exchange(false, std::memory_order_acquire)) {
    // A texture has become outdated - make sure whether textures are outdated
    // is rechecked in this draw and in subsequent ones to reload the new data
    // if needed.
    ResetTextureBindings();
    ++outdated_binding_resets_;
  }
  requested_bindings_ += uint32_t(rex::bit_count(used_texture_mask & ~texture_bindings_in_sync_));
  ++request_textures_calls_;

  // Update the texture keys and the textures.
  uint32_t bindings_changed = 0;
  // Reused (called for every draw, on the command processor thread).
  static thread_local std::vector<PendingTextureLoad> pending_texture_loads;
  static thread_local std::vector<PendingSharedMemoryRange> pending_shared_memory_ranges;
  pending_texture_loads.clear();
  pending_shared_memory_ranges.clear();
  auto queue_pending_texture_load = [&](Texture* texture) {
    if (texture == nullptr) {
      return;
    }
    for (const PendingTextureLoad& pending_load : pending_texture_loads) {
      if (pending_load.texture == texture) {
        return;
      }
    }
    PendingTextureLoad pending_load;
    PendingSharedMemoryRange pending_ranges[2];
    size_t pending_range_count = 0;
    if (!PrepareTextureLoad(*texture, pending_load, pending_ranges, pending_range_count)) {
      return;
    }
    pending_texture_loads.push_back(pending_load);
    for (size_t i = 0; i < pending_range_count; ++i) {
      pending_shared_memory_ranges.push_back(pending_ranges[i]);
    }
  };
  uint32_t textures_remaining = used_texture_mask & ~texture_bindings_in_sync_;
  uint32_t index = 0;
  while (rex::bit_scan_forward(textures_remaining, &index)) {
    uint32_t index_bit = UINT32_C(1) << index;
    textures_remaining &= ~index_bit;
    TextureBinding& binding = texture_bindings_[index];
    xenos::xe_gpu_texture_fetch_t fetch = regs.GetTextureFetch(index);
    TextureKey old_key = binding.key;
    uint8_t old_swizzled_signs = binding.swizzled_signs;
    BindingInfoFromFetchConstant(fetch, binding.key, &binding.swizzled_signs);
    texture_bindings_in_sync_ |= index_bit;
    // For FindOrCreateTexture below (a replacement's original is compared in
    // the colours this binding shows).
    replacement_lookup_swizzle_ = fetch.swizzle;
    if (!binding.key.is_valid) {
      if (old_key.is_valid) {
        bindings_changed |= index_bit;
      }
      binding.Reset();
      continue;
    }
    uint32_t old_host_swizzle = binding.host_swizzle;
    binding.host_swizzle = GuestToHostSwizzle(fetch.swizzle, GetHostFormatSwizzle(binding.key));
    if (binding.key == old_key && binding.texture && binding.texture->replacement_content_hash_) {
      binding.host_swizzle =
          ReplacementHostSwizzle(fetch.swizzle, GetHostFormatSwizzle(binding.key));
    }

    // Check if need to load the unsigned and the signed versions of the texture
    // (if the format is emulated with different host bit representations for
    // signed and unsigned - otherwise only the unsigned one is loaded).
    bool key_changed = binding.key != old_key;
    bool any_sign_was_not_signed = texture_util::IsAnySignNotSigned(old_swizzled_signs);
    bool any_sign_was_signed = texture_util::IsAnySignSigned(old_swizzled_signs);
    bool any_sign_is_not_signed = texture_util::IsAnySignNotSigned(binding.swizzled_signs);
    bool any_sign_is_signed = texture_util::IsAnySignSigned(binding.swizzled_signs);
    if (key_changed || binding.host_swizzle != old_host_swizzle ||
        any_sign_is_not_signed != any_sign_was_not_signed ||
        any_sign_is_signed != any_sign_was_signed) {
      bindings_changed |= index_bit;
    }
    bool load_unsigned_data = false, load_signed_data = false;
    if (IsSignedVersionSeparateForFormat(binding.key)) {
      // Can reuse previously loaded unsigned/signed versions if the key is the
      // same and the texture was previously bound as unsigned/signed
      // respectively (checking the previous values of signedness rather than
      // binding.texture != nullptr and binding.texture_signed != nullptr also
      // prevents repeated attempts to load the texture if it has failed to
      // load).
      if (any_sign_is_not_signed) {
        if (key_changed || !any_sign_was_not_signed) {
          binding.texture = FindOrCreateTexture(binding.key);
          load_unsigned_data = true;
        }
      } else {
        binding.texture = nullptr;
      }
      if (any_sign_is_signed) {
        if (key_changed || !any_sign_was_signed) {
          TextureKey signed_key = binding.key;
          signed_key.signed_separate = 1;
          binding.texture_signed = FindOrCreateTexture(signed_key);
          load_signed_data = true;
        }
      } else {
        binding.texture_signed = nullptr;
      }
    } else {
      // Same resource for both unsigned and signed, but descriptor formats may
      // be different.
      if (key_changed) {
        binding.texture = FindOrCreateTexture(binding.key);
        load_unsigned_data = true;
      }
      binding.texture_signed = nullptr;
    }
    // A replacement image holds the colours the game shows: guest data
    // stored in another component order (ARGB read with ZYXW) must not be
    // reordered once more.
    if (binding.texture && binding.texture->replacement_content_hash_) {
      const uint32_t replacement_swizzle =
          ReplacementHostSwizzle(fetch.swizzle, GetHostFormatSwizzle(binding.key));
      if (replacement_swizzle != binding.host_swizzle) {
        // Once per replacement (bindings are looked up again every frame).
        static std::unordered_set<uint64_t> logged;
        if (logged.size() < 256 && logged.insert(binding.texture->replacement_content_hash_).second) {
          REXLOG_INFO(
              "TextureReplacement: {:016x} (guest format {}, fetch swizzle {:03o}) shown in "
              "the replacement's own colour order",
              binding.texture->replacement_content_hash_, uint32_t(binding.key.format),
              uint32_t(fetch.swizzle));
        }
        binding.host_swizzle = replacement_swizzle;
        if (replacement_swizzle != old_host_swizzle) {
          bindings_changed |= index_bit;
        }
      }
    }
    if (binding.texture) {
      binding.texture->last_guest_swizzle_ = fetch.swizzle;
    }
    if (load_unsigned_data) {
      queue_pending_texture_load(binding.texture);
    }
    if (load_signed_data) {
      queue_pending_texture_load(binding.texture_signed);
    }
  }

  replacement_lookup_swizzle_ = 0x688;  // XYZW

  COUNT_profile_set("gpu/texture_cache/request_textures_pending_load_count",
                    uint32_t(pending_texture_loads.size()));
  COUNT_profile_set("gpu/texture_cache/request_textures_pending_range_count",
                    uint32_t(pending_shared_memory_ranges.size()));

  bool batched_shared_memory_request_succeeded = true;
  static thread_local std::vector<std::pair<uint32_t, uint32_t>> pending_shared_memory_range_pairs;
  pending_shared_memory_range_pairs.clear();
  if (!pending_shared_memory_ranges.empty()) {
    for (const PendingSharedMemoryRange& pending_range : pending_shared_memory_ranges) {
      pending_shared_memory_range_pairs.emplace_back(pending_range.start, pending_range.length);
    }
    batched_shared_memory_request_succeeded = shared_memory().RequestRanges(
        pending_shared_memory_range_pairs.data(), pending_shared_memory_range_pairs.size());
  }

  if (batched_shared_memory_request_succeeded) {
    for (const PendingTextureLoad& pending_load : pending_texture_loads) {
      CommitPreparedTextureLoad(pending_load);
    }
  } else {
    for (const PendingTextureLoad& pending_load : pending_texture_loads) {
      if (pending_load.texture != nullptr) {
        LoadTextureData(*pending_load.texture);
      }
    }
  }
  if (bindings_changed) {
    UpdateTextureBindingsImpl(bindings_changed);
  }
}

const char* TextureCache::TextureKey::GetLogDimensionName(xenos::DataDimension dimension) {
  switch (dimension) {
    case xenos::DataDimension::k1D:
      return "1D";
    case xenos::DataDimension::k2DOrStacked:
      return "2D";
    case xenos::DataDimension::k3D:
      return "3D";
    case xenos::DataDimension::kCube:
      return "cube";
    default:
      assert_unhandled_case(dimension);
      return "unknown";
  }
}

void TextureCache::TextureKey::LogAction(const char* action) const {
  REXGPU_TRACE(
      "{} {} {}{}x{}x{} {} {} texture with {} {}packed mip level{}, "
      "base at 0x{:08X} (pitch {}), mips at 0x{:08X}",
      action, tiled ? "tiled" : "linear", scaled_resolve ? "scaled " : "", GetWidth(), GetHeight(),
      GetDepthOrArraySize(), GetLogDimensionName(), FormatInfo::Get(format)->name,
      mip_max_level + 1, packed_mips ? "" : "un", mip_max_level != 0 ? "s" : "", base_page << 12,
      pitch << 5, mip_page << 12);
}

void TextureCache::Texture::LogAction(const char* action) const {
  REXGPU_TRACE(
      "{} {} {}{}x{}x{} {} {} texture with {} {}packed mip level{}, "
      "base at 0x{:08X} (pitch {}, size 0x{:08X}), mips at 0x{:08X} (size "
      "0x{:08X})",
      action, key_.tiled ? "tiled" : "linear", key_.scaled_resolve ? "scaled " : "",
      key_.GetWidth(), key_.GetHeight(), key_.GetDepthOrArraySize(), key_.GetLogDimensionName(),
      FormatInfo::Get(key_.format)->name, key_.mip_max_level + 1, key_.packed_mips ? "" : "un",
      key_.mip_max_level != 0 ? "s" : "", key_.base_page << 12, key_.pitch << 5, GetGuestBaseSize(),
      key_.mip_page << 12, GetGuestMipsSize());
}

// The texture must be in the recent usage list. Place it in front now because
// after creation, the texture will likely be used immediately, and it should
// not be destroyed immediately after creation if dropping of old textures is
// performed somehow. The list is maintained by the Texture, not the
// TextureCache itself (unlike the `textures_` container).
TextureCache::Texture::Texture(TextureCache& texture_cache, const TextureKey& key, bool track_usage)
    : texture_cache_(texture_cache),
      key_(key),
      guest_layout_(key.GetGuestLayout()),
      last_usage_submission_index_(texture_cache.current_submission_index_),
      last_usage_time_(texture_cache.current_submission_time_),
      used_previous_(track_usage ? texture_cache.texture_used_last_ : nullptr),
      used_next_(nullptr),
      in_usage_list_(track_usage) {
  if (track_usage) {
    if (texture_cache.texture_used_last_) {
      texture_cache.texture_used_last_->used_next_ = this;
    } else {
      texture_cache.texture_used_first_ = this;
    }
    texture_cache.texture_used_last_ = this;
  }

  // Never try to upload data that doesn't exist.
  base_outdated_ = guest_layout().base.level_data_extent_bytes != 0;
  mips_outdated_ = guest_layout().mips_total_extent_bytes != 0;
  outdated_mask_.store(
      (base_outdated_ ? kOutdatedBitBase : 0) | (mips_outdated_ ? kOutdatedBitMips : 0),
      std::memory_order_relaxed);
}

TextureCache::Texture::~Texture() {
  if (mips_watch_handle_) {
    texture_cache().shared_memory().UnwatchMemoryRange(mips_watch_handle_);
  }
  if (base_watch_handle_) {
    texture_cache().shared_memory().UnwatchMemoryRange(base_watch_handle_);
  }

  if (in_usage_list_) {
    if (used_previous_) {
      used_previous_->used_next_ = used_next_;
    } else {
      texture_cache_.texture_used_first_ = used_next_;
    }
    if (used_next_) {
      used_next_->used_previous_ = used_previous_;
    } else {
      texture_cache_.texture_used_last_ = used_previous_;
    }
  }

  texture_cache_.UpdateTexturesTotalHostMemoryUsage(0, host_memory_usage_);
}

void TextureCache::Texture::MakeUpToDateAndWatch(
    const std::unique_lock<std::recursive_mutex>& global_lock) {
  SharedMemory& shared_memory = texture_cache().shared_memory();
  if (base_outdated_) {
    assert_not_zero(GetGuestBaseSize());
    base_outdated_ = false;
    base_watch_handle_ = shared_memory.WatchMemoryRange(
        key().base_page << 12, GetGuestBaseSize(), TextureCache::WatchCallback, this, nullptr, 0);
    outdated_mask_.fetch_and(~kOutdatedBitBase, std::memory_order_release);
  }
  if (mips_outdated_) {
    assert_not_zero(GetGuestMipsSize());
    mips_outdated_ = false;
    mips_watch_handle_ = shared_memory.WatchMemoryRange(
        key().mip_page << 12, GetGuestMipsSize(), TextureCache::WatchCallback, this, nullptr, 1);
    outdated_mask_.fetch_and(~kOutdatedBitMips, std::memory_order_release);
  }
}

void TextureCache::Texture::MarkAsUsed() {
  if (!in_usage_list_) {
    return;
  }
  assert_true(last_usage_submission_index_ <= texture_cache_.current_submission_index_);
  // This is called very frequently, don't relink unless needed for caching.
  if (last_usage_submission_index_ >= texture_cache_.current_submission_index_) {
    return;
  }
  last_usage_submission_index_ = texture_cache_.current_submission_index_;
  last_usage_time_ = texture_cache_.current_submission_time_;
  if (used_next_ == nullptr) {
    // Already the most recently used.
    return;
  }
  if (used_previous_ != nullptr) {
    used_previous_->used_next_ = used_next_;
  } else {
    texture_cache_.texture_used_first_ = used_next_;
  }
  used_next_->used_previous_ = used_previous_;
  used_previous_ = texture_cache_.texture_used_last_;
  used_next_ = nullptr;
  texture_cache_.texture_used_last_->used_next_ = this;
  texture_cache_.texture_used_last_ = this;
}

void TextureCache::Texture::WatchCallback(
    [[maybe_unused]] const std::unique_lock<std::recursive_mutex>& global_lock, bool is_mip) {
  if (is_mip) {
    assert_not_zero(GetGuestMipsSize());
    mips_outdated_ = true;
    mips_watch_handle_ = nullptr;
    outdated_mask_.fetch_or(kOutdatedBitMips, std::memory_order_release);
  } else {
    assert_not_zero(GetGuestBaseSize());
    base_outdated_ = true;
    base_watch_handle_ = nullptr;
    outdated_mask_.fetch_or(kOutdatedBitBase, std::memory_order_release);
    texture_cache_.InvalidateHashCache(key().base_page);
  }
}

void TextureCache::WatchCallback(const std::unique_lock<std::recursive_mutex>& global_lock,
                                 void* context, void* data, uint64_t argument,
                                 bool invalidated_by_gpu) {
  Texture& texture = *static_cast<Texture*>(context);
  texture.WatchCallback(global_lock, argument != 0);
  texture.texture_cache().texture_became_outdated_.store(true, std::memory_order_release);
}

void TextureCache::DestroyAllTextures(bool from_destructor) {
  ResetTextureBindings(from_destructor);
  textures_.clear();
  COUNT_profile_set("gpu/texture_cache/textures", 0);
}

TextureCache::Texture* TextureCache::FindOrCreateTexture(TextureKey key) {
  // Check if the texture is a scaled resolve texture.
  if (IsDrawResolutionScaled() && key.tiled && IsScaledResolveSupportedForFormat(key)) {
    texture_util::TextureGuestLayout scaled_resolve_guest_layout = key.GetGuestLayout();
    if ((scaled_resolve_guest_layout.base.level_data_extent_bytes &&
         IsRangeScaledResolved(key.base_page << 12,
                               scaled_resolve_guest_layout.base.level_data_extent_bytes)) ||
        (scaled_resolve_guest_layout.mips_total_extent_bytes &&
         IsRangeScaledResolved(key.mip_page << 12,
                               scaled_resolve_guest_layout.mips_total_extent_bytes))) {
      key.scaled_resolve = 1;
    }
  }

  auto get_host_extent = [this, &key](uint32_t& host_width_out, uint32_t& host_height_out,
                                      uint32_t& host_depth_or_array_size_out) {
    host_width_out = key.GetWidth();
    host_height_out = key.GetHeight();
    if (key.scaled_resolve) {
      host_width_out *= draw_resolution_scale_x();
      host_height_out *= draw_resolution_scale_y();
    }
    host_depth_or_array_size_out = key.GetDepthOrArraySize();
  };
  auto try_drop_top_level = [&key]() {
    if (!key.mip_page || !key.mip_max_level) {
      return false;
    }
    uint32_t old_width = key.GetWidth();
    uint32_t old_height = key.GetHeight();
    uint32_t old_depth_or_array_size = key.GetDepthOrArraySize();
    uint32_t new_width = std::max(old_width >> 1, UINT32_C(1));
    uint32_t new_height = std::max(old_height >> 1, UINT32_C(1));
    uint32_t new_depth_or_array_size = key.dimension == xenos::DataDimension::k3D
                                           ? std::max(old_depth_or_array_size >> 1, UINT32_C(1))
                                           : old_depth_or_array_size;
    if (new_width == old_width && new_height == old_height &&
        new_depth_or_array_size == old_depth_or_array_size) {
      return false;
    }
    const FormatInfo* format_info = FormatInfo::Get(key.format);
    uint32_t rebased_pitch_texels = std::max(rex::next_pow2(old_width) >> 1, UINT32_C(1));
    rebased_pitch_texels = rex::align(rebased_pitch_texels, format_info->block_width);
    rebased_pitch_texels = rex::align(rebased_pitch_texels, xenos::kTextureTileWidthHeight);
    if (!key.tiled) {
      uint32_t rebased_pitch_blocks = rebased_pitch_texels / format_info->block_width;
      uint32_t rebased_row_pitch_bytes = rebased_pitch_blocks * format_info->bytes_per_block();
      rebased_row_pitch_bytes =
          rex::align(rebased_row_pitch_bytes, xenos::kTextureLinearRowAlignmentBytes);
      rebased_pitch_blocks = rebased_row_pitch_bytes / format_info->bytes_per_block();
      rebased_pitch_texels = rebased_pitch_blocks * format_info->block_width;
    }
    uint32_t rebased_pitch = std::max(rebased_pitch_texels >> 5, UINT32_C(1));
    if (rebased_pitch > ((UINT32_C(1) << 9) - 1)) {
      return false;
    }
    key.width_minus_1 = new_width - 1;
    key.height_minus_1 = new_height - 1;
    key.depth_or_array_size_minus_1 = new_depth_or_array_size - 1;
    key.base_page = key.mip_page;
    key.mip_page = 0;
    key.pitch = rebased_pitch;
    key.mip_max_level = 0;
    key.packed_mips = 0;
    return true;
  };
  uint32_t host_width, host_height, host_depth_or_array_size;
  get_host_extent(host_width, host_height, host_depth_or_array_size);
  // If the host can't support the full texture extent, first try using the
  // unscaled version of a scaled resolve texture, then fall back to the first
  // stored mip level only.
  uint32_t max_host_width_height = GetMaxHostTextureWidthHeight(key.dimension);
  uint32_t max_host_depth_or_array_size = GetMaxHostTextureDepthOrArraySize(key.dimension);
  while (true) {
    bool width_height_too_large =
        host_width > max_host_width_height || host_height > max_host_width_height;
    bool depth_or_array_too_large = host_depth_or_array_size > max_host_depth_or_array_size;
    if (!width_height_too_large && !depth_or_array_too_large) {
      break;
    }
    bool size_adjusted = false;
    if (key.scaled_resolve) {
      key.scaled_resolve = 0;
      size_adjusted = true;
    } else if ((width_height_too_large ||
                (depth_or_array_too_large && key.dimension == xenos::DataDimension::k3D)) &&
               try_drop_top_level()) {
      size_adjusted = true;
    }
    if (!size_adjusted) {
      return nullptr;
    }
    get_host_extent(host_width, host_height, host_depth_or_array_size);
  }

  // Look for a replacement using a stable hash of the original guest bytes.
  texture_util::TextureGuestLayout original_guest_layout{};
  bool has_replacement = false;
  uint64_t replacement_content_hash = 0;

  // (The program's in-memory replacements count too, with or without a pack.)
  const bool replacing = replacement_ &&
                         (REXCVAR_GET(texture_replace_enabled) || rex::graphics::HasTextureMemoryReplacements()) &&
                         key.base_page != 0 && !key.scaled_resolve;
  const TextureKey original_key = key;
  bool memo_final = false;  // the result can be remembered (not waiting for a decode)
  if (replacing) {
    auto memo = replacement_memo_.find(key);
    if (memo != replacement_memo_.end()) {
      auto texture_it = textures_.find(memo->second);
      if (texture_it != textures_.end() && !texture_it->second->outdated_mask()) {
        return texture_it->second.get();
      }
      replacement_memo_.erase(memo);
    }
  }
  if (replacing) {
    memo_final = true;
    original_guest_layout = key.GetGuestLayout();
    const uint32_t guest_size = original_guest_layout.base.level_data_extent_bytes;

    // Not render-to-texture data (resolved, and not written by the CPU since):
    // that changes from frame to frame, so its key would be hashed again on
    // every use - at 1x, where it isn't a scaled resolve texture, several
    // full-screen textures a frame took most of the frame time.
    if (guest_size > 0 && !IsRangeResolved(key.base_page << 12, guest_size)) {
      const uint32_t base_page = key.base_page;
      const uint8_t* guest_bytes = shared_memory().TranslatePhysical(key.base_page << 12);
      const uint64_t fingerprint =
          TextureReplacement::FingerprintGuestData(guest_bytes, guest_size);
      const uint64_t layout = ReplacementLayoutSignature(original_guest_layout, key.dimension,
                                                         key.format, key.tiled != 0);
      std::optional<CachedContentHash> cached_hash;
      {
        std::lock_guard lock(base_page_hash_cache_mutex_);
        if (auto it = base_page_hash_cache_.find(base_page); it != base_page_hash_cache_.end()) {
          cached_hash = it->second;
        }
      }

      rex::perf::TextureCacheStats& stats = rex::perf::GetTextureCacheStats();
      stats.replacement_lookups.fetch_add(1, std::memory_order_relaxed);
      if (REXCVAR_GET(texture_replace_stats)) {
        static auto next = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        static uint64_t lookups = 0, hashes = 0, hash_ns = 0;
        if (std::chrono::steady_clock::now() >= next) {
          next += std::chrono::seconds(5);
          const uint64_t l = stats.replacement_lookups, h = stats.replacement_hashes, n = stats.replacement_hash_ns;
          REXLOG_INFO("TextureReplacement: per second: {} lookups, {} hashes ({:.2f} ms hashing)", (l - lookups) / 5,
                      (h - hashes) / 5, double(n - hash_ns) / 5e6);
          lookups = l, hashes = h, hash_ns = n;
        }
      }
      if (cached_hash && cached_hash->size == guest_size && cached_hash->fingerprint == fingerprint &&
          cached_hash->layout == layout) {
        replacement_content_hash = cached_hash->hash;
      } else {
        const auto hash_start = std::chrono::steady_clock::now();
        replacement_content_hash =
            HashReplacementBase(original_guest_layout, key.dimension, key.format, key.tiled != 0,
                                guest_bytes, guest_size);
        {
          std::lock_guard lock(base_page_hash_cache_mutex_);
          base_page_hash_cache_[base_page] = {guest_size, fingerprint, layout, replacement_content_hash};
        }
        stats.replacement_hashes.fetch_add(1, std::memory_order_relaxed);
        stats.replacement_hash_ns.fetch_add(
            uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - hash_start)
                         .count()),
            std::memory_order_relaxed);
      }

      // Only the size here - the pixels are decoded when the texture is
      // loaded (and may have been dropped from RAM since).
      uint32_t replacement_width = 0, replacement_height = 0;
      // A JPEG (no alpha) can't stand in for a texture with alpha: its clear
      // parts would draw black (packs made before the loading screen's
      // "loading" label was kept as PNG). The game's own texture is used.
      using TF = xenos::TextureFormat;
      const bool guest_alpha = key.format == TF::k_DXT2_3 || key.format == TF::k_DXT4_5 ||
                               key.format == TF::k_8_8_8_8 || key.format == TF::k_4_4_4_4 ||
                               key.format == TF::k_1_5_5_5 || key.format == TF::k_2_10_10_10;
      bool jpeg_for_alpha = false;
      if (guest_alpha && replacement_->IsJpeg(replacement_content_hash)) {
        jpeg_for_alpha = true;
        static std::unordered_set<uint64_t> logged_jpeg;
        if (logged_jpeg.size() < 64 && logged_jpeg.insert(replacement_content_hash).second) {
          REXLOG_INFO("TextureReplacement: {:016x}.jpg not used: the texture has alpha", replacement_content_hash);
        }
      }
      if (!jpeg_for_alpha && replacement_->FindReplacementSize(replacement_content_hash, replacement_width,
                                                               replacement_height)) {
        // The key holds up to 8192 per side (bigger images are given at that
        // size already - TextureReplacement::FitReplacementSize).
        const uint32_t max_size = std::min(GetMaxHostTextureWidthHeight(key.dimension),
                                           xenos::kTexture2DCubeMaxWidthHeight);

        if (replacement_width > max_size || replacement_height > max_size) {
          static std::unordered_set<uint64_t> logged_too_big;
          if (logged_too_big.size() < 256 &&
              logged_too_big.insert(replacement_content_hash).second) {
            REXLOG_WARN(
                "TextureReplacement: {:016x} is {}x{}, more than this GPU's textures can be "
                "({} per side) - the original texture is shown",
                replacement_content_hash, replacement_width, replacement_height, max_size);
          }
        } else {
          const uint32_t original_width = key.GetWidth();
          const uint32_t original_height = key.GetHeight();

          TextureKey replacement_key = key;
          replacement_key.width_minus_1 = replacement_width - 1;
          replacement_key.height_minus_1 = replacement_height - 1;
          replacement_key.format = xenos::TextureFormat::k_8_8_8_8;
          // Mips (made when the replacement is loaded) down to the game's own
          // last level at the replacement's scale - level log2(scale) is the
          // original's base: a texture the game made without mips keeps the
          // original's detail at a distance (not blurred below it) and isn't
          // sampled finer than the original either (no extra shimmer); one
          // with a full chain gets a full chain.
          const float scale = std::max(float(replacement_width) / float(original_width),
                                       float(replacement_height) / float(original_height));
          const uint32_t scale_log2 =
              scale > 1.0f ? uint32_t(std::floor(std::log2(scale) + 0.01f)) : 0;
          replacement_key.mip_max_level =
              std::min({rex::log2_floor(std::max(replacement_width, replacement_height)),
                        uint32_t(xenos::kTextureMaxMips - 1),
                        uint32_t(key.mip_max_level) + scale_log2});
          replacement_key.replacement_id = uint32_t(replacement_content_hash & 0xFFFFFFF) | 1;

          // Premultiplied alpha (texture_replace_alpha): the original's data
          // for the loading thread to compare the replacement with, while the
          // guest memory holds it.
          if (textures_.find(replacement_key) == textures_.end() &&
              key.dimension == xenos::DataDimension::k2DOrStacked &&
              original_guest_layout.array_size == 1 && original_guest_layout.packed_level != 0 &&
              TextureReplacement::CanDecodeGuestFormat(key.format) &&
              replacement_->WantsGuestReference(replacement_content_hash)) {
            auto guest = std::make_shared<TextureReplacementGuest>();
            guest->bytes.assign(guest_bytes, guest_bytes + guest_size);
            guest->width = original_width;
            guest->height = original_height;
            guest->row_pitch_bytes = original_guest_layout.base.row_pitch_bytes;
            guest->tiled = key.tiled != 0;
            guest->format = key.format;
            guest->endianness = key.endianness;
            guest->swizzle = replacement_lookup_swizzle_;
            replacement_->SetGuestReference(replacement_content_hash, std::move(guest));
          }

          // Not decoded yet: the original texture this time, the replacement
          // decoded in the background (its ready callback has the bindings
          // looked up again). A replacement texture already made is used.
          if (REXCVAR_GET(texture_replace_async) &&
              textures_.find(replacement_key) == textures_.end() &&
              !replacement_->IsResident(replacement_content_hash)) {
            replacement_->RequestAsync(replacement_content_hash);
            memo_final = false;
          } else {
            key = replacement_key;
            has_replacement = true;
          }

          REXGPU_DEBUG(
              "TextureReplacement: injecting {}x{} replacement for original {}x{} "
              "(hash {:016x})",
              replacement_width, replacement_height,
              original_width, original_height, replacement_content_hash);
        }
      }
    }
  }

  // Try to find an existing texture.
  // TODO(Triang3l): Reuse a texture with mip_page unchanged, but base_page
  // previously 0, now not 0, to save memory - common case in streaming.
  auto remember = [&] {
    if (!memo_final) return;
    if (replacement_memo_.size() > 65536) replacement_memo_.clear();
    replacement_memo_[original_key] = key;
  };
  auto found_texture_it = textures_.find(key);
  if (found_texture_it != textures_.end()) {
    remember();
    return found_texture_it->second.get();
  }

  // Create the texture and add it to the map.
  Texture* texture;
  {
    const auto create_start = std::chrono::steady_clock::now();
    std::unique_ptr<Texture> new_texture = CreateTexture(key);
    const double create_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - create_start)
                                 .count();
    if (create_ms > 4.0) {
      REXLOG_INFO("Texture cache: creating a {}x{} texture took {:.1f} ms", key.GetWidth(),
                  key.GetHeight(), create_ms);
    }
    if (!new_texture) {
      key.LogAction("Failed to create");
      return nullptr;
    }
    assert_true(new_texture->key() == key);

    if (has_replacement) {
      new_texture->replacement_content_hash_ = replacement_content_hash;
      new_texture->OverrideGuestLayout(original_guest_layout);
    }

    texture = textures_.emplace(key, std::move(new_texture)).first->second.get();
  }
  COUNT_profile_set("gpu/texture_cache/textures", textures_.size());
  rex::perf::GetTextureCacheStats().created.fetch_add(1, std::memory_order_relaxed);
  texture->LogAction("Created");
  remember();
  return texture;
}

bool TextureCache::LoadTextureData(Texture& texture) {
  PendingTextureLoad pending_load;
  PendingSharedMemoryRange pending_ranges[2];
  size_t pending_range_count = 0;
  if (!PrepareTextureLoad(texture, pending_load, pending_ranges, pending_range_count)) {
    return true;
  }

  std::pair<uint32_t, uint32_t> pending_range_pairs[2];
  for (size_t i = 0; i < pending_range_count; ++i) {
    pending_range_pairs[i] = std::make_pair(pending_ranges[i].start, pending_ranges[i].length);
  }
  if (!shared_memory().RequestRanges(pending_range_pairs, pending_range_count)) {
    return false;
  }

  return CommitPreparedTextureLoad(pending_load);
}

void TextureCache::BindingInfoFromFetchConstant(const xenos::xe_gpu_texture_fetch_t& fetch,
                                                TextureKey& key_out, uint8_t* swizzled_signs_out) {
  // Reset the key and the signedness.
  key_out.MakeInvalid();
  if (swizzled_signs_out != nullptr) {
    *swizzled_signs_out = uint8_t(xenos::TextureSign::kUnsigned) * uint8_t(0b01010101);
  }

  switch (fetch.type) {
    case xenos::FetchConstantType::kTexture:
      break;
    case xenos::FetchConstantType::kInvalidTexture:
      if (REXCVAR_GET(gpu_allow_invalid_fetch_constants)) {
        break;
      }
      REXGPU_WARN(
          "Texture fetch constant ({:08X} {:08X} {:08X} {:08X} {:08X} {:08X}) "
          "has \"invalid\" type! This is incorrect behavior, but you can try "
          "bypassing this by launching Xenia with "
          "--gpu_allow_invalid_fetch_constants=true.",
          fetch.dword_0, fetch.dword_1, fetch.dword_2, fetch.dword_3, fetch.dword_4, fetch.dword_5);
      return;
    default:
      REXGPU_WARN(
          "Texture fetch constant ({:08X} {:08X} {:08X} {:08X} {:08X} {:08X}) "
          "is completely invalid!",
          fetch.dword_0, fetch.dword_1, fetch.dword_2, fetch.dword_3, fetch.dword_4, fetch.dword_5);
      return;
  }

  uint32_t width_minus_1, height_minus_1, depth_or_array_size_minus_1;
  uint32_t base_page, mip_page, mip_max_level;
  texture_util::GetSubresourcesFromFetchConstant(fetch, &width_minus_1, &height_minus_1,
                                                 &depth_or_array_size_minus_1, &base_page,
                                                 &mip_page, nullptr, &mip_max_level);
  if (base_page == 0 && mip_page == 0) {
    // No texture data at all.
    return;
  }
  if (fetch.dimension == xenos::DataDimension::k1D) {
    bool is_invalid_1d = false;
    // TODO(Triang3l): Support long 1D textures.
    if (width_minus_1 >= xenos::kTexture2DCubeMaxWidthHeight) {
      REXGPU_ERROR(
          "1D texture is too wide ({}) - ignoring! Report the game to Xenia "
          "developers",
          width_minus_1 + 1);
      is_invalid_1d = true;
    }
    assert_false(fetch.tiled);
    if (fetch.tiled) {
      REXGPU_ERROR(
          "1D texture has tiling enabled in the fetch constant, but this "
          "appears to be completely wrong - ignoring! Report the game to Xenia "
          "developers");
      is_invalid_1d = true;
    }
    assert_false(fetch.packed_mips);
    if (fetch.packed_mips) {
      REXGPU_ERROR(
          "1D texture has packed mips enabled in the fetch constant, but this "
          "appears to be completely wrong - ignoring! Report the game to Xenia "
          "developers");
      is_invalid_1d = true;
    }
    if (is_invalid_1d) {
      return;
    }
  }

  xenos::TextureFormat format = GetBaseFormat(fetch.format);

  key_out.base_page = base_page;
  key_out.mip_page = mip_page;
  key_out.dimension = fetch.dimension;
  key_out.width_minus_1 = width_minus_1;
  key_out.height_minus_1 = height_minus_1;
  key_out.depth_or_array_size_minus_1 = depth_or_array_size_minus_1;
  key_out.pitch = fetch.pitch;
  key_out.mip_max_level = mip_max_level;
  key_out.tiled = fetch.tiled;
  key_out.packed_mips = fetch.packed_mips;
  key_out.format = format;
  key_out.endianness = fetch.endianness;

  key_out.is_valid = 1;

  if (swizzled_signs_out != nullptr) {
    *swizzled_signs_out = texture_util::SwizzleSigns(fetch);
  }
}

void TextureCache::ResetTextureBindings(bool from_destructor) {
  ++fetch_constants_0_11_generation_;
  uint32_t bindings_reset = 0;
  for (size_t i = 0; i < texture_bindings_.size(); ++i) {
    TextureBinding& binding = texture_bindings_[i];
    if (!binding.key.is_valid) {
      continue;
    }
    binding.Reset();
    bindings_reset |= UINT32_C(1) << i;
  }
  texture_bindings_in_sync_ &= ~bindings_reset;
  if (!from_destructor && bindings_reset) {
    UpdateTextureBindingsImpl(bindings_reset);
  }
}

void TextureCache::UpdateTexturesTotalHostMemoryUsage(uint64_t add, uint64_t subtract) {
  textures_total_host_memory_usage_ = textures_total_host_memory_usage_ - subtract + add;
  COUNT_profile_set(
      "gpu/texture_cache/total_host_memory_usage_mb",
      uint32_t((textures_total_host_memory_usage_ + ((UINT32_C(1) << 20) - 1)) >> 20));
}

bool TextureCache::IsRangeScaledResolved(uint32_t start_unscaled, uint32_t length_unscaled) {
  return IsDrawResolutionScaled() && IsRangeResolved(start_unscaled, length_unscaled);
}

bool TextureCache::IsRangeResolved(uint32_t start_unscaled, uint32_t length_unscaled) {
  if (!scaled_resolve_pages_) {
    return false;
  }
  start_unscaled = std::min(start_unscaled, SharedMemory::kBufferSize);
  length_unscaled = std::min(length_unscaled, SharedMemory::kBufferSize - start_unscaled);
  if (!length_unscaled) {
    return false;
  }

  // Two-level check for faster rejection since resolve targets are usually
  // placed in relatively small and localized memory portions (confirmed by
  // testing - pretty much all times the deeper level was entered, the texture
  // was a resolve target).
  uint32_t page_first = start_unscaled >> 12;
  uint32_t page_last = (start_unscaled + length_unscaled - 1) >> 12;
  uint32_t block_first = page_first >> 5;
  uint32_t block_last = page_last >> 5;
  uint32_t l2_block_first = block_first >> 6;
  uint32_t l2_block_last = block_last >> 6;
  auto global_lock = global_critical_region_.Acquire();
  for (uint32_t i = l2_block_first; i <= l2_block_last; ++i) {
    uint64_t l2_block = scaled_resolve_pages_l2_[i];
    if (i == l2_block_first) {
      l2_block &= ~((UINT64_C(1) << (block_first & 63)) - 1);
    }
    if (i == l2_block_last && (block_last & 63) != 63) {
      l2_block &= (UINT64_C(1) << ((block_last & 63) + 1)) - 1;
    }
    uint32_t block_relative_index;
    while (rex::bit_scan_forward(l2_block, &block_relative_index)) {
      l2_block &= ~(UINT64_C(1) << block_relative_index);
      uint32_t block_index = (i << 6) + block_relative_index;
      uint32_t check_bits = UINT32_MAX;
      if (block_index == block_first) {
        check_bits &= ~((UINT32_C(1) << (page_first & 31)) - 1);
      }
      if (block_index == block_last && (page_last & 31) != 31) {
        check_bits &= (UINT32_C(1) << ((page_last & 31) + 1)) - 1;
      }
      if (scaled_resolve_pages_[block_index] & check_bits) {
        return true;
      }
    }
  }
  return false;
}

void TextureCache::ScaledResolveGlobalWatchCallbackThunk(
    const std::unique_lock<std::recursive_mutex>& global_lock, void* context,
    uint32_t address_first, uint32_t address_last, bool invalidated_by_gpu) {
  TextureCache* texture_cache = reinterpret_cast<TextureCache*>(context);
  texture_cache->ScaledResolveGlobalWatchCallback(global_lock, address_first, address_last,
                                                  invalidated_by_gpu);
}

void TextureCache::ScaledResolveGlobalWatchCallback(
    const std::unique_lock<std::recursive_mutex>& global_lock, uint32_t address_first,
    uint32_t address_last, bool invalidated_by_gpu) {
  if (invalidated_by_gpu) {
    // Resolves themselves do exactly the opposite of what this should do.
    return;
  }
  // Mark scaled resolve ranges as non-scaled. Textures themselves will be
  // invalidated by their shared memory watches.
  uint32_t resolve_page_first = address_first >> 12;
  uint32_t resolve_page_last = address_last >> 12;
  uint32_t resolve_block_first = resolve_page_first >> 5;
  uint32_t resolve_block_last = resolve_page_last >> 5;
  uint32_t resolve_l2_block_first = resolve_block_first >> 6;
  uint32_t resolve_l2_block_last = resolve_block_last >> 6;
  for (uint32_t i = resolve_l2_block_first; i <= resolve_l2_block_last; ++i) {
    uint64_t resolve_l2_block = scaled_resolve_pages_l2_[i];
    if (REXCVAR_GET(pre_mask_resolve_l2_block)) {
      // Pre-mask to only process blocks within the write range.
      if (i == resolve_l2_block_first) {
        resolve_l2_block &= ~((UINT64_C(1) << (resolve_block_first & 63)) - 1);
      }
      if (i == resolve_l2_block_last && (resolve_block_last & 63) != 63) {
        resolve_l2_block &= (UINT64_C(1) << ((resolve_block_last & 63) + 1)) - 1;
      }
    }
    uint32_t resolve_block_relative_index;
    while (rex::bit_scan_forward(resolve_l2_block, &resolve_block_relative_index)) {
      resolve_l2_block &= ~(UINT64_C(1) << resolve_block_relative_index);
      uint32_t resolve_block_index = (i << 6) + resolve_block_relative_index;
      uint32_t resolve_keep_bits = 0;
      if (resolve_block_index == resolve_block_first) {
        resolve_keep_bits |= (UINT32_C(1) << (resolve_page_first & 31)) - 1;
      }
      if (resolve_block_index == resolve_block_last && (resolve_page_last & 31) != 31) {
        resolve_keep_bits |= ~((UINT32_C(1) << ((resolve_page_last & 31) + 1)) - 1);
      }
      scaled_resolve_pages_[resolve_block_index] &= resolve_keep_bits;
      if (scaled_resolve_pages_[resolve_block_index] == 0) {
        scaled_resolve_pages_l2_[i] &= ~(UINT64_C(1) << resolve_block_relative_index);
      }
    }
  }
}

}  // namespace rex::graphics
