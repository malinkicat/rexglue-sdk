/**
 * @file        perf/frame_rate.h
 * @brief       Guest frame rate tracking, available in every build config.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace rex::perf {

// Called once per guest swap (VdSwap) by the command processor.
void RecordGuestSwap();

// Total guest swaps since startup.
uint64_t GetGuestSwapCount();

// Times between the latest guest swaps in milliseconds, oldest first (up to
// the last 255). Returns how many were written.
size_t GetGuestFrameTimes(float* out_ms, size_t max_count);

// Guest frontbuffer size (before resolution scaling) of the latest swap.
void RecordGuestFrontbuffer(uint32_t width, uint32_t height);

// Draw resolution scale the GPU backend renders with, and the one the config
// asked for (higher when an upscaler quality mode renders below it).
void SetDrawResolutionScale(uint32_t x, uint32_t y, uint32_t requested_x, uint32_t requested_y);

struct RenderInfo {
  uint32_t frontbuffer_width = 0;
  uint32_t frontbuffer_height = 0;
  uint32_t scale_x = 0;
  uint32_t scale_y = 0;
  uint32_t requested_scale_x = 0;
  uint32_t requested_scale_y = 0;
};
RenderInfo GetRenderInfo();

// Texture cache activity, for the perf_report debug command (totals since
// startup; the reader takes differences). Written by the GPU thread.
struct TextureCacheStats {
  std::atomic<uint64_t> created{0};             // host textures created
  std::atomic<uint64_t> destroyed{0};           // destroyed by the memory limits
  std::atomic<uint64_t> loaded{0};              // texture data uploads (guest or replacement)
  std::atomic<uint64_t> replacement_uploads{0}; // uploads from replacement images
  std::atomic<uint64_t> replacement_lookups{0}; // texture lookups checked for a replacement
  std::atomic<uint64_t> replacement_hashes{0};  // full replacement key hashes of guest data
  std::atomic<uint64_t> replacement_hash_ns{0};
  std::atomic<uint64_t> replacement_decodes{0}; // image files decoded on the GPU thread
  std::atomic<uint64_t> replacement_decode_ns{0};
  // Current values.
  std::atomic<uint64_t> memory_bytes{0};        // host memory of the cached textures
  std::atomic<uint32_t> limit_soft_mb{0};
  std::atomic<uint32_t> limit_hard_mb{0};
  std::atomic<uint64_t> vram_usage_bytes{0};    // the process's local video memory use
  std::atomic<uint64_t> vram_budget_bytes{0};   // the OS budget for it
};
TextureCacheStats& GetTextureCacheStats();

// GPU time per part of a frame, from timestamp queries (gpu_profile debug
// command). Off unless a measurement is running.
enum class GpuPass : uint32_t {
  kOther,          // everything not below (uploads, index conversion, barriers)
  kDraw,           // the game's draws
  kDrawMemexport,  // the game's draws exporting to memory (with their barriers)
  kRenderTargets,  // render target cache: ownership transfers between host RTs (color)
  kRtTransferDepth,    // ownership transfers into depth render targets (depth part)
  kRtTransferStencil,  // ... their stencil, bit by bit (no stencil reference output)
  kResolve,        // resolves: render target dump to the EDRAM buffer
  kResolveCopy,    // resolves: EDRAM buffer -> (scaled) resolve memory
  kResolveClear,   // clears done as part of resolves
  kTextureLoad,    // texture loading / untiling (incl. resolved textures)
  kSharedMemory,   // guest memory uploads for vertex buffers etc.
  kDlssInputs,     // DLSS/FSR: depth + motion vector compute
  kDlssEvaluate,   // NGX evaluate / FSR dispatch
  kDlssCopyBack,   // upscaler output back into the render target (+ downsample)
  kDlssMirror,     // HUD draws mirrored into the output-size picture
  kDlssFreezeHud,  // copy of the render target at its first resolve
  kDlssCompose,    // compose of the upscaled picture with the presented frame
  kSwap,           // gamma ramp / FXAA of the frontbuffer
  kPresenterGuest, // presenter: guest output effects (CAS, FSR, bilinear)
  kPresenterUi,    // presenter: overlays and UI
  kCount,
};
const char* GetGpuPassName(GpuPass pass);

struct GpuProfileStats {
  std::atomic<bool> enabled{false};
  std::atomic<uint64_t> pass_ns[size_t(GpuPass::kCount)] = {};
  std::atomic<uint64_t> pass_count[size_t(GpuPass::kCount)] = {};  // timestamped intervals
  // GPU time from the first to the last timestamp of each command processor
  // submission / presenter paint.
  std::atomic<uint64_t> submission_ns{0};
  std::atomic<uint64_t> submissions{0};
  std::atomic<uint64_t> submissions_unprofiled{0};
  std::atomic<uint64_t> paint_ns{0};
  std::atomic<uint64_t> paints{0};
  // GPU idle time between consecutive command processor submissions (presenter
  // paints in between included).
  std::atomic<uint64_t> gap_ns{0};
  // CPU side, command processor thread.
  std::atomic<uint64_t> fence_waits{0};      // waits for the GPU (submission fences)
  std::atomic<uint64_t> fence_wait_ns{0};
  std::atomic<uint64_t> occlusion_queries{0};
  std::atomic<uint64_t> draws{0};
  std::atomic<uint64_t> resolves{0};
  std::atomic<uint64_t> clears_in_place{0};
  std::atomic<uint64_t> memory_upload_batches{0};  // shared memory UploadRanges with work
  std::atomic<uint64_t> memory_upload_bytes{0};
  std::atomic<uint64_t> texture_loads{0};
  std::atomic<uint64_t> texture_load_bytes{0};  // host texture bytes written by loads  // clear draws done as clears (gpu_clear_draws_in_place)
  std::atomic<uint64_t> cp_submissions{0};
  // Presenter (UI thread): time in PaintAndPresent, and in Present itself.
  std::atomic<uint64_t> present_calls{0};
  std::atomic<uint64_t> present_cpu_ns{0};
};
GpuProfileStats& GetGpuProfileStats();
inline bool IsGpuProfiling() {
  return GetGpuProfileStats().enabled.load(std::memory_order_relaxed);
}

}  // namespace rex::perf
