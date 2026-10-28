/**
 * @file        core/perf/frame_rate.cpp
 * @brief       Guest frame rate tracking. See perf/frame_rate.h.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */
#include <rex/perf/frame_rate.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>

namespace rex::perf {

namespace {
// Times of the latest swaps, by swap number. Written by the command processor
// thread only.
constexpr size_t kSwapTimeHistory = 256;
std::array<std::atomic<int64_t>, kSwapTimeHistory> g_swap_times_ns{};
std::atomic<uint64_t> g_guest_swap_count{0};
std::atomic<uint32_t> g_frontbuffer_width{0};
std::atomic<uint32_t> g_frontbuffer_height{0};
std::atomic<uint32_t> g_scale_x{0};
std::atomic<uint32_t> g_scale_y{0};
std::atomic<uint32_t> g_requested_scale_x{0};
std::atomic<uint32_t> g_requested_scale_y{0};
}  // namespace

void RecordGuestSwap() {
  const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
  // The time first, so readers never see a count with its time missing.
  const uint64_t index = g_guest_swap_count.load(std::memory_order_relaxed);
  g_swap_times_ns[index % kSwapTimeHistory].store(now, std::memory_order_relaxed);
  g_guest_swap_count.store(index + 1, std::memory_order_release);
}

uint64_t GetGuestSwapCount() {
  return g_guest_swap_count.load(std::memory_order_relaxed);
}

size_t GetGuestFrameTimes(float* out_ms, size_t max_count) {
  const uint64_t count = g_guest_swap_count.load(std::memory_order_acquire);
  if (!out_ms || count < 2) {
    return 0;
  }
  const size_t frames =
      size_t(std::min<uint64_t>({uint64_t(max_count), count - 1, kSwapTimeHistory - 1}));
  for (size_t i = 0; i < frames; ++i) {
    const uint64_t newer = count - frames + i;
    const int64_t interval_ns =
        g_swap_times_ns[newer % kSwapTimeHistory].load(std::memory_order_relaxed) -
        g_swap_times_ns[(newer - 1) % kSwapTimeHistory].load(std::memory_order_relaxed);
    out_ms[i] = float(std::max<int64_t>(interval_ns, 0)) * 1.0e-6f;
  }
  return frames;
}

void RecordGuestFrontbuffer(uint32_t width, uint32_t height) {
  g_frontbuffer_width.store(width, std::memory_order_relaxed);
  g_frontbuffer_height.store(height, std::memory_order_relaxed);
}

void SetDrawResolutionScale(uint32_t x, uint32_t y, uint32_t requested_x, uint32_t requested_y) {
  g_scale_x.store(x, std::memory_order_relaxed);
  g_scale_y.store(y, std::memory_order_relaxed);
  g_requested_scale_x.store(requested_x, std::memory_order_relaxed);
  g_requested_scale_y.store(requested_y, std::memory_order_relaxed);
}

RenderInfo GetRenderInfo() {
  RenderInfo info;
  info.frontbuffer_width = g_frontbuffer_width.load(std::memory_order_relaxed);
  info.frontbuffer_height = g_frontbuffer_height.load(std::memory_order_relaxed);
  info.scale_x = g_scale_x.load(std::memory_order_relaxed);
  info.scale_y = g_scale_y.load(std::memory_order_relaxed);
  info.requested_scale_x = g_requested_scale_x.load(std::memory_order_relaxed);
  info.requested_scale_y = g_requested_scale_y.load(std::memory_order_relaxed);
  return info;
}

TextureCacheStats& GetTextureCacheStats() {
  static TextureCacheStats stats;
  return stats;
}

const char* GetGpuPassName(GpuPass pass) {
  switch (pass) {
    case GpuPass::kOther:
      return "other";
    case GpuPass::kDraw:
      return "draws";
    case GpuPass::kDrawMemexport:
      return "draws_memexport";
    case GpuPass::kRenderTargets:
      return "rt_transfers";
    case GpuPass::kRtTransferDepth:
      return "rt_transfers_depth";
    case GpuPass::kRtTransferStencil:
      return "rt_transfers_stencil";
    case GpuPass::kResolve:
      return "resolve_dumps";
    case GpuPass::kResolveCopy:
      return "resolve_copies";
    case GpuPass::kResolveClear:
      return "resolve_clears";
    case GpuPass::kTextureLoad:
      return "texture_loads";
    case GpuPass::kSharedMemory:
      return "memory_uploads";
    case GpuPass::kDlssInputs:
      return "dlss_inputs";
    case GpuPass::kDlssEvaluate:
      return "dlss_evaluate";
    case GpuPass::kDlssCopyBack:
      return "dlss_copy_back";
    case GpuPass::kDlssMirror:
      return "dlss_hud_mirror";
    case GpuPass::kDlssFreezeHud:
      return "dlss_freeze_hud";
    case GpuPass::kDlssCompose:
      return "dlss_compose";
    case GpuPass::kSwap:
      return "swap_gamma";
    case GpuPass::kPresenterGuest:
      return "present_effects";
    case GpuPass::kPresenterUi:
      return "present_ui";
    default:
      return "?";
  }
}

GpuProfileStats& GetGpuProfileStats() {
  static GpuProfileStats stats;
  return stats;
}

}  // namespace rex::perf
