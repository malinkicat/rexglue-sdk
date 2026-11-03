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
#include <set>
#include <chrono>
#include <intrin.h>
#include <cfloat>
#include <cmath>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <bit>
#include <chrono>
#include <map>
#include <unordered_set>
#include <cstdarg>
#include <cstring>
#include <sstream>
#include <utility>

#include <rex/hash.h>
#include <rex/assert.h>
#include <rex/cvar.h>
#include <rex/ui/flags.h>
#include <rex/dbg.h>
#include <rex/perf/counter.h>
#include <rex/graphics/d3d12/command_processor.h>
#include <rex/perf/frame_rate.h>
#include <rex/graphics/d3d12/graphics_system.h>
#include <rex/graphics/d3d12/shader.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/xenos.h>
#include <rex/kernel/xboxkrnl/video.h>
#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/ui/d3d12/d3d12_presenter.h>
#include <rex/ui/d3d12/d3d12_util.h>

REXCVAR_DEFINE_DOUBLE(ui_scale, 1.0, "GPU",
                      "Menus and HUD drawn at this size (0.5-1.3), about the screen's centre");

// --ui_scale changed while running (the game's menu), as a plain C export
// looked up by name (the GPU plugin keeps its own copy of its cvars).
extern "C" __declspec(dllexport) void ReXSetUiScale(double scale) { REXCVAR_SET(ui_scale, std::clamp(scale, 0.5, 1.3)); }
REXCVAR_DEFINE_INT32(ultrawide_pin_log, 0, "GPU",
                      "Debug: log the UI draws' sprite matrices (c0, c1) for this many draws");
REXCVAR_DEFINE_INT32(ui_scale_log_after, 0, "GPU",
                      "Debug: with ultrawide_pin_log and ui_scale, log UI draws only after this many seconds");
REXCVAR_DEFINE_BOOL(ultrawide_hud_pin, false, "GPU",
                    "With ultrawide_squeeze: UI sprites in the left / right third go to the screen's "
                    "edges (set by the game while riding; menus stay centred)");
REXCVAR_DEFINE_BOOL(ultrawide_hud_clip, false, "GPU",
                    "With ultrawide_squeeze: the UI is clipped to the 16:9 frame, as on a 16:9 screen "
                    "(set by the game while riding: HUD sliding in stays out of the sides; menus' art "
                    "may run into them)");
REXCVAR_DEFINE_BOOL(d3d12_gpu_done_log, false, "GPU",
                    "Debug: gpudone.csv - when the GPU finished each frame (its closing submission's fence), "
                    "on the steady clock (as swaptimes.csv's steady_ns and present_log.csv)");
REXCVAR_DEFINE_INT32(d3d12_queued_frames, 3, "GPU",
                     "How many frames the GPU may run behind the game (1-3): fewer keeps each frame "
                     "finished sooner after it's handed over, for vsync presents that must make their refresh "
                     "(the Steam Deck's GPU clocks just high enough to keep a full backlog)");
REXCVAR_DEFINE_DOUBLE(tall_squeeze, 1.0, "GPU",
                      "Squeeze the scene and UI vertically by this factor (the screen's aspect / 16:9, "
                      "e.g. 0.9 for 16:10) with present_letterbox=false: a taller view instead of bars; 1 = off");
REXCVAR_DEFINE_DOUBLE(ultrawide_squeeze, 1.0, "GPU",
                      "Experiment: squeeze every draw horizontally by this factor (16:9 / the "
                      "window's aspect) for ultrawide with present_letterbox=false; 1 = off");
REXCVAR_DEFINE_BOOL(d3d12_bindless, true, "GPU/D3D12", "Use bindless resources where available")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_BOOL(d3d12_readback_memexport, false, "GPU/D3D12",
                    "Read data written by memory export in shaders on the CPU")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(d3d12_draw_timing, false, "GPU/D3D12",
                    "Debug: log the command processor's time per IssueDraw section (ms per frame)");
REXCVAR_DEFINE_BOOL(d3d12_draw_timing_sections, true, "GPU/D3D12",
                    "Debug: with d3d12_draw_timing, also time each IssueDraw section (the "
                    "timer reads cost about 1 ms per frame); false logs only the command "
                    "processor's busy time per frame and per draw");
REXCVAR_DEFINE_BOOL(d3d12_debug_prim_stats, false, "GPU/D3D12",
                    "Debug: log draws per frame by primitive type, and skip the types whose bits "
                    "are set in debug_skip_prims.txt (hex mask, re-read every 60 frames)");
REXCVAR_DEFINE_BOOL(d3d12_readback_resolve, false, "GPU/D3D12",
                    "Read render-to-texture results on the CPU")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// amped3 debug (--d3d12_draw_timing): rdtsc time per IssueDraw section, logged
// as ms per frame every 120 swaps - how much of the command processor's draw
// work could move to another thread.
namespace {
enum DrawTimingSection {
  kDrawTimingSetup,       // entry to primitive processing (shaders, clears in place)
  kDrawTimingProcess,     // primitive processing
  kDrawTimingProcessRt,   // render target cache update (and what's between)
  kDrawTimingPipeline,    // translations + ConfigurePipeline
  kDrawTimingTextures,    // RequestTextures
  kDrawTimingViewport,    // viewport, scissor, fixed function state
  kDrawTimingSysConst,    // UpdateSystemConstantValues
  kDrawTimingBindings,    // UpdateBindings
  kDrawTimingVertexBuffers,  // vertex fetch constants -> shared memory ranges
  kDrawTimingTail,        // memexport, index buffer, barriers, the draw
  kDrawTimingCount
};
const char* const kDrawTimingNames[kDrawTimingCount] = {
    "setup", "process", "rt", "pipeline", "textures", "viewport", "sysconst", "bindings",
    "vertexbuffers", "tail"};
uint64_t g_draw_timing[kDrawTimingCount];
uint64_t g_amped3_copy_tsc;
}  // namespace
namespace {
// --d3d12_gpu_done_log: a thread waits for each frame's closing fence value
// and notes when it completed.
std::mutex g_gpu_done_mutex;
std::condition_variable g_gpu_done_cv;
std::deque<uint64_t> g_gpu_done_queue;
bool g_gpu_done_started = false;
void LogGpuDone(ID3D12Fence* fence, uint64_t value) {
  std::lock_guard<std::mutex> lock(g_gpu_done_mutex);
  if (!g_gpu_done_started) {
    g_gpu_done_started = true;
    fence->AddRef();
    std::thread([fence] {
      HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
      std::FILE* file = std::fopen("gpudone.csv", "w");
      if (!file || !event) return;
      std::fputs("fence,steady_ns\n", file);
      for (uint64_t n = 1;; ++n) {
        uint64_t value;
        {
          std::unique_lock<std::mutex> wait_lock(g_gpu_done_mutex);
          g_gpu_done_cv.wait(wait_lock, [] { return !g_gpu_done_queue.empty(); });
          value = g_gpu_done_queue.front();
          g_gpu_done_queue.pop_front();
        }
        if (fence->GetCompletedValue() < value && SUCCEEDED(fence->SetEventOnCompletion(value, event))) {
          WaitForSingleObject(event, 1000);
        }
        std::fprintf(file, "%llu,%lld\n", (unsigned long long)value,
                     (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count());
        if (n % 120 == 0) std::fflush(file);
      }
    }).detach();
  }
  g_gpu_done_queue.push_back(value);
  g_gpu_done_cv.notify_one();
}
}  // namespace

namespace rex::graphics::d3d12 {
extern uint64_t g_amped3_upload_valid_tsc, g_amped3_upload_copy_tsc, g_amped3_upload_pages,
    g_amped3_upload_calls;
}  // namespace rex::graphics::d3d12
namespace rex::graphics {
extern uint64_t g_amped3_cp_busy_tsc;
extern bool g_amped3_count_packets;
extern uint64_t g_amped3_opcode_counts[128];
extern uint64_t g_amped3_type0_packets, g_amped3_type0_registers;
extern uint64_t g_amped3_cp_exec_busy_tsc;
extern bool amped3_make_valid_timing;
extern uint64_t amped3_make_valid_lock_tsc, amped3_make_valid_protect_tsc;
}  // namespace rex::graphics
namespace {
// amped3 debug (--d3d12_debug_prim_stats): draws per primitive type, and the
// types to skip (finding which kind of draw makes an artefact).
uint64_t g_prim_counts[64];
uint64_t g_prim_skip_mask;
// Also skip draws numbered [g_skip_first, g_skip_end) within the frame (counting
// all draws, from 0 after each swap), and the frame's draw count.
uint32_t g_skip_first, g_skip_end, g_frame_draw_index, g_frame_draws_last;
// Pixel shaders (ucode hashes) of the triangle-list draws seen, logged every 120
// frames, and the ones whose draws are skipped (further lines of the file).
std::set<uint64_t> g_ps_seen, g_ps_skip;
uint64_t g_draw_timing_last;
uint64_t g_draw_timing_draws;
bool g_draw_timing_on;
inline void DrawTimingMark(DrawTimingSection section) {
  if (g_draw_timing_on) {
    uint64_t now = __rdtsc();
    g_draw_timing[section] += now - g_draw_timing_last;
    g_draw_timing_last = now;
  }
}
}  // namespace

REXCVAR_DEFINE_INT32(gpu_debug_log_draws, 0, "GPU/Debug",
                     "Log the state of every draw for this many frames (counts down)");

REXCVAR_DEFINE_BOOL(gpu_pass_profile, false, "GPU/Debug",
                    "Log the GPU time per frame of each kind of pass every 300 frames");

REXCVAR_DEFINE_BOOL(gpu_pass_profile_passes, true, "GPU/Debug",
                    "With gpu_pass_profile, time each kind of pass (a GPU timestamp at every "
                    "pass switch, several per draw); otherwise only whole submissions (GPU busy "
                    "time and gaps, without distorting it)");

REXCVAR_DEFINE_BOOL(gpu_clear_draws_in_place, true, "GPU/D3D12",
                    "Do clears the game draws as constant rectangles (like the Xbox 360 Direct3D 9 "
                    "Clear through a 4x MSAA view) as render target clears where the data "
                    "already is, without moving it between render targets")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(d3d12_submission_thread, true, "GPU/D3D12",
                    "Execute and submit command lists on another thread, so the command "
                    "processor continues with the next commands meanwhile");

REXCVAR_DEFINE_BOOL(d3d12_reuse_float_constants, true, "GPU/D3D12",
                    "Reuse the previous upload of a shader stage's float constants when the "
                    "same values are needed again in the same frame")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(d3d12_present_on_submission_thread, true, "GPU/D3D12",
                    "With the submission thread, also present from it (in order after the "
                    "frame's command lists), so the command processor doesn't wait for them "
                    "at swaps");

REXCVAR_DEFINE_INT32(d3d12_submit_every_draws, 0, "GPU/D3D12",
                     "Also submit the command list every N draws (0 = off): smaller submissions let a "
                     "compositor sharing the GPU (gamescope on the Steam Deck) run its frame in between "
                     "instead of waiting behind a whole frame of ours and missing the refresh");
REXCVAR_DEFINE_BOOL(d3d12_submit_on_primary_buffer_end, true, "GPU/D3D12",
                    "Submit command list when PM4 primary buffer ends")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DECLARE(bool, gpu_skip_overwritten_transfers);
REXCVAR_DECLARE(int32_t, gpu_screen_extents);
REXCVAR_DEFINE_STRING(dlss_handoff, "resolve", "GPU/Debug",
                      "Where the scene is handed to DLSS: hud (the first HUD draw) or resolve (the "
                      "last frontbuffer-wide color resolve before the HUD)")
    .allowed({"hud", "resolve"});
REXCVAR_DEFINE_INT32(dlss_trace_frame, 0, "GPU/Debug",
                     "Log every frontbuffer-wide draw and every resolve of this frame number (0 = off)");
REXCVAR_DEFINE_INT32(scene_render_percent, 100, "GPU",
                     "Experimental: draw the 3D scene into the top-left part of the frame, this "
                     "percent of the size (an upscaler brings it back to full size)")
    .range(50, 100);
REXCVAR_DEFINE_BOOL(dlss_log, false, "GPU/Debug",
                    "Log the DLSS draw classification once a second (scene draws, HUD draws, hand-offs)");
namespace {
struct DlssStats {
  uint32_t frames, scene, hud, taken, wide, narrow, fb_width;
} g_dlss_stats;
// The scene's resolve: the last frontbuffer-wide color resolve before the HUD
// (Amped 3 resolves G-buffer-like targets of the same width earlier in the
// frame), learnt from the previous frame.
uint32_t g_dlss_wide_resolves = 0;      // this frame so far
uint32_t g_dlss_resolves_before_hud = 0;  // this frame, at its first HUD draw
uint32_t g_dlss_scene_resolve = 0;        // learnt: 1-based index, 0 = unknown
uint32_t g_dlss_frame = 0;

bool DlssTracing() {
  const int32_t f = REXCVAR_GET(dlss_trace_frame);
  return f && uint32_t(f) == g_dlss_frame;
}
}  // namespace
REXCVAR_DEFINE_BOOL(amped3_hide_hud, false, "GPU/D3D12",
                    "Hide the game's HUD and text (its 2D sprite and text vertex shaders), e.g. "
                    "for screenshots");
REXCVAR_DEFINE_BOOL(amped3_cull_subpixel_sprites, true, "GPU/D3D12",
                    "Skip the game's sprite draws (vertex shader CB134112C994DF08) whose whole "
                    "quad is under a pixel: it hides a flare by shrinking it to ~0.4 px, which "
                    "rasterized as a stray pink pixel at (527, 602)");
REXCVAR_DEFINE_BOOL(d3d12_debug_pink_probe, false, "GPU/D3D12",
                    "Debug (with d3d12_debug_ps_extents): each frame skip a random half of that "
                    "shader's draws, read back the output pixel at (527, 602) and log, per draw "
                    "index, the pink frames among the frames it was skipped in");
REXCVAR_DEFINE_BOOL(d3d12_debug_ps_extents_skip_unmeasured, false, "GPU/D3D12",
                    "Debug: skip the d3d12_debug_ps_extents shader's draws whose rectangle the "
                    "CPU can't estimate");
REXCVAR_DEFINE_BOOL(d3d12_debug_ps_extents_skip_full, false, "GPU/D3D12",
                    "Debug: skip the full-screen draws of the d3d12_debug_ps_extents shader");
REXCVAR_DEFINE_STRING(d3d12_debug_ps_extents, "", "GPU/D3D12",
                      "Debug: log the screen rectangle of draws with this pixel shader hash");
REXCVAR_DEFINE_BOOL(gpu_tile_replay_verify, false, "GPU",
                    "gpu_tile_replay: compare what tile 2-3 draws derive with tile 1's record");
REXCVAR_DEFINE_BOOL(gpu_tile_replay_apply, false, "GPU",
                    "gpu_tile_replay: skip deriving the state of eligible tile 2-3 draws "
                    "(primitive processing, pipeline, bindings) and use tile 1's");
REXCVAR_DEFINE_BOOL(gpu_tile_replay, false, "GPU",
                    "amped3: reuse tile 1's per-draw work for the same draws in tiles 2 and 3 of "
                    "the game's predicated tiling (statistics only for now)");
REXCVAR_DEFINE_BOOL(gpu_screen_extents_predict, false, "GPU",
                    "gpu_screen_extents=4: write the predicted extents (else only the "
                    "statistics are gathered and the whole screen is written)");
REXCVAR_DEFINE_BOOL(gpu_screen_extents_motion, false, "GPU",
                    "gpu_screen_extents=4: predict with each draw's last motion");
REXCVAR_DEFINE_INT32(gpu_screen_extents_margin, 4, "GPU",
                     "gpu_screen_extents=4: 8x8 tiles added on each side of the predicted extents");
REXCVAR_DEFINE_INT32(gpu_screen_extents_flush, 256, "GPU",
                     "gpu_screen_extents=3: submit the GPU work after every this many measured "
                     "draws, so less of it is left to await when the guest needs the extents "
                     "(0: only then)");

REXCVAR_DEFINE_BOOL(gpu_lod_free_texture_fetches, true, "GPU",
                    "Pixel shaders sample textures that have a single mip level, the same "
                    "minification and magnification filter and no anisotropic filtering at LOD 0 "
                    "instead of computing gradients (the same result, less shader work)");
namespace rex::graphics::d3d12 {

// Generated with `xb buildshaders`.
namespace shaders {
#include "../shaders/bytecode/d3d12_5_1/apply_gamma_pwl_cs.h"
#include "../shaders/bytecode/d3d12_5_1/apply_gamma_pwl_fxaa_luma_cs.h"
#include "../shaders/bytecode/d3d12_5_1/apply_gamma_table_cs.h"
#include "../shaders/bytecode/d3d12_5_1/apply_gamma_table_fxaa_luma_cs.h"
#include "../shaders/bytecode/d3d12_5_1/fxaa_cs.h"
#include "../shaders/bytecode/d3d12_5_1/fxaa_extreme_cs.h"
#include "../shaders/bytecode/d3d12_5_1/smaa_blend_cs.h"
#include "../shaders/bytecode/d3d12_5_1/smaa_edges_cs.h"
#include "../shaders/bytecode/d3d12_5_1/smaa_weights_cs.h"
#include "../../../thirdparty/smaa/Textures/AreaTex.h"
#include "../../../thirdparty/smaa/Textures/SearchTex.h"
#include "../shaders/bytecode/d3d12_5_1/resolve_downscale_cs.h"
}  // namespace shaders

D3D12CommandProcessor::D3D12CommandProcessor(D3D12GraphicsSystem* graphics_system,
                                             system::KernelState* kernel_state)
    : CommandProcessor(graphics_system, kernel_state), deferred_command_list_(*this) {
  legacy_readback_memexport_cvar_name_ = "d3d12_readback_memexport";
}
D3D12CommandProcessor::~D3D12CommandProcessor() = default;

void D3D12CommandProcessor::UpdateDebugMarkersEnabled() {
  debug_markers_enabled_ = IsGpuDebugMarkersEnabled();
}

void D3D12CommandProcessor::PushDebugMarker(const char* format, ...) {
  if (!debug_markers_enabled_) {
    return;
  }
  char label[256];
  va_list args;
  va_start(args, format);
  vsnprintf(label, sizeof(label), format, args);
  va_end(args);
  deferred_command_list_.BeginDebugMarker(label);
}

void D3D12CommandProcessor::PopDebugMarker() {
  if (!debug_markers_enabled_) {
    return;
  }
  deferred_command_list_.EndDebugMarker();
}

void D3D12CommandProcessor::InsertDebugMarker(const char* format, ...) {
  if (!debug_markers_enabled_) {
    return;
  }
  char label[256];
  va_list args;
  va_start(args, format);
  vsnprintf(label, sizeof(label), format, args);
  va_end(args);
  deferred_command_list_.InsertDebugMarker(label);
}

void D3D12CommandProcessor::ClearCaches() {
  CommandProcessor::ClearCaches();
  InvalidateAllVertexBufferResidency();
  cache_clear_requested_ = true;
}

void D3D12CommandProcessor::InvalidateGpuMemory() {
  if (shared_memory_) {
    shared_memory_->InvalidateAllPages();
  }
}

void D3D12CommandProcessor::InvalidateAllVertexBufferResidency() {
  vertex_buffers_in_sync_[0] = 0;
  vertex_buffers_in_sync_[1] = 0;
  for (VertexBufferState& state : vertex_buffer_states_) {
    state.address = UINT32_MAX;
    state.size = UINT32_MAX;
  }
}

void D3D12CommandProcessor::InvalidateVertexBufferResidency(uint32_t vfetch_index) {
  if (vfetch_index >= vertex_buffer_states_.size()) {
    return;
  }
  vertex_buffers_in_sync_[vfetch_index >> 6] &= ~(uint64_t(1) << (vfetch_index & 63));
}

void D3D12CommandProcessor::InvalidateVertexBufferResidencyRange(uint32_t first_vfetch,
                                                                 uint32_t last_vfetch) {
  if (first_vfetch > last_vfetch) {
    std::swap(first_vfetch, last_vfetch);
  }
  if (first_vfetch >= vertex_buffer_states_.size()) {
    return;
  }
  last_vfetch = std::min(last_vfetch, uint32_t(vertex_buffer_states_.size() - 1));
  // Whole 64-bit words of the bitmap at once.
  for (uint32_t word = first_vfetch >> 6; word <= last_vfetch >> 6; ++word) {
    uint64_t mask = ~uint64_t(0);
    if (word == first_vfetch >> 6) {
      mask &= ~uint64_t(0) << (first_vfetch & 63);
    }
    if (word == last_vfetch >> 6) {
      mask &= ~uint64_t(0) >> (63 - (last_vfetch & 63));
    }
    vertex_buffers_in_sync_[word] &= ~mask;
  }
}

void D3D12CommandProcessor::InitializeShaderStorage(const std::filesystem::path& cache_root,
                                                    uint32_t title_id, bool blocking) {
  CommandProcessor::InitializeShaderStorage(cache_root, title_id, blocking);
  pipeline_cache_->InitializeShaderStorage(cache_root, title_id, blocking);
}

bool D3D12CommandProcessor::ExecutePacketType3_EVENT_WRITE_ZPD(memory::RingBuffer* reader,
                                                               uint32_t packet, uint32_t count) {
  if (!REXCVAR_GET(occlusion_query_enable) || !occlusion_query_resources_available_) {
    return CommandProcessor::ExecutePacketType3_EVENT_WRITE_ZPD(reader, packet, count);
  }

  const uint32_t kQueryFinished = rex::byte_swap(0xFFFFFEED);
  assert_true(count == 1);
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);

  uint32_t sample_count_addr = register_file_->values[XE_GPU_REG_RB_SAMPLE_COUNT_ADDR];
  auto* sample_counts =
      memory_->TranslatePhysical<xenos::xe_gpu_depth_sample_counts*>(sample_count_addr);
  if (!sample_counts) {
    DisableHostOcclusionQueries();
    return true;
  }

  auto write_fallback_result = [sample_counts, kQueryFinished]() -> bool {
    auto fake_sample_count = REXCVAR_GET(query_occlusion_fake_sample_count);
    if (fake_sample_count < 0) {
      return true;
    }
    bool is_end_via_z_pass =
        sample_counts->ZPass_A == kQueryFinished || sample_counts->ZPass_B == kQueryFinished;
    bool is_end_via_z_fail =
        sample_counts->ZFail_A == kQueryFinished || sample_counts->ZFail_B == kQueryFinished;
    std::memset(sample_counts, 0, sizeof(xenos::xe_gpu_depth_sample_counts));
    if (is_end_via_z_pass || is_end_via_z_fail) {
      sample_counts->ZPass_A = fake_sample_count;
      sample_counts->Total_A = fake_sample_count;
    }
    return true;
  };

  bool is_end_via_z_pass =
      sample_counts->ZPass_A == kQueryFinished || sample_counts->ZPass_B == kQueryFinished;
  bool is_end_via_z_fail =
      sample_counts->ZFail_A == kQueryFinished || sample_counts->ZFail_B == kQueryFinished;
  bool is_end = is_end_via_z_pass || is_end_via_z_fail;

  if (!is_end) {
    if (active_occlusion_query_.valid &&
        active_occlusion_query_.sample_count_address != sample_count_addr) {
      DisableHostOcclusionQueries();
      return write_fallback_result();
    }
    if (!BeginGuestOcclusionQuery(sample_count_addr)) {
      return write_fallback_result();
    }
    return true;
  }

  if (!active_occlusion_query_.valid ||
      active_occlusion_query_.sample_count_address != sample_count_addr) {
    DisableHostOcclusionQueries();
    return write_fallback_result();
  }

  if (!EndGuestOcclusionQuery(sample_count_addr, sample_counts)) {
    return write_fallback_result();
  }

  return true;
}

bool D3D12CommandProcessor::PushTransitionBarrier(ID3D12Resource* resource,
                                                  D3D12_RESOURCE_STATES old_state,
                                                  D3D12_RESOURCE_STATES new_state,
                                                  UINT subresource) {
  if (old_state == new_state) {
    return false;
  }
  D3D12_RESOURCE_BARRIER barrier;
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  barrier.Transition.pResource = resource;
  barrier.Transition.Subresource = subresource;
  barrier.Transition.StateBefore = old_state;
  barrier.Transition.StateAfter = new_state;
  barriers_.push_back(barrier);
  return true;
}

void D3D12CommandProcessor::PushAliasingBarrier(ID3D12Resource* old_resource,
                                                ID3D12Resource* new_resource) {
  D3D12_RESOURCE_BARRIER barrier;
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
  barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  barrier.Aliasing.pResourceBefore = old_resource;
  barrier.Aliasing.pResourceAfter = new_resource;
  barriers_.push_back(barrier);
}

void D3D12CommandProcessor::PushUAVBarrier(ID3D12Resource* resource) {
  D3D12_RESOURCE_BARRIER barrier;
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  barrier.UAV.pResource = resource;
  barriers_.push_back(barrier);
}

void D3D12CommandProcessor::SubmitBarriers() {
  UINT barrier_count = UINT(barriers_.size());
  if (barrier_count != 0) {
    deferred_command_list_.D3DResourceBarrier(barrier_count, barriers_.data());
    barriers_.clear();
  }
}

ID3D12RootSignature* D3D12CommandProcessor::GetRootSignature(const DxbcShader* vertex_shader,
                                                             const DxbcShader* pixel_shader,
                                                             bool tessellated) {
  if (bindless_resources_used_) {
    return tessellated ? root_signature_bindless_ds_ : root_signature_bindless_vs_;
  }

  D3D12_SHADER_VISIBILITY vertex_visibility =
      tessellated ? D3D12_SHADER_VISIBILITY_DOMAIN : D3D12_SHADER_VISIBILITY_VERTEX;

  uint32_t texture_count_vertex =
      uint32_t(vertex_shader->GetTextureBindingsAfterTranslation().size());
  uint32_t sampler_count_vertex =
      uint32_t(vertex_shader->GetSamplerBindingsAfterTranslation().size());
  uint32_t texture_count_pixel =
      pixel_shader ? uint32_t(pixel_shader->GetTextureBindingsAfterTranslation().size()) : 0;
  uint32_t sampler_count_pixel =
      pixel_shader ? uint32_t(pixel_shader->GetSamplerBindingsAfterTranslation().size()) : 0;

  // Better put the pixel texture/sampler in the lower bits probably because it
  // changes often.
  uint32_t index = 0;
  uint32_t index_offset = 0;
  index |= texture_count_pixel << index_offset;
  index_offset += D3D12Shader::kMaxTextureBindingIndexBits;
  index |= sampler_count_pixel << index_offset;
  index_offset += D3D12Shader::kMaxSamplerBindingIndexBits;
  index |= texture_count_vertex << index_offset;
  index_offset += D3D12Shader::kMaxTextureBindingIndexBits;
  index |= sampler_count_vertex << index_offset;
  index_offset += D3D12Shader::kMaxSamplerBindingIndexBits;
  index |= uint32_t(vertex_visibility == D3D12_SHADER_VISIBILITY_DOMAIN) << index_offset;
  ++index_offset;
  assert_true(index_offset <= 32);

  // Try an existing root signature.
  auto it = root_signatures_bindful_.find(index);
  if (it != root_signatures_bindful_.end()) {
    return it->second;
  }

  // Create a new one.
  D3D12_ROOT_SIGNATURE_DESC desc;
  D3D12_ROOT_PARAMETER parameters[kRootParameter_Bindful_Count_Max];
  desc.NumParameters = kRootParameter_Bindful_Count_Base;
  desc.pParameters = parameters;
  desc.NumStaticSamplers = 0;
  desc.pStaticSamplers = nullptr;
  desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

  // Base parameters.

  // Fetch constants.
  {
    auto& parameter = parameters[kRootParameter_Bindful_FetchConstants];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameter.Descriptor.ShaderRegister =
        uint32_t(DxbcShaderTranslator::CbufferRegister::kFetchConstants);
    parameter.Descriptor.RegisterSpace = 0;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }

  // Vertex float constants.
  {
    auto& parameter = parameters[kRootParameter_Bindful_FloatConstantsVertex];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameter.Descriptor.ShaderRegister =
        uint32_t(DxbcShaderTranslator::CbufferRegister::kFloatConstants);
    parameter.Descriptor.RegisterSpace = 0;
    parameter.ShaderVisibility = vertex_visibility;
  }

  // Pixel float constants.
  {
    auto& parameter = parameters[kRootParameter_Bindful_FloatConstantsPixel];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameter.Descriptor.ShaderRegister =
        uint32_t(DxbcShaderTranslator::CbufferRegister::kFloatConstants);
    parameter.Descriptor.RegisterSpace = 0;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
  }

  // System constants.
  {
    auto& parameter = parameters[kRootParameter_Bindful_SystemConstants];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameter.Descriptor.ShaderRegister =
        uint32_t(DxbcShaderTranslator::CbufferRegister::kSystemConstants);
    parameter.Descriptor.RegisterSpace = 0;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }

  // Bool and loop constants.
  {
    auto& parameter = parameters[kRootParameter_Bindful_BoolLoopConstants];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameter.Descriptor.ShaderRegister =
        uint32_t(DxbcShaderTranslator::CbufferRegister::kBoolLoopConstants);
    parameter.Descriptor.RegisterSpace = 0;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }

  // Shared memory and, if ROVs are used, EDRAM.
  D3D12_DESCRIPTOR_RANGE shared_memory_and_edram_ranges[3];
  {
    auto& parameter = parameters[kRootParameter_Bindful_SharedMemoryAndEdram];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameter.DescriptorTable.NumDescriptorRanges = 2;
    parameter.DescriptorTable.pDescriptorRanges = shared_memory_and_edram_ranges;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    shared_memory_and_edram_ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    shared_memory_and_edram_ranges[0].NumDescriptors = 1;
    shared_memory_and_edram_ranges[0].BaseShaderRegister =
        uint32_t(DxbcShaderTranslator::SRVMainRegister::kSharedMemory);
    shared_memory_and_edram_ranges[0].RegisterSpace =
        uint32_t(DxbcShaderTranslator::SRVSpace::kMain);
    shared_memory_and_edram_ranges[0].OffsetInDescriptorsFromTableStart = 0;
    shared_memory_and_edram_ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    shared_memory_and_edram_ranges[1].NumDescriptors = 1;
    shared_memory_and_edram_ranges[1].BaseShaderRegister =
        UINT(DxbcShaderTranslator::UAVRegister::kSharedMemory);
    shared_memory_and_edram_ranges[1].RegisterSpace = 0;
    shared_memory_and_edram_ranges[1].OffsetInDescriptorsFromTableStart = 1;
    if (render_target_cache_->GetPath() == RenderTargetCache::Path::kPixelShaderInterlock) {
      ++parameter.DescriptorTable.NumDescriptorRanges;
      shared_memory_and_edram_ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
      shared_memory_and_edram_ranges[2].NumDescriptors = 1;
      shared_memory_and_edram_ranges[2].BaseShaderRegister =
          UINT(DxbcShaderTranslator::UAVRegister::kEdram);
      shared_memory_and_edram_ranges[2].RegisterSpace = 0;
      shared_memory_and_edram_ranges[2].OffsetInDescriptorsFromTableStart = 2;
    }
  }

  // amped3: screen extents slot.
  {
    auto& parameter = parameters[kRootParameter_Bindful_ScreenExtents];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    parameter.Descriptor.ShaderRegister = UINT(DxbcShaderTranslator::UAVRegister::kScreenExtents);
    parameter.Descriptor.RegisterSpace = 0;
    parameter.ShaderVisibility = vertex_visibility;
  }

  // Extra parameters.

  // Pixel textures.
  D3D12_DESCRIPTOR_RANGE range_textures_pixel;
  if (texture_count_pixel > 0) {
    auto& parameter = parameters[desc.NumParameters];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameter.DescriptorTable.NumDescriptorRanges = 1;
    parameter.DescriptorTable.pDescriptorRanges = &range_textures_pixel;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    range_textures_pixel.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range_textures_pixel.NumDescriptors = texture_count_pixel;
    range_textures_pixel.BaseShaderRegister =
        uint32_t(DxbcShaderTranslator::SRVMainRegister::kBindfulTexturesStart);
    range_textures_pixel.RegisterSpace = uint32_t(DxbcShaderTranslator::SRVSpace::kMain);
    range_textures_pixel.OffsetInDescriptorsFromTableStart = 0;
    ++desc.NumParameters;
  }

  // Pixel samplers.
  D3D12_DESCRIPTOR_RANGE range_samplers_pixel;
  if (sampler_count_pixel > 0) {
    auto& parameter = parameters[desc.NumParameters];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameter.DescriptorTable.NumDescriptorRanges = 1;
    parameter.DescriptorTable.pDescriptorRanges = &range_samplers_pixel;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    range_samplers_pixel.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
    range_samplers_pixel.NumDescriptors = sampler_count_pixel;
    range_samplers_pixel.BaseShaderRegister = 0;
    range_samplers_pixel.RegisterSpace = 0;
    range_samplers_pixel.OffsetInDescriptorsFromTableStart = 0;
    ++desc.NumParameters;
  }

  // Vertex textures.
  D3D12_DESCRIPTOR_RANGE range_textures_vertex;
  if (texture_count_vertex > 0) {
    auto& parameter = parameters[desc.NumParameters];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameter.DescriptorTable.NumDescriptorRanges = 1;
    parameter.DescriptorTable.pDescriptorRanges = &range_textures_vertex;
    parameter.ShaderVisibility = vertex_visibility;
    range_textures_vertex.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range_textures_vertex.NumDescriptors = texture_count_vertex;
    range_textures_vertex.BaseShaderRegister =
        uint32_t(DxbcShaderTranslator::SRVMainRegister::kBindfulTexturesStart);
    range_textures_vertex.RegisterSpace = uint32_t(DxbcShaderTranslator::SRVSpace::kMain);
    range_textures_vertex.OffsetInDescriptorsFromTableStart = 0;
    ++desc.NumParameters;
  }

  // Vertex samplers.
  D3D12_DESCRIPTOR_RANGE range_samplers_vertex;
  if (sampler_count_vertex > 0) {
    auto& parameter = parameters[desc.NumParameters];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameter.DescriptorTable.NumDescriptorRanges = 1;
    parameter.DescriptorTable.pDescriptorRanges = &range_samplers_vertex;
    parameter.ShaderVisibility = vertex_visibility;
    range_samplers_vertex.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
    range_samplers_vertex.NumDescriptors = sampler_count_vertex;
    range_samplers_vertex.BaseShaderRegister = 0;
    range_samplers_vertex.RegisterSpace = 0;
    range_samplers_vertex.OffsetInDescriptorsFromTableStart = 0;
    ++desc.NumParameters;
  }

  ID3D12RootSignature* root_signature =
      ui::d3d12::util::CreateRootSignature(GetD3D12Provider(), desc);
  if (root_signature == nullptr) {
    REXGPU_ERROR(
        "Failed to create a root signature with {} pixel textures, {} pixel "
        "samplers, {} vertex textures and {} vertex samplers",
        texture_count_pixel, sampler_count_pixel, texture_count_vertex, sampler_count_vertex);
    return nullptr;
  }
  root_signatures_bindful_.emplace(index, root_signature);
  return root_signature;
}

uint32_t D3D12CommandProcessor::GetRootBindfulExtraParameterIndices(
    const DxbcShader* vertex_shader, const DxbcShader* pixel_shader,
    RootBindfulExtraParameterIndices& indices_out) {
  uint32_t index = kRootParameter_Bindful_Count_Base;
  if (pixel_shader && !pixel_shader->GetTextureBindingsAfterTranslation().empty()) {
    indices_out.textures_pixel = index++;
  } else {
    indices_out.textures_pixel = RootBindfulExtraParameterIndices::kUnavailable;
  }
  if (pixel_shader && !pixel_shader->GetSamplerBindingsAfterTranslation().empty()) {
    indices_out.samplers_pixel = index++;
  } else {
    indices_out.samplers_pixel = RootBindfulExtraParameterIndices::kUnavailable;
  }
  if (!vertex_shader->GetTextureBindingsAfterTranslation().empty()) {
    indices_out.textures_vertex = index++;
  } else {
    indices_out.textures_vertex = RootBindfulExtraParameterIndices::kUnavailable;
  }
  if (!vertex_shader->GetSamplerBindingsAfterTranslation().empty()) {
    indices_out.samplers_vertex = index++;
  } else {
    indices_out.samplers_vertex = RootBindfulExtraParameterIndices::kUnavailable;
  }
  return index;
}

uint64_t D3D12CommandProcessor::RequestViewBindfulDescriptors(
    uint64_t previous_heap_index, uint32_t count_for_partial_update, uint32_t count_for_full_update,
    D3D12_CPU_DESCRIPTOR_HANDLE& cpu_handle_out, D3D12_GPU_DESCRIPTOR_HANDLE& gpu_handle_out) {
  assert_false(bindless_resources_used_);
  assert_true(submission_open_);
  uint32_t descriptor_index;
  uint64_t current_heap_index = view_bindful_heap_pool_->Request(
      frame_current_, previous_heap_index, count_for_partial_update, count_for_full_update,
      descriptor_index);
  if (current_heap_index == ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid) {
    // There was an error.
    return ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid;
  }
  ID3D12DescriptorHeap* heap = view_bindful_heap_pool_->GetLastRequestHeap();
  if (view_bindful_heap_current_ != heap) {
    view_bindful_heap_current_ = heap;
    deferred_command_list_.SetDescriptorHeaps(view_bindful_heap_current_,
                                              sampler_bindful_heap_current_);
  }
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  cpu_handle_out = provider.OffsetViewDescriptor(
      view_bindful_heap_pool_->GetLastRequestHeapCPUStart(), descriptor_index);
  gpu_handle_out = provider.OffsetViewDescriptor(
      view_bindful_heap_pool_->GetLastRequestHeapGPUStart(), descriptor_index);
  return current_heap_index;
}

uint32_t D3D12CommandProcessor::RequestPersistentViewBindlessDescriptor() {
  assert_true(bindless_resources_used_);
  if (!view_bindless_heap_free_.empty()) {
    uint32_t descriptor_index = view_bindless_heap_free_.back();
    view_bindless_heap_free_.pop_back();
    return descriptor_index;
  }
  if (view_bindless_heap_allocated_ >= kViewBindlessHeapSize) {
    return UINT32_MAX;
  }
  return view_bindless_heap_allocated_++;
}

void D3D12CommandProcessor::ReleaseViewBindlessDescriptorImmediately(uint32_t descriptor_index) {
  assert_true(bindless_resources_used_);
  view_bindless_heap_free_.push_back(descriptor_index);
}

bool D3D12CommandProcessor::RequestOneUseSingleViewDescriptors(
    uint32_t count, ui::d3d12::util::DescriptorCpuGpuHandlePair* handles_out) {
  assert_true(submission_open_);
  if (!count) {
    return true;
  }
  assert_not_null(handles_out);
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  if (bindless_resources_used_) {
    // Request separate bindless descriptors that will be freed when this
    // submission is completed by the GPU.
    if (count >
        kViewBindlessHeapSize - view_bindless_heap_allocated_ + view_bindless_heap_free_.size()) {
      return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t descriptor_index;
      if (!view_bindless_heap_free_.empty()) {
        descriptor_index = view_bindless_heap_free_.back();
        view_bindless_heap_free_.pop_back();
      } else {
        descriptor_index = view_bindless_heap_allocated_++;
      }
      view_bindless_one_use_descriptors_.push_back(
          std::make_pair(descriptor_index, submission_current_));
      handles_out[i] = std::make_pair(
          provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_, descriptor_index),
          provider.OffsetViewDescriptor(view_bindless_heap_gpu_start_, descriptor_index));
    }
  } else {
    // Request a range within the current heap for bindful resources path.
    D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle_start;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle_start;
    if (RequestViewBindfulDescriptors(ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid, count,
                                      count, cpu_handle_start, gpu_handle_start) ==
        ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid) {
      return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
      handles_out[i] = std::make_pair(provider.OffsetViewDescriptor(cpu_handle_start, i),
                                      provider.OffsetViewDescriptor(gpu_handle_start, i));
    }
  }
  return true;
}

ui::d3d12::util::DescriptorCpuGpuHandlePair D3D12CommandProcessor::GetSystemBindlessViewHandlePair(
    SystemBindlessView view) const {
  assert_true(bindless_resources_used_);
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  return std::make_pair(
      provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_, uint32_t(view)),
      provider.OffsetViewDescriptor(view_bindless_heap_gpu_start_, uint32_t(view)));
}

ui::d3d12::util::DescriptorCpuGpuHandlePair
D3D12CommandProcessor::GetSharedMemoryUintPow2BindlessSRVHandlePair(
    uint32_t element_size_bytes_pow2) const {
  SystemBindlessView view;
  switch (element_size_bytes_pow2) {
    case 2:
      view = SystemBindlessView::kSharedMemoryR32UintSRV;
      break;
    case 3:
      view = SystemBindlessView::kSharedMemoryR32G32UintSRV;
      break;
    case 4:
      view = SystemBindlessView::kSharedMemoryR32G32B32A32UintSRV;
      break;
    default:
      assert_unhandled_case(element_size_bytes_pow2);
      view = SystemBindlessView::kSharedMemoryR32UintSRV;
  }
  return GetSystemBindlessViewHandlePair(view);
}

ui::d3d12::util::DescriptorCpuGpuHandlePair
D3D12CommandProcessor::GetSharedMemoryUintPow2BindlessUAVHandlePair(
    uint32_t element_size_bytes_pow2) const {
  SystemBindlessView view;
  switch (element_size_bytes_pow2) {
    case 2:
      view = SystemBindlessView::kSharedMemoryR32UintUAV;
      break;
    case 3:
      view = SystemBindlessView::kSharedMemoryR32G32UintUAV;
      break;
    case 4:
      view = SystemBindlessView::kSharedMemoryR32G32B32A32UintUAV;
      break;
    default:
      assert_unhandled_case(element_size_bytes_pow2);
      view = SystemBindlessView::kSharedMemoryR32UintUAV;
  }
  return GetSystemBindlessViewHandlePair(view);
}

ui::d3d12::util::DescriptorCpuGpuHandlePair
D3D12CommandProcessor::GetEdramUintPow2BindlessSRVHandlePair(
    uint32_t element_size_bytes_pow2) const {
  SystemBindlessView view;
  switch (element_size_bytes_pow2) {
    case 2:
      view = SystemBindlessView::kEdramR32UintSRV;
      break;
    case 3:
      view = SystemBindlessView::kEdramR32G32UintSRV;
      break;
    case 4:
      view = SystemBindlessView::kEdramR32G32B32A32UintSRV;
      break;
    default:
      assert_unhandled_case(element_size_bytes_pow2);
      view = SystemBindlessView::kEdramR32UintSRV;
  }
  return GetSystemBindlessViewHandlePair(view);
}

ui::d3d12::util::DescriptorCpuGpuHandlePair
D3D12CommandProcessor::GetEdramUintPow2BindlessUAVHandlePair(
    uint32_t element_size_bytes_pow2) const {
  SystemBindlessView view;
  switch (element_size_bytes_pow2) {
    case 2:
      view = SystemBindlessView::kEdramR32UintUAV;
      break;
    case 3:
      view = SystemBindlessView::kEdramR32G32UintUAV;
      break;
    case 4:
      view = SystemBindlessView::kEdramR32G32B32A32UintUAV;
      break;
    default:
      assert_unhandled_case(element_size_bytes_pow2);
      view = SystemBindlessView::kEdramR32UintUAV;
  }
  return GetSystemBindlessViewHandlePair(view);
}

uint64_t D3D12CommandProcessor::RequestSamplerBindfulDescriptors(
    uint64_t previous_heap_index, uint32_t count_for_partial_update, uint32_t count_for_full_update,
    D3D12_CPU_DESCRIPTOR_HANDLE& cpu_handle_out, D3D12_GPU_DESCRIPTOR_HANDLE& gpu_handle_out) {
  assert_false(bindless_resources_used_);
  assert_true(submission_open_);
  uint32_t descriptor_index;
  uint64_t current_heap_index = sampler_bindful_heap_pool_->Request(
      frame_current_, previous_heap_index, count_for_partial_update, count_for_full_update,
      descriptor_index);
  if (current_heap_index == ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid) {
    // There was an error.
    return ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid;
  }
  ID3D12DescriptorHeap* heap = sampler_bindful_heap_pool_->GetLastRequestHeap();
  if (sampler_bindful_heap_current_ != heap) {
    sampler_bindful_heap_current_ = heap;
    deferred_command_list_.SetDescriptorHeaps(view_bindful_heap_current_,
                                              sampler_bindful_heap_current_);
  }
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  cpu_handle_out = provider.OffsetSamplerDescriptor(
      sampler_bindful_heap_pool_->GetLastRequestHeapCPUStart(), descriptor_index);
  gpu_handle_out = provider.OffsetSamplerDescriptor(
      sampler_bindful_heap_pool_->GetLastRequestHeapGPUStart(), descriptor_index);
  return current_heap_index;
}

ID3D12Resource* D3D12CommandProcessor::RequestScratchGPUBuffer(uint32_t size,
                                                               D3D12_RESOURCE_STATES state) {
  assert_true(submission_open_);
  assert_false(scratch_buffer_used_);
  if (!submission_open_ || scratch_buffer_used_ || size == 0) {
    return nullptr;
  }

  if (size <= scratch_buffer_size_) {
    PushTransitionBarrier(scratch_buffer_, scratch_buffer_state_, state);
    scratch_buffer_state_ = state;
    scratch_buffer_used_ = true;
    return scratch_buffer_;
  }

  size = rex::align(size, kScratchBufferSizeIncrement);

  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  D3D12_RESOURCE_DESC buffer_desc;
  ui::d3d12::util::FillBufferResourceDesc(buffer_desc, size,
                                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  ID3D12Resource* buffer;
  if (FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesDefault,
                                             provider.GetHeapFlagCreateNotZeroed(), &buffer_desc,
                                             state, nullptr, IID_PPV_ARGS(&buffer)))) {
    REXGPU_ERROR("Failed to create a {} MB scratch GPU buffer", size >> 20);
    return nullptr;
  }
  if (scratch_buffer_ != nullptr) {
    resources_for_deletion_.emplace_back(submission_current_, scratch_buffer_);
  }
  scratch_buffer_ = buffer;
  scratch_buffer_size_ = size;
  scratch_buffer_state_ = state;
  scratch_buffer_used_ = true;
  return scratch_buffer_;
}

void D3D12CommandProcessor::ReleaseScratchGPUBuffer(ID3D12Resource* buffer,
                                                    D3D12_RESOURCE_STATES new_state) {
  assert_true(submission_open_);
  assert_true(scratch_buffer_used_);
  scratch_buffer_used_ = false;
  if (buffer == scratch_buffer_) {
    scratch_buffer_state_ = new_state;
  }
}

void D3D12CommandProcessor::SetExternalPipeline(ID3D12PipelineState* pipeline) {
  if (current_external_pipeline_ != pipeline) {
    current_external_pipeline_ = pipeline;
    current_guest_pipeline_ = nullptr;
    deferred_command_list_.D3DSetPipelineState(pipeline);
  }
}

void D3D12CommandProcessor::InvalidateStateAfterExternalCommands() {
  ff_viewport_update_needed_ = true;
  ff_scissor_update_needed_ = true;
  ff_blend_factor_update_needed_ = true;
  ff_stencil_ref_update_needed_ = true;
  viewport_cache_valid_ = false;
  current_guest_pipeline_ = nullptr;
  current_external_pipeline_ = nullptr;
  current_graphics_root_signature_ = nullptr;
  current_graphics_root_up_to_date_ = 0;
  primitive_topology_ = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
  vertex_buffers_in_sync_[0] = 0;
  vertex_buffers_in_sync_[1] = 0;
  if (bindless_resources_used_) {
    deferred_command_list_.SetDescriptorHeaps(view_bindless_heap_, sampler_bindless_heap_current_);
  } else if (view_bindful_heap_current_ || sampler_bindful_heap_current_) {
    deferred_command_list_.SetDescriptorHeaps(view_bindful_heap_current_,
                                              sampler_bindful_heap_current_);
  }
  render_target_cache_->InvalidateCommandListRenderTargets();
}

void D3D12CommandProcessor::SetExternalGraphicsRootSignature(ID3D12RootSignature* root_signature) {
  if (current_graphics_root_signature_ != root_signature) {
    current_graphics_root_signature_ = root_signature;
    tile_replay_root_signature_ = nullptr;
    deferred_command_list_.D3DSetGraphicsRootSignature(root_signature);
  }
  // Force-invalidate because setting a non-guest root signature.
  current_graphics_root_up_to_date_ = 0;
}

void D3D12CommandProcessor::SetViewport(const D3D12_VIEWPORT& viewport) {
  ff_viewport_update_needed_ |= ff_viewport_.TopLeftX != viewport.TopLeftX;
  ff_viewport_update_needed_ |= ff_viewport_.TopLeftY != viewport.TopLeftY;
  ff_viewport_update_needed_ |= ff_viewport_.Width != viewport.Width;
  ff_viewport_update_needed_ |= ff_viewport_.Height != viewport.Height;
  ff_viewport_update_needed_ |= ff_viewport_.MinDepth != viewport.MinDepth;
  ff_viewport_update_needed_ |= ff_viewport_.MaxDepth != viewport.MaxDepth;
  if (ff_viewport_update_needed_) {
    ff_viewport_ = viewport;
    deferred_command_list_.RSSetViewport(ff_viewport_);
    ff_viewport_update_needed_ = false;
  }
}

void D3D12CommandProcessor::SetScissorRect(const D3D12_RECT& scissor_rect) {
  ff_scissor_update_needed_ |= ff_scissor_.left != scissor_rect.left;
  ff_scissor_update_needed_ |= ff_scissor_.top != scissor_rect.top;
  ff_scissor_update_needed_ |= ff_scissor_.right != scissor_rect.right;
  ff_scissor_update_needed_ |= ff_scissor_.bottom != scissor_rect.bottom;
  if (ff_scissor_update_needed_) {
    ff_scissor_ = scissor_rect;
    deferred_command_list_.RSSetScissorRect(ff_scissor_);
    ff_scissor_update_needed_ = false;
  }
}

void D3D12CommandProcessor::SetStencilReference(uint32_t stencil_ref) {
  ff_stencil_ref_update_needed_ |= ff_stencil_ref_ != stencil_ref;
  if (ff_stencil_ref_update_needed_) {
    ff_stencil_ref_ = stencil_ref;
    deferred_command_list_.D3DOMSetStencilRef(stencil_ref);
    ff_stencil_ref_update_needed_ = false;
  }
}

void D3D12CommandProcessor::SetPrimitiveTopology(D3D12_PRIMITIVE_TOPOLOGY primitive_topology) {
  if (primitive_topology_ != primitive_topology) {
    primitive_topology_ = primitive_topology;
    deferred_command_list_.D3DIASetPrimitiveTopology(primitive_topology);
  }
}

std::string D3D12CommandProcessor::GetWindowTitleText() const {
  std::ostringstream title;
  title << "Direct3D 12";
  if (render_target_cache_) {
    // Rasterizer-ordered views are a feature very rarely used as of 2020 and
    // that faces adoption complications (outside of Direct3D - on Vulkan - at
    // least), but crucial to Xenia - raise awareness of its usage.
    // https://github.com/KhronosGroup/Vulkan-Ecosystem/issues/27#issuecomment-455712319
    // "In Xenia's title bar "D3D12 ROV" can be seen, which was a surprise, as I
    //  wasn't aware that Xenia D3D12 backend was using Raster Order Views
    //  feature" - oscarbg in that issue.
    switch (render_target_cache_->GetPath()) {
      case RenderTargetCache::Path::kHostRenderTargets:
        title << " - RTV/DSV";
        break;
      case RenderTargetCache::Path::kPixelShaderInterlock:
        title << " - ROV";
        break;
      default:
        break;
    }
    uint32_t draw_resolution_scale_x =
        texture_cache_ ? texture_cache_->draw_resolution_scale_x() : 1;
    uint32_t draw_resolution_scale_y =
        texture_cache_ ? texture_cache_->draw_resolution_scale_y() : 1;
    if (draw_resolution_scale_x > 1 || draw_resolution_scale_y > 1) {
      title << ' ' << draw_resolution_scale_x << 'x' << draw_resolution_scale_y;
    }
  }
  return title.str();
}

bool D3D12CommandProcessor::SetupContext() {
  if (!CommandProcessor::SetupContext()) {
    REXGPU_ERROR("Failed to initialize base command processor context");
    return false;
  }
  InvalidateAllVertexBufferResidency();
  UpdateDebugMarkersEnabled();

  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  ID3D12CommandQueue* direct_queue = provider.GetDirectQueue();

  fence_completion_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (fence_completion_event_ == nullptr) {
    REXGPU_ERROR("Failed to create the fence completion event");
    return false;
  }
  if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&submission_fence_)))) {
    REXGPU_ERROR("Failed to create the submission fence");
    return false;
  }
  if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                 IID_PPV_ARGS(&queue_operations_since_submission_fence_)))) {
    REXGPU_ERROR(
        "Failed to create the fence for awaiting queue operations done since "
        "the latest submission");
    return false;
  }

  // Create the command list and one allocator because it's needed for a command
  // list.
  ID3D12CommandAllocator* command_allocator;
  if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            IID_PPV_ARGS(&command_allocator)))) {
    REXGPU_ERROR("Failed to create a command allocator");
    return false;
  }
  command_allocator_writable_first_ = new CommandAllocator;
  command_allocator_writable_first_->command_allocator = command_allocator;
  command_allocator_writable_first_->last_usage_submission = 0;
  command_allocator_writable_first_->next = nullptr;
  command_allocator_writable_last_ = command_allocator_writable_first_;
  if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, command_allocator,
                                       nullptr, IID_PPV_ARGS(&command_list_)))) {
    REXGPU_ERROR("Failed to create the graphics command list");
    return false;
  }
  // Initially in open state, wait until a deferred command list submission.
  command_list_->Close();
  // Optional - added in Creators Update (SDK 10.0.15063.0).
  command_list_->QueryInterface(IID_PPV_ARGS(&command_list_1_));

  bindless_resources_used_ = REXCVAR_GET(d3d12_bindless) &&
                             provider.GetResourceBindingTier() >= D3D12_RESOURCE_BINDING_TIER_2;

  // Get the draw resolution scale for the render target cache and the texture
  // cache.
  uint32_t draw_resolution_scale_x, draw_resolution_scale_y;
  bool draw_resolution_scale_not_clamped =
      TextureCache::GetConfigDrawResolutionScale(draw_resolution_scale_x, draw_resolution_scale_y);
  if (!D3D12TextureCache::ClampDrawResolutionScaleToMaxSupported(
          draw_resolution_scale_x, draw_resolution_scale_y, provider)) {
    draw_resolution_scale_not_clamped = false;
  }
  if (!draw_resolution_scale_not_clamped) {
    REXGPU_WARN(
        "The requested draw resolution scale is not supported by the device or "
        "the emulator, reducing to {}x{}",
        draw_resolution_scale_x, draw_resolution_scale_y);
  }

  shared_memory_ = std::make_unique<D3D12SharedMemory>(*this, *memory_);
  if (!shared_memory_->Initialize()) {
    REXGPU_ERROR("Failed to initialize shared memory");
    return false;
  }

  // Initialize the render target cache before configuring binding - need to
  // know if using rasterizer-ordered views for the bindless root signature.
  render_target_cache_ = std::make_unique<D3D12RenderTargetCache>(
      *register_file_, *memory_, draw_resolution_scale_x, draw_resolution_scale_y, *this,
      bindless_resources_used_);
  if (!render_target_cache_->Initialize()) {
    REXGPU_ERROR("Failed to initialize the render target cache");
    return false;
  }

  // Initialize resource binding.
  constant_buffer_pool_ = std::make_unique<ui::d3d12::D3D12UploadBufferPool>(
      provider, std::max(ui::d3d12::D3D12UploadBufferPool::kDefaultPageSize,
                         sizeof(float) * 4 * D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT));
  if (bindless_resources_used_) {
    D3D12_DESCRIPTOR_HEAP_DESC view_bindless_heap_desc;
    view_bindless_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    view_bindless_heap_desc.NumDescriptors = kViewBindlessHeapSize;
    view_bindless_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    view_bindless_heap_desc.NodeMask = 0;
    if (FAILED(device->CreateDescriptorHeap(&view_bindless_heap_desc,
                                            IID_PPV_ARGS(&view_bindless_heap_)))) {
      REXGPU_ERROR("Failed to create the bindless CBV/SRV/UAV descriptor heap");
      return false;
    }
    view_bindless_heap_cpu_start_ = view_bindless_heap_->GetCPUDescriptorHandleForHeapStart();
    view_bindless_heap_gpu_start_ = view_bindless_heap_->GetGPUDescriptorHandleForHeapStart();
    view_bindless_heap_allocated_ = uint32_t(SystemBindlessView::kCount);

    D3D12_DESCRIPTOR_HEAP_DESC sampler_bindless_heap_desc;
    sampler_bindless_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    sampler_bindless_heap_desc.NumDescriptors = kSamplerHeapSize;
    sampler_bindless_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    sampler_bindless_heap_desc.NodeMask = 0;
    if (FAILED(device->CreateDescriptorHeap(&sampler_bindless_heap_desc,
                                            IID_PPV_ARGS(&sampler_bindless_heap_current_)))) {
      REXGPU_ERROR("Failed to create the bindless sampler descriptor heap");
      return false;
    }
    sampler_bindless_heap_cpu_start_ =
        sampler_bindless_heap_current_->GetCPUDescriptorHandleForHeapStart();
    sampler_bindless_heap_gpu_start_ =
        sampler_bindless_heap_current_->GetGPUDescriptorHandleForHeapStart();
    sampler_bindless_heap_allocated_ = 0;
  } else {
    view_bindful_heap_pool_ = std::make_unique<ui::d3d12::D3D12DescriptorHeapPool>(
        device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kViewBindfulHeapSize);
    sampler_bindful_heap_pool_ = std::make_unique<ui::d3d12::D3D12DescriptorHeapPool>(
        device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, kSamplerHeapSize);
  }

  if (bindless_resources_used_) {
    // Global bindless resource root signatures.
    // No CBV or UAV descriptor ranges with any descriptors to be allocated
    // dynamically (via RequestPersistentViewBindlessDescriptor or
    // RequestOneUseSingleViewDescriptors) should be here, because they would
    // overlap the unbounded SRV range, which is not allowed on Nvidia Fermi!
    D3D12_ROOT_SIGNATURE_DESC root_signature_bindless_desc;
    D3D12_ROOT_PARAMETER
    root_parameters_bindless[kRootParameter_Bindless_Count];
    root_signature_bindless_desc.NumParameters = kRootParameter_Bindless_Count;
    root_signature_bindless_desc.pParameters = root_parameters_bindless;
    root_signature_bindless_desc.NumStaticSamplers = 0;
    root_signature_bindless_desc.pStaticSamplers = nullptr;
    root_signature_bindless_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    // Fetch constants.
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_FetchConstants];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
      parameter.Descriptor.ShaderRegister =
          uint32_t(DxbcShaderTranslator::CbufferRegister::kFetchConstants);
      parameter.Descriptor.RegisterSpace = 0;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    // Vertex float constants.
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_FloatConstantsVertex];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
      parameter.Descriptor.ShaderRegister =
          uint32_t(DxbcShaderTranslator::CbufferRegister::kFloatConstants);
      parameter.Descriptor.RegisterSpace = 0;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    }
    // Pixel float constants.
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_FloatConstantsPixel];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
      parameter.Descriptor.ShaderRegister =
          uint32_t(DxbcShaderTranslator::CbufferRegister::kFloatConstants);
      parameter.Descriptor.RegisterSpace = 0;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    }
    // Pixel shader descriptor indices.
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_DescriptorIndicesPixel];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
      parameter.Descriptor.ShaderRegister =
          uint32_t(DxbcShaderTranslator::CbufferRegister::kDescriptorIndices);
      parameter.Descriptor.RegisterSpace = 0;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    }
    // Vertex shader descriptor indices.
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_DescriptorIndicesVertex];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
      parameter.Descriptor.ShaderRegister =
          uint32_t(DxbcShaderTranslator::CbufferRegister::kDescriptorIndices);
      parameter.Descriptor.RegisterSpace = 0;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    }
    // System constants.
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_SystemConstants];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
      parameter.Descriptor.ShaderRegister =
          uint32_t(DxbcShaderTranslator::CbufferRegister::kSystemConstants);
      parameter.Descriptor.RegisterSpace = 0;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    // Bool and loop constants.
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_BoolLoopConstants];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
      parameter.Descriptor.ShaderRegister =
          uint32_t(DxbcShaderTranslator::CbufferRegister::kBoolLoopConstants);
      parameter.Descriptor.RegisterSpace = 0;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    // Shared memory SRV and UAV.
    D3D12_DESCRIPTOR_RANGE root_shared_memory_view_ranges[2];
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_SharedMemory];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      parameter.DescriptorTable.NumDescriptorRanges =
          uint32_t(rex::countof(root_shared_memory_view_ranges));
      parameter.DescriptorTable.pDescriptorRanges = root_shared_memory_view_ranges;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      {
        auto& range = root_shared_memory_view_ranges[0];
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range.NumDescriptors = 1;
        range.BaseShaderRegister = UINT(DxbcShaderTranslator::SRVMainRegister::kSharedMemory);
        range.RegisterSpace = UINT(DxbcShaderTranslator::SRVSpace::kMain);
        range.OffsetInDescriptorsFromTableStart = 0;
      }
      {
        auto& range = root_shared_memory_view_ranges[1];
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        range.NumDescriptors = 1;
        range.BaseShaderRegister = UINT(DxbcShaderTranslator::UAVRegister::kSharedMemory);
        range.RegisterSpace = 0;
        range.OffsetInDescriptorsFromTableStart = 1;
      }
    }
    // Sampler heap.
    D3D12_DESCRIPTOR_RANGE root_bindless_sampler_range;
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_SamplerHeap];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      // Will be appending.
      parameter.DescriptorTable.NumDescriptorRanges = 1;
      parameter.DescriptorTable.pDescriptorRanges = &root_bindless_sampler_range;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      root_bindless_sampler_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
      root_bindless_sampler_range.NumDescriptors = UINT_MAX;
      root_bindless_sampler_range.BaseShaderRegister = 0;
      root_bindless_sampler_range.RegisterSpace = 0;
      root_bindless_sampler_range.OffsetInDescriptorsFromTableStart = 0;
    }
    // View heap.
    D3D12_DESCRIPTOR_RANGE root_bindless_view_ranges[4];
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_ViewHeap];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      // Will be appending.
      parameter.DescriptorTable.NumDescriptorRanges = 0;
      parameter.DescriptorTable.pDescriptorRanges = root_bindless_view_ranges;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      // EDRAM.
      if (render_target_cache_->GetPath() == RenderTargetCache::Path::kPixelShaderInterlock) {
        assert_true(parameter.DescriptorTable.NumDescriptorRanges <
                    rex::countof(root_bindless_view_ranges));
        auto& range = root_bindless_view_ranges[parameter.DescriptorTable.NumDescriptorRanges++];
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        range.NumDescriptors = 1;
        range.BaseShaderRegister = UINT(DxbcShaderTranslator::UAVRegister::kEdram);
        range.RegisterSpace = 0;
        range.OffsetInDescriptorsFromTableStart = UINT(SystemBindlessView::kEdramR32UintUAV);
      }
      // Used UAV and SRV ranges must not overlap on Nvidia Fermi, so textures
      // have OffsetInDescriptorsFromTableStart after all static descriptors of
      // other types.
      // 2D array textures.
      {
        assert_true(parameter.DescriptorTable.NumDescriptorRanges <
                    rex::countof(root_bindless_view_ranges));
        auto& range = root_bindless_view_ranges[parameter.DescriptorTable.NumDescriptorRanges++];
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range.NumDescriptors = UINT_MAX;
        range.BaseShaderRegister = 0;
        range.RegisterSpace = UINT(DxbcShaderTranslator::SRVSpace::kBindlessTextures2DArray);
        range.OffsetInDescriptorsFromTableStart = UINT(SystemBindlessView::kUnboundedSRVsStart);
      }
      // 3D textures.
      {
        assert_true(parameter.DescriptorTable.NumDescriptorRanges <
                    rex::countof(root_bindless_view_ranges));
        auto& range = root_bindless_view_ranges[parameter.DescriptorTable.NumDescriptorRanges++];
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range.NumDescriptors = UINT_MAX;
        range.BaseShaderRegister = 0;
        range.RegisterSpace = UINT(DxbcShaderTranslator::SRVSpace::kBindlessTextures3D);
        range.OffsetInDescriptorsFromTableStart = UINT(SystemBindlessView::kUnboundedSRVsStart);
      }
      // Cube textures.
      {
        assert_true(parameter.DescriptorTable.NumDescriptorRanges <
                    rex::countof(root_bindless_view_ranges));
        auto& range = root_bindless_view_ranges[parameter.DescriptorTable.NumDescriptorRanges++];
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range.NumDescriptors = UINT_MAX;
        range.BaseShaderRegister = 0;
        range.RegisterSpace = UINT(DxbcShaderTranslator::SRVSpace::kBindlessTexturesCube);
        range.OffsetInDescriptorsFromTableStart = UINT(SystemBindlessView::kUnboundedSRVsStart);
      }
    }
    // amped3: screen extents slot.
    {
      auto& parameter = root_parameters_bindless[kRootParameter_Bindless_ScreenExtents];
      parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
      parameter.Descriptor.ShaderRegister =
          UINT(DxbcShaderTranslator::UAVRegister::kScreenExtents);
      parameter.Descriptor.RegisterSpace = 0;
      parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    root_signature_bindless_vs_ =
        ui::d3d12::util::CreateRootSignature(provider, root_signature_bindless_desc);
    if (!root_signature_bindless_vs_) {
      REXGPU_ERROR(
          "Failed to create the global root signature for bindless resources, "
          "the version for use without tessellation");
      return false;
    }
    root_parameters_bindless[kRootParameter_Bindless_FloatConstantsVertex].ShaderVisibility =
        D3D12_SHADER_VISIBILITY_DOMAIN;
    root_parameters_bindless[kRootParameter_Bindless_DescriptorIndicesVertex].ShaderVisibility =
        D3D12_SHADER_VISIBILITY_DOMAIN;
    root_signature_bindless_ds_ =
        ui::d3d12::util::CreateRootSignature(provider, root_signature_bindless_desc);
    if (!root_signature_bindless_ds_) {
      REXGPU_ERROR(
          "Failed to create the global root signature for bindless resources, "
          "the version for use with tessellation");
      return false;
    }
  }

  primitive_processor_ =
      std::make_unique<D3D12PrimitiveProcessor>(*register_file_, *memory_, *shared_memory_, *this);
  if (!primitive_processor_->Initialize()) {
    REXGPU_ERROR("Failed to initialize the geometric primitive processor");
    return false;
  }

  texture_cache_ =
      D3D12TextureCache::Create(*register_file_, *shared_memory_, draw_resolution_scale_x,
                                draw_resolution_scale_y, *this, bindless_resources_used_);
  if (!texture_cache_) {
    REXGPU_ERROR("Failed to initialize the texture cache");
    return false;
  }

  pipeline_cache_ = std::make_unique<PipelineCache>(
      *this, *register_file_, *render_target_cache_.get(), bindless_resources_used_);
  if (!pipeline_cache_->Initialize()) {
    REXGPU_ERROR("Failed to initialize the graphics pipeline cache");
    return false;
  }

  D3D12_HEAP_FLAGS heap_flag_create_not_zeroed = provider.GetHeapFlagCreateNotZeroed();

  // Create gamma ramp resources.
  gamma_ramp_256_entry_table_up_to_date_ = false;
  gamma_ramp_pwl_up_to_date_ = false;
  D3D12_RESOURCE_DESC gamma_ramp_buffer_desc;
  ui::d3d12::util::FillBufferResourceDesc(gamma_ramp_buffer_desc, (256 + 128 * 3) * 4,
                                          D3D12_RESOURCE_FLAG_NONE);
  // The first action will be uploading.
  gamma_ramp_buffer_state_ = D3D12_RESOURCE_STATE_COPY_DEST;
  if (FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesDefault,
                                             heap_flag_create_not_zeroed, &gamma_ramp_buffer_desc,
                                             gamma_ramp_buffer_state_, nullptr,
                                             IID_PPV_ARGS(&gamma_ramp_buffer_)))) {
    REXGPU_ERROR("Failed to create the gamma ramp buffer");
    return false;
  }
  // The upload buffer is frame-buffered.
  gamma_ramp_buffer_desc.Width *= kQueueFrames;
  if (FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesUpload,
                                             heap_flag_create_not_zeroed, &gamma_ramp_buffer_desc,
                                             D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                             IID_PPV_ARGS(&gamma_ramp_upload_buffer_)))) {
    REXGPU_ERROR("Failed to create the gamma ramp upload buffer");
    return false;
  }
  if (FAILED(gamma_ramp_upload_buffer_->Map(
          0, nullptr, reinterpret_cast<void**>(&gamma_ramp_upload_buffer_mapping_)))) {
    REXGPU_ERROR("Failed to map the gamma ramp upload buffer");
    gamma_ramp_upload_buffer_mapping_ = nullptr;
    return false;
  }

  // amped3 (gpu_screen_extents=3): screen extents slots.
  {
    D3D12_RESOURCE_DESC screen_extents_buffer_desc;
    ui::d3d12::util::FillBufferResourceDesc(screen_extents_buffer_desc,
                                            kScreenExtentsSlots * sizeof(uint32_t) * 4,
                                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (FAILED(device->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesDefault, heap_flag_create_not_zeroed,
            &screen_extents_buffer_desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&screen_extents_buffer_)))) {
      REXGPU_ERROR("Failed to create the screen extents buffer");
      return false;
    }
    screen_extents_buffer_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    void* mapping;
    if (FAILED(device->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesUpload, heap_flag_create_not_zeroed,
            &screen_extents_buffer_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&screen_extents_zero_buffer_))) ||
        FAILED(screen_extents_zero_buffer_->Map(0, nullptr, &mapping))) {
      REXGPU_ERROR("Failed to create the screen extents zero buffer");
      return false;
    }
    std::memset(mapping, 0, screen_extents_buffer_desc.Width);
    screen_extents_zero_buffer_->Unmap(0, nullptr);
    screen_extents_buffer_desc.Width *= kScreenExtentsReadbacks;
    if (FAILED(device->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesReadback, heap_flag_create_not_zeroed,
            &screen_extents_buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&screen_extents_readback_buffer_))) ||
        FAILED(screen_extents_readback_buffer_->Map(0, nullptr, &mapping))) {
      REXGPU_ERROR("Failed to create the screen extents readback buffer");
      return false;
    }
    screen_extents_readback_mapping_ = static_cast<const uint32_t*>(mapping);
    screen_extents_slots_used_ = 0;
    screen_extents_root_slot_ = 0;
    screen_extents_draw_.slot = UINT32_MAX;
    screen_extents_pending_.clear();
    screen_extents_batches_.clear();
    screen_extents_known_.clear();
  }

  // Initialize compute pipelines for output with gamma ramp.
  D3D12_ROOT_PARAMETER
  apply_gamma_root_parameters[UINT(ApplyGammaRootParameter::kCount)];
  {
    D3D12_ROOT_PARAMETER& apply_gamma_root_parameter_constants =
        apply_gamma_root_parameters[UINT(ApplyGammaRootParameter::kConstants)];
    apply_gamma_root_parameter_constants.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    apply_gamma_root_parameter_constants.Constants.ShaderRegister = 0;
    apply_gamma_root_parameter_constants.Constants.RegisterSpace = 0;
    apply_gamma_root_parameter_constants.Constants.Num32BitValues =
        sizeof(ApplyGammaConstants) / sizeof(uint32_t);
    apply_gamma_root_parameter_constants.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_DESCRIPTOR_RANGE apply_gamma_root_descriptor_range_dest;
  apply_gamma_root_descriptor_range_dest.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  apply_gamma_root_descriptor_range_dest.NumDescriptors = 1;
  apply_gamma_root_descriptor_range_dest.BaseShaderRegister = 0;
  apply_gamma_root_descriptor_range_dest.RegisterSpace = 0;
  apply_gamma_root_descriptor_range_dest.OffsetInDescriptorsFromTableStart = 0;
  {
    D3D12_ROOT_PARAMETER& apply_gamma_root_parameter_dest =
        apply_gamma_root_parameters[UINT(ApplyGammaRootParameter::kDestination)];
    apply_gamma_root_parameter_dest.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    apply_gamma_root_parameter_dest.DescriptorTable.NumDescriptorRanges = 1;
    apply_gamma_root_parameter_dest.DescriptorTable.pDescriptorRanges =
        &apply_gamma_root_descriptor_range_dest;
    apply_gamma_root_parameter_dest.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_DESCRIPTOR_RANGE apply_gamma_root_descriptor_range_source;
  apply_gamma_root_descriptor_range_source.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  apply_gamma_root_descriptor_range_source.NumDescriptors = 1;
  apply_gamma_root_descriptor_range_source.BaseShaderRegister = 1;
  apply_gamma_root_descriptor_range_source.RegisterSpace = 0;
  apply_gamma_root_descriptor_range_source.OffsetInDescriptorsFromTableStart = 0;
  {
    D3D12_ROOT_PARAMETER& apply_gamma_root_parameter_source =
        apply_gamma_root_parameters[UINT(ApplyGammaRootParameter::kSource)];
    apply_gamma_root_parameter_source.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    apply_gamma_root_parameter_source.DescriptorTable.NumDescriptorRanges = 1;
    apply_gamma_root_parameter_source.DescriptorTable.pDescriptorRanges =
        &apply_gamma_root_descriptor_range_source;
    apply_gamma_root_parameter_source.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_DESCRIPTOR_RANGE apply_gamma_root_descriptor_range_ramp;
  apply_gamma_root_descriptor_range_ramp.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  apply_gamma_root_descriptor_range_ramp.NumDescriptors = 1;
  apply_gamma_root_descriptor_range_ramp.BaseShaderRegister = 0;
  apply_gamma_root_descriptor_range_ramp.RegisterSpace = 0;
  apply_gamma_root_descriptor_range_ramp.OffsetInDescriptorsFromTableStart = 0;
  {
    D3D12_ROOT_PARAMETER& apply_gamma_root_parameter_gamma_ramp =
        apply_gamma_root_parameters[UINT(ApplyGammaRootParameter::kRamp)];
    apply_gamma_root_parameter_gamma_ramp.ParameterType =
        D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    apply_gamma_root_parameter_gamma_ramp.DescriptorTable.NumDescriptorRanges = 1;
    apply_gamma_root_parameter_gamma_ramp.DescriptorTable.pDescriptorRanges =
        &apply_gamma_root_descriptor_range_ramp;
    apply_gamma_root_parameter_gamma_ramp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_ROOT_SIGNATURE_DESC apply_gamma_root_signature_desc;
  apply_gamma_root_signature_desc.NumParameters = UINT(ApplyGammaRootParameter::kCount);
  apply_gamma_root_signature_desc.pParameters = apply_gamma_root_parameters;
  apply_gamma_root_signature_desc.NumStaticSamplers = 0;
  apply_gamma_root_signature_desc.pStaticSamplers = nullptr;
  apply_gamma_root_signature_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
  *(apply_gamma_root_signature_.ReleaseAndGetAddressOf()) =
      ui::d3d12::util::CreateRootSignature(provider, apply_gamma_root_signature_desc);
  if (!apply_gamma_root_signature_) {
    REXGPU_ERROR("Failed to create the gamma ramp application root signature");
    return false;
  }
  *(apply_gamma_table_pipeline_.ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
      device, shaders::apply_gamma_table_cs, sizeof(shaders::apply_gamma_table_cs),
      apply_gamma_root_signature_.Get());
  if (!apply_gamma_table_pipeline_) {
    REXGPU_ERROR(
        "Failed to create the 256-entry table gamma ramp application compute "
        "pipeline");
    return false;
  }
  *(apply_gamma_table_fxaa_luma_pipeline_.ReleaseAndGetAddressOf()) =
      ui::d3d12::util::CreateComputePipeline(device, shaders::apply_gamma_table_fxaa_luma_cs,
                                             sizeof(shaders::apply_gamma_table_fxaa_luma_cs),
                                             apply_gamma_root_signature_.Get());
  if (!apply_gamma_table_fxaa_luma_pipeline_) {
    REXGPU_ERROR(
        "Failed to create the 256-entry table gamma ramp application compute "
        "pipeline with perceptual luma output");
    return false;
  }
  *(apply_gamma_pwl_pipeline_.ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
      device, shaders::apply_gamma_pwl_cs, sizeof(shaders::apply_gamma_pwl_cs),
      apply_gamma_root_signature_.Get());
  if (!apply_gamma_pwl_pipeline_) {
    REXGPU_ERROR("Failed to create the PWL gamma ramp application compute pipeline");
    return false;
  }
  *(apply_gamma_pwl_fxaa_luma_pipeline_.ReleaseAndGetAddressOf()) =
      ui::d3d12::util::CreateComputePipeline(device, shaders::apply_gamma_pwl_fxaa_luma_cs,
                                             sizeof(shaders::apply_gamma_pwl_fxaa_luma_cs),
                                             apply_gamma_root_signature_.Get());
  if (!apply_gamma_pwl_fxaa_luma_pipeline_) {
    REXGPU_ERROR(
        "Failed to create the PWL gamma ramp application compute pipeline with "
        "perceptual luma output");
    return false;
  }

  // Initialize compute pipelines for post-processing anti-aliasing.
  D3D12_ROOT_PARAMETER fxaa_root_parameters[UINT(FxaaRootParameter::kCount)];
  {
    D3D12_ROOT_PARAMETER& fxaa_root_parameter_constants =
        fxaa_root_parameters[UINT(ApplyGammaRootParameter::kConstants)];
    fxaa_root_parameter_constants.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    fxaa_root_parameter_constants.Constants.ShaderRegister = 0;
    fxaa_root_parameter_constants.Constants.RegisterSpace = 0;
    fxaa_root_parameter_constants.Constants.Num32BitValues =
        sizeof(FxaaConstants) / sizeof(uint32_t);
    fxaa_root_parameter_constants.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_DESCRIPTOR_RANGE fxaa_root_descriptor_range_dest;
  fxaa_root_descriptor_range_dest.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  fxaa_root_descriptor_range_dest.NumDescriptors = 1;
  fxaa_root_descriptor_range_dest.BaseShaderRegister = 0;
  fxaa_root_descriptor_range_dest.RegisterSpace = 0;
  fxaa_root_descriptor_range_dest.OffsetInDescriptorsFromTableStart = 0;
  {
    D3D12_ROOT_PARAMETER& fxaa_root_parameter_dest =
        fxaa_root_parameters[UINT(FxaaRootParameter::kDestination)];
    fxaa_root_parameter_dest.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    fxaa_root_parameter_dest.DescriptorTable.NumDescriptorRanges = 1;
    fxaa_root_parameter_dest.DescriptorTable.pDescriptorRanges = &fxaa_root_descriptor_range_dest;
    fxaa_root_parameter_dest.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_DESCRIPTOR_RANGE fxaa_root_descriptor_range_source;
  fxaa_root_descriptor_range_source.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  fxaa_root_descriptor_range_source.NumDescriptors = 1;
  fxaa_root_descriptor_range_source.BaseShaderRegister = 0;
  fxaa_root_descriptor_range_source.RegisterSpace = 0;
  fxaa_root_descriptor_range_source.OffsetInDescriptorsFromTableStart = 0;
  {
    D3D12_ROOT_PARAMETER& fxaa_root_parameter_source =
        fxaa_root_parameters[UINT(FxaaRootParameter::kSource)];
    fxaa_root_parameter_source.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    fxaa_root_parameter_source.DescriptorTable.NumDescriptorRanges = 1;
    fxaa_root_parameter_source.DescriptorTable.pDescriptorRanges =
        &fxaa_root_descriptor_range_source;
    fxaa_root_parameter_source.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_STATIC_SAMPLER_DESC fxaa_root_sampler;
  fxaa_root_sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
  fxaa_root_sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
  fxaa_root_sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
  fxaa_root_sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
  fxaa_root_sampler.MipLODBias = 0.0f;
  fxaa_root_sampler.MaxAnisotropy = 1;
  fxaa_root_sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
  fxaa_root_sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
  fxaa_root_sampler.MinLOD = 0.0f;
  fxaa_root_sampler.MaxLOD = 0.0f;
  fxaa_root_sampler.ShaderRegister = 0;
  fxaa_root_sampler.RegisterSpace = 0;
  fxaa_root_sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  D3D12_ROOT_SIGNATURE_DESC fxaa_root_signature_desc;
  fxaa_root_signature_desc.NumParameters = UINT(FxaaRootParameter::kCount);
  fxaa_root_signature_desc.pParameters = fxaa_root_parameters;
  fxaa_root_signature_desc.NumStaticSamplers = 1;
  fxaa_root_signature_desc.pStaticSamplers = &fxaa_root_sampler;
  fxaa_root_signature_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
  *(fxaa_root_signature_.ReleaseAndGetAddressOf()) =
      ui::d3d12::util::CreateRootSignature(provider, fxaa_root_signature_desc);
  if (!fxaa_root_signature_) {
    REXGPU_ERROR("Failed to create the FXAA root signature");
    return false;
  }
  *(fxaa_pipeline_.ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
      device, shaders::fxaa_cs, sizeof(shaders::fxaa_cs), fxaa_root_signature_.Get());
  if (!fxaa_pipeline_) {
    REXGPU_ERROR("Failed to create the FXAA compute pipeline");
    return false;
  }
  *(fxaa_extreme_pipeline_.ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
      device, shaders::fxaa_extreme_cs, sizeof(shaders::fxaa_extreme_cs),
      fxaa_root_signature_.Get());
  if (!fxaa_pipeline_) {
    REXGPU_ERROR("Failed to create the extreme-quality FXAA compute pipeline");
    return false;
  }

  if (!SetupSmaa()) {
    return false;
  }

  // Resolve downscale compute pipeline for scaled readback resolve.
  D3D12_ROOT_PARAMETER
  resolve_downscale_root_parameters[UINT(ResolveDownscaleRootParameter::kCount)];
  {
    D3D12_ROOT_PARAMETER& constants_parameter =
        resolve_downscale_root_parameters[UINT(ResolveDownscaleRootParameter::kConstants)];
    constants_parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    constants_parameter.Constants.ShaderRegister = 0;
    constants_parameter.Constants.RegisterSpace = 0;
    constants_parameter.Constants.Num32BitValues =
        sizeof(ResolveDownscaleConstants) / sizeof(uint32_t);
    constants_parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_DESCRIPTOR_RANGE resolve_downscale_source_range;
  resolve_downscale_source_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  resolve_downscale_source_range.NumDescriptors = 1;
  resolve_downscale_source_range.BaseShaderRegister = 0;
  resolve_downscale_source_range.RegisterSpace = 0;
  resolve_downscale_source_range.OffsetInDescriptorsFromTableStart = 0;
  {
    D3D12_ROOT_PARAMETER& source_parameter =
        resolve_downscale_root_parameters[UINT(ResolveDownscaleRootParameter::kSource)];
    source_parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    source_parameter.DescriptorTable.NumDescriptorRanges = 1;
    source_parameter.DescriptorTable.pDescriptorRanges = &resolve_downscale_source_range;
    source_parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_DESCRIPTOR_RANGE resolve_downscale_destination_range;
  resolve_downscale_destination_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  resolve_downscale_destination_range.NumDescriptors = 1;
  resolve_downscale_destination_range.BaseShaderRegister = 0;
  resolve_downscale_destination_range.RegisterSpace = 0;
  resolve_downscale_destination_range.OffsetInDescriptorsFromTableStart = 0;
  {
    D3D12_ROOT_PARAMETER& destination_parameter =
        resolve_downscale_root_parameters[UINT(ResolveDownscaleRootParameter::kDestination)];
    destination_parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    destination_parameter.DescriptorTable.NumDescriptorRanges = 1;
    destination_parameter.DescriptorTable.pDescriptorRanges = &resolve_downscale_destination_range;
    destination_parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_ROOT_SIGNATURE_DESC resolve_downscale_root_signature_desc;
  resolve_downscale_root_signature_desc.NumParameters = UINT(ResolveDownscaleRootParameter::kCount);
  resolve_downscale_root_signature_desc.pParameters = resolve_downscale_root_parameters;
  resolve_downscale_root_signature_desc.NumStaticSamplers = 0;
  resolve_downscale_root_signature_desc.pStaticSamplers = nullptr;
  resolve_downscale_root_signature_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
  *(resolve_downscale_root_signature_.ReleaseAndGetAddressOf()) =
      ui::d3d12::util::CreateRootSignature(provider, resolve_downscale_root_signature_desc);
  if (resolve_downscale_root_signature_) {
    *(resolve_downscale_pipeline_.ReleaseAndGetAddressOf()) =
        ui::d3d12::util::CreateComputePipeline(device, shaders::resolve_downscale_cs,
                                               sizeof(shaders::resolve_downscale_cs),
                                               resolve_downscale_root_signature_.Get());
  }
  if (!resolve_downscale_root_signature_ || !resolve_downscale_pipeline_) {
    resolve_downscale_pipeline_.Reset();
    resolve_downscale_root_signature_.Reset();
    REXGPU_WARN("Failed to initialize D3D12 resolve-downscale readback pipeline");
  }

  dlss_ = std::make_unique<D3D12Dlss>(*this);
  if (!dlss_->Initialize()) {
    dlss_.reset();
  }

  if (bindless_resources_used_) {
    // Create the system bindless descriptors once all resources are
    // initialized.
    // kNullRawSRV.
    ui::d3d12::util::CreateBufferRawSRV(
        device,
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kNullRawSRV)),
        nullptr, 0);
    // kNullRawUAV.
    ui::d3d12::util::CreateBufferRawUAV(
        device,
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kNullRawUAV)),
        nullptr, 0);
    // kNullTexture2DArray.
    D3D12_SHADER_RESOURCE_VIEW_DESC null_srv_desc;
    null_srv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    null_srv_desc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(
        D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0,
        D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0);
    null_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    null_srv_desc.Texture2DArray.MostDetailedMip = 0;
    null_srv_desc.Texture2DArray.MipLevels = 1;
    null_srv_desc.Texture2DArray.FirstArraySlice = 0;
    null_srv_desc.Texture2DArray.ArraySize = 1;
    null_srv_desc.Texture2DArray.PlaneSlice = 0;
    null_srv_desc.Texture2DArray.ResourceMinLODClamp = 0.0f;
    device->CreateShaderResourceView(
        nullptr, &null_srv_desc,
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kNullTexture2DArray)));
    // kNullTexture3D.
    null_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
    null_srv_desc.Texture3D.MostDetailedMip = 0;
    null_srv_desc.Texture3D.MipLevels = 1;
    null_srv_desc.Texture3D.ResourceMinLODClamp = 0.0f;
    device->CreateShaderResourceView(
        nullptr, &null_srv_desc,
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kNullTexture3D)));
    // kNullTextureCube.
    null_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
    null_srv_desc.TextureCube.MostDetailedMip = 0;
    null_srv_desc.TextureCube.MipLevels = 1;
    null_srv_desc.TextureCube.ResourceMinLODClamp = 0.0f;
    device->CreateShaderResourceView(
        nullptr, &null_srv_desc,
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kNullTextureCube)));
    // kSharedMemoryRawSRV.
    shared_memory_->WriteRawSRVDescriptor(provider.OffsetViewDescriptor(
        view_bindless_heap_cpu_start_, uint32_t(SystemBindlessView::kSharedMemoryRawSRV)));
    // kSharedMemoryR32UintSRV.
    shared_memory_->WriteUintPow2SRVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kSharedMemoryR32UintSRV)),
        2);
    // kSharedMemoryR32G32UintSRV.
    shared_memory_->WriteUintPow2SRVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kSharedMemoryR32G32UintSRV)),
        3);
    // kSharedMemoryR32G32B32A32UintSRV.
    shared_memory_->WriteUintPow2SRVDescriptor(
        provider.OffsetViewDescriptor(
            view_bindless_heap_cpu_start_,
            uint32_t(SystemBindlessView::kSharedMemoryR32G32B32A32UintSRV)),
        4);
    // kSharedMemoryRawUAV.
    shared_memory_->WriteRawUAVDescriptor(provider.OffsetViewDescriptor(
        view_bindless_heap_cpu_start_, uint32_t(SystemBindlessView::kSharedMemoryRawUAV)));
    // kSharedMemoryR32UintUAV.
    shared_memory_->WriteUintPow2UAVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kSharedMemoryR32UintUAV)),
        2);
    // kSharedMemoryR32G32UintUAV.
    shared_memory_->WriteUintPow2UAVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kSharedMemoryR32G32UintUAV)),
        3);
    // kSharedMemoryR32G32B32A32UintUAV.
    shared_memory_->WriteUintPow2UAVDescriptor(
        provider.OffsetViewDescriptor(
            view_bindless_heap_cpu_start_,
            uint32_t(SystemBindlessView::kSharedMemoryR32G32B32A32UintUAV)),
        4);
    // kEdramRawSRV.
    render_target_cache_->WriteEdramRawSRVDescriptor(provider.OffsetViewDescriptor(
        view_bindless_heap_cpu_start_, uint32_t(SystemBindlessView::kEdramRawSRV)));
    // kEdramR32UintSRV.
    render_target_cache_->WriteEdramUintPow2SRVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kEdramR32UintSRV)),
        2);
    // kEdramR32G32UintSRV.
    render_target_cache_->WriteEdramUintPow2SRVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kEdramR32G32UintSRV)),
        3);
    // kEdramR32G32B32A32UintSRV.
    render_target_cache_->WriteEdramUintPow2SRVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kEdramR32G32B32A32UintSRV)),
        4);
    // kEdramRawUAV.
    render_target_cache_->WriteEdramRawUAVDescriptor(provider.OffsetViewDescriptor(
        view_bindless_heap_cpu_start_, uint32_t(SystemBindlessView::kEdramRawUAV)));
    // kEdramR32UintUAV.
    render_target_cache_->WriteEdramUintPow2UAVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kEdramR32UintUAV)),
        2);
    // kEdramR32G32UintUAV.
    render_target_cache_->WriteEdramUintPow2UAVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kEdramR32G32UintUAV)),
        3);
    // kEdramR32G32B32A32UintUAV.
    render_target_cache_->WriteEdramUintPow2UAVDescriptor(
        provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                      uint32_t(SystemBindlessView::kEdramR32G32B32A32UintUAV)),
        4);
    // kGammaRampTableSRV.
    WriteGammaRampSRV(
        false, provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                             uint32_t(SystemBindlessView::kGammaRampTableSRV)));
    // kGammaRampPWLSRV.
    WriteGammaRampSRV(
        true, provider.OffsetViewDescriptor(view_bindless_heap_cpu_start_,
                                            uint32_t(SystemBindlessView::kGammaRampPWLSRV)));
  }

  occlusion_query_resources_available_ = InitializeOcclusionQueryResources();

  // Just not to expose uninitialized memory.
  std::memset(&system_constants_, 0, sizeof(system_constants_));

  return true;
}

void D3D12CommandProcessor::ShutdownContext() {
  if (presenter_tail_on_submission_thread_) {
    // Paint inline again (after the pending tail) before the thread goes away.
    ui::Presenter* presenter = graphics_system_ ? graphics_system_->presenter() : nullptr;
    if (presenter) {
      presenter->SetGuestOutputRefreshTailRunner(nullptr);
    }
    presenter_tail_on_submission_thread_ = false;
  }
  AwaitAllQueueOperationsCompletion();
  ShutdownSubmissionThread();
  if (dlss_) {
    dlss_->Shutdown();
    dlss_.reset();
  }
  InvalidateAllVertexBufferResidency();
  ShutdownOcclusionQueryResources();

  ui::d3d12::util::ReleaseAndNull(readback_buffer_);
  readback_buffer_size_ = 0;
  for (auto& resolve_readback_pair : readback_buffers_) {
    auto& readback = resolve_readback_pair.second;
    for (uint32_t i = 0; i < 2; ++i) {
      if (readback.buffers[i]) {
        if (readback.mapped_data[i]) {
          readback.buffers[i]->Unmap(0, nullptr);
          readback.mapped_data[i] = nullptr;
        }
        readback.buffers[i]->Release();
        readback.buffers[i] = nullptr;
      }
      readback.sizes[i] = 0;
      readback.submission_written[i] = 0;
      readback.written_size[i] = 0;
    }
  }
  readback_buffers_.clear();
  for (auto& memexport_readback_pair : memexport_readback_buffers_) {
    auto& readback = memexport_readback_pair.second;
    for (uint32_t i = 0; i < 2; ++i) {
      if (readback.buffers[i]) {
        if (readback.mapped_data[i]) {
          readback.buffers[i]->Unmap(0, nullptr);
          readback.mapped_data[i] = nullptr;
        }
        readback.buffers[i]->Release();
        readback.buffers[i] = nullptr;
      }
      readback.sizes[i] = 0;
      readback.submission_written[i] = 0;
      readback.written_size[i] = 0;
    }
  }
  memexport_readback_buffers_.clear();

  ui::d3d12::util::ReleaseAndNull(scratch_buffer_);
  scratch_buffer_size_ = 0;
  resolve_downscale_buffer_size_ = 0;
  resolve_downscale_buffer_.Reset();

  for (const std::pair<uint64_t, ID3D12Resource*>& resource_for_deletion :
       resources_for_deletion_) {
    resource_for_deletion.second->Release();
  }
  resources_for_deletion_.clear();

  fxaa_source_texture_submission_ = 0;
  fxaa_source_texture_.Reset();

  fxaa_extreme_pipeline_.Reset();
  fxaa_pipeline_.Reset();
  for (auto& pipeline : smaa_pipelines_) {
    pipeline.Reset();
  }
  smaa_root_signature_.Reset();
  smaa_area_texture_.Reset();
  smaa_search_texture_.Reset();
  smaa_lut_upload_buffer_.Reset();
  smaa_luts_uploaded_ = false;
  for (auto& texture : smaa_textures_) {
    texture.Reset();
  }
  smaa_textures_submission_ = 0;
  fxaa_root_signature_.Reset();
  resolve_downscale_pipeline_.Reset();
  resolve_downscale_root_signature_.Reset();

  apply_gamma_pwl_fxaa_luma_pipeline_.Reset();
  apply_gamma_pwl_pipeline_.Reset();
  apply_gamma_table_fxaa_luma_pipeline_.Reset();
  apply_gamma_table_pipeline_.Reset();
  apply_gamma_root_signature_.Reset();

  // Unmapping will be done implicitly by the destruction.
  gamma_ramp_upload_buffer_mapping_ = nullptr;
  gamma_ramp_upload_buffer_.Reset();

  screen_extents_readback_mapping_ = nullptr;
  screen_extents_readback_buffer_.Reset();
  screen_extents_zero_buffer_.Reset();
  screen_extents_buffer_.Reset();
  gamma_ramp_buffer_.Reset();

  texture_cache_.reset();

  pipeline_cache_.reset();

  primitive_processor_.reset();

  // Shut down binding - bindless descriptors may be owned by subsystems like
  // the texture cache.

  // Root signatures are used by pipelines, thus freed after the pipelines.
  ui::d3d12::util::ReleaseAndNull(root_signature_bindless_ds_);
  ui::d3d12::util::ReleaseAndNull(root_signature_bindless_vs_);
  for (auto it : root_signatures_bindful_) {
    it.second->Release();
  }
  root_signatures_bindful_.clear();

  if (bindless_resources_used_) {
    texture_cache_bindless_sampler_map_.clear();
    for (const auto& sampler_bindless_heap_overflowed : sampler_bindless_heaps_overflowed_) {
      sampler_bindless_heap_overflowed.first->Release();
    }
    sampler_bindless_heaps_overflowed_.clear();
    sampler_bindless_heap_allocated_ = 0;
    ui::d3d12::util::ReleaseAndNull(sampler_bindless_heap_current_);
    view_bindless_one_use_descriptors_.clear();
    view_bindless_heap_free_.clear();
    ui::d3d12::util::ReleaseAndNull(view_bindless_heap_);
  } else {
    sampler_bindful_heap_pool_.reset();
    view_bindful_heap_pool_.reset();
  }
  constant_buffer_pool_.reset();

  render_target_cache_.reset();

  shared_memory_.reset();

  deferred_command_list_.Reset();
  ui::d3d12::util::ReleaseAndNull(command_list_1_);
  ui::d3d12::util::ReleaseAndNull(command_list_);
  ClearCommandAllocatorCache();

  frame_open_ = false;
  frame_current_ = 1;
  frame_completed_ = 0;
  std::memset(closed_frame_submissions_, 0, sizeof(closed_frame_submissions_));

  // First release the fences since they may reference fence_completion_event_.

  queue_operations_done_since_submission_signal_ = false;
  queue_operations_since_submission_fence_last_ = 0;
  ui::d3d12::util::ReleaseAndNull(queue_operations_since_submission_fence_);

  ui::d3d12::util::ReleaseAndNull(submission_fence_);
  submission_open_ = false;
  submission_current_ = 1;
  submission_completed_ = 0;

  if (fence_completion_event_) {
    CloseHandle(fence_completion_event_);
    fence_completion_event_ = nullptr;
  }

  device_removed_ = false;

  CommandProcessor::ShutdownContext();
}

void D3D12CommandProcessor::WriteRegister(uint32_t index, uint32_t value) {
  CommandProcessor::WriteRegister(index, value);

  if (index >= XE_GPU_REG_SHADER_CONSTANT_000_X && index <= XE_GPU_REG_SHADER_CONSTANT_511_W) {
    if (frame_open_) {
      uint32_t float_constant_index = (index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
      if (float_constant_index >= 256) {
        float_constant_index -= 256;
        if (current_float_constant_map_pixel_[float_constant_index >> 6] &
            (1ull << (float_constant_index & 63))) {
          cbuffer_binding_float_pixel_.up_to_date = false;
        }
      } else {
        if (current_float_constant_map_vertex_[float_constant_index >> 6] &
            (1ull << (float_constant_index & 63))) {
          cbuffer_binding_float_vertex_.up_to_date = false;
        }
      }
    }
  } else if (index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
             index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    cbuffer_binding_bool_loop_.up_to_date = false;
  } else if (index >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
             index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) {
    cbuffer_binding_fetch_.up_to_date = false;
    if (texture_cache_ != nullptr) {
      texture_cache_->TextureFetchConstantWritten((index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) /
                                                  6);
    }
    InvalidateVertexBufferResidency((index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) / 2);
  }
}

void D3D12CommandProcessor::WriteRegistersFromMem(uint32_t start_index, uint32_t* base,
                                                  uint32_t num_registers) {
  if (!num_registers) {
    return;
  }
  uint32_t end_index = start_index + num_registers - 1;

  auto range_has_any_constant_usage = [](const uint64_t* usage_map, uint32_t first_constant,
                                         uint32_t last_constant) -> bool {
    if (first_constant > last_constant) {
      return false;
    }
    uint32_t first_word = first_constant >> 6;
    uint32_t last_word = last_constant >> 6;
    uint32_t first_bit = first_constant & 63;
    uint32_t last_bit = last_constant & 63;
    if (first_word == last_word) {
      uint32_t bit_count = last_bit - first_bit + 1;
      uint64_t mask = bit_count == 64 ? UINT64_MAX : ((UINT64_C(1) << bit_count) - 1) << first_bit;
      return (usage_map[first_word] & mask) != 0;
    }
    if (usage_map[first_word] & (UINT64_MAX << first_bit)) {
      return true;
    }
    for (uint32_t word = first_word + 1; word < last_word; ++word) {
      if (usage_map[word]) {
        return true;
      }
    }
    uint64_t last_mask = last_bit == 63 ? UINT64_MAX : ((UINT64_C(1) << (last_bit + 1)) - 1);
    return (usage_map[last_word] & last_mask) != 0;
  };

  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_000_X &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_511_W) {
    memory::copy_and_swap(register_file_->values + start_index, base, num_registers);
    if (frame_open_) {
      uint32_t first_float_constant = (start_index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
      uint32_t last_float_constant = (end_index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
      if (first_float_constant < 256) {
        uint32_t last_vertex_constant = std::min(last_float_constant, 255u);
        if (range_has_any_constant_usage(current_float_constant_map_vertex_, first_float_constant,
                                         last_vertex_constant)) {
          cbuffer_binding_float_vertex_.up_to_date = false;
        }
      }
      if (last_float_constant >= 256) {
        uint32_t first_pixel_constant =
            first_float_constant >= 256 ? first_float_constant - 256 : 0;
        uint32_t last_pixel_constant = last_float_constant - 256;
        if (range_has_any_constant_usage(current_float_constant_map_pixel_, first_pixel_constant,
                                         last_pixel_constant)) {
          cbuffer_binding_float_pixel_.up_to_date = false;
        }
      }
    }
    return;
  }

  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    memory::copy_and_swap(register_file_->values + start_index, base, num_registers);
    cbuffer_binding_bool_loop_.up_to_date = false;
    return;
  }

  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) {
    memory::copy_and_swap(register_file_->values + start_index, base, num_registers);
    cbuffer_binding_fetch_.up_to_date = false;
    uint32_t first_fetch_dword = start_index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0;
    uint32_t last_fetch_dword = end_index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0;
    if (texture_cache_) {
      texture_cache_->TextureFetchConstantsWritten(first_fetch_dword / 6, last_fetch_dword / 6);
    }
    InvalidateVertexBufferResidencyRange(first_fetch_dword / 2, last_fetch_dword / 2);
    return;
  }

  CommandProcessor::WriteRegistersFromMem(start_index, base, num_registers);
}

void D3D12CommandProcessor::OnGammaRamp256EntryTableValueWritten() {
  gamma_ramp_256_entry_table_up_to_date_ = false;
}

void D3D12CommandProcessor::OnGammaRampPWLValueWritten() {
  gamma_ramp_pwl_up_to_date_ = false;
}

namespace {
// amped3 pink probe (d3d12_debug_pink_probe): the draws of the
// d3d12_debug_ps_extents shader skipped in the frame being drawn, by index.
constexpr uint32_t kPinkProbeIndices = 512, kPinkProbeSlots = 16;
uint64_t g_pink_skip[kPinkProbeIndices / 64] = {};
uint32_t g_pink_draws = 0;
struct PinkProbeSlot {
  uint64_t submission = 0;
  uint64_t skip[kPinkProbeIndices / 64];
  uint32_t draws;
};
PinkProbeSlot g_pink_slots[kPinkProbeSlots];
uint32_t g_pink_skipped_frames[kPinkProbeIndices], g_pink_skipped_pink[kPinkProbeIndices];
// Frames in which the draw index existed and wasn't skipped, and pink ones.
uint32_t g_pink_drawn_frames[kPinkProbeIndices], g_pink_drawn_pink[kPinkProbeIndices];
// Pink frames by the shader's draw count.
std::map<uint32_t, std::pair<uint32_t, uint32_t>> g_pink_by_draws;
uint32_t g_pink_frames = 0, g_pink_pink = 0, g_pink_max_draws = 0;
Microsoft::WRL::ComPtr<ID3D12Resource> g_pink_buffer;
const uint32_t* g_pink_mapping = nullptr;
}  // namespace

void D3D12CommandProcessor::Amped3PinkProbe(ID3D12Resource* output, uint32_t width,
                                            uint32_t height) {
  if (!g_pink_buffer) {
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 512 * kPinkProbeSlots;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    void* mapping;
    if (FAILED(GetD3D12Provider().GetDevice()->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesReadback, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&g_pink_buffer))) ||
        FAILED(g_pink_buffer->Map(0, nullptr, &mapping))) {
      g_pink_buffer.Reset();
      return;
    }
    g_pink_mapping = static_cast<const uint32_t*>(mapping);
  }
  // Results of finished frames.
  for (uint32_t i = 0; i < kPinkProbeSlots; ++i) {
    PinkProbeSlot& slot = g_pink_slots[i];
    if (!slot.submission || slot.submission > submission_completed_) {
      continue;
    }
    slot.submission = 0;
    const uint32_t v = g_pink_mapping[i * 128];
    const uint32_t r = (v & 1023) >> 2, g = ((v >> 10) & 1023) >> 2, b = ((v >> 20) & 1023) >> 2;
    const bool pink = r > 190 && g < 160 && b > 140 && r > g + 60 && b > g + 30;
    auto& by_draws = g_pink_by_draws[slot.draws];
    ++by_draws.first;
    by_draws.second += pink;
    // Gameplay frames only (menus have few of these draws and real pink).
    if (slot.draws < 8) {
      continue;
    }
    ++g_pink_frames;
    g_pink_pink += pink;
    g_pink_max_draws = std::max(g_pink_max_draws, slot.draws);
    for (uint32_t j = 0; j < std::min(slot.draws, kPinkProbeIndices); ++j) {
      if ((slot.skip[j >> 6] >> (j & 63)) & 1) {
        ++g_pink_skipped_frames[j];
        g_pink_skipped_pink[j] += pink;
      } else {
        ++g_pink_drawn_frames[j];
        g_pink_drawn_pink[j] += pink;
      }
    }
    if (g_pink_frames % 600 == 0) {
      // Candidates: indices never pink while skipped, ordered by how often
      // they were skipped.
      std::string line;
      for (uint32_t j = 0; j < std::min(g_pink_max_draws, kPinkProbeIndices); ++j) {
        if (g_pink_skipped_frames[j] >= 8) {
          // Pink rate while skipped vs while drawn.
          line += fmt::format(" {}:{:.0f}/{:.0f}%", j,
                              100.0 * g_pink_skipped_pink[j] / g_pink_skipped_frames[j],
                              g_pink_drawn_frames[j]
                                  ? 100.0 * g_pink_drawn_pink[j] / g_pink_drawn_frames[j]
                                  : 0.0);
        }
      }
      std::string draws_line;
      for (const auto& [draws, counts] : g_pink_by_draws) {
        if (counts.first >= 20) {
          draws_line += fmt::format(" {}:{}/{}", draws, counts.second, counts.first);
        }
      }
      REXGPU_INFO("pink_probe: pink frames by draw count (draws:pink/frames):{}", draws_line);
      REXGPU_INFO("pink_probe: {} frames, {} pink ({:.0f}%), up to {} draws/frame, last pixel "
                  "{},{},{}; pink % skipped/drawn by index:{}",
                  g_pink_frames, g_pink_pink, 100.0 * g_pink_pink / g_pink_frames,
                  g_pink_max_draws, r, g, b, line);
    }
  }
  // This frame's pixel, then the next frame's random skips.
  uint32_t free_slot = UINT32_MAX;
  for (uint32_t i = 0; i < kPinkProbeSlots; ++i) {
    if (!g_pink_slots[i].submission) {
      free_slot = i;
      break;
    }
  }
  if (free_slot != UINT32_MAX) {
    PinkProbeSlot& slot = g_pink_slots[free_slot];
    std::memcpy(slot.skip, g_pink_skip, sizeof(g_pink_skip));
    slot.draws = g_pink_draws;
    slot.submission = submission_current_;
    D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
    dst.pResource = g_pink_buffer.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Offset = free_slot * 512;
    dst.PlacedFootprint.Footprint = {ui::d3d12::D3D12Presenter::kGuestOutputFormat, 1, 1, 1, 256};
    src.pResource = output;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    const uint32_t x = 527 * width / 1280, y = 602 * height / 720;
    D3D12_BOX box = {x, y, 0, x + 1, y + 1, 1};
    PushTransitionBarrier(output, ui::d3d12::D3D12Presenter::kGuestOutputInternalState,
                          D3D12_RESOURCE_STATE_COPY_SOURCE);
    SubmitBarriers();
    deferred_command_list_.D3DCopyTextureRegion(&dst, 0, 0, 0, &src, &box);
    PushTransitionBarrier(output, D3D12_RESOURCE_STATE_COPY_SOURCE,
                          ui::d3d12::D3D12Presenter::kGuestOutputInternalState);
  }
  static uint64_t seed = 0x9E3779B97F4A7C15ull;
  for (uint64_t& word : g_pink_skip) {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    word = free_slot != UINT32_MAX ? seed : 0;
  }
  g_pink_draws = 0;
}

void D3D12CommandProcessor::IssueSwap(uint32_t frontbuffer_ptr, uint32_t frontbuffer_width,
                                      uint32_t frontbuffer_height) {
  // The guest frontbuffer size (DLSS tells the scene's render target by it).
  rex::perf::RecordGuestFrontbuffer(frontbuffer_width, frontbuffer_height);
  rex::graphics::amped3_make_valid_timing = REXCVAR_GET(d3d12_draw_timing);
  if (REXCVAR_GET(d3d12_debug_prim_stats)) {
    static uint32_t prim_swaps = 0;
    g_frame_draws_last = g_frame_draw_index;
    g_frame_draw_index = 0;
    if (++prim_swaps % 30 == 0) {
      // "mask [first end]": primitive types (hex) and a range of draw numbers.
      uint64_t mask = 0;
      uint32_t first = 0, end = 0;
      if (std::FILE* f = std::fopen("debug_skip_prims.txt", "r")) {
        unsigned long long value = 0;
        if (std::fscanf(f, "%llx", &value) == 1) {
          mask = value;
          unsigned a = 0, b = 0;
          if (std::fscanf(f, "%u %u", &a, &b) == 2) {
            first = a;
            end = b;
          }
          std::set<uint64_t> ps_skip;
          while (std::fscanf(f, "%llx", &value) == 1) {
            ps_skip.insert(value);
          }
          if (ps_skip != g_ps_skip) {
            g_ps_skip = std::move(ps_skip);
            REXGPU_INFO("d3d12_debug_prim_stats: skipping {} pixel shaders", g_ps_skip.size());
          }
        }
        std::fclose(f);
      }
      if (mask != g_prim_skip_mask || first != g_skip_first || end != g_skip_end) {
        g_prim_skip_mask = mask;
        g_skip_first = first;
        g_skip_end = end;
        REXGPU_INFO("d3d12_debug_prim_stats: skipping types {:X} draws {}-{} (frame had {} draws)",
                    mask, first, end, g_frame_draws_last);
      }
    }
    if (prim_swaps % 120 == 0) {
      std::string counts;
      for (uint32_t i = 0; i < 64; ++i) {
        if (g_prim_counts[i]) {
          counts += fmt::format(" {:02X}:{:.1f}", i, double(g_prim_counts[i]) / 120);
        }
      }
      REXGPU_INFO("d3d12_debug_prim_stats: draws per frame by primitive type:{}", counts);
      std::string shaders;
      for (uint64_t hash : g_ps_seen) {
        shaders += fmt::format(" {:016X}", hash);
      }
      REXGPU_INFO("d3d12_debug_prim_stats: triangle-list pixel shaders:{}", shaders);
      g_ps_seen.clear();
      std::fill(std::begin(g_prim_counts), std::end(g_prim_counts), 0);
    }
  }
  rex::graphics::g_amped3_count_packets = REXCVAR_GET(d3d12_draw_timing);
  if (REXCVAR_GET(d3d12_draw_timing)) {
    static uint32_t swaps = 0;
    static uint64_t tsc_start = 0;
    static auto clock_start = std::chrono::steady_clock::now();
    if (!tsc_start) {
      tsc_start = __rdtsc();
    } else if (++swaps == 120) {
      double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - clock_start)
                           .count();
      double tsc_per_ms = double(__rdtsc() - tsc_start) / (seconds * 1000.0);
      uint64_t total = 0;
      std::string parts;
      for (uint32_t i = 0; i < kDrawTimingCount; ++i) {
        total += g_draw_timing[i];
        parts += fmt::format(" {} {:.2f}", kDrawTimingNames[i],
                             double(g_draw_timing[i]) / tsc_per_ms / swaps);
      }
      REXGPU_INFO("d3d12_draw_timing: {:.0f} draws/frame, IssueDraw {:.2f} ms/frame:{}",
                  double(g_draw_timing_draws) / swaps, double(total) / tsc_per_ms / swaps, parts);
      REXGPU_INFO(
          "d3d12_draw_timing: uploads {:.1f}/frame, {:.0f} pages/frame, make valid {:.2f} ms, "
          "copy {:.2f} ms per frame",
          double(g_amped3_upload_calls) / swaps, double(g_amped3_upload_pages) / swaps,
          double(g_amped3_upload_valid_tsc) / tsc_per_ms / swaps,
          double(g_amped3_upload_copy_tsc) / tsc_per_ms / swaps);
      REXGPU_INFO(
          "d3d12_draw_timing: command processor busy {:.2f} ms/frame ({:.0f} ns/draw), "
          "IssueCopy {:.2f} ms",
          double(rex::graphics::g_amped3_cp_busy_tsc) / tsc_per_ms / swaps,
          g_draw_timing_draws ? double(rex::graphics::g_amped3_cp_busy_tsc) / tsc_per_ms * 1e6 /
                                    double(g_draw_timing_draws)
                              : 0.0,
          double(g_amped3_copy_tsc) / tsc_per_ms / swaps);
      rex::graphics::g_amped3_cp_busy_tsc = 0;
      if (rex::graphics::g_amped3_cp_exec_busy_tsc) {
        REXGPU_INFO("d3d12_draw_timing: gpu_cp_split executor busy {:.2f} ms/frame ({:.0f} ns/draw)",
                    double(rex::graphics::g_amped3_cp_exec_busy_tsc) / tsc_per_ms / swaps,
                    g_draw_timing_draws
                        ? double(rex::graphics::g_amped3_cp_exec_busy_tsc) / tsc_per_ms * 1e6 /
                              double(g_draw_timing_draws)
                        : 0.0);
        rex::graphics::g_amped3_cp_exec_busy_tsc = 0;
      }
      {
        std::string opcodes;
        for (uint32_t i = 0; i < 128; ++i) {
          if (rex::graphics::g_amped3_opcode_counts[i]) {
            opcodes += fmt::format(" {:02X}:{:.1f}", i,
                                   double(rex::graphics::g_amped3_opcode_counts[i]) / swaps);
          }
        }
        REXGPU_INFO("d3d12_draw_timing: per frame type-0 {:.0f} ({:.0f} registers), type-3:{}",
                    double(rex::graphics::g_amped3_type0_packets) / swaps,
                    double(rex::graphics::g_amped3_type0_registers) / swaps, opcodes);
        std::fill(std::begin(rex::graphics::g_amped3_opcode_counts),
                  std::end(rex::graphics::g_amped3_opcode_counts), 0);
        rex::graphics::g_amped3_type0_packets = rex::graphics::g_amped3_type0_registers = 0;
      }
      g_amped3_copy_tsc = 0;
      REXGPU_INFO("d3d12_draw_timing: make valid: lock + bitmaps {:.2f} ms, protect {:.2f} ms",
                  double(rex::graphics::amped3_make_valid_lock_tsc) / tsc_per_ms / swaps,
                  double(rex::graphics::amped3_make_valid_protect_tsc) / tsc_per_ms / swaps);
      rex::graphics::amped3_make_valid_lock_tsc = rex::graphics::amped3_make_valid_protect_tsc = 0;
      g_amped3_upload_calls = g_amped3_upload_pages = g_amped3_upload_valid_tsc =
          g_amped3_upload_copy_tsc = 0;
      std::fill(std::begin(g_draw_timing), std::end(g_draw_timing), 0);
      g_draw_timing_draws = 0;
      swaps = 0;
      tsc_start = __rdtsc();
      clock_start = std::chrono::steady_clock::now();
    }
  }
  SCOPE_profile_cpu_f("gpu");
  vertex_buffers_in_sync_[0] = 0;
  vertex_buffers_in_sync_[1] = 0;

  if (!graphics_system_)
    return;
  ui::Presenter* presenter = graphics_system_->presenter();
  if (!presenter) {
    REXGPU_ERROR("IssueSwap: presenter is null");
    return;
  }
  // A pre-rendered video on a screen wider or taller than 16:9 is shown
  // letterboxed, never stretched: present_letterbox turned on while one plays
  // (only when that changes), then back to the player's setting. (Not a
  // Presenter member: the game is built against the stock SDK's layout.)
  {
    static bool video_shown = false;
    static bool letterbox_setting = true;
    // A video's frames don't all draw it (decoding lags now and then): it has
    // ended after 30 frames without.
    static int frames_without_video = 0;
    // ...but 3D without a video for 3 frames in a row is the game proper (the
    // in-engine story after its video): full screen at once, not with the
    // video's bars for another half second.
    static int scene_frames = 0;
    if (picture_frame_ && !scene_frame_) video_frame_ = true;  // a loading screen
    scene_frames = scene_frame_ && !video_frame_ ? scene_frames + 1 : 0;
    picture_frame_ = scene_frame_ = false;
    frames_without_video = video_frame_ ? 0 : frames_without_video + 1;
    const bool video = video_frame_ || (video_shown && frames_without_video < 30 && scene_frames < 3);
    video_frame_ = video;
    if (video != video_shown) {
      if (video) letterbox_setting = REXCVAR_GET(present_letterbox);
      video_shown = video;
      const bool ok = rex::cvar::SetFlagByName("present_letterbox",
                                               video_shown || letterbox_setting ? "true" : "false");
      REXGPU_INFO("Video {}: present_letterbox {}{}", video_shown ? "starts" : "ends",
                  REXCVAR_GET(present_letterbox), ok ? "" : " (not set)");
    }
  }
  presenter_video_frame_ = video_frame_;
  video_frame_ = false;

  // In case the swap command is the only one in the frame.
  if (!BeginSubmission(true)) {
    REXGPU_ERROR("IssueSwap: BeginSubmission failed");
    return;
  }

  // Obtain the actual swap source texture size (resolution-scaled if it's a
  // resolve destination, or not otherwise).
  if (REXCVAR_GET(gpu_pass_profile)) {
    // GPU time per frame of each pass, every 300 frames.
    rex::perf::GpuProfileStats& stats = rex::perf::GetGpuProfileStats();
    static uint32_t frames = 0;
    // CPU cycles of this (command processor) thread, for A/B comparisons that
    // don't depend on the scene as much as the frame rate does (per draw).
    static ULONG64 cycles_start = 0;
    ULONG64 cycles_now = 0;
    QueryThreadCycleTime(GetCurrentThread(), &cycles_now);
    gpu_profile_per_pass_ = REXCVAR_GET(gpu_pass_profile_passes);
    if (!stats.enabled.exchange(true)) {
      frames = 0;
      cycles_start = cycles_now;
    } else if (++frames == 300) {
      const uint64_t draws = std::exchange(issued_draw_count_, 0);
      const uint64_t submissions_timed = stats.submissions.exchange(0);
      REXGPU_INFO("gpu_pass_profile: command processor {:.2f} Mcycles/frame, {:.0f} cycles/draw; "
                  "GPU busy {:.2f} ms/frame ({} submissions timed), idle between submissions "
                  "{:.2f} ms/frame, CPU waits for the GPU {:.2f} ms/frame ({})",
                  double(cycles_now - cycles_start) / 1e6 / frames,
                  draws ? double(cycles_now - cycles_start) / double(draws) : 0.0,
                  double(stats.submission_ns.exchange(0)) / 1e6 / frames, submissions_timed,
                  double(stats.gap_ns.exchange(0)) / 1e6 / frames,
                  double(stats.fence_wait_ns.exchange(0)) / 1e6 / frames,
                  stats.fence_waits.exchange(0));
      cycles_start = cycles_now;
      std::string line;
      double total = 0.0;
      for (uint32_t i = 0; i < uint32_t(rex::perf::GpuPass::kCount); ++i) {
        const uint64_t ns = stats.pass_ns[i].exchange(0);
        const uint64_t count = stats.pass_count[i].exchange(0);
        total += double(ns);
        if (ns) {
          line += fmt::format(" {}={:.2f}ms/{}", rex::perf::GetGpuPassName(rex::perf::GpuPass(i)),
                              double(ns) / 1e6 / frames, count / frames);
        }
      }
      REXGPU_INFO("gpu_pass_profile: {:.2f} ms/frame:{} (clears in place {}/frame, direct "
                  "resolves pending {}, submissions {:.1f}/frame, texture binding resets "
                  "{:.1f}/frame, bindings re-evaluated {:.1f}/frame, draws {:.0f}/frame, "
                  "float constant uploads {:.0f}/frame ({:.0f} KB), reused {:.0f}/frame)",
                  total / 1e6 / frames, line, stats.clears_in_place.exchange(0) / frames,
                  texture_cache_->GetPendingDirectResolveCount(),
                  double(stats.cp_submissions.exchange(0)) / frames,
                  double(std::exchange(texture_cache_->outdated_binding_resets_, 0)) / frames,
                  double(std::exchange(texture_cache_->requested_bindings_, 0)) / frames,
                  double(draws) / frames,
                  double(std::exchange(float_constant_uploads_, 0)) / frames,
                  double(std::exchange(float_constant_upload_bytes_, 0)) / 1024.0 / frames,
                  double(std::exchange(float_constant_reuses_, 0)) / frames);
      frames = 0;
    }
  }
  D3D12_SHADER_RESOURCE_VIEW_DESC swap_texture_srv_desc;
  xenos::TextureFormat frontbuffer_format;
  uint32_t frontbuffer_width_unscaled = 0, frontbuffer_height_unscaled = 0;
  SetGpuProfilePass(rex::perf::GpuPass::kTextureLoad);
  ID3D12Resource* swap_texture_resource =
      texture_cache_->RequestSwapTexture(swap_texture_srv_desc, frontbuffer_format,
                                         &frontbuffer_width_unscaled, &frontbuffer_height_unscaled);
  SetGpuProfilePass(rex::perf::GpuPass::kSwap);
  if (!swap_texture_resource) {
    // Dump texture fetch constant 0 for debugging
    const auto& regs = *register_file_;
    auto fetch = regs.GetTextureFetch(0);
    REXGPU_ERROR(
        "IssueSwap: RequestSwapTexture failed - fetch0: {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
        fetch.dword_0, fetch.dword_1, fetch.dword_2, fetch.dword_3, fetch.dword_4, fetch.dword_5);
    return;
  }
  D3D12_RESOURCE_DESC swap_texture_desc = swap_texture_resource->GetDesc();
  // The swap gamma / FXAA pass samples source texels by pixel index, but swap
  // textures may be allocation-padded. Prefer the active frontbuffer region
  // from the swap packet, scaled proportionally to the actual source texture.
  uint32_t source_width_scaled = uint32_t(swap_texture_desc.Width);
  uint32_t source_height_scaled = uint32_t(swap_texture_desc.Height);
  auto get_active_swap_dimension = [](uint32_t packet_unscaled, uint32_t source_unscaled,
                                      uint32_t source_scaled) -> uint32_t {
    if (!source_scaled) {
      return 0;
    }
    uint32_t active_unscaled = packet_unscaled ? packet_unscaled : source_unscaled;
    if (!active_unscaled) {
      return source_scaled;
    }
    if (source_unscaled) {
      active_unscaled = std::min(active_unscaled, source_unscaled);
      uint64_t active_scaled =
          (uint64_t(active_unscaled) * source_scaled + (source_unscaled >> 1)) / source_unscaled;
      return uint32_t(std::clamp<uint64_t>(active_scaled, 1, source_scaled));
    }
    return std::min(active_unscaled, source_scaled);
  };
  uint32_t guest_output_width =
      get_active_swap_dimension(frontbuffer_width, frontbuffer_width_unscaled, source_width_scaled);
  uint32_t guest_output_height = get_active_swap_dimension(
      frontbuffer_height, frontbuffer_height_unscaled, source_height_scaled);
  if (!guest_output_width) {
    guest_output_width = source_width_scaled
                             ? source_width_scaled
                             : (frontbuffer_width ? frontbuffer_width : frontbuffer_width_unscaled);
  }
  if (!guest_output_height) {
    guest_output_height = source_height_scaled ? source_height_scaled
                                               : (frontbuffer_height ? frontbuffer_height
                                                                     : frontbuffer_height_unscaled);
  }
  bool swap_source_scaled = frontbuffer_width_unscaled && frontbuffer_height_unscaled &&
                            (source_width_scaled != frontbuffer_width_unscaled ||
                             source_height_scaled != frontbuffer_height_unscaled);
  if (texture_cache_->IsDrawResolutionScaled() && !swap_source_scaled) {
    static bool draw_scale_swap_unscaled_logged = false;
    if (!draw_scale_swap_unscaled_logged) {
      draw_scale_swap_unscaled_logged = true;
      REXGPU_WARN(
          "D3D12 draw resolution scaling is enabled, but the swap source is "
          "unscaled ({}x{}). This title may be presenting from an unscaled "
          "resolve path.",
          guest_output_width, guest_output_height);
    }
  }

  system::X_VIDEO_MODE video_mode;
  kernel::xboxkrnl::VdQueryVideoMode(&video_mode);
  uint32_t display_width = std::max(uint32_t(1), uint32_t(video_mode.display_width));
  uint32_t display_height = std::max(uint32_t(1), uint32_t(video_mode.display_height));
  if (presenter_video_frame_) {
    // A pre-rendered video: 16:9 whatever the window (letterboxed, see above).
    display_width = 16;
    display_height = 9;
  }

  // Frame generation's presents wait for vsync: never on the thread that feeds
  // the GPU.
  static const bool frame_generation = rex::cvar::GetFlagByName("frame_generation") == "true";
  const bool tail_on_submission_thread =
      REXCVAR_GET(d3d12_present_on_submission_thread) && UseSubmissionThread() && !frame_generation;
  if (tail_on_submission_thread != presenter_tail_on_submission_thread_) {
    if (tail_on_submission_thread) {
      presenter->SetGuestOutputRefreshTailRunner(
          [this](std::function<void()> tail) { EnqueueSubmissionCallback(std::move(tail)); });
    } else {
      presenter->SetGuestOutputRefreshTailRunner(nullptr);
    }
    presenter_tail_on_submission_thread_ = tail_on_submission_thread;
  }

  presenter->RefreshGuestOutput(
      guest_output_width, guest_output_height, display_width, display_height,
      [this, &swap_texture_srv_desc, frontbuffer_format, swap_texture_resource, guest_output_width,
       guest_output_height](ui::Presenter::GuestOutputRefreshContext& context) -> bool {
        const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
        ID3D12Device* device = provider.GetDevice();

        SwapPostEffect swap_post_effect = GetActualSwapPostEffect();
        bool use_smaa = swap_post_effect == SwapPostEffect::kSmaa && smaa_root_signature_;
        // (SMAA reads the same gamma-applied source texture as FXAA.)
        bool use_fxaa = swap_post_effect == SwapPostEffect::kFxaa ||
                        swap_post_effect == SwapPostEffect::kFxaaExtreme || use_smaa;
        if (use_fxaa) {
          // Make sure the texture of the correct size is available for FXAA.
          if (fxaa_source_texture_) {
            D3D12_RESOURCE_DESC fxaa_source_texture_desc = fxaa_source_texture_->GetDesc();
            if (fxaa_source_texture_desc.Width != guest_output_width ||
                fxaa_source_texture_desc.Height != guest_output_height) {
              if (submission_completed_ < fxaa_source_texture_submission_) {
                fxaa_source_texture_->AddRef();
                resources_for_deletion_.emplace_back(fxaa_source_texture_submission_,
                                                     fxaa_source_texture_.Get());
              }
              fxaa_source_texture_.Reset();
              fxaa_source_texture_submission_ = 0;
            }
          }
          if (!fxaa_source_texture_) {
            D3D12_RESOURCE_DESC fxaa_source_texture_desc;
            fxaa_source_texture_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            fxaa_source_texture_desc.Alignment = 0;
            fxaa_source_texture_desc.Width = guest_output_width;
            fxaa_source_texture_desc.Height = guest_output_height;
            fxaa_source_texture_desc.DepthOrArraySize = 1;
            fxaa_source_texture_desc.MipLevels = 1;
            fxaa_source_texture_desc.Format = kFxaaSourceTextureFormat;
            fxaa_source_texture_desc.SampleDesc.Count = 1;
            fxaa_source_texture_desc.SampleDesc.Quality = 0;
            fxaa_source_texture_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            fxaa_source_texture_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            if (FAILED(device->CreateCommittedResource(
                    &ui::d3d12::util::kHeapPropertiesDefault, provider.GetHeapFlagCreateNotZeroed(),
                    &fxaa_source_texture_desc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    nullptr, IID_PPV_ARGS(&fxaa_source_texture_)))) {
              REXGPU_ERROR("Failed to create the FXAA input texture");
              swap_post_effect = SwapPostEffect::kNone;
              use_fxaa = false;
            }
          }
        }

        // This is according to D3D::InitializePresentationParameters from a
        // game executable, which initializes the 256-entry table gamma ramp for
        // 8_8_8_8 output and the PWL gamma ramp for 2_10_10_10.
        // TODO(Triang3l): Choose between the table and PWL based on
        // DC_LUTA_CONTROL, support both for all formats (and also different
        // increments for PWL).
        bool use_pwl_gamma_ramp =
            frontbuffer_format == xenos::TextureFormat::k_2_10_10_10 ||
            frontbuffer_format == xenos::TextureFormat::k_2_10_10_10_AS_16_16_16_16;

        context.SetIs8bpc(!use_pwl_gamma_ramp && !use_fxaa);

        // Upload the new gamma ramp, using the upload buffer for the current
        // frame (will close the frame after this anyway, so can't write
        // multiple times per frame).
        if (!(use_pwl_gamma_ramp ? gamma_ramp_pwl_up_to_date_
                                 : gamma_ramp_256_entry_table_up_to_date_)) {
          uint32_t gamma_ramp_offset_bytes = use_pwl_gamma_ramp ? 256 * 4 : 0;
          uint32_t gamma_ramp_upload_offset_bytes =
              uint32_t(frame_current_ % kQueueFrames) * ((256 + 128 * 3) * 4) +
              gamma_ramp_offset_bytes;
          uint32_t gamma_ramp_size_bytes = (use_pwl_gamma_ramp ? 128 * 3 : 256) * 4;
          if (std::endian::native != std::endian::little && use_pwl_gamma_ramp) {
            // R16G16 is first R16, where the shader expects the base, and
            // second G16, where the delta should be, but gamma_ramp_pwl_rgb()
            // is an array of 32-bit DC_LUT_PWL_DATA registers - swap 16 bits in
            // each 32.
            auto gamma_ramp_pwl_upload_buffer = reinterpret_cast<reg::DC_LUT_PWL_DATA*>(
                gamma_ramp_upload_buffer_mapping_ + gamma_ramp_upload_offset_bytes);
            const reg::DC_LUT_PWL_DATA* gamma_ramp_pwl = gamma_ramp_pwl_rgb();
            for (size_t i = 0; i < 128 * 3; ++i) {
              reg::DC_LUT_PWL_DATA& gamma_ramp_pwl_upload_buffer_entry =
                  gamma_ramp_pwl_upload_buffer[i];
              reg::DC_LUT_PWL_DATA gamma_ramp_pwl_entry = gamma_ramp_pwl[i];
              gamma_ramp_pwl_upload_buffer_entry.base = gamma_ramp_pwl_entry.delta;
              gamma_ramp_pwl_upload_buffer_entry.delta = gamma_ramp_pwl_entry.base;
            }
          } else {
            std::memcpy(gamma_ramp_upload_buffer_mapping_ + gamma_ramp_upload_offset_bytes,
                        use_pwl_gamma_ramp ? static_cast<const void*>(gamma_ramp_pwl_rgb())
                                           : static_cast<const void*>(gamma_ramp_256_entry_table()),
                        gamma_ramp_size_bytes);
          }
          PushTransitionBarrier(gamma_ramp_buffer_.Get(), gamma_ramp_buffer_state_,
                                D3D12_RESOURCE_STATE_COPY_DEST);
          gamma_ramp_buffer_state_ = D3D12_RESOURCE_STATE_COPY_DEST;
          SubmitBarriers();
          deferred_command_list_.D3DCopyBufferRegion(
              gamma_ramp_buffer_.Get(), gamma_ramp_offset_bytes, gamma_ramp_upload_buffer_.Get(),
              gamma_ramp_upload_offset_bytes, gamma_ramp_size_bytes);
          (use_pwl_gamma_ramp ? gamma_ramp_pwl_up_to_date_
                              : gamma_ramp_256_entry_table_up_to_date_) = true;
        }

        // Destination, source, and if bindful, gamma ramp.
        ui::d3d12::util::DescriptorCpuGpuHandlePair apply_gamma_descriptors[3];
        ui::d3d12::util::DescriptorCpuGpuHandlePair apply_gamma_descriptor_gamma_ramp;
        if (!RequestOneUseSingleViewDescriptors(bindless_resources_used_ ? 2 : 3,
                                                apply_gamma_descriptors)) {
          return false;
        }
        // Must not call anything that can change the descriptor heap from now
        // on!
        if (bindless_resources_used_) {
          apply_gamma_descriptor_gamma_ramp = GetSystemBindlessViewHandlePair(
              use_pwl_gamma_ramp ? SystemBindlessView::kGammaRampPWLSRV
                                 : SystemBindlessView::kGammaRampTableSRV);
        } else {
          apply_gamma_descriptor_gamma_ramp = apply_gamma_descriptors[2];
          WriteGammaRampSRV(use_pwl_gamma_ramp, apply_gamma_descriptor_gamma_ramp.first);
        }

        ID3D12Resource* guest_output_resource =
            static_cast<ui::d3d12::D3D12Presenter::D3D12GuestOutputRefreshContext&>(context)
                .resource_uav_capable();

        if (use_fxaa) {
          fxaa_source_texture_submission_ = submission_current_;
        }

        ID3D12Resource* apply_gamma_dest =
            use_fxaa ? fxaa_source_texture_.Get() : guest_output_resource;
        D3D12_RESOURCE_STATES apply_gamma_dest_initial_state =
            use_fxaa ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                     : ui::d3d12::D3D12Presenter::kGuestOutputInternalState;
        static_cast<ui::d3d12::D3D12Presenter::D3D12GuestOutputRefreshContext&>(context)
            .resource_uav_capable();
        PushTransitionBarrier(apply_gamma_dest, apply_gamma_dest_initial_state,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        // From now on, even in case of failure, apply_gamma_dest must be
        // transitioned back to apply_gamma_dest_initial_state!
        D3D12_UNORDERED_ACCESS_VIEW_DESC apply_gamma_dest_uav_desc;
        apply_gamma_dest_uav_desc.Format =
            use_fxaa ? kFxaaSourceTextureFormat : ui::d3d12::D3D12Presenter::kGuestOutputFormat;
        apply_gamma_dest_uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        apply_gamma_dest_uav_desc.Texture2D.MipSlice = 0;
        apply_gamma_dest_uav_desc.Texture2D.PlaneSlice = 0;
        device->CreateUnorderedAccessView(apply_gamma_dest, nullptr, &apply_gamma_dest_uav_desc,
                                          apply_gamma_descriptors[0].first);

        device->CreateShaderResourceView(swap_texture_resource, &swap_texture_srv_desc,
                                         apply_gamma_descriptors[1].first);

        PushTransitionBarrier(gamma_ramp_buffer_.Get(), gamma_ramp_buffer_state_,
                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gamma_ramp_buffer_state_ = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        deferred_command_list_.D3DSetComputeRootSignature(apply_gamma_root_signature_.Get());
        ApplyGammaConstants apply_gamma_constants;
        apply_gamma_constants.size[0] = guest_output_width;
        apply_gamma_constants.size[1] = guest_output_height;
        deferred_command_list_.D3DSetComputeRoot32BitConstants(
            UINT(ApplyGammaRootParameter::kConstants),
            sizeof(apply_gamma_constants) / sizeof(uint32_t), &apply_gamma_constants, 0);
        deferred_command_list_.D3DSetComputeRootDescriptorTable(
            UINT(ApplyGammaRootParameter::kDestination), apply_gamma_descriptors[0].second);
        deferred_command_list_.D3DSetComputeRootDescriptorTable(
            UINT(ApplyGammaRootParameter::kSource), apply_gamma_descriptors[1].second);
        deferred_command_list_.D3DSetComputeRootDescriptorTable(
            UINT(ApplyGammaRootParameter::kRamp), apply_gamma_descriptor_gamma_ramp.second);
        ID3D12PipelineState* apply_gamma_pipeline;
        if (use_pwl_gamma_ramp) {
          apply_gamma_pipeline = use_fxaa ? apply_gamma_pwl_fxaa_luma_pipeline_.Get()
                                          : apply_gamma_pwl_pipeline_.Get();
        } else {
          apply_gamma_pipeline = use_fxaa ? apply_gamma_table_fxaa_luma_pipeline_.Get()
                                          : apply_gamma_table_pipeline_.Get();
        }
        SetExternalPipeline(apply_gamma_pipeline);
        SubmitBarriers();
        uint32_t group_count_x = (guest_output_width + 15) / 16;
        uint32_t group_count_y = (guest_output_height + 7) / 8;
        deferred_command_list_.D3DDispatch(group_count_x, group_count_y, 1);

        // amped3: SMAA instead of FXAA.
        if (use_smaa) {
          PushTransitionBarrier(apply_gamma_dest, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                apply_gamma_dest_initial_state);
          PushTransitionBarrier(guest_output_resource,
                                ui::d3d12::D3D12Presenter::kGuestOutputInternalState,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
          bool smaa_applied =
              ApplySmaa(guest_output_resource, guest_output_width, guest_output_height);
          PushTransitionBarrier(guest_output_resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                ui::d3d12::D3D12Presenter::kGuestOutputInternalState);
          if (!smaa_applied) {
            PushTransitionBarrier(apply_gamma_dest, apply_gamma_dest_initial_state,
                                  D3D12_RESOURCE_STATE_COPY_SOURCE);
            PushTransitionBarrier(guest_output_resource,
                                  ui::d3d12::D3D12Presenter::kGuestOutputInternalState,
                                  D3D12_RESOURCE_STATE_COPY_DEST);
            SubmitBarriers();
            deferred_command_list_.D3DCopyResource(guest_output_resource, apply_gamma_dest);
            PushTransitionBarrier(apply_gamma_dest, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                  apply_gamma_dest_initial_state);
            PushTransitionBarrier(guest_output_resource, D3D12_RESOURCE_STATE_COPY_DEST,
                                  ui::d3d12::D3D12Presenter::kGuestOutputInternalState);
          }
        } else
        // Apply FXAA.
        if (use_fxaa) {
          // Destination and source.
          ui::d3d12::util::DescriptorCpuGpuHandlePair fxaa_descriptors[2];
          if (!RequestOneUseSingleViewDescriptors(uint32_t(rex::countof(fxaa_descriptors)),
                                                  fxaa_descriptors)) {
            // Failed to obtain descriptors for FXAA - just copy after gamma
            // ramp application without applying FXAA.
            PushTransitionBarrier(apply_gamma_dest, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_STATE_COPY_SOURCE);
            PushTransitionBarrier(guest_output_resource,
                                  ui::d3d12::D3D12Presenter::kGuestOutputInternalState,
                                  D3D12_RESOURCE_STATE_COPY_DEST);
            SubmitBarriers();
            deferred_command_list_.D3DCopyResource(guest_output_resource, apply_gamma_dest);
            PushTransitionBarrier(apply_gamma_dest, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                  apply_gamma_dest_initial_state);
            PushTransitionBarrier(guest_output_resource, D3D12_RESOURCE_STATE_COPY_DEST,
                                  ui::d3d12::D3D12Presenter::kGuestOutputInternalState);
            return false;
          } else {
            assert_true(apply_gamma_dest_initial_state ==
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            PushTransitionBarrier(apply_gamma_dest, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  apply_gamma_dest_initial_state);
            PushTransitionBarrier(guest_output_resource,
                                  ui::d3d12::D3D12Presenter::kGuestOutputInternalState,
                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            // From now on, even in case of failure, guest_output_resource must
            // be transitioned back to kGuestOutputInternalState!
            deferred_command_list_.D3DSetComputeRootSignature(fxaa_root_signature_.Get());
            FxaaConstants fxaa_constants;
            fxaa_constants.size[0] = guest_output_width;
            fxaa_constants.size[1] = guest_output_height;
            fxaa_constants.size_inv[0] = 1.0f / float(fxaa_constants.size[0]);
            fxaa_constants.size_inv[1] = 1.0f / float(fxaa_constants.size[1]);
            deferred_command_list_.D3DSetComputeRoot32BitConstants(
                UINT(FxaaRootParameter::kConstants), sizeof(fxaa_constants) / sizeof(uint32_t),
                &fxaa_constants, 0);
            D3D12_UNORDERED_ACCESS_VIEW_DESC fxaa_dest_uav_desc;
            fxaa_dest_uav_desc.Format = ui::d3d12::D3D12Presenter::kGuestOutputFormat;
            fxaa_dest_uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            fxaa_dest_uav_desc.Texture2D.MipSlice = 0;
            fxaa_dest_uav_desc.Texture2D.PlaneSlice = 0;
            device->CreateUnorderedAccessView(guest_output_resource, nullptr, &fxaa_dest_uav_desc,
                                              fxaa_descriptors[0].first);
            deferred_command_list_.D3DSetComputeRootDescriptorTable(
                UINT(FxaaRootParameter::kDestination), fxaa_descriptors[0].second);
            D3D12_SHADER_RESOURCE_VIEW_DESC fxaa_source_srv_desc;
            fxaa_source_srv_desc.Format = kFxaaSourceTextureFormat;
            fxaa_source_srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            fxaa_source_srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            fxaa_source_srv_desc.Texture2D.MostDetailedMip = 0;
            fxaa_source_srv_desc.Texture2D.MipLevels = 1;
            fxaa_source_srv_desc.Texture2D.PlaneSlice = 0;
            fxaa_source_srv_desc.Texture2D.ResourceMinLODClamp = 0.0f;
            device->CreateShaderResourceView(fxaa_source_texture_.Get(), &fxaa_source_srv_desc,
                                             fxaa_descriptors[1].first);
            deferred_command_list_.D3DSetComputeRootDescriptorTable(
                UINT(FxaaRootParameter::kSource), fxaa_descriptors[1].second);
            SetExternalPipeline(swap_post_effect == SwapPostEffect::kFxaaExtreme
                                    ? fxaa_extreme_pipeline_.Get()
                                    : fxaa_pipeline_.Get());
            SubmitBarriers();
            deferred_command_list_.D3DDispatch(group_count_x, group_count_y, 1);
            PushTransitionBarrier(guest_output_resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  ui::d3d12::D3D12Presenter::kGuestOutputInternalState);
          }
        } else {
          assert_true(apply_gamma_dest_initial_state ==
                      ui::d3d12::D3D12Presenter::kGuestOutputInternalState);
          PushTransitionBarrier(apply_gamma_dest, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                apply_gamma_dest_initial_state);
        }

        if (REXCVAR_GET(d3d12_debug_pink_probe)) {
          Amped3PinkProbe(guest_output_resource, guest_output_width, guest_output_height);
        }
        // Need to submit all the commands before giving the image back to the
        // presenter so it can submit its own commands for displaying it to the
        // queue.
        SubmitBarriers();
        EndSubmission(true);
        return true;
      });

  // End the frame even if did not present for any reason (the image refresher
  // was not called), to prevent leaking per-frame resources.
  EndSubmission(true);

  if (dlss_) {
    dlss_->EndFrame();
  }
  // The scene's resolve: the last single-sampled frontbuffer-wide color 0
  // resolve before the frame's final one (the frontbuffer copy).
  if (g_dlss_wide_resolves >= 2) {
    g_dlss_scene_resolve = g_dlss_wide_resolves - 1;
  }
  g_dlss_wide_resolves = 0;
  g_dlss_resolves_before_hud = 0;
  ++g_dlss_frame;
  if (REXCVAR_GET(dlss_log) && ++g_dlss_stats.frames % 60 == 0) {
    REXGPU_INFO("DLSS draws (60 frames): fb width {}, wide-target draws {}, narrow {}, scene {}, HUD {}, "
                "hand-offs {}, wants scene {}, scene resolve #{}, frame {}",
                g_dlss_stats.fb_width, g_dlss_stats.wide, g_dlss_stats.narrow, g_dlss_stats.scene,
                g_dlss_stats.hud, g_dlss_stats.taken, dlss_ ? int(dlss_->WantsScene()) : -1,
                g_dlss_scene_resolve, g_dlss_frame);
    g_dlss_stats = {g_dlss_stats.frames};
  }
}

bool D3D12CommandProcessor::SetupSmaa() {
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  D3D12_ROOT_PARAMETER parameters[UINT(SmaaRootParameter::kCount)];
  D3D12_DESCRIPTOR_RANGE ranges[4];
  {
    auto& parameter = parameters[UINT(SmaaRootParameter::kConstants)];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameter.Constants.ShaderRegister = 0;
    parameter.Constants.RegisterSpace = 0;
    parameter.Constants.Num32BitValues = 2;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  for (uint32_t i = 0; i < 4; ++i) {
    D3D12_DESCRIPTOR_RANGE& range = ranges[i];
    range.RangeType = i < 3 ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV : D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    range.NumDescriptors = 1;
    range.BaseShaderRegister = i < 3 ? i : 0;
    range.RegisterSpace = 0;
    range.OffsetInDescriptorsFromTableStart = 0;
    auto& parameter = parameters[UINT(SmaaRootParameter::kSource0) + i];
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameter.DescriptorTable.NumDescriptorRanges = 1;
    parameter.DescriptorTable.pDescriptorRanges = &range;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_STATIC_SAMPLER_DESC samplers[2] = {};
  for (uint32_t i = 0; i < 2; ++i) {
    D3D12_STATIC_SAMPLER_DESC& sampler = samplers[i];
    sampler.Filter =
        i ? D3D12_FILTER_MIN_MAG_MIP_POINT : D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxAnisotropy = 1;
    sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
    sampler.ShaderRegister = i;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_ROOT_SIGNATURE_DESC desc;
  desc.NumParameters = UINT(SmaaRootParameter::kCount);
  desc.pParameters = parameters;
  desc.NumStaticSamplers = 2;
  desc.pStaticSamplers = samplers;
  desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
  *(smaa_root_signature_.ReleaseAndGetAddressOf()) =
      ui::d3d12::util::CreateRootSignature(provider, desc);
  if (!smaa_root_signature_) {
    // Optional: FXAA is used instead.
    REXGPU_ERROR("Failed to create the SMAA root signature");
    return true;
  }
  const std::pair<const void*, size_t> shaders_bytecode[3] = {
      {shaders::smaa_edges_cs, sizeof(shaders::smaa_edges_cs)},
      {shaders::smaa_weights_cs, sizeof(shaders::smaa_weights_cs)},
      {shaders::smaa_blend_cs, sizeof(shaders::smaa_blend_cs)},
  };
  for (uint32_t i = 0; i < 3; ++i) {
    *(smaa_pipelines_[i].ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
        device, shaders_bytecode[i].first, shaders_bytecode[i].second, smaa_root_signature_.Get());
    if (!smaa_pipelines_[i]) {
      REXGPU_ERROR("Failed to create the SMAA compute pipelines");
      smaa_root_signature_.Reset();
      return true;
    }
  }
  // Lookup textures and their upload buffer (rows aligned to 256 bytes).
  struct Lut {
    Microsoft::WRL::ComPtr<ID3D12Resource>* texture;
    DXGI_FORMAT format;
    uint32_t width, height, pitch;
    const unsigned char* bytes;
  } luts[2] = {
      {&smaa_area_texture_, DXGI_FORMAT_R8G8_UNORM, AREATEX_WIDTH, AREATEX_HEIGHT, AREATEX_PITCH,
       shaders::areaTexBytes},
      {&smaa_search_texture_, DXGI_FORMAT_R8_UNORM, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT,
       SEARCHTEX_PITCH, shaders::searchTexBytes},
  };
  uint32_t upload_size = 0;
  for (const Lut& lut : luts) {
    upload_size += rex::align(lut.pitch, uint32_t(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)) * lut.height;
    upload_size = rex::align(upload_size, uint32_t(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT));
  }
  D3D12_RESOURCE_DESC buffer_desc;
  ui::d3d12::util::FillBufferResourceDesc(buffer_desc, upload_size, D3D12_RESOURCE_FLAG_NONE);
  uint8_t* mapping;
  if (FAILED(device->CreateCommittedResource(
          &ui::d3d12::util::kHeapPropertiesUpload, provider.GetHeapFlagCreateNotZeroed(),
          &buffer_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
          IID_PPV_ARGS(&smaa_lut_upload_buffer_))) ||
      FAILED(smaa_lut_upload_buffer_->Map(0, nullptr, reinterpret_cast<void**>(&mapping)))) {
    REXGPU_ERROR("Failed to create the SMAA lookup texture upload buffer");
    smaa_root_signature_.Reset();
    return true;
  }
  uint32_t offset = 0;
  for (const Lut& lut : luts) {
    D3D12_RESOURCE_DESC texture_desc = {};
    texture_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture_desc.Width = lut.width;
    texture_desc.Height = lut.height;
    texture_desc.DepthOrArraySize = 1;
    texture_desc.MipLevels = 1;
    texture_desc.Format = lut.format;
    texture_desc.SampleDesc.Count = 1;
    if (FAILED(device->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesDefault, provider.GetHeapFlagCreateNotZeroed(),
            &texture_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(lut.texture->ReleaseAndGetAddressOf())))) {
      REXGPU_ERROR("Failed to create an SMAA lookup texture");
      smaa_root_signature_.Reset();
      return true;
    }
    uint32_t row_pitch = rex::align(lut.pitch, uint32_t(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT));
    for (uint32_t y = 0; y < lut.height; ++y) {
      std::memcpy(mapping + offset + y * row_pitch, lut.bytes + y * lut.pitch, lut.pitch);
    }
    offset = rex::align(offset + row_pitch * lut.height,
                        uint32_t(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT));
  }
  smaa_lut_upload_buffer_->Unmap(0, nullptr);
  smaa_luts_uploaded_ = false;
  return true;
}

bool D3D12CommandProcessor::ApplySmaa(ID3D12Resource* destination, uint32_t width,
                                      uint32_t height) {
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  if (!smaa_luts_uploaded_) {
    uint32_t offset = 0;
    ID3D12Resource* textures[2] = {smaa_area_texture_.Get(), smaa_search_texture_.Get()};
    for (ID3D12Resource* texture : textures) {
      D3D12_RESOURCE_DESC texture_desc = texture->GetDesc();
      D3D12_TEXTURE_COPY_LOCATION dest, source;
      dest.pResource = texture;
      dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      dest.SubresourceIndex = 0;
      source.pResource = smaa_lut_upload_buffer_.Get();
      source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      source.PlacedFootprint.Offset = offset;
      source.PlacedFootprint.Footprint.Format = texture_desc.Format;
      source.PlacedFootprint.Footprint.Width = uint32_t(texture_desc.Width);
      source.PlacedFootprint.Footprint.Height = texture_desc.Height;
      source.PlacedFootprint.Footprint.Depth = 1;
      uint32_t pitch = uint32_t(texture_desc.Width) *
                       (texture_desc.Format == DXGI_FORMAT_R8G8_UNORM ? 2 : 1);
      source.PlacedFootprint.Footprint.RowPitch =
          rex::align(pitch, uint32_t(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT));
      deferred_command_list_.CopyTexture(dest, source);
      offset = rex::align(offset + source.PlacedFootprint.Footprint.RowPitch * texture_desc.Height,
                          uint32_t(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT));
      PushTransitionBarrier(texture, D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    smaa_luts_uploaded_ = true;
  }
  // Edges and weights textures of the output size.
  for (auto& texture : smaa_textures_) {
    if (texture) {
      D3D12_RESOURCE_DESC desc = texture->GetDesc();
      if (desc.Width != width || desc.Height != height) {
        if (submission_completed_ < smaa_textures_submission_) {
          texture->AddRef();
          resources_for_deletion_.emplace_back(smaa_textures_submission_, texture.Get());
        }
        texture.Reset();
      }
    }
    if (!texture) {
      D3D12_RESOURCE_DESC desc = {};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = width;
      desc.Height = height;
      desc.DepthOrArraySize = 1;
      desc.MipLevels = 1;
      desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      desc.SampleDesc.Count = 1;
      desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
      if (FAILED(device->CreateCommittedResource(
              &ui::d3d12::util::kHeapPropertiesDefault, provider.GetHeapFlagCreateNotZeroed(),
              &desc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
              IID_PPV_ARGS(texture.ReleaseAndGetAddressOf())))) {
        REXGPU_ERROR("Failed to create an SMAA texture");
        return false;
      }
    }
  }
  smaa_textures_submission_ = submission_current_;
  // Per pass: three sources and the destination.
  ui::d3d12::util::DescriptorCpuGpuHandlePair descriptors[12];
  if (!RequestOneUseSingleViewDescriptors(12, descriptors)) {
    return false;
  }
  auto srv = [&](ID3D12Resource* resource, DXGI_FORMAT format,
                 const ui::d3d12::util::DescriptorCpuGpuHandlePair& handle) {
    D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
    desc.Format = format;
    desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(resource, &desc, handle.first);
  };
  auto uav = [&](ID3D12Resource* resource, DXGI_FORMAT format,
                 const ui::d3d12::util::DescriptorCpuGpuHandlePair& handle) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
    desc.Format = format;
    desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(resource, nullptr, &desc, handle.first);
  };
  ID3D12Resource* color = fxaa_source_texture_.Get();
  ID3D12Resource* edges = smaa_textures_[0].Get();
  ID3D12Resource* weights = smaa_textures_[1].Get();
  const DXGI_FORMAT kRgba8 = DXGI_FORMAT_R8G8B8A8_UNORM;
  // Edges: color -> edges.
  srv(color, kFxaaSourceTextureFormat, descriptors[0]);
  srv(color, kFxaaSourceTextureFormat, descriptors[1]);
  srv(color, kFxaaSourceTextureFormat, descriptors[2]);
  uav(edges, kRgba8, descriptors[3]);
  // Weights: edges, AreaTex, SearchTex -> weights.
  srv(edges, kRgba8, descriptors[4]);
  srv(smaa_area_texture_.Get(), DXGI_FORMAT_R8G8_UNORM, descriptors[5]);
  srv(smaa_search_texture_.Get(), DXGI_FORMAT_R8_UNORM, descriptors[6]);
  uav(weights, kRgba8, descriptors[7]);
  // Blend: color, weights -> destination.
  srv(color, kFxaaSourceTextureFormat, descriptors[8]);
  srv(weights, kRgba8, descriptors[9]);
  srv(weights, kRgba8, descriptors[10]);
  uav(destination, ui::d3d12::D3D12Presenter::kGuestOutputFormat, descriptors[11]);

  deferred_command_list_.D3DSetComputeRootSignature(smaa_root_signature_.Get());
  uint32_t constants[2] = {width, height};
  deferred_command_list_.D3DSetComputeRoot32BitConstants(UINT(SmaaRootParameter::kConstants), 2,
                                                         constants, 0);
  uint32_t groups_x = (width + 7) / 8, groups_y = (height + 7) / 8;
  ID3D12Resource* pass_targets[2] = {edges, weights};
  for (uint32_t pass = 0; pass < 3; ++pass) {
    if (pass < 2) {
      PushTransitionBarrier(pass_targets[pass], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    for (uint32_t i = 0; i < 4; ++i) {
      deferred_command_list_.D3DSetComputeRootDescriptorTable(
          UINT(SmaaRootParameter::kSource0) + i, descriptors[pass * 4 + i].second);
    }
    SetExternalPipeline(smaa_pipelines_[pass].Get());
    SubmitBarriers();
    deferred_command_list_.D3DDispatch(groups_x, groups_y, 1);
    if (pass < 2) {
      PushTransitionBarrier(pass_targets[pass], D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
  }
  return true;
}

void D3D12CommandProcessor::OnPrimaryBufferEnd() {
  if (REXCVAR_GET(d3d12_submit_on_primary_buffer_end) && submission_open_ &&
      CanEndSubmissionImmediately()) {
    EndSubmission(false);
  }
}

void D3D12CommandProcessor::Amped3OnTileChange(uint32_t new_tile) {
  if (!REXCVAR_GET(gpu_tile_replay)) {
    return;
  }
  const RegisterFile& regs = *register_file_;
  tile_replay_cursor_ = 0;
  tile_replay_divergent_.clear();
  tile_replay_divergent_constants_.clear();
  std::memset(tile_replay_divergent_float_, 0, sizeof(tile_replay_divergent_float_));
  tile_replay_divergent_fetch_ = 0;
  tile_replay_disabled_ = false;
  if (new_tile == 1) {
    tile_replay_start_registers_.assign(regs.values, regs.values + RegisterFile::kRegisterCount);
    tile_replay_start_shaders_[0] = active_vertex_shader();
    tile_replay_start_shaders_[1] = active_pixel_shader();
    tile_replay_addresses_.clear();
    tile_replay_watch_values_.clear();
    tile_replay_watch_hashes_.clear();
    tile_replay_srv_keys_.clear();
    // This frame's watch set.
    if (tile_replay_watch_position_.empty()) {
      tile_replay_watch_position_.assign(RegisterFile::kRegisterCount, -1);
    }
    for (uint32_t index : tile_replay_watch_) {
      tile_replay_watch_position_[index] = -1;
    }
    tile_replay_watch_ = tile_replay_watch_next_;
    for (size_t i = 0; i < tile_replay_watch_.size(); ++i) {
      tile_replay_watch_position_[tile_replay_watch_[i]] = int8_t(i);
    }
    std::memcpy(tile_replay_watch_float_, tile_replay_watch_float_next_,
                sizeof(tile_replay_watch_float_));
    tile_replay_watch_fetch_ = tile_replay_watch_fetch_next_;
    if (!(++tile_replay_stats_[7] % 300)) {
      const uint64_t* s = tile_replay_stats_;
      uint64_t later = std::max(s[0] + s[1] + s[2] + s[3] + s[4] + s[5], uint64_t(1));
      std::vector<std::pair<uint64_t, uint32_t>> blocking;
      for (auto& b : tile_replay_blocking_registers_) {
        blocking.emplace_back(b.second, b.first);
      }
      std::sort(blocking.rbegin(), blocking.rend());
      std::string top;
      for (size_t i = 0; i < blocking.size() && i < 8; ++i) {
        top += fmt::format(" {:04X}{}:{}", blocking[i].second & 0xFFFF,
                           (blocking[i].second >> 31) ? "(watched)" : "", blocking[i].first);
      }
      REXGPU_INFO(
          "gpu_tile_replay: 300 frames: tile 1 draws {}/frame; tiles 2-3 draws {}/frame, "
          "eligible {:.1f}%, replayed {}/frame; not: no tile 1 draw {:.1f}%, registers {:.1f}%, "
          "constants {:.1f}%, too much differing {:.1f}%, shaders {:.1f}%; fell back: render "
          "targets {}, textures {}; watching {} registers, blocking:{}",
          s[6] / 300, later / 300, 100.0 * s[0] / later, tile_replay_replayed_ / 300,
          100.0 * s[1] / later, 100.0 * s[2] / later, 100.0 * s[3] / later, 100.0 * s[4] / later,
          100.0 * s[5] / later, tile_replay_fallbacks_[0], tile_replay_fallbacks_[1],
          tile_replay_watch_.size(), top);
      std::memset(tile_replay_stats_, 0, sizeof(tile_replay_stats_));
      tile_replay_blocking_registers_.clear();
      tile_replay_replayed_ = 0;
      std::memset(tile_replay_fallbacks_, 0, sizeof(tile_replay_fallbacks_));
    }
    return;
  }
  if (new_tile < 2 || tile_replay_start_registers_.empty()) {
    return;
  }
  // Registers that legitimately differ between the tiles or don't affect
  // drawing: everything below the context registers (control, coherency,
  // counters), the window offset and scissor (derived again for each draw),
  // the resolve destination (copies always run normally), event initiators
  // and the index buffer registers (set by each indexed draw packet).
  auto per_tile = [](uint32_t index) {
    return index < 0x2000 || index == XE_GPU_REG_PA_SC_WINDOW_OFFSET ||
           index == XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL ||
           index == XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR || index == XE_GPU_REG_RB_COPY_DEST_BASE ||
           index == XE_GPU_REG_VGT_EVENT_INITIATOR || index == XE_GPU_REG_VGT_DMA_BASE ||
           index == XE_GPU_REG_VGT_DMA_SIZE;
  };
  const uint32_t* start = tile_replay_start_registers_.data();
  for (uint32_t i = 0x2000; i < RegisterFile::kRegisterCount; ++i) {
    if (regs.values[i] == start[i] || per_tile(i)) {
      continue;
    }
    if (i >= XE_GPU_REG_SHADER_CONSTANT_000_X && i < XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) {
      uint32_t constant = (i - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
      tile_replay_divergent_float_[constant >> 6] |= uint64_t(1) << (constant & 63);
      tile_replay_divergent_constants_.emplace_back(i, regs.values[i]);
    } else if (i >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
               i < XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + 32 * 6) {
      tile_replay_divergent_fetch_ |= uint32_t(1)
                                      << ((i - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) / 6);
      tile_replay_divergent_constants_.emplace_back(i, regs.values[i]);
    } else {
      tile_replay_divergent_.emplace_back(i, regs.values[i]);
    }
  }
  if (new_tile == 2) {
    // Watch them in the next frame's tile 1.
    tile_replay_watch_next_.clear();
    for (const std::pair<uint32_t, uint32_t>& divergent : tile_replay_divergent_) {
      if (tile_replay_watch_next_.size() < 32) {
        tile_replay_watch_next_.push_back(divergent.first);
      }
    }
    std::memcpy(tile_replay_watch_float_next_, tile_replay_divergent_float_,
                sizeof(tile_replay_watch_float_next_));
    tile_replay_watch_fetch_next_ = tile_replay_divergent_fetch_;
  }
  // Watched ones are compared with tile 1 draw by draw; the rest must be
  // written again before a draw that depends on them.
  tile_replay_divergent_.erase(
      std::remove_if(tile_replay_divergent_.begin(), tile_replay_divergent_.end(),
                     [this](const std::pair<uint32_t, uint32_t>& divergent) {
                       return tile_replay_watch_position_[divergent.first] >= 0;
                     }),
      tile_replay_divergent_.end());
  for (uint32_t i = 0; i < 8; ++i) {
    tile_replay_divergent_float_[i] &= ~tile_replay_watch_float_[i];
  }
  tile_replay_divergent_fetch_ &= ~tile_replay_watch_fetch_;
  tile_replay_divergent_constants_.erase(
      std::remove_if(tile_replay_divergent_constants_.begin(),
                     tile_replay_divergent_constants_.end(),
                     [this](const std::pair<uint32_t, uint32_t>& divergent) {
                       uint32_t i = divergent.first;
                       if (i < XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) {
                         uint32_t c = (i - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
                         return ((tile_replay_watch_float_[c >> 6] >> (c & 63)) & 1) != 0;
                       }
                       return ((tile_replay_watch_fetch_ >>
                                ((i - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) / 6)) &
                               1) != 0;
                     }),
      tile_replay_divergent_constants_.end());
  tile_replay_disabled_ = tile_replay_divergent_.size() > 64;
  const Shader* shaders[2] = {active_vertex_shader(), active_pixel_shader()};
  for (uint32_t i = 0; i < 2; ++i) {
    tile_replay_shader_divergent_[i] = shaders[i] != tile_replay_start_shaders_[i];
    tile_replay_divergent_shaders_[i] = shaders[i];
  }
}

uint64_t D3D12CommandProcessor::TileReplayWatchedConstantsHash(const Shader* vertex_shader,
                                                               const Shader* pixel_shader,
                                                               uint32_t used_fetch_mask) const {
  const RegisterFile& regs = *register_file_;
  // (Multiply-xor: these are a few registers per draw, compared exactly once.)
  uint64_t hash = 0;
  auto mix = [&hash](const uint32_t* values, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
      hash = (hash ^ values[i]) * 0x9E3779B97F4A7C15ull;
      hash ^= hash >> 29;
    }
  };
  for (uint32_t stage = 0; stage < 2; ++stage) {
    const Shader* shader = stage ? pixel_shader : vertex_shader;
    if (!shader) {
      continue;
    }
    const uint64_t* used = shader->constant_register_map().float_bitmap;
    for (uint32_t i = 0; i < 4; ++i) {
      uint64_t bits = used[i] & tile_replay_watch_float_[stage * 4 + i];
      uint32_t j;
      while (rex::bit_scan_forward(bits, &j)) {
        bits &= bits - 1;
        uint32_t constant = stage * 256 + i * 64 + j;
        mix(&regs.values[XE_GPU_REG_SHADER_CONSTANT_000_X + constant * 4], 4);
      }
    }
  }
  uint32_t fetch = used_fetch_mask & tile_replay_watch_fetch_;
  uint32_t j;
  while (rex::bit_scan_forward(fetch, &j)) {
    fetch &= fetch - 1;
    mix(&regs.values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + j * 6], 6);
  }
  return hash;
}

bool D3D12CommandProcessor::TileReplayBindings(const TileReplayRecord& record,
                                               const D3D12Shader* vertex_shader,
                                               const D3D12Shader* pixel_shader) {
  // The textures must be the same as tile 1's (bindless descriptor indices).
  const D3D12TextureCache::TextureSRVKey* keys = tile_replay_srv_keys_.data() + record.srv_keys_start;
  if ((record.srv_key_counts[0] &&
       !texture_cache_->AreActiveTextureSRVKeysUpToDate(
           keys, vertex_shader->GetTextureBindingsAfterTranslation().data(),
           record.srv_key_counts[0])) ||
      (record.srv_key_counts[1] &&
       !texture_cache_->AreActiveTextureSRVKeysUpToDate(
           keys + record.srv_key_counts[0], pixel_shader->GetTextureBindingsAfterTranslation().data(),
           record.srv_key_counts[1]))) {
    return false;
  }
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  if (current_graphics_root_signature_ != record.root_signature) {
    current_graphics_root_signature_ = record.root_signature;
    current_graphics_root_up_to_date_ = 0;
    deferred_command_list_.D3DSetGraphicsRootSignature(record.root_signature);
  }
  if (tile_replay_root_signature_ != current_graphics_root_signature_) {
    tile_replay_root_signature_ = current_graphics_root_signature_;
    std::memset(tile_replay_root_addresses_, 0, sizeof(tile_replay_root_addresses_));
  }
  // System constants: this tile's (the viewport differs).
  if (!cbuffer_binding_system_.up_to_date) {
    uint8_t* system_constants = constant_buffer_pool_->Request(
        frame_current_, sizeof(system_constants_), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
        nullptr, nullptr, &cbuffer_binding_system_.address);
    if (system_constants == nullptr) {
      return false;
    }
    std::memcpy(system_constants, &system_constants_, sizeof(system_constants_));
    cbuffer_binding_system_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_SystemConstants);
  }
  if (!(current_graphics_root_up_to_date_ & (1u << kRootParameter_Bindless_SystemConstants))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
        kRootParameter_Bindless_SystemConstants, cbuffer_binding_system_.address);
    current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_SystemConstants;
  }
  // Tile 1's constant buffers (the same data, uploaded in this frame). The
  // regular path's own bindings aren't in these root parameters anymore.
  static const uint32_t kParameters[6] = {
      kRootParameter_Bindless_FetchConstants,         kRootParameter_Bindless_FloatConstantsVertex,
      kRootParameter_Bindless_FloatConstantsPixel,    kRootParameter_Bindless_BoolLoopConstants,
      kRootParameter_Bindless_DescriptorIndicesPixel, kRootParameter_Bindless_DescriptorIndicesVertex,
  };
  for (uint32_t i = 0; i < 6; ++i) {
    uint32_t bit = 1u << kParameters[i];
    if ((current_graphics_root_up_to_date_ & bit) ||
        tile_replay_root_addresses_[i] != record.constant_buffers[i]) {
      deferred_command_list_.D3DSetGraphicsRootConstantBufferView(kParameters[i],
                                                                  record.constant_buffers[i]);
      tile_replay_root_addresses_[i] = record.constant_buffers[i];
      current_graphics_root_up_to_date_ &= ~bit;
    }
  }
  // As UpdateBindings: shared memory as an SRV, the heaps, the extents slot.
  if (!current_shared_memory_binding_is_uav_.has_value() ||
      current_shared_memory_binding_is_uav_.value()) {
    current_shared_memory_binding_is_uav_ = false;
    current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_SharedMemory);
  }
  if (!(current_graphics_root_up_to_date_ & (1u << kRootParameter_Bindless_SharedMemory))) {
    deferred_command_list_.D3DSetGraphicsRootDescriptorTable(
        kRootParameter_Bindless_SharedMemory,
        provider.OffsetViewDescriptor(
            view_bindless_heap_gpu_start_,
            uint32_t(SystemBindlessView::kSharedMemoryRawSRVAndNullRawUAVStart)));
    current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_SharedMemory;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << kRootParameter_Bindless_SamplerHeap))) {
    deferred_command_list_.D3DSetGraphicsRootDescriptorTable(kRootParameter_Bindless_SamplerHeap,
                                                             sampler_bindless_heap_gpu_start_);
    current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_SamplerHeap;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << kRootParameter_Bindless_ViewHeap))) {
    deferred_command_list_.D3DSetGraphicsRootDescriptorTable(kRootParameter_Bindless_ViewHeap,
                                                             view_bindless_heap_gpu_start_);
    current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_ViewHeap;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << kRootParameter_Bindless_ScreenExtents))) {
    deferred_command_list_.D3DSetGraphicsRootUnorderedAccessView(
        kRootParameter_Bindless_ScreenExtents,
        screen_extents_buffer_->GetGPUVirtualAddress() + uint64_t(screen_extents_root_slot_) * 16);
    current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_ScreenExtents;
  }
  return true;
}


bool D3D12CommandProcessor::TileReplayEligible(const Shader* vertex_shader,
                                               const Shader* pixel_shader,
                                               uint32_t used_texture_mask) {
  const RegisterFile& regs = *register_file_;
  tile_replay_current_ = -1;
  if (amped3_tile_ < 1 || tile_replay_start_registers_.empty()) {
    return false;
  }
  uint32_t used_fetch_mask = used_texture_mask;
  const Shader::ConstantRegisterMap& vertex_map = vertex_shader->constant_register_map();
  for (uint32_t i = 0; i < rex::countof(vertex_map.vertex_fetch_bitmap); ++i) {
    uint32_t bits = vertex_map.vertex_fetch_bitmap[i];
    uint32_t j;
    while (rex::bit_scan_forward(bits, &j)) {
      bits &= ~(uint32_t(1) << j);
      used_fetch_mask |= uint32_t(1) << ((i * 32 + j) / 3);
    }
  }
  if (amped3_tile_ == 1) {
    uint32_t record = uint32_t(tile_replay_addresses_.size());
    tile_replay_addresses_.push_back(amped3_draw_address_);
    for (uint32_t index : tile_replay_watch_) {
      tile_replay_watch_values_.push_back(regs.values[index]);
    }
    tile_replay_watch_hashes_.push_back(
        TileReplayWatchedConstantsHash(vertex_shader, pixel_shader, used_fetch_mask));
    if (tile_replay_records_.size() <= record) {
      tile_replay_records_.resize(record + 1);
    }
    tile_replay_records_[record].valid = false;
    tile_replay_current_ = int32_t(record);
    ++tile_replay_stats_[6];
    return false;
  }
  if (tile_replay_disabled_) {
    ++tile_replay_stats_[4];
    return false;
  }
  // The same draw as tile 1's at this position (or a few further, if this
  // tile skipped some).
  uint32_t record = tile_replay_cursor_;
  while (record < tile_replay_addresses_.size() && record < tile_replay_cursor_ + 8 &&
         tile_replay_addresses_[record] != amped3_draw_address_) {
    ++record;
  }
  if (record >= tile_replay_addresses_.size() ||
      tile_replay_addresses_[record] != amped3_draw_address_) {
    ++tile_replay_stats_[1];
    return false;
  }
  tile_replay_cursor_ = record + 1;
  const Shader* active_shaders[2] = {active_vertex_shader(), active_pixel_shader()};
  for (uint32_t i = 0; i < 2; ++i) {
    if (tile_replay_shader_divergent_[i] &&
        active_shaders[i] != tile_replay_divergent_shaders_[i]) {
      tile_replay_shader_divergent_[i] = false;
    }
  }
  if (tile_replay_shader_divergent_[0] || (pixel_shader && tile_replay_shader_divergent_[1])) {
    ++tile_replay_stats_[5];
    return false;
  }
  // Watched registers: as at the same draw in tile 1.
  const uint32_t* watch_values = tile_replay_watch_values_.data() + size_t(record) * tile_replay_watch_.size();
  for (size_t i = 0; i < tile_replay_watch_.size(); ++i) {
    if (regs.values[tile_replay_watch_[i]] != watch_values[i]) {
      ++tile_replay_stats_[2];
      ++tile_replay_blocking_registers_[tile_replay_watch_[i] | (uint32_t(1) << 31)];
      return false;
    }
  }
  // Others: written again since the tile's start.
  tile_replay_divergent_.erase(
      std::remove_if(tile_replay_divergent_.begin(), tile_replay_divergent_.end(),
                     [&regs](const std::pair<uint32_t, uint32_t>& divergent) {
                       return regs.values[divergent.first] != divergent.second;
                     }),
      tile_replay_divergent_.end());
  if (!tile_replay_divergent_.empty()) {
    ++tile_replay_stats_[2];
    ++tile_replay_blocking_registers_[tile_replay_divergent_.front().first];
    return false;
  }
  // Unwatched constants this draw uses: written again since the tile's start.
  bool uses_divergent = (used_fetch_mask & tile_replay_divergent_fetch_) != 0;
  for (uint32_t i = 0; i < 4 && !uses_divergent; ++i) {
    uses_divergent = (vertex_map.float_bitmap[i] & tile_replay_divergent_float_[i]) ||
                     (pixel_shader && (pixel_shader->constant_register_map().float_bitmap[i] &
                                       tile_replay_divergent_float_[4 + i]));
  }
  if (uses_divergent) {
    std::memset(tile_replay_divergent_float_, 0, sizeof(tile_replay_divergent_float_));
    tile_replay_divergent_fetch_ = 0;
    tile_replay_divergent_constants_.erase(
        std::remove_if(tile_replay_divergent_constants_.begin(),
                       tile_replay_divergent_constants_.end(),
                       [&regs](const std::pair<uint32_t, uint32_t>& divergent) {
                         return regs.values[divergent.first] != divergent.second;
                       }),
        tile_replay_divergent_constants_.end());
    for (const std::pair<uint32_t, uint32_t>& divergent : tile_replay_divergent_constants_) {
      uint32_t i = divergent.first;
      if (i < XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) {
        uint32_t c = (i - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
        tile_replay_divergent_float_[c >> 6] |= uint64_t(1) << (c & 63);
      } else {
        tile_replay_divergent_fetch_ |= uint32_t(1)
                                        << ((i - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) / 6);
      }
    }
    uses_divergent = (used_fetch_mask & tile_replay_divergent_fetch_) != 0;
    for (uint32_t i = 0; i < 4 && !uses_divergent; ++i) {
      uses_divergent = (vertex_map.float_bitmap[i] & tile_replay_divergent_float_[i]) ||
                       (pixel_shader && (pixel_shader->constant_register_map().float_bitmap[i] &
                                         tile_replay_divergent_float_[4 + i]));
    }
    if (uses_divergent) {
      ++tile_replay_stats_[3];
      return false;
    }
  }
  // Watched constants this draw uses: as at the same draw in tile 1.
  if (TileReplayWatchedConstantsHash(vertex_shader, pixel_shader, used_fetch_mask) !=
      tile_replay_watch_hashes_[record]) {
    ++tile_replay_stats_[3];
    return false;
  }
  ++tile_replay_stats_[0];
  if (tile_replay_records_[record].valid) {
    tile_replay_current_ = int32_t(record);
  }
  return true;
}

void D3D12CommandProcessor::TileReplayHashes(const D3D12Shader* vertex_shader,
                                             const D3D12Shader* pixel_shader,
                                             uint64_t hashes_out[6]) const {
  const RegisterFile& regs = *register_file_;
  // Float constants the shaders use.
  for (uint32_t stage = 0; stage < 2; ++stage) {
    const D3D12Shader* shader = stage ? pixel_shader : vertex_shader;
    uint64_t hash = 0;
    if (shader) {
      const uint64_t* bitmap = shader->constant_register_map().float_bitmap;
      uint32_t base = stage ? XE_GPU_REG_SHADER_CONSTANT_256_X : XE_GPU_REG_SHADER_CONSTANT_000_X;
      for (uint32_t i = 0; i < 256; ++i) {
        if ((bitmap[i >> 6] >> (i & 63)) & 1) {
          hash = XXH3_64bits_withSeed(&regs.values[base + i * 4], sizeof(uint32_t) * 4, hash);
        }
      }
    }
    hashes_out[stage] = hash;
  }
  hashes_out[2] = XXH3_64bits(&regs.values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0],
                              sizeof(uint32_t) * 32 * 6);
  hashes_out[3] = XXH3_64bits(&regs.values[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031],
                              sizeof(uint32_t) * (8 + 32));
  // System constants other than the viewport-derived ones.
  DxbcShaderTranslator::SystemConstants system = system_constants_;
  std::memset(system.ndc_scale, 0, sizeof(system.ndc_scale));
  std::memset(system.ndc_offset, 0, sizeof(system.ndc_offset));
  std::memset(system.point_screen_diameter_to_ndc_radius, 0,
              sizeof(system.point_screen_diameter_to_ndc_radius));
  hashes_out[4] = XXH3_64bits(&system, sizeof(system));
  // Texture and sampler bindings.
  uint64_t hash = 0;
  if (vertex_shader) {
    size_t count = vertex_shader->GetTextureBindingsAfterTranslation().size();
    hash = XXH3_64bits_withSeed(current_texture_srv_keys_vertex_.data(),
                                sizeof(D3D12TextureCache::TextureSRVKey) * count, hash);
    count = vertex_shader->GetSamplerBindingsAfterTranslation().size();
    hash = XXH3_64bits_withSeed(current_sampler_bindless_indices_vertex_.data(),
                                sizeof(uint32_t) * count, hash);
  }
  if (pixel_shader) {
    size_t count = pixel_shader->GetTextureBindingsAfterTranslation().size();
    hash = XXH3_64bits_withSeed(current_texture_srv_keys_pixel_.data(),
                                sizeof(D3D12TextureCache::TextureSRVKey) * count, hash);
    count = pixel_shader->GetSamplerBindingsAfterTranslation().size();
    hash = XXH3_64bits_withSeed(current_sampler_bindless_indices_pixel_.data(),
                                sizeof(uint32_t) * count, hash);
  }
  hashes_out[5] = hash;
}

void D3D12CommandProcessor::Amped3RecordScreenExtents(uint32_t address) {
  if (screen_extents_draw_.slot == UINT32_MAX) {
    ++screen_extents_stats_[3];
    return;
  }
  ScreenExtentsDraw& draw = screen_extents_draw_;
  draw.address = address;
  if (REXCVAR_GET(gpu_screen_extents) == 4) {
    // Identity: the draw's inputs, and which occurrence of them this is.
    uint32_t& occurrence = screen_extents_occurrences_[draw.key];
    draw.key = (draw.key ^ occurrence++) * 0x9E3779B97F4A7C15ull;
    auto known = screen_extents_known_.find(draw.key);
    if (known != screen_extents_known_.end()) {
      const ScreenExtentsKnown& k = known->second;
      for (uint32_t i = 0; i < 4; ++i) {
        draw.predicted[0][i] = k.tiles[i];
        draw.predicted[1][i] = k.tiles[i];
        if (k.prev_resolve != UINT32_MAX && k.resolve > k.prev_resolve) {
          // Motion per resolve, carried on to this one.
          draw.predicted[1][i] +=
              (int32_t(k.tiles[i]) - int32_t(k.prev_tiles[i])) *
              int32_t(screen_extents_resolves_ + 1 - k.resolve) /
              int32_t(k.resolve - k.prev_resolve);
        }
      }
      if (REXCVAR_GET(gpu_screen_extents_predict)) {
        const int32_t margin = std::max(REXCVAR_GET(gpu_screen_extents_margin), 0);
        const int32_t tile_max = (xenos::kTexture2DCubeMaxWidthHeight - 1) >> 3;
        const int32_t* p = draw.predicted[REXCVAR_GET(gpu_screen_extents_motion) ? 1 : 0];
        auto clamp = [tile_max](int32_t v) { return uint16_t(std::min(std::max(v, 0), tile_max)); };
        uint16_t extents[6] = {
            clamp(p[0] - margin), clamp(p[1] + margin), clamp(p[2] - margin),
            clamp(p[3] + margin), 0, 1,
        };
        memory::copy_and_swap_16_unaligned(memory_->TranslatePhysical(address), extents,
                                           rex::countof(extents));
      }
    } else {
      draw.predicted[0][0] = 1;
      draw.predicted[0][1] = 0;
    }
    screen_extents_pending_.push_back(draw);
    draw.slot = UINT32_MAX;
    return;
  }
  screen_extents_pending_.push_back(draw);
  draw.slot = UINT32_MAX;
  int32_t flush = REXCVAR_GET(gpu_screen_extents_flush);
  if (flush > 0 && screen_extents_pending_.size() % uint32_t(flush) == 0 && submission_open_) {
    EndSubmission(false);
  }
}

bool D3D12CommandProcessor::ScreenExtentsFromSlot(const ScreenExtentsDraw& draw,
                                                  const uint32_t* slot,
                                                  uint16_t tiles_out[4]) const {
  if (!slot[0]) {
    // No vertex.
    return false;
  }
  // 1/4096 from -4, rounded down - minimums inverted, maximums one step up.
  float ndc[4] = {
      float(0x10000 - slot[0]) * (1.0f / 4096.0f) - 4.0f,
      float(slot[2] + 1) * (1.0f / 4096.0f) - 4.0f,
      float(0x10000 - slot[1]) * (1.0f / 4096.0f) - 4.0f,
      float(slot[3] + 1) * (1.0f / 4096.0f) - 4.0f,
  };
  auto to_tiles = [](float value, bool max) {
    float clamped = std::min(std::max(max ? std::ceil(value) : std::floor(value), 0.0f),
                             float(xenos::kTexture2DCubeMaxWidthHeight - 1));
    return uint16_t(uint32_t(clamped) >> 3);
  };
  for (uint32_t i = 0; i < 2; ++i) {
    float a = ndc[i * 2] * draw.scale[i] + draw.offset[i];
    float b = ndc[i * 2 + 1] * draw.scale[i] + draw.offset[i];
    tiles_out[i * 2] = to_tiles(std::min(a, b), false);
    tiles_out[i * 2 + 1] = to_tiles(std::max(a, b), true);
  }
  return true;
}

void D3D12CommandProcessor::ScreenExtentsProcessBatches(bool await_oldest) {
  if (screen_extents_batches_.empty()) {
    return;
  }
  CheckSubmissionFence(await_oldest ? screen_extents_batches_.front().submission : 0);
  while (!screen_extents_batches_.empty() &&
         screen_extents_batches_.front().submission <= submission_completed_) {
    ScreenExtentsBatch& batch = screen_extents_batches_.front();
    const uint32_t* region =
        screen_extents_readback_mapping_ + size_t(batch.region) * kScreenExtentsSlots * 4;
    for (const ScreenExtentsDraw& draw : batch.draws) {
      uint16_t tiles[4];
      if (!ScreenExtentsFromSlot(draw, region + draw.slot * 4, tiles)) {
        screen_extents_known_.erase(draw.key);
        continue;
      }
      // The tiles (horizontal bands of 32 8x8 tiles: rows 0-255, 256-511,
      // 512-) a vertical tile range covers, as bits 0-2.
      auto bands = [](int32_t min_y, int32_t max_y) -> uint32_t {
        auto band = [](int32_t y) { return std::min(std::max(y, 0) / 32, 2); };
        if (max_y < 0 || min_y > max_y) {
          return 0;
        }
        uint32_t first = band(min_y), last = band(max_y);
        return ((2u << last) - 1) & ~((1u << first) - 1);
      };
      uint32_t covered = bands(tiles[2], tiles[3]);
      if (draw.predicted[0][0] <= draw.predicted[0][1]) {
        ++screen_extents_predicted_;
        screen_extents_ideal_skips_ += rex::bit_count(~covered & 0b110u);
        for (uint32_t v = 0; v < 2; ++v) {
          for (uint32_t m = 0; m < rex::countof(kScreenExtentsMargins); ++m) {
            int32_t margin = kScreenExtentsMargins[m];
            uint32_t predicted = bands(draw.predicted[v][2] - margin, draw.predicted[v][3] + margin);
            screen_extents_skips_[v][m] += rex::bit_count(~predicted & 0b110u);
            screen_extents_wrong_[v][m] += (covered & ~predicted & 0b110u) != 0;
          }
        }
      }
      // (New entries are zeroed; resolve numbers start at 1.)
      ScreenExtentsKnown& known = screen_extents_known_[draw.key];
      if (!known.resolve) {
        known.prev_resolve = UINT32_MAX;
      } else {
        std::memcpy(known.prev_tiles, known.tiles, sizeof(known.prev_tiles));
        known.prev_resolve = known.resolve;
      }
      std::memcpy(known.tiles, tiles, sizeof(tiles));
      known.resolve = batch.resolve;
      ++screen_extents_stats_[1];
    }
    screen_extents_batches_.pop_front();
  }
}

void D3D12CommandProcessor::Amped3ResolveScreenExtents() {
  uint32_t slots = screen_extents_slots_used_;
  if (!slots) {
    return;
  }
  // The next measured draw zeroes the slots again.
  screen_extents_slots_used_ = 0;
  screen_extents_draw_.slot = UINT32_MAX;
  screen_extents_occurrences_.clear();
  if (screen_extents_pending_.empty() || !BeginSubmission(true)) {
    screen_extents_pending_.clear();
    return;
  }
  bool predict_mode = REXCVAR_GET(gpu_screen_extents) == 4;
  uint32_t region = 0;
  if (predict_mode) {
    // Measurements for the next frames: read back without waiting.
    if (screen_extents_batches_.size() >= kScreenExtentsReadbacks) {
      ScreenExtentsProcessBatches(true);
    }
    region = screen_extents_next_region_;
    screen_extents_next_region_ = (region + 1) % kScreenExtentsReadbacks;
  }
  PushTransitionBarrier(screen_extents_buffer_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE);
  SubmitBarriers();
  deferred_command_list_.D3DCopyBufferRegion(
      screen_extents_readback_buffer_.Get(),
      uint64_t(region) * kScreenExtentsSlots * sizeof(uint32_t) * 4, screen_extents_buffer_.Get(),
      0, slots * sizeof(uint32_t) * 4);
  PushTransitionBarrier(screen_extents_buffer_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  SubmitBarriers();
  screen_extents_stats_[0] += screen_extents_pending_.size();
  ++screen_extents_resolves_;
  if (predict_mode) {
    ScreenExtentsBatch batch;
    batch.submission = submission_current_;
    batch.region = region;
    batch.resolve = screen_extents_resolves_;
    batch.draws.swap(screen_extents_pending_);
    screen_extents_batches_.push_back(std::move(batch));
    ScreenExtentsProcessBatches(false);
    if (!(screen_extents_resolves_ % 600)) {
      for (auto it = screen_extents_known_.begin(); it != screen_extents_known_.end();) {
        if (screen_extents_resolves_ - it->second.resolve > 60) {
          it = screen_extents_known_.erase(it);
        } else {
          ++it;
        }
      }
    }
  } else {
    auto wait_start = std::chrono::steady_clock::now();
    if (!AwaitAllQueueOperationsCompletion()) {
      screen_extents_pending_.clear();
      return;
    }
    screen_extents_stats_[2] += uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                                             std::chrono::steady_clock::now() - wait_start)
                                             .count());
    for (const ScreenExtentsDraw& draw : screen_extents_pending_) {
      uint16_t tiles[4];
      if (!ScreenExtentsFromSlot(draw, screen_extents_readback_mapping_ + draw.slot * 4, tiles)) {
        continue;
      }
      uint16_t extents[6] = {tiles[0], tiles[1], tiles[2], tiles[3], 0, 1};
      memory::copy_and_swap_16_unaligned(memory_->TranslatePhysical(draw.address), extents,
                                         rex::countof(extents));
      ++screen_extents_stats_[1];
    }
    screen_extents_pending_.clear();
  }
  if (!(screen_extents_resolves_ % 300)) {
    REXGPU_INFO(
        "gpu_screen_extents: 300 resolves: {} measured draws, {} measurements used, {} "
        "unmeasured, {} us average wait, {} known draws",
        screen_extents_stats_[0], screen_extents_stats_[1], screen_extents_stats_[3],
        screen_extents_stats_[2] / 300, screen_extents_known_.size());
    if (screen_extents_predicted_) {
      std::string line;
      for (uint32_t v = 0; v < 2; ++v) {
        line += v ? " | with motion:" : " | as is:";
        for (uint32_t m = 0; m < rex::countof(kScreenExtentsMargins); ++m) {
          line += fmt::format(" m{} {:.1f}%/{:.2f}%", kScreenExtentsMargins[m],
                              100.0 * screen_extents_skips_[v][m] /
                                  std::max(screen_extents_ideal_skips_, uint64_t(1)),
                              100.0 * screen_extents_wrong_[v][m] / screen_extents_predicted_);
        }
      }
      REXGPU_INFO(
          "gpu_screen_extents: {} predicted draws ({}/frame), ideal tile 2/3 skips {}/frame; "
          "margin: kept skips / draws missing from a tile{}",
          screen_extents_predicted_, screen_extents_predicted_ / 300,
          screen_extents_ideal_skips_ / 300, line);
    }
    std::memset(screen_extents_stats_, 0, sizeof(screen_extents_stats_));
    std::memset(screen_extents_skips_, 0, sizeof(screen_extents_skips_));
    std::memset(screen_extents_wrong_, 0, sizeof(screen_extents_wrong_));
    screen_extents_ideal_skips_ = 0;
    screen_extents_predicted_ = 0;
  }
}

// The game's sprite vertex shader: position = (c0..c3) * (vertex xyz, 1) over a
// 4-vertex quad in vertex fetch constant 95 (stride 8 dwords, xyz first).
bool D3D12CommandProcessor::Amped3IsSubpixelSprite(const Shader& vertex_shader) {
  if (vertex_shader.ucode_data_hash() != 0xCB134112C994DF08ull) {
    return false;
  }
  const RegisterFile& regs = *register_file_;
  const auto& bindings = vertex_shader.vertex_bindings();
  if (bindings.size() != 1 || bindings[0].stride_words < 3) {
    return false;
  }
  xenos::xe_gpu_vertex_fetch_t vf = regs.GetVertexFetch(bindings[0].fetch_constant);
  const uint32_t stride = bindings[0].stride_words;
  if (vf.type != xenos::FetchConstantType::kVertex || vf.size < stride * 4) {
    if (REXCVAR_GET(d3d12_debug_pink_probe)) {
      static int logs = 0;
      if (logs++ < 10) {
        REXGPU_INFO("subpixel sprite check: fetch type {} size {} stride {}", uint32_t(vf.type),
                    uint32_t(vf.size), stride);
      }
    }
    return false;
  }
  const uint32_t* data = memory_->TranslatePhysical<const uint32_t*>(vf.address << 2);
  const float x_scale = regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_XSCALE);
  const float y_scale = regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YSCALE);
  float x_min = FLT_MAX, x_max = -FLT_MAX, y_min = FLT_MAX, y_max = -FLT_MAX;
  for (uint32_t v = 0; v < vf.size / stride; ++v) {
    float in[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    for (uint32_t k = 0; k < 3; ++k) {
      uint32_t word = xenos::GpuSwap(data[v * stride + k], vf.endian);
      std::memcpy(&in[k], &word, sizeof(float));
    }
    float out[4];
    for (uint32_t row = 0; row < 4; ++row) {
      out[row] = 0.0f;
      for (uint32_t k = 0; k < 4; ++k) {
        out[row] += in[k] * regs.Get<float>(XE_GPU_REG_SHADER_CONSTANT_000_X + 4 * row + k);
      }
    }
    if (!(out[3] > 0.0f)) {
      return false;  // Clipped or degenerate: leave it to the GPU.
    }
    const float x = out[0] / out[3] * x_scale, y = out[1] / out[3] * y_scale;
    x_min = std::min(x_min, x);
    x_max = std::max(x_max, x);
    y_min = std::min(y_min, y);
    y_max = std::max(y_max, y);
  }
  if (REXCVAR_GET(d3d12_debug_pink_probe)) {
    static int logs = 0;
    if (x_max - x_min < 2.0f && y_max - y_min < 2.0f && logs++ < 40) {
      REXGPU_INFO("subpixel sprite check: {:.3f}-{:.3f} x {:.3f}-{:.3f} ({} vertices)", x_min,
                  x_max, y_min, y_max, vf.size / stride);
    }
  }
  return x_max - x_min < 1.0f && y_max - y_min < 1.0f;
}

bool D3D12CommandProcessor::Amped3ScreenExtents(float extents_out[4]) {
  Shader* vertex_shader = active_vertex_shader();
  return vertex_shader && render_target_cache_ &&
         render_target_cache_->amped3_draw_extent_estimator().Amped3EstimateScreenExtents(
             *vertex_shader, extents_out);
}

Shader* D3D12CommandProcessor::LoadShader(xenos::ShaderType shader_type, uint32_t guest_address,
                                          const uint32_t* host_address, uint32_t dword_count) {
  return pipeline_cache_->LoadShader(shader_type, host_address, dword_count);
}

uint32_t D3D12CommandProcessor::GetLodFreeFetchMask(const D3D12Shader& pixel_shader) {
  // The previous draw's result if nothing it depends on changed (most draws).
  // RequestTextures below is still called after a texture became outdated,
  // as ResetTextureBindings changes the generation.
  // (The generation is read after RequestTextures in the computation below,
  // which may itself reset the bindings.)
  if (lod_free_cache_shader_ == &pixel_shader &&
      lod_free_cache_generation_ == texture_cache_->fetch_constants_0_11_generation() &&
      !texture_cache_->IsAnyTextureOutdatedPending()) {
    return lod_free_cache_mask_;
  }
  const uint32_t mask = ComputeLodFreeFetchMask(pixel_shader);
  lod_free_cache_shader_ = &pixel_shader;
  lod_free_cache_generation_ = texture_cache_->fetch_constants_0_11_generation();
  lod_free_cache_mask_ = mask;
  return mask;
}

uint32_t D3D12CommandProcessor::ComputeLodFreeFetchMask(const D3D12Shader& pixel_shader) {
  // Texture fetches whose result can't depend on the level of detail: 1D / 2D,
  // a single mip level in the fetch constant and in the texture actually bound
  // (a texture pack replacement may have more), the same minification and
  // magnification filter (only one level - the level of detail would only
  // choose between them), no anisotropic filtering (it uses the gradients
  // even with one level). As in D3D12TextureCache::GetSamplerParameters.
  const RegisterFile& regs = *register_file_;
  uint32_t candidates = 0, lod_dependent = 0;
  for (const Shader::TextureBinding& binding : pixel_shader.texture_bindings()) {
    const uint32_t fetch_constant = binding.fetch_constant;
    if (fetch_constant >= 12) {
      continue;
    }
    const ParsedTextureFetchInstruction& fetch_instr = binding.fetch_instr;
    if (fetch_instr.opcode != ucode::FetchOpcode::kTextureFetch) {
      continue;
    }
    const uint32_t fetch_constant_bit = uint32_t(1) << fetch_constant;
    bool lod_free = fetch_instr.dimension == xenos::FetchOpDimension::k1D ||
                    fetch_instr.dimension == xenos::FetchOpDimension::k2D;
    if (lod_free) {
      xenos::xe_gpu_texture_fetch_t fetch = regs.GetTextureFetch(fetch_constant);
      xenos::TextureFilter mag_filter =
          fetch_instr.attributes.mag_filter == xenos::TextureFilter::kUseFetchConst
              ? fetch.mag_filter
              : fetch_instr.attributes.mag_filter;
      xenos::TextureFilter min_filter =
          fetch_instr.attributes.min_filter == xenos::TextureFilter::kUseFetchConst
              ? fetch.min_filter
              : fetch_instr.attributes.min_filter;
      xenos::AnisoFilter aniso_filter =
          fetch_instr.attributes.aniso_filter == xenos::AnisoFilter::kUseFetchConst
              ? fetch.aniso_filter
              : fetch_instr.attributes.aniso_filter;
      uint32_t mip_min_level, mip_max_level;
      texture_util::GetSubresourcesFromFetchConstant(fetch, nullptr, nullptr, nullptr, nullptr,
                                                     nullptr, &mip_min_level, &mip_max_level);
      lod_free = mip_max_level == 0 && mip_min_level == 0 &&
                 aniso_filter == xenos::AnisoFilter::kDisabled &&
                 (mag_filter == xenos::TextureFilter::kLinear) ==
                     (min_filter == xenos::TextureFilter::kLinear);
    }
    if (lod_free) {
      candidates |= fetch_constant_bit;
    } else {
      lod_dependent |= fetch_constant_bit;
    }
  }
  candidates &= ~lod_dependent;
  if (!candidates) {
    return 0;
  }
  // The textures that will be bound (requesting them is incremental - the
  // draw's own request later only adds the rest).
  SetGpuProfilePass(rex::perf::GpuPass::kTextureLoad);
  texture_cache_->RequestTextures(candidates);
  SetGpuProfilePass(rex::perf::GpuPass::kOther);
  uint32_t lod_free_mask = 0;
  uint32_t fetch_constant;
  while (rex::bit_scan_forward(candidates, &fetch_constant)) {
    candidates &= ~(uint32_t(1) << fetch_constant);
    if (texture_cache_->IsActiveTextureSingleLevel(fetch_constant)) {
      lod_free_mask |= uint32_t(1) << fetch_constant;
    }
  }
  return lod_free_mask;
}


// --ui_scale: a UI sprite's extent on screen (NDC min x, max x, min y, max y)
// from its vertex buffer's positions (the first attribute, float3) through
// its matrix (c0, c1, c3); the vertices read, 0 if it can't tell.
int D3D12CommandProcessor::UiSpriteExtents(const Shader& vertex_shader, float extents[4]) {
  extents[0] = extents[2] = 1e9f;
  extents[1] = extents[3] = -1e9f;
  const auto& bindings = vertex_shader.vertex_bindings();
  if (bindings.empty() || bindings[0].attributes.empty()) return 0;
  const auto& binding = bindings[0];
  const auto& attr = binding.attributes[0].fetch_instr.attributes;
  if (attr.data_format != xenos::VertexFormat::k_32_32_32_FLOAT &&
      attr.data_format != xenos::VertexFormat::k_32_32_32_32_FLOAT) {
    return 0;
  }
  const auto& regs = *register_file_;
  xenos::xe_gpu_vertex_fetch_t vf = regs.GetVertexFetch(binding.fetch_constant);
  if (!binding.stride_words) return 0;
  const uint32_t count = std::min(uint32_t(vf.size) / binding.stride_words, 4096u);
  const uint32_t* data = memory_->TranslatePhysical<const uint32_t*>(vf.address << 2);
  auto c = [&](uint32_t i) { return regs.Get<float>(XE_GPU_REG_SHADER_CONSTANT_000_X + i); };
  for (uint32_t v = 0; v < count; ++v) {
    float p[3];
    for (uint32_t k = 0; k < 3; ++k) {
      const uint32_t word = xenos::GpuSwap(data[v * binding.stride_words + attr.offset + k], vf.endian);
      std::memcpy(&p[k], &word, 4);
    }
    const float w = c(12) * p[0] + c(13) * p[1] + c(14) * p[2] + c(15);
    if (w <= 0.0f) continue;
    const float x = (c(0) * p[0] + c(1) * p[1] + c(2) * p[2] + c(3)) / w;
    const float y = (c(4) * p[0] + c(5) * p[1] + c(6) * p[2] + c(7)) / w;
    extents[0] = std::min(extents[0], x);
    extents[1] = std::max(extents[1], x);
    extents[2] = std::min(extents[2], y);
    extents[3] = std::max(extents[3], y);
  }
  return int(count);
}

bool D3D12CommandProcessor::IssueDraw(xenos::PrimitiveType primitive_type, uint32_t index_count,
                                      IndexBufferInfo* index_buffer_info,
                                      bool major_mode_explicit) {
  g_draw_timing_on = REXCVAR_GET(d3d12_draw_timing) && REXCVAR_GET(d3d12_draw_timing_sections);
  if (g_draw_timing_on) {
    g_draw_timing_last = __rdtsc();
  }
  g_draw_timing_draws += REXCVAR_GET(d3d12_draw_timing);
  // --d3d12_submit_every_draws: a submission boundary every N draws.
  {
    static uint32_t draws_since_submit = 0;
    const int32_t every = REXCVAR_GET(d3d12_submit_every_draws);
    if (every > 0 && ++draws_since_submit >= uint32_t(every)) {
      draws_since_submit = 0;
      if (submission_open_ && CanEndSubmissionImmediately()) EndSubmission(false);
    }
  }
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  ID3D12Device* device = GetD3D12Provider().GetDevice();
  const RegisterFile& regs = *register_file_;
  ++issued_draw_count_;
  screen_extents_draw_.slot = UINT32_MAX;
  tile_replay_current_ = -1;

  xenos::EdramMode edram_mode = regs.Get<reg::RB_MODECONTROL>().edram_mode;
  if (edram_mode == xenos::EdramMode::kCopy) {
    // Special copy handling.
    uint64_t copy_start = __rdtsc();
    bool copy_result = IssueCopy();
    g_amped3_copy_tsc += __rdtsc() - copy_start;
    return copy_result;
  }

  // (After the copies: skipping those would freeze the picture.)
  if (REXCVAR_GET(d3d12_debug_prim_stats)) {
    uint32_t type = uint32_t(primitive_type) & 63;
    ++g_prim_counts[type];
    uint32_t draw_index = g_frame_draw_index++;
    if (((g_prim_skip_mask >> type) & 1) ||
        (draw_index >= g_skip_first && draw_index < g_skip_end)) {
      return true;
    }
  }

  bool surface_pitch_is_zero = regs.Get<reg::RB_SURFACE_INFO>().surface_pitch == 0;

  // Vertex shader analysis.
  auto vertex_shader = static_cast<D3D12Shader*>(active_vertex_shader());
  if (!vertex_shader) {
    // Always need a vertex shader.
    return false;
  }
  pipeline_cache_->AnalyzeShaderUcode(*vertex_shader);
  bool memexport_used_vertex = vertex_shader->memexport_eM_written() != 0;

  // Pixel shader analysis.
  bool primitive_polygonal = draw_util::IsPrimitivePolygonal(regs);
  bool is_rasterization_done = draw_util::IsRasterizationPotentiallyDone(regs, primitive_polygonal);
  if (surface_pitch_is_zero && is_rasterization_done) {
    // Doesn't actually draw.
    // Unlikely that zero would even really be legal though.
    return true;
  }
  D3D12Shader* pixel_shader = nullptr;
  if (is_rasterization_done) {
    // See xenos::EdramMode for explanation why the pixel shader is only used
    // when it's kColorDepth here.
    if (edram_mode == xenos::EdramMode::kColorDepth) {
      pixel_shader = static_cast<D3D12Shader*>(active_pixel_shader());
      if (pixel_shader) {
        pipeline_cache_->AnalyzeShaderUcode(*pixel_shader);
        if (!draw_util::IsPixelShaderNeededWithRasterization(*pixel_shader, regs)) {
          pixel_shader = nullptr;
        }
      }
    }
  } else {
    // Disabling pixel shader for this case is also required by the pipeline
    // cache.
    if (!memexport_used_vertex) {
      // This draw has no effect.
      return true;
    }
  }
  if (REXCVAR_GET(d3d12_debug_prim_stats) && primitive_type == xenos::PrimitiveType::kTriangleList) {
    uint64_t ps_hash = pixel_shader ? pixel_shader->ucode_data_hash() : 0;
    g_ps_seen.insert(ps_hash);
    if (g_ps_skip.count(ps_hash)) {
      return true;
    }
  }
  if (REXCVAR_GET(amped3_cull_subpixel_sprites) && Amped3IsSubpixelSprite(*vertex_shader)) {
    return true;
  }
  if (REXCVAR_GET(amped3_hide_hud)) {
    const uint64_t vs_hash = vertex_shader->ucode_data_hash();
    if (vs_hash == 0xCB134112C994DF08ull || vs_hash == 0x50BC47DED5DB7325ull) {
      return true;
    }
  }
  // DLSS (see D3D12Dlss): the scene draws - depth test on and not always
  // passing, into the frontbuffer-wide render target - and the first HUD draw
  // after them, where the scene is handed over: for Amped 3, the first draw
  // with its HUD vertex shaders (sprites and text).
  ultrawide_draw_ = false;
  ui_scale_draw_ = false;
  // --ui_scale: the UI on the frame (its three vertex shaders: sprites, text,
  // and shapes/outlined text placed by their vertices) shrinks or grows about the
  // frame's centre, so a layout made of several draws keeps its shape. Full-screen
  // layers (menu backgrounds, fades: covering the whole frame, from their
  // vertices) keep their size.
  if (std::abs(REXCVAR_GET(ui_scale) - 1.0) > 0.001 && is_rasterization_done && !video_frame_ &&
      !presenter_video_frame_) {
    const uint64_t vs = vertex_shader->ucode_data_hash();
    const uint32_t frontbuffer_width = rex::perf::GetRenderInfo().frontbuffer_width;
    static const auto log_start = std::chrono::steady_clock::now();
    if (REXCVAR_GET(ultrawide_pin_log) > 0 && frontbuffer_width &&
        std::chrono::steady_clock::now() - log_start > std::chrono::seconds(REXCVAR_GET(ui_scale_log_after)) &&
        regs.Get<reg::RB_SURFACE_INFO>().surface_pitch >= frontbuffer_width) {
      auto c = [&](uint32_t i) { return regs.Get<float>(XE_GPU_REG_SHADER_CONSTANT_000_X + i); };
      float ext[4];
      const int verts = UiSpriteExtents(*vertex_shader, ext);
      REXGPU_INFO("ui_scale_log: vs {:016X} ps {:016X} c0 {:.3f} {:.3f} {:.3f} {:.3f} c1 {:.3f} {:.3f} {:.3f} {:.3f} c3 {:.3f} {:.3f} {:.3f} {:.3f} prim {} count {} verts {} ext {:.2f} {:.2f} {:.2f} {:.2f}",
                  vs, pixel_shader ? pixel_shader->ucode_data_hash() : 0, c(0), c(1), c(2), c(3), c(4), c(5), c(6),
                  c(7), c(12), c(13), c(14), c(15), uint32_t(primitive_type), index_count, verts, ext[0], ext[1], ext[2], ext[3]);
      rex::cvar::SetFlagByName("ultrawide_pin_log", std::to_string(REXCVAR_GET(ultrawide_pin_log) - 1));
    }
    if ((vs == 0xCB134112C994DF08ull || vs == 0x50BC47DED5DB7325ull || vs == 0x4607896AFDF26D1Bull ||
         vs == 0x3A233AC80251CB6Aull) &&
        frontbuffer_width && regs.Get<reg::RB_SURFACE_INFO>().surface_pitch >= frontbuffer_width) {
      float e[4];
      const bool full_screen =
          UiSpriteExtents(*vertex_shader, e) && e[0] <= -0.95f && e[1] >= 0.95f && e[2] <= -0.95f && e[3] >= 0.95f;
      ui_scale_draw_ = !full_screen;
    }
  }
  if ((REXCVAR_GET(ultrawide_squeeze) != 1.0 || REXCVAR_GET(tall_squeeze) != 1.0) && is_rasterization_done) {
    const uint32_t frontbuffer_width = rex::perf::GetRenderInfo().frontbuffer_width;
    const uint64_t vs_hash = vertex_shader->ucode_data_hash();
    // (4607...: text and the HUD's numbers, e.g. the score and the radio's.
    // 3A23...: the score's second layer, the yellow of a banked score:
    // unsqueezed it sat off the pink one, a second score over the first.)
    const bool ui_shader = vs_hash == 0xCB134112C994DF08ull || vs_hash == 0x50BC47DED5DB7325ull ||
                           vs_hash == 0x4607896AFDF26D1Bull || vs_hash == 0x3A233AC80251CB6Aull;
    // Amped 3's pre-rendered videos (attract, intro, title background): kept
    // 16:9, with bars (squeezing their quads doesn't work: they're drawn in
    // more than one pass).
    // (6EAD12...: the title screen's moving background.)
    const bool video_shader = vs_hash == 0xDA562C2172F5F825ull || vs_hash == 0x6EAD12298B35F2F8ull;
    const reg::RB_DEPTHCONTROL depth_control = draw_util::GetNormalizedDepthControl(regs);
    const bool scene_geometry = depth_control.z_enable && depth_control.zfunc != xenos::CompareFunction::kAlways;
    // A frame showing a video is presented letterboxed instead, unsqueezed.
    if (video_shader) video_frame_ = true;
    // A loading screen (its picture, 84E84A..., in a frame without 3D) too, so
    // it keeps 16:9 between the menus' and the story's videos instead of going
    // from bars to stretched and back. (The game draws that shader in rides
    // as well, with the scene.)
    if (vs_hash == 0x84E84A34660A2FD1ull) picture_frame_ = true;
    if (scene_geometry) scene_frame_ = true;
    // (Also while the last frame was one: the title screen draws its UI
    // before its moving background.)
    ultrawide_draw_ = !video_frame_ && !presenter_video_frame_ && frontbuffer_width &&
                      regs.Get<reg::RB_SURFACE_INFO>().surface_pitch >= frontbuffer_width &&
                      (scene_geometry || ui_shader);
    // The UI sprite's extents across the 16:9 frame (-1..1), from its
    // vertices: the UI vertex shaders place it with a 4x4 matrix in c0-c3,
    // and text sets its glyphs' positions in the vertices.
    float e[4];
    ultrawide_ui_draw_ = ultrawide_draw_ && ui_shader;
    const bool have_extents = ultrawide_draw_ && ui_shader && UiSpriteExtents(*vertex_shader, e) && e[0] <= e[1];
    // UI parked beside the 16:9 frame (the radio popup and others while
    // hidden, or before they slide in) is off a 16:9 screen; squeezed, it
    // would show in the wider screen's sides.
    if (have_extents && (e[1] <= -0.999f || e[0] >= 0.999f)) {
      ultrawide_draw_ = false;
      return true;
    }
    // UI covering the whole frame (the fades and flashes of a wipeout or a
    // reset) covers the whole screen: squeezed, it darkened only the 16:9
    // band, and the strips above and below flickered like bars.
    if (have_extents && e[0] <= -0.95f && e[1] >= 0.95f && e[2] <= -0.95f && e[3] >= 0.95f) {
      ultrawide_draw_ = false;
      ultrawide_ui_draw_ = false;
    }
    // The HUD at the screen's edges while riding (the game sets the flag):
    // a sprite whose middle lands in the left / right third of the frame
    // (c0.w, the matrix's origin, if its vertices can't be read).
    ultrawide_pin_ = 0.0f;
    if (ultrawide_draw_ && ui_shader && REXCVAR_GET(ultrawide_hud_pin)) {
      const float x = have_extents ? 0.5f * (e[0] + e[1]) : regs.Get<float>(XE_GPU_REG_SHADER_CONSTANT_000_X + 3);
      ultrawide_pin_ = x < -1.0f / 3.0f ? -1.0f : x > 1.0f / 3.0f ? 1.0f : 0.0f;
    }
    if (REXCVAR_GET(ultrawide_pin_log) > 0 && frontbuffer_width &&
        regs.Get<reg::RB_SURFACE_INFO>().surface_pitch >= frontbuffer_width) {
      auto c = [&](uint32_t i) { return regs.Get<float>(XE_GPU_REG_SHADER_CONSTANT_000_X + i); };
      REXGPU_INFO("ultrawide_pin_log: vs {:016X} ps {:016X} squeezed {} depth {} c0 {:.3f} {:.3f} {:.3f} {:.3f} c1 {:.3f} {:.3f} {:.3f} {:.3f} pin {}",
                  vs_hash, pixel_shader ? pixel_shader->ucode_data_hash() : 0, ultrawide_draw_, scene_geometry,
                  c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), ultrawide_pin_);
      rex::cvar::SetFlagByName("ultrawide_pin_log", std::to_string(REXCVAR_GET(ultrawide_pin_log) - 1));
    }
  }
  bool dlss_take_scene = false;
  dlss_scene_draw_ = false;
  if (dlss_ && is_rasterization_done) {
    const reg::RB_DEPTHCONTROL depth_control = draw_util::GetNormalizedDepthControl(regs);
    const uint32_t frontbuffer_width = rex::perf::GetRenderInfo().frontbuffer_width;
    const uint64_t vs_hash = vertex_shader->ucode_data_hash();
    const bool hud_shader = vs_hash == 0xCB134112C994DF08ull || vs_hash == 0x50BC47DED5DB7325ull;
    g_dlss_stats.fb_width = frontbuffer_width;
    if (DlssTracing()) {
      const auto rb_color_info = regs.Get<reg::RB_COLOR_INFO>();
      REXGPU_INFO("trace draw: pitch {} vs {:016X} ps {:016X} z {} func {} zwrite {} color base {} fmt {} "
                  "mask {:X} blend {:08X}",
                  regs.Get<reg::RB_SURFACE_INFO>().surface_pitch, vs_hash,
                  pixel_shader ? pixel_shader->ucode_data_hash() : 0, int(depth_control.z_enable),
                  int(depth_control.zfunc), int(depth_control.z_write_enable), rb_color_info.color_base,
                  int(rb_color_info.color_format), regs[XE_GPU_REG_RB_COLOR_MASK],
                  regs[XE_GPU_REG_RB_BLENDCONTROL0]);
    }
    if (frontbuffer_width && regs.Get<reg::RB_SURFACE_INFO>().surface_pitch >= frontbuffer_width) {
      ++g_dlss_stats.wide;
    } else {
      ++g_dlss_stats.narrow;
    }
    const uint32_t pitch = regs.Get<reg::RB_SURFACE_INFO>().surface_pitch;
    // Amped 3 draws the 3D scene into its tiled surface (3 columns, pitch 480;
    // pitch 1280 once joined by --gpu_tile_collapse).
    const bool scene_surface = frontbuffer_width && pitch * 3 >= frontbuffer_width;
    if (scene_surface) {
      if (depth_control.z_enable && depth_control.zfunc != xenos::CompareFunction::kAlways && !hud_shader) {
        ++g_dlss_stats.scene;
        float camera_constants[16];
        for (uint32_t i = 0; i < 16; ++i) {
          camera_constants[i] = regs.Get<float>(XE_GPU_REG_SHADER_CONSTANT_000_X + i);
        }
        // The object: its vertex shader, indices and first vertex buffer.
        uint64_t object = vs_hash ^ (uint64_t(index_count) << 32);
        object = object * 0x9E3779B97F4A7C15ull ^
                 (index_buffer_info ? uint64_t(index_buffer_info->guest_base) : 0);
        object = object * 0x9E3779B97F4A7C15ull ^ regs[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0];
        object = object * 0x9E3779B97F4A7C15ull ^ regs[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_1];
        dlss_->OnSceneDraw(camera_constants, object);
        dlss_scene_draw_ = true;
      } else if (hud_shader) {
        // Only Amped 3's HUD shaders: its sky, fog and other full-screen 3D
        // passes also test depth with "always" and come before the scene ends.
        ++g_dlss_stats.hud;
        dlss_take_scene = REXCVAR_GET(dlss_handoff) == "hud" && dlss_->WantsScene();
        g_dlss_stats.taken += dlss_take_scene;
      }
    }
  }
  bool memexport_used_pixel = pixel_shader && (pixel_shader->memexport_eM_written() != 0);
  bool memexport_used = memexport_used_vertex || memexport_used_pixel;
  // amped3 debug (--d3d12_debug_ps_extents=<pixel shader hash>): the screen
  // rectangle of that shader's draws (the CPU-run vertex shader), to find a
  // stray one-pixel draw.
  if (!REXCVAR_GET(d3d12_debug_ps_extents).empty() && pixel_shader &&
      fmt::format("{:016X}", pixel_shader->ucode_data_hash()) ==
          REXCVAR_GET(d3d12_debug_ps_extents)) {
    static int logs = 0;
    // pink_skip.txt (next to the exe, re-read every 2 s): skip the draws of
    // this shader with these indices in the frame (bisecting a stray pixel).
    {
      static uint64_t last_check = 0;
      static uint32_t skip_lo = UINT32_MAX, skip_hi = 0;
      static uint32_t frame_index = 0, last_frame = 0;
      uint64_t now = GetTickCount64();
      if (now - last_check > 2000) {
        last_check = now;
        skip_lo = UINT32_MAX;
        skip_hi = 0;
        if (std::FILE* f = std::fopen("pink_skip.txt", "r")) {
          unsigned lo, hi;
          if (std::fscanf(f, "%u %u", &lo, &hi) == 2) {
            skip_lo = lo;
            skip_hi = hi;
          }
          std::fclose(f);
        }
      }
      uint32_t frame = uint32_t(GetCurrentFrame());
      if (frame != last_frame) {
        last_frame = frame;
        frame_index = 0;
      }
      uint32_t index = frame_index++;
      if (index >= skip_lo && index < skip_hi) {
        return true;
      }
      if (REXCVAR_GET(d3d12_debug_pink_probe)) {
        const uint32_t probe_index = g_pink_draws++;
        // The draws that make the pink pixel (indices 19-35): what they are.
        static int detail_logs = 0;
        if (detail_logs < 40 && g_pink_pink > 200) {
          ++detail_logs;
          auto initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();
          auto vtx_cntl = regs.Get<reg::PA_SU_VTX_CNTL>();
          std::string verts;
          for (const auto& binding : vertex_shader->vertex_bindings()) {
            xenos::xe_gpu_vertex_fetch_t vf = regs.GetVertexFetch(binding.fetch_constant);
            verts += fmt::format(" [fetch {} stride {} size {} endian {}:", binding.fetch_constant,
                                 binding.stride_words, uint32_t(vf.size), uint32_t(vf.endian));
            const uint32_t* data = memory_->TranslatePhysical<const uint32_t*>(vf.address << 2);
            for (uint32_t w = 0; w < std::min(binding.stride_words * 4, uint32_t(vf.size)); ++w) {
              uint32_t v = xenos::GpuSwap(data[w], vf.endian);
              float f;
              std::memcpy(&f, &v, 4);
              verts += fmt::format(" {:g}", f);
            }
            verts += "]";
          }
          {
            xenos::xe_gpu_texture_fetch_t tf = regs.GetTextureFetch(0);
            verts += fmt::format(" [tf0 dim {} fmt {} {}x{} base {:08X} mip {:08X} levels {}-{} "
                                 "clamp {} {} filter {} {} mip_filter {} swizzle {:03X} "
                                 "lod_bias {} exp_adjust {}]",
                                 uint32_t(tf.dimension), uint32_t(tf.format),
                                 uint32_t(tf.size_2d.width) + 1, uint32_t(tf.size_2d.height) + 1,
                                 uint32_t(tf.base_address) << 12, uint32_t(tf.mip_address) << 12,
                                 uint32_t(tf.mip_min_level), uint32_t(tf.mip_max_level),
                                 uint32_t(tf.clamp_x), uint32_t(tf.clamp_y),
                                 uint32_t(tf.min_filter), uint32_t(tf.mag_filter),
                                 uint32_t(tf.mip_filter), uint32_t(tf.swizzle),
                                 int32_t(tf.lod_bias), int32_t(tf.exp_adjust));
          }
          for (uint32_t c : {0u, 1u, 2u, 3u, 4u, 5u, 8u}) {
            verts += fmt::format(" c{}=", c);
            for (uint32_t k = 0; k < 4; ++k) {
              verts += fmt::format("{}{:g}", k ? "," : "",
                                   regs.Get<float>(XE_GPU_REG_SHADER_CONSTANT_000_X + 4 * c + k));
            }
          }
          REXGPU_INFO("pink_probe draw {}: prim {} source {} indices {} pix_center {} "
                      "vte {:08X} scale {:g} {:g} offset {:g} {:g} window offset {:08X} "
                      "scissor {:08X} {:08X} vs {:016X}{}",
                      probe_index, uint32_t(primitive_type), uint32_t(initiator.source_select),
                      index_count, uint32_t(vtx_cntl.pix_center), regs[XE_GPU_REG_PA_CL_VTE_CNTL],
                      regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_XSCALE),
                      regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YSCALE),
                      regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_XOFFSET),
                      regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YOFFSET),
                      regs[XE_GPU_REG_PA_SC_WINDOW_OFFSET], regs[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL],
                      regs[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR], vertex_shader->ucode_data_hash(),
                      verts);
          if (detail_logs == 1) {
            REXGPU_INFO("pink_probe: vertex shader:\n{}", vertex_shader->ucode_disassembly());
          }
        }
        if (probe_index < kPinkProbeIndices && ((g_pink_skip[probe_index >> 6] >> (probe_index & 63)) & 1)) {
          return true;
        }
      }
    }
    float e[4];
    bool measured = Amped3ScreenExtents(e);
    if (!measured && REXCVAR_GET(d3d12_debug_ps_extents_skip_unmeasured)) {
      static int unmeasured_logs = 0;
      if (unmeasured_logs++ < 20) {
        REXGPU_INFO("ps_extents: unmeasured draw prim {} indices {} vs {:016X} skipped",
                    uint32_t(primitive_type), index_count, vertex_shader->ucode_data_hash());
      }
      return true;
    }
    if (measured) {
      // Near the pink pixel at (527, 602): always (rate-limited per spot).
      if (e[0] <= 530 && e[1] >= 524 && e[2] <= 605 && e[3] >= 599 && e[1] - e[0] < 1000) {
        static int near_logs = 0;
        if (near_logs++ < 200) {
          REXGPU_INFO("ps_extents NEAR: prim {} indices {} x {:.3f}-{:.3f} y {:.3f}-{:.3f} vs {:016X}",
                      uint32_t(primitive_type), index_count, e[0], e[1], e[2], e[3],
                      vertex_shader->ucode_data_hash());
        }
      }
      if (logs < 3000) {
      ++logs;
      REXGPU_INFO("ps_extents: prim {} indices {} x {:.2f}-{:.2f} y {:.2f}-{:.2f} vs {:016X}",
                  uint32_t(primitive_type), index_count, e[0], e[1], e[2], e[3],
                  vertex_shader->ucode_data_hash());
      }
      static bool disassembly_logged = false;
      if (!disassembly_logged) {
        disassembly_logged = true;
        REXGPU_INFO("ps_extents: pixel shader:\n{}\nvertex shader:\n{}",
                    pixel_shader->ucode_disassembly(), vertex_shader->ucode_disassembly());
      }
      if (e[1] - e[0] > 1000 && REXCVAR_GET(d3d12_debug_ps_extents_skip_full)) {
        return true;
      }
      if (e[1] - e[0] > 1000) {
        // The full-screen one: its textures.
        uint32_t mask = pixel_shader->GetUsedTextureMaskAfterTranslation();
        uint32_t i;
        while (rex::bit_scan_forward(mask, &i)) {
          mask &= mask - 1;
          xenos::xe_gpu_texture_fetch_t f = regs.GetTextureFetch(i);
          REXGPU_INFO("ps_extents:   fetch {} dim {} format {} size {}x{} base {:08X} endian {} "
                      "swizzle {:03X} clamp {} {} filter {} {}",
                      i, uint32_t(f.dimension), uint32_t(f.format), uint32_t(f.size_2d.width) + 1,
                      uint32_t(f.size_2d.height) + 1, uint32_t(f.base_address) << 12, uint32_t(f.endianness),
                      uint32_t(f.swizzle), uint32_t(f.clamp_x), uint32_t(f.clamp_y),
                      uint32_t(f.min_filter), uint32_t(f.mag_filter));
        }
      }
    }
  }
  if (REXCVAR_GET(gpu_tile_replay) && amped3_tile_) {
    TileReplayEligible(vertex_shader, pixel_shader,
                       vertex_shader->GetUsedTextureMaskAfterTranslation() |
                           (pixel_shader ? pixel_shader->GetUsedTextureMaskAfterTranslation() : 0));
  }

  if (!BeginSubmission(true)) {
    return false;
  }

  SetGpuProfilePass(rex::perf::GpuPass::kOther);

  if (!memexport_used && !active_occlusion_query_.valid && REXCVAR_GET(gpu_clear_draws_in_place) &&
      render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets &&
      TryClearDrawInPlace(*vertex_shader, pixel_shader, draw_util::GetNormalizedDepthControl(regs))) {
    return true;
  }

  // amped3 tile replay: this tile 2-3 draw derives the same as tile 1's at the
  // same packet (gpu_tile_replay_verify), so take what that derived.
  const TileReplayRecord* replay = nullptr;
  if (tile_replay_current_ >= 0 && amped3_tile_ >= 2 && REXCVAR_GET(gpu_tile_replay_apply) &&
      !memexport_used && !active_occlusion_query_.valid && bindless_resources_used_ &&
      tile_replay_records_[tile_replay_current_].sampler_heap_generation ==
          tile_replay_sampler_heap_generation_) {
    replay = &tile_replay_records_[tile_replay_current_];
  }

  DrawTimingMark(kDrawTimingSetup);
  // Process primitives.
  PrimitiveProcessor::ProcessingResult primitive_processing_result;
  if (replay) {
    primitive_processing_result = replay->primitive;
    if (primitive_processing_result.index_buffer_type ==
        PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA) {
      uint32_t index_size = primitive_processing_result.host_index_format ==
                                    xenos::IndexFormat::kInt16
                                ? sizeof(uint16_t)
                                : sizeof(uint32_t);
      if (!shared_memory_->RequestRange(primitive_processing_result.guest_index_base,
                                        primitive_processing_result.host_draw_vertex_count *
                                            index_size)) {
        return false;
      }
    }
  } else if (!primitive_processor_->Process(primitive_processing_result)) {
    return false;
  }
  DrawTimingMark(kDrawTimingProcess);
  if (!primitive_processing_result.host_draw_vertex_count) {
    // Nothing to draw.
    return true;
  }

  reg::RB_DEPTHCONTROL normalized_depth_control = draw_util::GetNormalizedDepthControl(regs);

  // Shader modifications (not needed when replaying).
  DxbcShaderTranslator::Modification vertex_shader_modification(0);
  DxbcShaderTranslator::Modification pixel_shader_modification(0);
  bool measure_screen_extents = false;
  auto compute_shader_modifications = [&]() {
  uint32_t ps_param_gen_pos = UINT32_MAX;
  uint32_t interpolator_mask =
      pixel_shader ? (vertex_shader->writes_interpolators() &
                      pixel_shader->GetInterpolatorInputMask(regs.Get<reg::SQ_PROGRAM_CNTL>(),
                                                             regs.Get<reg::SQ_CONTEXT_MISC>(),
                                                             ps_param_gen_pos))
                   : 0;
  vertex_shader_modification =
      pipeline_cache_->GetCurrentVertexShaderModification(
          *vertex_shader, primitive_processing_result.host_vertex_shader_type, interpolator_mask);
  // amped3 (gpu_screen_extents=3): measure the screen extents of the draws in
  // the first tile's pass. Only for triangles drawn directly from the guest
  // vertex shader's positions (no point sprites, rectangles or tessellation).
  if (amped3_extents_pass_ && screen_extents_slots_used_ < kScreenExtentsSlots &&
      REXCVAR_GET(gpu_screen_extents) >= 3 &&
      primitive_processing_result.host_vertex_shader_type ==
          Shader::HostVertexShaderType::kVertex) {
    switch (primitive_type) {
      case xenos::PrimitiveType::kTriangleList:
      case xenos::PrimitiveType::kTriangleFan:
      case xenos::PrimitiveType::kTriangleStrip:
        measure_screen_extents = true;
        vertex_shader_modification.vertex.screen_extents = 1;
        break;
      default:
        break;
    }
  }
  pixel_shader_modification =
      pixel_shader
          ? pipeline_cache_->GetCurrentPixelShaderModification(
                *pixel_shader, interpolator_mask, ps_param_gen_pos, normalized_depth_control)
          : DxbcShaderTranslator::Modification(0);
  if (pixel_shader && REXCVAR_GET(gpu_lod_free_texture_fetches)) {
    pixel_shader_modification.pixel.lod_free_fetch_mask = GetLodFreeFetchMask(*pixel_shader);
  }
  };
  if (!replay) {
    compute_shader_modifications();
  }

  // Set up the render targets - this may perform dispatches and draws.
  uint32_t normalized_color_mask =
      pixel_shader ? draw_util::GetNormalizedColorMask(regs, pixel_shader->writes_color_targets())
                   : 0;
  // Color render targets this draw overwrites with no dependency on their old
  // contents - the render target cache doesn't transfer old data into the
  // part it certainly covers (gpu_skip_overwritten_transfers).
  {
    uint32_t draw_overwritten_color_rts = 0;
    if (REXCVAR_GET(gpu_skip_overwritten_transfers) && pixel_shader && is_rasterization_done &&
        !memexport_used && !pixel_shader->kills_pixels() &&
        render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets &&
        (!normalized_depth_control.z_enable ||
         normalized_depth_control.zfunc == xenos::CompareFunction::kAlways) &&
        !normalized_depth_control.stencil_enable) {
      auto rb_colorcontrol = regs.Get<reg::RB_COLORCONTROL>();
      if ((!rb_colorcontrol.alpha_test_enable ||
           rb_colorcontrol.alpha_func == xenos::CompareFunction::kAlways) &&
          !rb_colorcontrol.alpha_to_mask_enable) {
        for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
          if (((normalized_color_mask >> (4 * i)) & 0b1111) != 0b1111) {
            continue;
          }
          auto blend_control =
              regs.Get<reg::RB_BLENDCONTROL>(reg::RB_BLENDCONTROL::rt_register_indices[i]);
          if (blend_control.color_srcblend == xenos::BlendFactor::kOne &&
              blend_control.color_destblend == xenos::BlendFactor::kZero &&
              blend_control.color_comb_fcn == xenos::BlendOp::kAdd &&
              blend_control.alpha_srcblend == xenos::BlendFactor::kOne &&
              blend_control.alpha_destblend == xenos::BlendFactor::kZero &&
              blend_control.alpha_comb_fcn == xenos::BlendOp::kAdd) {
            draw_overwritten_color_rts |= uint32_t(1) << i;
          }
        }
      }
    }
    render_target_cache_->SetDrawOverwrittenColorRenderTargets(draw_overwritten_color_rts);
  }
  SetGpuProfilePass(rex::perf::GpuPass::kRenderTargets);
  if (dlss_take_scene) {
    const rex::perf::RenderInfo render_info = rex::perf::GetRenderInfo();
    uint32_t scene_width = render_info.frontbuffer_width * texture_cache_->draw_resolution_scale_x();
    uint32_t scene_height =
        render_info.frontbuffer_height * texture_cache_->draw_resolution_scale_y();
    render_target_cache_->RequestSceneCallback(
        [this, scene_width, scene_height](ID3D12Resource* color, ID3D12Resource* depth,
                                          D3D12_CPU_DESCRIPTOR_HANDLE depth_srv) {
          (void)depth;
          return dlss_->ProcessScene(color, depth_srv, scene_width, scene_height);
        });
  }
  if (!render_target_cache_->Update(is_rasterization_done, normalized_depth_control,
                                    normalized_color_mask, *vertex_shader)) {
    return false;
  }
  DrawTimingMark(kDrawTimingProcessRt);

  // Create the pipeline (for this, need the actually used render target formats
  // from the render target cache), translating the shaders - doing this now to
  // obtain the used textures.
  uint32_t bound_depth_and_color_render_target_bits;
  uint32_t bound_depth_and_color_render_target_formats[1 + xenos::kMaxColorRenderTargets];
  bool host_render_targets_used =
      render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets;
  if (host_render_targets_used) {
    bound_depth_and_color_render_target_bits =
        render_target_cache_->GetLastUpdateBoundRenderTargets(
            bound_depth_and_color_render_target_formats);
  } else {
    bound_depth_and_color_render_target_bits = 0;
  }
  if (replay && (replay->render_target_bits != bound_depth_and_color_render_target_bits ||
                 std::memcmp(replay->render_target_formats,
                             bound_depth_and_color_render_target_formats,
                             sizeof(replay->render_target_formats)))) {
    // Different render targets than in tile 1: derive normally from here.
    ++tile_replay_fallbacks_[0];
    replay = nullptr;
    compute_shader_modifications();
  }
  D3D12Shader::D3D12Translation* vertex_shader_translation;
  D3D12Shader::D3D12Translation* pixel_shader_translation;
  void* pipeline_handle;
  ID3D12RootSignature* root_signature;
  if (replay) {
    vertex_shader_translation = static_cast<D3D12Shader::D3D12Translation*>(
        const_cast<void*>(replay->vertex_translation));
    pixel_shader_translation = static_cast<D3D12Shader::D3D12Translation*>(
        const_cast<void*>(replay->pixel_translation));
    pipeline_handle = replay->pipeline_handle;
    root_signature = replay->root_signature;
  } else {
    vertex_shader_translation = static_cast<D3D12Shader::D3D12Translation*>(
        vertex_shader->GetOrCreateTranslation(vertex_shader_modification.value));
    pixel_shader_translation =
        pixel_shader ? static_cast<D3D12Shader::D3D12Translation*>(
                           pixel_shader->GetOrCreateTranslation(pixel_shader_modification.value))
                     : nullptr;
    if (!pipeline_cache_->ConfigurePipeline(
            vertex_shader_translation, pixel_shader_translation, primitive_processing_result,
            normalized_depth_control, normalized_color_mask,
            bound_depth_and_color_render_target_bits, bound_depth_and_color_render_target_formats,
            &pipeline_handle, &root_signature)) {
      return false;
    }
  }
  DrawTimingMark(kDrawTimingPipeline);
  if (REXCVAR_GET(async_shader_compilation) &&
      pipeline_cache_->GetD3D12PipelineByHandle(pipeline_handle) == nullptr) {
    return true;
  }

  if (measure_screen_extents) {
    if (!screen_extents_slots_used_) {
      PushTransitionBarrier(screen_extents_buffer_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_COPY_DEST);
      SubmitBarriers();
      deferred_command_list_.D3DCopyBufferRegion(screen_extents_buffer_.Get(), 0,
                                                 screen_extents_zero_buffer_.Get(), 0,
                                                 kScreenExtentsSlots * sizeof(uint32_t) * 4);
      PushTransitionBarrier(screen_extents_buffer_.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      SubmitBarriers();
    }
    screen_extents_draw_.slot = screen_extents_slots_used_++;
    {
      // gpu_screen_extents=4: the draw's identity across frames.
      uint64_t key = uint64_t(reinterpret_cast<uintptr_t>(vertex_shader));
      key = key * 0x100000001B3ull ^ uint64_t(reinterpret_cast<uintptr_t>(pixel_shader));
      key = key * 0x100000001B3ull ^ regs[XE_GPU_REG_VGT_DMA_BASE];
      key = key * 0x100000001B3ull ^ regs[XE_GPU_REG_VGT_DMA_SIZE];
      key = key * 0x100000001B3ull ^ index_count;
      const Shader::ConstantRegisterMap& extents_constant_map =
          vertex_shader->constant_register_map();
      for (uint32_t i = 0; i < rex::countof(extents_constant_map.vertex_fetch_bitmap); ++i) {
        uint32_t j;
        if (rex::bit_scan_forward(extents_constant_map.vertex_fetch_bitmap[i], &j)) {
          key = key * 0x100000001B3ull ^ regs.GetVertexFetch(i * 32 + j).address;
          break;
        }
      }
      screen_extents_draw_.key = key;
    }
    // Guest X/W and Y/W to pixels, as DrawExtentEstimator::
    // Amped3EstimateScreenExtents.
    auto pa_cl_vte_cntl = regs.Get<reg::PA_CL_VTE_CNTL>();
    screen_extents_draw_.scale[0] =
        pa_cl_vte_cntl.vport_x_scale_ena ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_XSCALE) : 1.0f;
    screen_extents_draw_.scale[1] =
        pa_cl_vte_cntl.vport_y_scale_ena ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YSCALE) : 1.0f;
    screen_extents_draw_.offset[0] = pa_cl_vte_cntl.vport_x_offset_ena
                                         ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_XOFFSET)
                                         : 0.0f;
    screen_extents_draw_.offset[1] = pa_cl_vte_cntl.vport_y_offset_ena
                                         ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YOFFSET)
                                         : 0.0f;
    if (regs.Get<reg::PA_SU_SC_MODE_CNTL>().vtx_window_offset_enable) {
      auto pa_sc_window_offset = regs.Get<reg::PA_SC_WINDOW_OFFSET>();
      screen_extents_draw_.offset[0] += float(pa_sc_window_offset.window_x_offset);
      screen_extents_draw_.offset[1] += float(pa_sc_window_offset.window_y_offset);
    }
    if (screen_extents_root_slot_ != screen_extents_draw_.slot) {
      screen_extents_root_slot_ = screen_extents_draw_.slot;
      current_graphics_root_up_to_date_ &=
          ~(1u << (bindless_resources_used_ ? kRootParameter_Bindless_ScreenExtents
                                            : kRootParameter_Bindful_ScreenExtents));
    }
  }

  // Direct resolve data (D3D12TextureCache) that the vertex fetches and the
  // memory exports of this draw may read goes to the guest memory copy now:
  // from binding the pipeline on, the shared memory requests below must not
  // record dispatches or change the descriptor heap.
  if (texture_cache_->HasPendingDirectResolves()) {
    if (memexport_used) {
      std::pair<uint32_t, uint32_t> everything(0, SharedMemory::kBufferSize);
      texture_cache_->WriteOutDirectResolves(&everything, 1);
    } else {
      // Not value-initialized (std::pair would zero all 96 on every draw):
      // only the first vertex_range_count are written and read.
      union VertexRangeStorage {
        VertexRangeStorage() {}
        std::pair<uint32_t, uint32_t> ranges[96];
      } vertex_range_storage;
      auto& vertex_ranges = vertex_range_storage.ranges;
      size_t vertex_range_count = 0;
      const Shader::ConstantRegisterMap& vertex_constant_map =
          vertex_shader->constant_register_map();
      for (uint32_t i = 0; i < rex::countof(vertex_constant_map.vertex_fetch_bitmap); ++i) {
        uint32_t vfetch_bits_remaining = vertex_constant_map.vertex_fetch_bitmap[i];
        uint32_t j;
        while (rex::bit_scan_forward(vfetch_bits_remaining, &j)) {
          vfetch_bits_remaining &= ~(uint32_t(1) << j);
          xenos::xe_gpu_vertex_fetch_t vfetch_constant = regs.GetVertexFetch(i * 32 + j);
          if (vertex_range_count < rex::countof(vertex_ranges)) {
            vertex_ranges[vertex_range_count++] =
                std::make_pair(vfetch_constant.address << 2, vfetch_constant.size << 2);
          }
        }
      }
      texture_cache_->WriteOutDirectResolves(vertex_ranges, vertex_range_count);
    }
  }

  // Update the textures - this may bind pipelines.
  uint32_t used_texture_mask =
      vertex_shader->GetUsedTextureMaskAfterTranslation() |
      (pixel_shader != nullptr ? pixel_shader->GetUsedTextureMaskAfterTranslation() : 0);
  SetGpuProfilePass(rex::perf::GpuPass::kTextureLoad);
  texture_cache_->RequestTextures(used_texture_mask);
  SetGpuProfilePass(rex::perf::GpuPass::kOther);
  DrawTimingMark(kDrawTimingTextures);

  // Bind the pipeline after configuring it and doing everything that may bind
  // other pipelines.
  if (current_guest_pipeline_ != pipeline_handle) {
    deferred_command_list_.SetPipelineStateHandle(reinterpret_cast<void*>(pipeline_handle));
    current_guest_pipeline_ = pipeline_handle;
    current_external_pipeline_ = nullptr;
  }

  // Get dynamic rasterizer state.
  uint32_t draw_resolution_scale_x = texture_cache_->draw_resolution_scale_x();
  uint32_t draw_resolution_scale_y = texture_cache_->draw_resolution_scale_y();

  bool convert_z_to_float24 =
      host_render_targets_used && render_target_cache_->depth_float24_convert_in_pixel_shader();
  bool ps_writes_depth = pixel_shader && pixel_shader->writes_depth();

  // Build a cache key from all viewport-affecting state to skip redundant
  // recalculation when the viewport registers haven't changed between draws.
  ViewportCacheKey viewport_key;
  viewport_key.pa_cl_clip_cntl = regs[XE_GPU_REG_PA_CL_CLIP_CNTL];
  viewport_key.pa_cl_vte_cntl = regs[XE_GPU_REG_PA_CL_VTE_CNTL];
  viewport_key.pa_su_sc_mode_cntl = regs[XE_GPU_REG_PA_SU_SC_MODE_CNTL];
  viewport_key.pa_su_vtx_cntl = regs[XE_GPU_REG_PA_SU_VTX_CNTL];
  viewport_key.pa_sc_window_offset = regs[XE_GPU_REG_PA_SC_WINDOW_OFFSET];
  viewport_key.normalized_depth_control = normalized_depth_control.value;
  std::memcpy(viewport_key.vport_regs, &regs[XE_GPU_REG_PA_CL_VPORT_XSCALE],
              sizeof(viewport_key.vport_regs));
  viewport_key.flags = (uint32_t(convert_z_to_float24) << 0) |
                       (uint32_t(host_render_targets_used) << 1) | (uint32_t(ps_writes_depth) << 2);

  draw_util::ViewportInfo viewport_info;
  if (viewport_cache_valid_ && viewport_key == previous_viewport_key_) {
    viewport_info = previous_viewport_info_;
  } else {
    draw_util::GetHostViewportInfo(regs, draw_resolution_scale_x, draw_resolution_scale_y, true,
                                   D3D12_VIEWPORT_BOUNDS_MAX, D3D12_VIEWPORT_BOUNDS_MAX, false,
                                   normalized_depth_control, convert_z_to_float24,
                                   host_render_targets_used, ps_writes_depth, viewport_info);
    previous_viewport_key_ = viewport_key;
    previous_viewport_info_ = viewport_info;
    viewport_cache_valid_ = true;
  }

  draw_util::Scissor scissor;
  draw_util::GetScissor(regs, scissor);
  scissor.offset[0] *= draw_resolution_scale_x;
  scissor.offset[1] *= draw_resolution_scale_y;
  scissor.extent[0] *= draw_resolution_scale_x;
  scissor.extent[1] *= draw_resolution_scale_y;

  // Squeezed HUD is clipped to the 16:9 frame's columns while riding, as a
  // 16:9 screen clips it: HUD sliding in from beside the frame would show in
  // the wider screen's sides (menus aren't: their art running past the
  // frame looks better than cut). The squeeze scales host clip space about
  // the viewport's centre, so the frame's edges land at k * (+-ndc_scale +
  // ndc_offset). HUD pinned to the screen's edges isn't clipped.
  // Only draws whose viewport spans the frame: the frame's edges are found
  // from the viewport, so a HUD element drawn into a viewport of its own (the
  // score counter's numbers) had its own edges taken for the frame's and was
  // cut down to a sliver.
  const uint32_t frame_width = rex::perf::GetRenderInfo().frontbuffer_width * draw_resolution_scale_x;
  if (ultrawide_draw_ && ultrawide_ui_draw_ && ultrawide_pin_ == 0.0f && REXCVAR_GET(ultrawide_hud_clip) &&
      viewport_info.xy_extent[0] && (!frame_width || viewport_info.xy_extent[0] * 10 >= frame_width * 9)) {
    const float k = float(REXCVAR_GET(ultrawide_squeeze));
    auto column = [&](float edge) {
      const float ndc = k * (edge * viewport_info.ndc_scale[0] + viewport_info.ndc_offset[0]);
      return float(viewport_info.xy_offset[0]) + (ndc + 1.0f) * 0.5f * float(viewport_info.xy_extent[0]);
    };
    const float a = column(-1.0f), b = column(1.0f);
    const uint32_t lo = uint32_t(std::max(0.0f, std::floor(std::min(a, b))));
    const uint32_t hi = uint32_t(std::max(0.0f, std::ceil(std::max(a, b))));
    const uint32_t start = std::max(scissor.offset[0], lo);
    const uint32_t end = std::min(scissor.offset[0] + scissor.extent[0], hi);
    scissor.offset[0] = start;
    scissor.extent[0] = end > start ? end - start : 0;
  }

  // Rendering the scene below the output resolution (--scene_render_percent):
  // the scene's draws - on the scene surface, before its resolve - go into the
  // top-left part; passes that use the pixel position stay consistent.
  if (REXCVAR_GET(scene_render_percent) < 100 && g_dlss_scene_resolve &&
      g_dlss_wide_resolves < g_dlss_scene_resolve) {
    const uint32_t frontbuffer_width = rex::perf::GetRenderInfo().frontbuffer_width;
    if (frontbuffer_width && regs.Get<reg::RB_SURFACE_INFO>().surface_pitch * 3 >= frontbuffer_width) {
      const float f = REXCVAR_GET(scene_render_percent) / 100.0f;
      for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t end = viewport_info.xy_offset[i] + viewport_info.xy_extent[i];
        viewport_info.xy_offset[i] = uint32_t(viewport_info.xy_offset[i] * f);
        viewport_info.xy_extent[i] = uint32_t(std::ceil(end * f)) - viewport_info.xy_offset[i];
        const uint32_t scissor_end = scissor.offset[i] + scissor.extent[i];
        scissor.offset[i] = uint32_t(scissor.offset[i] * f);
        scissor.extent[i] = uint32_t(std::ceil(scissor_end * f)) - scissor.offset[i];
      }
    }
  }

  // Update viewport, scissor, blend factor and stencil reference.
  UpdateFixedFunctionState(viewport_info, scissor, primitive_polygonal, normalized_depth_control);
  DrawTimingMark(kDrawTimingViewport);

  // Update system constants before uploading them.
  // TODO(Triang3l): With ROV, pass the disabled render target mask for safety.
  UpdateSystemConstantValues(memexport_used, primitive_polygonal,
                             primitive_processing_result.line_loop_closing_index,
                             primitive_processing_result.host_shader_index_endian, viewport_info,
                             used_texture_mask, normalized_depth_control, normalized_color_mask);
  DrawTimingMark(kDrawTimingSysConst);

  // Update constant buffers, descriptors and root parameters.
  if (replay && !TileReplayBindings(*replay, vertex_shader, pixel_shader)) {
    ++tile_replay_fallbacks_[1];
    replay = nullptr;
  }
  if (replay) {
    ++tile_replay_replayed_;
  } else if (!UpdateBindings(vertex_shader, pixel_shader, root_signature, memexport_used)) {
    return false;
  }
  DrawTimingMark(kDrawTimingBindings);
  // Must not call anything that can change the descriptor heap from now on!
  struct DirectResolveWriteOutBlock {
    D3D12TextureCache& texture_cache;
    explicit DirectResolveWriteOutBlock(D3D12TextureCache& cache) : texture_cache(cache) {
      texture_cache.SetDirectResolveWriteOutBlocked(true);
    }
    ~DirectResolveWriteOutBlock() { texture_cache.SetDirectResolveWriteOutBlocked(false); }
  } direct_resolve_write_out_block(*texture_cache_);

  // Ensure vertex buffers are resident.
  SetGpuProfilePass(rex::perf::GpuPass::kSharedMemory);
  const Shader::ConstantRegisterMap& constant_map_vertex = vertex_shader->constant_register_map();
  for (uint32_t i = 0; i < rex::countof(constant_map_vertex.vertex_fetch_bitmap); ++i) {
    uint32_t vfetch_bits_remaining = constant_map_vertex.vertex_fetch_bitmap[i];
    uint32_t j;
    while (rex::bit_scan_forward(vfetch_bits_remaining, &j)) {
      vfetch_bits_remaining &= ~(uint32_t(1) << j);
      uint32_t vfetch_index = i * 32 + j;
      uint64_t vfetch_bit = uint64_t(1) << (vfetch_index & 63);
      if (vertex_buffers_in_sync_[vfetch_index >> 6] & vfetch_bit) {
        continue;
      }
      xenos::xe_gpu_vertex_fetch_t vfetch_constant = regs.GetVertexFetch(vfetch_index);
      switch (vfetch_constant.type) {
        case xenos::FetchConstantType::kVertex:
          break;
        case xenos::FetchConstantType::kInvalidVertex:
          if (REXCVAR_GET(gpu_allow_invalid_fetch_constants)) {
            break;
          }
          REXGPU_WARN(
              "Vertex fetch constant {} ({:08X} {:08X}) has \"invalid\" type! "
              "This is incorrect behavior, but you can try bypassing this by "
              "launching Xenia with --gpu_allow_invalid_fetch_constants=true.",
              vfetch_index, vfetch_constant.dword_0, vfetch_constant.dword_1);
          return false;
        default:
          REXGPU_WARN("Vertex fetch constant {} ({:08X} {:08X}) is completely invalid!",
                      vfetch_index, vfetch_constant.dword_0, vfetch_constant.dword_1);
          return false;
      }
      VertexBufferState& state = vertex_buffer_states_[vfetch_index];
      if (state.address == vfetch_constant.address && state.size == vfetch_constant.size) {
        vertex_buffers_in_sync_[vfetch_index >> 6] |= vfetch_bit;
        continue;
      }
      if (!shared_memory_->RequestRange(vfetch_constant.address << 2, vfetch_constant.size << 2)) {
        REXGPU_ERROR(
            "Failed to request vertex buffer at 0x{:08X} (size {}) in the "
            "shared memory",
            vfetch_constant.address << 2, vfetch_constant.size << 2);
        return false;
      }
      state.address = vfetch_constant.address;
      state.size = vfetch_constant.size;
      vertex_buffers_in_sync_[vfetch_index >> 6] |= vfetch_bit;
    }
  }

  DrawTimingMark(kDrawTimingVertexBuffers);
  // Gather memexport ranges and ensure the heaps for them are resident, and
  // also load the data surrounding the export and to fill the regions that
  // won't be modified by the shaders.
  memexport_ranges_.clear();
  if (memexport_used_vertex) {
    draw_util::AddMemExportRanges(regs, *vertex_shader, memexport_ranges_);
  }
  if (memexport_used_pixel) {
    draw_util::AddMemExportRanges(regs, *pixel_shader, memexport_ranges_);
  }
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    if (!shared_memory_->RequestRange(memexport_range.base_address_dwords << 2,
                                      memexport_range.size_bytes)) {
      REXGPU_ERROR(
          "Failed to request memexport stream at 0x{:08X} (size {}) in the "
          "shared memory",
          memexport_range.base_address_dwords << 2, memexport_range.size_bytes);
      return false;
    }
  }
  if (memexport_used && memexport_ranges_.empty()) {
    if (!shared_memory_->RequestRange(0, SharedMemory::kBufferSize)) {
      REXGPU_ERROR(
          "Failed to request full shared memory residency for unresolved "
          "memexport destinations");
      return false;
    }
  }
  texture_cache_->SetDirectResolveWriteOutBlocked(false);

  // Primitive topology.
  D3D_PRIMITIVE_TOPOLOGY primitive_topology;
  if (primitive_processing_result.IsTessellated()) {
    switch (primitive_processing_result.host_primitive_type) {
      // TODO(Triang3l): Support all primitive types.
      case xenos::PrimitiveType::kTriangleList:
        primitive_topology = D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST;
        break;
      case xenos::PrimitiveType::kQuadList:
        primitive_topology = D3D_PRIMITIVE_TOPOLOGY_4_CONTROL_POINT_PATCHLIST;
        break;
      case xenos::PrimitiveType::kTrianglePatch:
        primitive_topology =
            (regs.Get<reg::VGT_HOS_CNTL>().tess_mode == xenos::TessellationMode::kAdaptive)
                ? D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST
                : D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST;
        break;
      case xenos::PrimitiveType::kQuadPatch:
        primitive_topology =
            (regs.Get<reg::VGT_HOS_CNTL>().tess_mode == xenos::TessellationMode::kAdaptive)
                ? D3D_PRIMITIVE_TOPOLOGY_4_CONTROL_POINT_PATCHLIST
                : D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST;
        break;
      default:
        REXGPU_ERROR(
            "Host tessellated primitive type {} returned by the primitive "
            "processor is not supported by the Direct3D 12 command processor",
            uint32_t(primitive_processing_result.host_primitive_type));
        assert_unhandled_case(primitive_processing_result.host_primitive_type);
        return false;
    }
  } else {
    switch (primitive_processing_result.host_primitive_type) {
      case xenos::PrimitiveType::kPointList:
        primitive_topology = D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
        break;
      case xenos::PrimitiveType::kLineList:
        primitive_topology = D3D_PRIMITIVE_TOPOLOGY_LINELIST;
        break;
      case xenos::PrimitiveType::kLineStrip:
        primitive_topology = D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
        break;
      case xenos::PrimitiveType::kTriangleList:
      case xenos::PrimitiveType::kRectangleList:
        primitive_topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        break;
      case xenos::PrimitiveType::kTriangleStrip:
        primitive_topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
        break;
      case xenos::PrimitiveType::kQuadList:
        primitive_topology = D3D_PRIMITIVE_TOPOLOGY_LINELIST_ADJ;
        break;
      default:
        REXGPU_ERROR(
            "Host primitive type {} returned by the primitive processor is not "
            "supported by the Direct3D 12 command processor",
            uint32_t(primitive_processing_result.host_primitive_type));
        assert_unhandled_case(primitive_processing_result.host_primitive_type);
        return false;
    }
  }
  rex::perf::GpuPass draw_profile_pass =
      memexport_used ? rex::perf::GpuPass::kDrawMemexport : rex::perf::GpuPass::kDraw;
  SetGpuProfilePass(draw_profile_pass);
  if (rex::perf::IsGpuProfiling()) {
    rex::perf::GetGpuProfileStats().draws.fetch_add(1, std::memory_order_relaxed);
  }
  SetPrimitiveTopology(primitive_topology);
  // Must not call anything that may change the primitive topology from now on!

  // Draw.
  if (primitive_processing_result.index_buffer_type ==
      PrimitiveProcessor::ProcessedIndexBufferType::kNone) {
    if (memexport_used) {
      shared_memory_->UseForWriting();
    } else {
      shared_memory_->UseForReading();
    }
    SubmitBarriers();
    PROFILE_DRAW_CALL();
    PROFILE_VERTICES(primitive_processing_result.host_draw_vertex_count);
    deferred_command_list_.D3DDrawInstanced(primitive_processing_result.host_draw_vertex_count, 1,
                                            0, 0);
  } else {
    D3D12_INDEX_BUFFER_VIEW index_buffer_view;
    index_buffer_view.SizeInBytes = primitive_processing_result.host_draw_vertex_count;
    if (primitive_processing_result.host_index_format == xenos::IndexFormat::kInt16) {
      index_buffer_view.SizeInBytes *= sizeof(uint16_t);
      index_buffer_view.Format = DXGI_FORMAT_R16_UINT;
    } else {
      index_buffer_view.SizeInBytes *= sizeof(uint32_t);
      index_buffer_view.Format = DXGI_FORMAT_R32_UINT;
    }
    ID3D12Resource* scratch_index_buffer = nullptr;
    switch (primitive_processing_result.index_buffer_type) {
      case PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA: {
        if (memexport_used) {
          // If the shared memory is a UAV, it can't be used as an index buffer
          // (UAV is a read/write state, index buffer is a read-only state).
          // Need to copy the indices to a buffer in the index buffer state.
          scratch_index_buffer = RequestScratchGPUBuffer(index_buffer_view.SizeInBytes,
                                                         D3D12_RESOURCE_STATE_COPY_DEST);
          if (scratch_index_buffer == nullptr) {
            return false;
          }
          shared_memory_->UseAsCopySource();
          SubmitBarriers();
          deferred_command_list_.D3DCopyBufferRegion(
              scratch_index_buffer, 0, shared_memory_->GetBuffer(),
              primitive_processing_result.guest_index_base, index_buffer_view.SizeInBytes);
          PushTransitionBarrier(scratch_index_buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                D3D12_RESOURCE_STATE_INDEX_BUFFER);
          index_buffer_view.BufferLocation = scratch_index_buffer->GetGPUVirtualAddress();
        } else {
          index_buffer_view.BufferLocation =
              shared_memory_->GetGPUAddress() + primitive_processing_result.guest_index_base;
        }
      } break;
      case PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted:
        index_buffer_view.BufferLocation = primitive_processor_->GetConvertedIndexBufferGpuAddress(
            primitive_processing_result.host_index_buffer_handle);
        break;
      case PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForAuto:
      case PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForDMA:
        index_buffer_view.BufferLocation = primitive_processor_->GetBuiltinIndexBufferGpuAddress(
            primitive_processing_result.host_index_buffer_handle);
        break;
      default:
        assert_unhandled_case(primitive_processing_result.index_buffer_type);
        return false;
    }
    deferred_command_list_.D3DIASetIndexBuffer(&index_buffer_view);
    if (memexport_used) {
      shared_memory_->UseForWriting();
    } else {
      shared_memory_->UseForReading();
    }
    SubmitBarriers();
    PROFILE_DRAW_CALL();
    PROFILE_VERTICES(primitive_processing_result.host_draw_vertex_count);
    deferred_command_list_.D3DDrawIndexedInstanced(
        primitive_processing_result.host_draw_vertex_count, 1, 0, 0, 0);
    if (scratch_index_buffer != nullptr) {
      ReleaseScratchGPUBuffer(scratch_index_buffer, D3D12_RESOURCE_STATE_INDEX_BUFFER);
    }
  }

  if (memexport_used) {
    // Make sure this memexporting draw is ordered with other work using shared
    // memory as a UAV.
    // TODO(Triang3l): Find some PM4 command that can be used for indication of
    // when memexports should be awaited?
    shared_memory_->MarkUAVWritesCommitNeeded();
    // Invalidate textures in memexported memory and watch for changes.
    if (!memexport_ranges_.empty()) {
      for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
        shared_memory_->RangeWrittenByGpu(memexport_range.base_address_dwords << 2,
                                          memexport_range.size_bytes);
      }
    } else {
      // Stream constants can be invalid or dynamic, so exact destinations may
      // be unknown. Keep invalidation conservative in this case.
      shared_memory_->RangeWrittenByGpu(0, SharedMemory::kBufferSize);
    }
    if (IsReadbackMemexportEnabled(REXCVAR_GET(d3d12_readback_memexport)) &&
        !memexport_ranges_.empty()) {
      uint32_t memexport_total_size = 0;
      for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
        memexport_total_size += memexport_range.size_bytes;
      }
      if (memexport_total_size != 0) {
        if (REXCVAR_GET(readback_memexport_fast)) {
          IssueDraw_MemexportReadbackFastPath(memexport_total_size);
        } else {
          IssueDraw_MemexportReadbackFullPath(memexport_total_size);
        }
      }
    }
  }

  if (tile_replay_current_ >= 0 && !memexport_used && !replay) {
    TileReplayRecord rec;
    rec.valid = true;
    rec.pipeline_handle = pipeline_handle;
    rec.root_signature = root_signature;
    rec.vertex_translation = vertex_shader_translation;
    rec.pixel_translation = pixel_shader_translation;
    rec.primitive = primitive_processing_result;
    rec.topology = primitive_topology;
    rec.normalized_depth_control = normalized_depth_control.value;
    rec.normalized_color_mask = normalized_color_mask;
    rec.used_texture_mask = used_texture_mask;
    rec.render_target_bits = bound_depth_and_color_render_target_bits;
    std::memcpy(rec.render_target_formats, bound_depth_and_color_render_target_formats,
                sizeof(rec.render_target_formats));
    std::memset(rec.hashes, 0, sizeof(rec.hashes));
    rec.constant_buffers[0] = cbuffer_binding_fetch_.address;
    rec.constant_buffers[1] = cbuffer_binding_float_vertex_.address;
    rec.constant_buffers[2] = cbuffer_binding_float_pixel_.address;
    rec.constant_buffers[3] = cbuffer_binding_bool_loop_.address;
    rec.constant_buffers[4] = cbuffer_binding_descriptor_indices_pixel_.address;
    rec.constant_buffers[5] = cbuffer_binding_descriptor_indices_vertex_.address;
    rec.sampler_heap_generation = tile_replay_sampler_heap_generation_;
    rec.srv_keys_start = uint32_t(tile_replay_srv_keys_.size());
    rec.srv_key_counts[0] = uint32_t(vertex_shader->GetTextureBindingsAfterTranslation().size());
    rec.srv_key_counts[1] =
        pixel_shader ? uint32_t(pixel_shader->GetTextureBindingsAfterTranslation().size()) : 0;
    if (amped3_tile_ == 1) {
      tile_replay_srv_keys_.insert(tile_replay_srv_keys_.end(),
                                   current_texture_srv_keys_vertex_.begin(),
                                   current_texture_srv_keys_vertex_.begin() + rec.srv_key_counts[0]);
      tile_replay_srv_keys_.insert(tile_replay_srv_keys_.end(),
                                   current_texture_srv_keys_pixel_.begin(),
                                   current_texture_srv_keys_pixel_.begin() + rec.srv_key_counts[1]);
    }
    if (REXCVAR_GET(gpu_tile_replay_verify)) {
      TileReplayHashes(vertex_shader, pixel_shader, rec.hashes);
    }
    if (amped3_tile_ == 1) {
      tile_replay_records_[tile_replay_current_] = rec;
    } else if (REXCVAR_GET(gpu_tile_replay_verify)) {
      const TileReplayRecord& t1 = tile_replay_records_[tile_replay_current_];
      const PrimitiveProcessor::ProcessingResult& a = t1.primitive;
      const PrimitiveProcessor::ProcessingResult& b = rec.primitive;
      bool mismatch[16] = {
          t1.pipeline_handle != rec.pipeline_handle || t1.root_signature != rec.root_signature,
          t1.vertex_translation != rec.vertex_translation ||
              t1.pixel_translation != rec.pixel_translation,
          a.host_primitive_type != b.host_primitive_type ||
              a.host_vertex_shader_type != b.host_vertex_shader_type ||
              a.host_draw_vertex_count != b.host_draw_vertex_count ||
              a.index_buffer_type != b.index_buffer_type ||
              a.guest_index_base != b.guest_index_base ||
              a.host_index_format != b.host_index_format ||
              a.host_shader_index_endian != b.host_shader_index_endian ||
              a.line_loop_closing_index != b.line_loop_closing_index ||
              a.host_index_buffer_handle != b.host_index_buffer_handle ||
              t1.topology != rec.topology,
          t1.normalized_depth_control != rec.normalized_depth_control ||
              t1.normalized_color_mask != rec.normalized_color_mask,
          t1.used_texture_mask != rec.used_texture_mask,
          t1.render_target_bits != rec.render_target_bits ||
              std::memcmp(t1.render_target_formats, rec.render_target_formats,
                          sizeof(rec.render_target_formats)) != 0,
          t1.hashes[0] != rec.hashes[0] || t1.hashes[1] != rec.hashes[1],
          t1.hashes[2] != rec.hashes[2],
          t1.hashes[3] != rec.hashes[3],
          t1.hashes[4] != rec.hashes[4],
          t1.hashes[5] != rec.hashes[5],
      };
      ++tile_replay_verified_;
      for (uint32_t i = 0; i < 16; ++i) {
        tile_replay_mismatches_[i] += mismatch[i];
      }
      if (!(tile_replay_verified_ % 100000)) {
        const uint64_t* m = tile_replay_mismatches_;
        REXGPU_INFO(
            "gpu_tile_replay_verify: {} draws: mismatches: pipeline {}, translations {}, "
            "primitive/index {}, depth/color state {}, textures used {}, render targets {}, "
            "float constants {}, fetch constants {}, bool/loop {}, system constants {}, "
            "texture/sampler bindings {}",
            tile_replay_verified_, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9],
            m[10]);
      }
    }
  }
  tile_replay_current_ = -1;

  DrawTimingMark(kDrawTimingTail);
  return true;
}

bool D3D12CommandProcessor::IssueDraw_MemexportReadbackFullPath(uint32_t total_size) {
  if (!total_size || memexport_ranges_.empty()) {
    return true;
  }

  ID3D12Resource* readback_buffer = RequestReadbackBuffer(total_size);
  if (!readback_buffer) {
    return true;
  }

  shared_memory_->UseAsCopySource();
  SubmitBarriers();
  ID3D12Resource* shared_memory_buffer = shared_memory_->GetBuffer();
  uint32_t readback_buffer_offset = 0;
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    deferred_command_list_.D3DCopyBufferRegion(
        readback_buffer, readback_buffer_offset, shared_memory_buffer,
        memexport_range.base_address_dwords << 2, memexport_range.size_bytes);
    readback_buffer_offset += memexport_range.size_bytes;
  }

  if (!AwaitAllQueueOperationsCompletion()) {
    return true;
  }

  D3D12_RANGE readback_range = {};
  readback_range.Begin = 0;
  readback_range.End = total_size;
  void* readback_mapping = nullptr;
  if (FAILED(readback_buffer->Map(0, &readback_range, &readback_mapping))) {
    return true;
  }

  const uint8_t* readback_bytes = reinterpret_cast<const uint8_t*>(readback_mapping);
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    std::memcpy(memory_->TranslatePhysical(memexport_range.base_address_dwords << 2),
                readback_bytes, memexport_range.size_bytes);
    readback_bytes += memexport_range.size_bytes;
  }

  D3D12_RANGE readback_write_range = {};
  readback_buffer->Unmap(0, &readback_write_range);
  return true;
}

bool D3D12CommandProcessor::IssueDraw_MemexportReadbackFastPath(uint32_t total_size) {
  if (!total_size || memexport_ranges_.empty()) {
    return true;
  }

  const uint64_t readback_key =
      MakeMemexportReadbackKey(memexport_ranges_.front().base_address_dwords, total_size);
  ReadbackBuffer& readback = memexport_readback_buffers_[readback_key];
  readback.last_used_frame = frame_current_;

  auto ensure_readback_slot = [&](uint32_t index, uint32_t size) -> bool {
    if (readback.buffers[index] && readback.mapped_data[index] && size <= readback.sizes[index]) {
      return true;
    }

    const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
    ID3D12Device* device = provider.GetDevice();
    D3D12_RESOURCE_DESC buffer_desc;
    ui::d3d12::util::FillBufferResourceDesc(buffer_desc, size, D3D12_RESOURCE_FLAG_NONE);
    ID3D12Resource* buffer = nullptr;
    if (FAILED(device->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesReadback, provider.GetHeapFlagCreateNotZeroed(),
            &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&buffer)))) {
      return false;
    }

    D3D12_RANGE read_range = {0, size};
    void* mapped_data = nullptr;
    if (FAILED(buffer->Map(0, &read_range, &mapped_data))) {
      buffer->Release();
      return false;
    }

    if (readback.buffers[index]) {
      if (!AwaitAllQueueOperationsCompletion()) {
        buffer->Unmap(0, nullptr);
        buffer->Release();
        return false;
      }
      if (readback.mapped_data[index]) {
        readback.buffers[index]->Unmap(0, nullptr);
      }
      readback.buffers[index]->Release();
    }

    readback.buffers[index] = buffer;
    readback.mapped_data[index] = mapped_data;
    readback.sizes[index] = size;
    readback.submission_written[index] = 0;
    readback.written_size[index] = 0;
    return true;
  };

  const uint32_t write_index = readback.current_index;
  const uint32_t read_index = 1 - write_index;
  const uint32_t readback_size = AlignReadbackBufferSize(total_size);
  if (!ensure_readback_slot(write_index, readback_size)) {
    return IssueDraw_MemexportReadbackFullPath(total_size);
  }

  shared_memory_->UseAsCopySource();
  SubmitBarriers();
  ID3D12Resource* shared_memory_buffer = shared_memory_->GetBuffer();
  uint32_t readback_offset = 0;
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    deferred_command_list_.D3DCopyBufferRegion(
        readback.buffers[write_index], readback_offset, shared_memory_buffer,
        memexport_range.base_address_dwords << 2, memexport_range.size_bytes);
    readback_offset += memexport_range.size_bytes;
  }
  readback.submission_written[write_index] = submission_current_;
  readback.written_size[write_index] = total_size;

  CheckSubmissionFence(0);
  bool previous_slot_ready = readback.buffers[read_index] && readback.mapped_data[read_index] &&
                             total_size <= readback.sizes[read_index] &&
                             total_size <= readback.written_size[read_index] &&
                             readback.submission_written[read_index] &&
                             readback.submission_written[read_index] <= submission_completed_;
  if (!previous_slot_ready) {
    IssueDraw_MemexportReadbackFullPath(total_size);
    readback.current_index = read_index;
    return true;
  }

  const uint8_t* readback_bytes = static_cast<const uint8_t*>(readback.mapped_data[read_index]);
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    std::memcpy(memory_->TranslatePhysical(memexport_range.base_address_dwords << 2),
                readback_bytes, memexport_range.size_bytes);
    readback_bytes += memexport_range.size_bytes;
  }
  readback.current_index = read_index;
  return true;
}

bool D3D12CommandProcessor::IssueCopy() {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES
  if (!BeginSubmission(true)) {
    return false;
  }
  SetGpuProfilePass(rex::perf::GpuPass::kResolve);
  if (rex::perf::IsGpuProfiling()) {
    rex::perf::GetGpuProfileStats().resolves.fetch_add(1, std::memory_order_relaxed);
  }
  // DLSS: Amped 3 copies the 3D scene out of its tiled surface (resolves of
  // color 0, color 1 and depth, several times a frame) and composites the copy
  // into the frontbuffer target, which is copied out last; the scene is handed
  // over at the last color 0 resolve before that (its index learnt last frame;
  // only tile 1, collapsed to the full width, counts).
  if (DlssTracing()) {
    const RegisterFile& tregs = *register_file_;
    REXGPU_INFO("trace resolve: pitch {} src {} color base {} depth base {} dest {:08X} dest pitch {}",
                tregs.Get<reg::RB_SURFACE_INFO>().surface_pitch,
                tregs.Get<reg::RB_COPY_CONTROL>().copy_src_select, tregs.Get<reg::RB_COLOR_INFO>().color_base,
                tregs.Get<reg::RB_DEPTH_INFO>().depth_base, tregs[XE_GPU_REG_RB_COPY_DEST_BASE],
                tregs.Get<reg::RB_COPY_DEST_PITCH>().copy_dest_pitch);
  }
  {
    const RegisterFile& regs = *register_file_;
    const uint32_t frontbuffer_width = rex::perf::GetRenderInfo().frontbuffer_width;
    if (frontbuffer_width && regs.Get<reg::RB_COPY_CONTROL>().copy_src_select == 0 &&
        render_target_cache_->SceneColorTargetFits(frontbuffer_width) &&
        ++g_dlss_wide_resolves == g_dlss_scene_resolve) {
      if (dlss_ && REXCVAR_GET(dlss_handoff) == "resolve" && dlss_->WantsScene()) {
        const rex::perf::RenderInfo render_info = rex::perf::GetRenderInfo();
        const uint32_t scene_width = render_info.frontbuffer_width * texture_cache_->draw_resolution_scale_x();
        const uint32_t scene_height = render_info.frontbuffer_height * texture_cache_->draw_resolution_scale_y();
        render_target_cache_->HandOverScene(
            [this, scene_width, scene_height](ID3D12Resource* color, ID3D12Resource* depth,
                                              D3D12_CPU_DESCRIPTOR_HANDLE depth_srv) {
              (void)depth;
              return dlss_->ProcessScene(color, depth_srv, scene_width, scene_height);
            });
      }
    }
  }
  ReadbackResolveMode readback_mode = GetReadbackResolveMode(REXCVAR_GET(d3d12_readback_resolve));
  if (readback_mode == ReadbackResolveMode::kDisabled) {
    uint32_t written_address, written_length;
    return render_target_cache_->Resolve(*memory_, *shared_memory_, *texture_cache_,
                                         written_address, written_length);
  }
  return IssueCopy_ReadbackResolvePath();
}

bool D3D12CommandProcessor::IssueCopy_ReadbackResolvePath() {
  uint32_t written_address, written_length;
  if (!render_target_cache_->Resolve(*memory_, *shared_memory_, *texture_cache_, written_address,
                                     written_length)) {
    return false;
  }

  if (!written_length) {
    return true;
  }

  if (!memory_->TranslatePhysical(written_address)) {
    return true;
  }

  bool is_scaled = texture_cache_->IsDrawResolutionScaled();
  uint64_t resolve_key = MakeReadbackResolveKey(written_address, written_length);
  ReadbackBuffer& rb = readback_buffers_[resolve_key];
  rb.last_used_frame = frame_current_;

  uint32_t write_index = rb.current_index;
  uint32_t size = AlignReadbackBufferSize(written_length);

  if (size > rb.sizes[write_index]) {
    const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
    ID3D12Device* device = provider.GetDevice();
    D3D12_RESOURCE_DESC buffer_desc;
    ui::d3d12::util::FillBufferResourceDesc(buffer_desc, size, D3D12_RESOURCE_FLAG_NONE);
    ID3D12Resource* buffer = nullptr;
    if (FAILED(device->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesReadback, provider.GetHeapFlagCreateNotZeroed(),
            &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&buffer)))) {
      REXGPU_ERROR("Failed to create a {} MB readback buffer", size >> 20);
      return true;
    }
    if (rb.buffers[write_index]) {
      if (rb.mapped_data[write_index]) {
        rb.buffers[write_index]->Unmap(0, nullptr);
        rb.mapped_data[write_index] = nullptr;
      }
      rb.buffers[write_index]->Release();
    }
    rb.buffers[write_index] = buffer;
    rb.sizes[write_index] = size;
    D3D12_RANGE read_range = {0, size};
    if (FAILED(buffer->Map(0, &read_range, &rb.mapped_data[write_index]))) {
      REXGPU_ERROR("Failed to persistently map resolve readback buffer");
      rb.mapped_data[write_index] = nullptr;
    }
  }

  if (!rb.buffers[write_index]) {
    return true;
  }

  if (is_scaled) {
    if (!resolve_downscale_pipeline_ || !resolve_downscale_root_signature_) {
      return true;
    }

    reg::RB_COPY_DEST_INFO copy_dest_info = register_file_->Get<reg::RB_COPY_DEST_INFO>();
    const FormatInfo* format_info = FormatInfo::Get(uint32_t(copy_dest_info.copy_dest_format));
    uint32_t bits_per_pixel = format_info->bits_per_pixel;
    if (bits_per_pixel != 8 && bits_per_pixel != 16 && bits_per_pixel != 32 &&
        bits_per_pixel != 64) {
      return true;
    }

    uint32_t pixel_size_log2;
    if (!rex::bit_scan_forward(bits_per_pixel >> 3, &pixel_size_log2)) {
      return true;
    }
    uint32_t tile_size_1x = 32 * 32 * (uint32_t(1) << pixel_size_log2);
    uint32_t tile_count = written_length / tile_size_1x;
    if (!tile_count) {
      return true;
    }

    uint32_t scaled_length = uint32_t(texture_cache_->GetCurrentScaledResolveRangeLengthScaled());
    uint64_t scaled_address = texture_cache_->GetCurrentScaledResolveRangeStartScaled();
    if (!scaled_length) {
      return true;
    }

    uint32_t downscale_buffer_size = AlignReadbackBufferSize(written_length);
    if (downscale_buffer_size > resolve_downscale_buffer_size_) {
      const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
      ID3D12Device* device = provider.GetDevice();
      D3D12_RESOURCE_DESC buffer_desc;
      ui::d3d12::util::FillBufferResourceDesc(buffer_desc, downscale_buffer_size,
                                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
      ID3D12Resource* buffer = nullptr;
      if (FAILED(device->CreateCommittedResource(
              &ui::d3d12::util::kHeapPropertiesDefault, provider.GetHeapFlagCreateNotZeroed(),
              &buffer_desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
              IID_PPV_ARGS(&buffer)))) {
        REXGPU_ERROR("Failed to create a {} MB resolve downscale buffer",
                     downscale_buffer_size >> 20);
        return true;
      }
      if (resolve_downscale_buffer_) {
        resources_for_deletion_.emplace_back(GetCurrentSubmission(),
                                             resolve_downscale_buffer_.Detach());
      }
      resolve_downscale_buffer_.Attach(buffer);
      resolve_downscale_buffer_size_ = downscale_buffer_size;
    }

    if (!resolve_downscale_buffer_) {
      return true;
    }

    ID3D12Resource* scaled_resolve_buffer = texture_cache_->GetCurrentScaledResolveBufferResource();
    size_t scaled_resolve_buffer_index = texture_cache_->GetCurrentScaledResolveBufferIndexPublic();
    if (!scaled_resolve_buffer) {
      return true;
    }
    uint64_t scaled_buffer_base = uint64_t(scaled_resolve_buffer_index) << 30;
    if (scaled_address < scaled_buffer_base) {
      return true;
    }
    uint64_t source_offset = scaled_address - scaled_buffer_base;

    ui::d3d12::util::DescriptorCpuGpuHandlePair downscale_descriptors[2];
    if (!RequestOneUseSingleViewDescriptors(2, downscale_descriptors)) {
      return true;
    }

    const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
    ID3D12Device* device = provider.GetDevice();
    uint32_t aligned_scaled_length =
        rex::align(scaled_length, uint32_t(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT));
    ui::d3d12::util::CreateBufferRawSRV(device, downscale_descriptors[0].first,
                                        scaled_resolve_buffer, aligned_scaled_length,
                                        source_offset);
    uint32_t aligned_written_length =
        rex::align(written_length, uint32_t(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT));
    ui::d3d12::util::CreateBufferRawUAV(device, downscale_descriptors[1].first,
                                        resolve_downscale_buffer_.Get(), aligned_written_length, 0);

    PushUAVBarrier(scaled_resolve_buffer);
    texture_cache_->TransitionCurrentScaledResolveRange(
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    SubmitBarriers();

    SetExternalPipeline(resolve_downscale_pipeline_.Get());
    deferred_command_list_.D3DSetComputeRootSignature(resolve_downscale_root_signature_.Get());
    ResolveDownscaleConstants constants;
    constants.scale_x = texture_cache_->draw_resolution_scale_x();
    constants.scale_y = texture_cache_->draw_resolution_scale_y();
    constants.pixel_size_log2 = pixel_size_log2;
    constants.tile_count = tile_count;
    constants.half_pixel_offset = (REXCVAR_GET(readback_resolve_half_pixel_offset) &&
                                   (constants.scale_x > 1 || constants.scale_y > 1))
                                      ? 1u
                                      : 0u;
    deferred_command_list_.D3DSetComputeRoot32BitConstants(
        UINT(ResolveDownscaleRootParameter::kConstants), sizeof(constants) / sizeof(uint32_t),
        &constants, 0);
    deferred_command_list_.D3DSetComputeRootDescriptorTable(
        UINT(ResolveDownscaleRootParameter::kSource), downscale_descriptors[0].second);
    deferred_command_list_.D3DSetComputeRootDescriptorTable(
        UINT(ResolveDownscaleRootParameter::kDestination), downscale_descriptors[1].second);
    deferred_command_list_.D3DDispatch(tile_count, 1, 1);

    PushUAVBarrier(resolve_downscale_buffer_.Get());
    PushTransitionBarrier(resolve_downscale_buffer_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_COPY_SOURCE);
    SubmitBarriers();
    deferred_command_list_.D3DCopyBufferRegion(rb.buffers[write_index], 0,
                                               resolve_downscale_buffer_.Get(), 0, written_length);
    PushTransitionBarrier(resolve_downscale_buffer_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    texture_cache_->TransitionCurrentScaledResolveRange(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    SubmitBarriers();
  } else {
    shared_memory_->UseAsCopySource();
    SubmitBarriers();
    ID3D12Resource* shared_memory_buffer = shared_memory_->GetBuffer();
    deferred_command_list_.D3DCopyBufferRegion(rb.buffers[write_index], 0, shared_memory_buffer,
                                               written_address, written_length);
  }

  ReadbackResolveMode readback_mode = GetReadbackResolveMode(REXCVAR_GET(d3d12_readback_resolve));
  bool use_delayed_sync =
      readback_mode == ReadbackResolveMode::kFast || readback_mode == ReadbackResolveMode::kSome;
  uint32_t read_index = write_index;
  if (use_delayed_sync) {
    read_index = 1 - write_index;
  } else if (!AwaitAllQueueOperationsCompletion()) {
    return true;
  }

  bool is_cache_miss = false;
  if (use_delayed_sync && (!rb.buffers[read_index] || written_length > rb.sizes[read_index] ||
                           !rb.mapped_data[read_index])) {
    is_cache_miss = true;
    read_index = write_index;
    if (!AwaitAllQueueOperationsCompletion()) {
      return true;
    }
  }

  bool should_copy = (readback_mode == ReadbackResolveMode::kSome) ? is_cache_miss : true;
  if (should_copy && rb.buffers[read_index] && written_length <= rb.sizes[read_index] &&
      rb.mapped_data[read_index]) {
    uint8_t* destination = memory_->TranslatePhysical(written_address);
    if (destination) {
      std::memcpy(destination, static_cast<uint8_t*>(rb.mapped_data[read_index]), written_length);
    }
  }

  rb.current_index = 1 - rb.current_index;
  return true;
}

void D3D12CommandProcessor::CheckSubmissionFence(uint64_t await_submission) {
  if (await_submission >= submission_current_) {
    if (submission_open_) {
      EndSubmission(false);
    }
    // Ending an open submission should result in queue operations done directly
    // (like UpdateTileMappings) to be tracked within the scope of that
    // submission, but just in case of a failure, or queue operations being done
    // outside of a submission, await explicitly.
    if (queue_operations_done_since_submission_signal_) {
      UINT64 fence_value = ++queue_operations_since_submission_fence_last_;
      ID3D12CommandQueue* direct_queue = GetD3D12Provider().GetDirectQueue();
      if (SUCCEEDED(direct_queue->Signal(queue_operations_since_submission_fence_, fence_value) &&
                    SUCCEEDED(queue_operations_since_submission_fence_->SetEventOnCompletion(
                        fence_value, fence_completion_event_)))) {
        PROFILE_CMD_BUFFER_STALL();
        WaitForSingleObject(fence_completion_event_, INFINITE);
        queue_operations_done_since_submission_signal_ = false;
      } else {
        REXGPU_ERROR(
            "Failed to await an out-of-submission queue operation completion "
            "Direct3D 12 fence");
      }
    }
    // A submission won't be ended if it hasn't been started, or if ending
    // has failed - clamp the index.
    await_submission = submission_current_ - 1;
  }

  uint64_t submission_completed_before = submission_completed_;
  submission_completed_ = submission_fence_->GetCompletedValue();
  if (submission_completed_ < await_submission) {
    if (SUCCEEDED(
            submission_fence_->SetEventOnCompletion(await_submission, fence_completion_event_))) {
      PROFILE_CMD_BUFFER_STALL();
      const bool profiling = rex::perf::IsGpuProfiling();
      const auto wait_start =
          profiling ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
      WaitForSingleObject(fence_completion_event_, INFINITE);
      if (profiling) {
        rex::perf::GpuProfileStats& stats = rex::perf::GetGpuProfileStats();
        stats.fence_waits.fetch_add(1, std::memory_order_relaxed);
        stats.fence_wait_ns.fetch_add(
            uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - wait_start)
                         .count()),
            std::memory_order_relaxed);
      }
      submission_completed_ = submission_fence_->GetCompletedValue();
    }
  }
  if (submission_completed_ < await_submission) {
    REXGPU_ERROR("Failed to await a submission completion Direct3D 12 fence");
  }
  if (submission_completed_ <= submission_completed_before) {
    // Not updated - no need to reclaim or download things.
    return;
  }

  // Reclaim command allocators.
  while (command_allocator_submitted_first_) {
    if (command_allocator_submitted_first_->last_usage_submission > submission_completed_) {
      break;
    }
    if (command_allocator_writable_last_) {
      command_allocator_writable_last_->next = command_allocator_submitted_first_;
    } else {
      command_allocator_writable_first_ = command_allocator_submitted_first_;
    }
    command_allocator_writable_last_ = command_allocator_submitted_first_;
    command_allocator_submitted_first_ = command_allocator_submitted_first_->next;
    command_allocator_writable_last_->next = nullptr;
  }
  if (!command_allocator_submitted_first_) {
    command_allocator_submitted_last_ = nullptr;
  }

  // Release single-use bindless descriptors.
  while (!view_bindless_one_use_descriptors_.empty()) {
    if (view_bindless_one_use_descriptors_.front().second > submission_completed_) {
      break;
    }
    ReleaseViewBindlessDescriptorImmediately(view_bindless_one_use_descriptors_.front().first);
    view_bindless_one_use_descriptors_.pop_front();
  }

  // Delete transient resources marked for deletion.
  while (!resources_for_deletion_.empty()) {
    if (resources_for_deletion_.front().first > submission_completed_) {
      break;
    }
    resources_for_deletion_.front().second->Release();
    resources_for_deletion_.pop_front();
  }

  shared_memory_->CompletedSubmissionUpdated();

  render_target_cache_->CompletedSubmissionUpdated();

  primitive_processor_->CompletedSubmissionUpdated();

  texture_cache_->CompletedSubmissionUpdated(submission_completed_);

  GpuProfileReadCompleted();
}

void D3D12CommandProcessor::LogDeviceRemovalDiagnostics(ID3D12Device* device, HRESULT reason) {
  const char* reason_str = "Unknown";
  switch (reason) {
    case DXGI_ERROR_DEVICE_HUNG:
      reason_str = "DEVICE_HUNG (TDR - GPU command took too long)";
      break;
    case DXGI_ERROR_DEVICE_REMOVED:
      reason_str = "DEVICE_REMOVED (driver internal error or hot-unplug)";
      break;
    case DXGI_ERROR_DEVICE_RESET:
      reason_str = "DEVICE_RESET (bad GPU command)";
      break;
    case DXGI_ERROR_DRIVER_INTERNAL_ERROR:
      reason_str = "DRIVER_INTERNAL_ERROR";
      break;
    case DXGI_ERROR_INVALID_CALL:
      reason_str = "INVALID_CALL";
      break;
  }
  REXGPU_ERROR("D3D12 device removed: HRESULT 0x{:08X} - {}", static_cast<unsigned>(reason),
               reason_str);

  Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedData> dred;
  if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dred)))) {
    return;
  }

  D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT breadcrumbs = {};
  if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput(&breadcrumbs))) {
    for (const D3D12_AUTO_BREADCRUMB_NODE* node = breadcrumbs.pHeadAutoBreadcrumbNode; node;
         node = node->pNext) {
      if (!node->pLastBreadcrumbValue || !node->pCommandHistory ||
          *node->pLastBreadcrumbValue == 0) {
        continue;
      }
      REXGPU_ERROR("DRED breadcrumb: completed {} of {} ops", *node->pLastBreadcrumbValue,
                   node->BreadcrumbCount);
      uint32_t last = std::min(*node->pLastBreadcrumbValue, node->BreadcrumbCount);
      uint32_t start = last > 3 ? last - 3 : 0;
      uint32_t end = std::min(last + 1, node->BreadcrumbCount);
      for (uint32_t i = start; i < end; i++) {
        REXGPU_ERROR("  [{}] op type {}{}", i, static_cast<int>(node->pCommandHistory[i]),
                     i == last ? " <-- FAULT" : "");
      }
    }
  }

  D3D12_DRED_PAGE_FAULT_OUTPUT page_fault = {};
  if (SUCCEEDED(dred->GetPageFaultAllocationOutput(&page_fault)) && page_fault.PageFaultVA != 0) {
    REXGPU_ERROR("DRED page fault at VA 0x{:016X}", page_fault.PageFaultVA);
  }
}

bool D3D12CommandProcessor::BeginSubmission(bool is_guest_command) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  if (device_removed_) {
    return false;
  }

  bool is_opening_frame = is_guest_command && !frame_open_;
  if (submission_open_ && !is_opening_frame) {
    return true;
  }

  // Check if the device is still available.
  ID3D12Device* device = GetD3D12Provider().GetDevice();
  HRESULT device_removed_reason = device->GetDeviceRemovedReason();
  if (FAILED(device_removed_reason)) {
    device_removed_ = true;
    LogDeviceRemovalDiagnostics(device, device_removed_reason);
    if (graphics_system_) {
      graphics_system_->OnHostGpuLossFromAnyThread(device_removed_reason !=
                                                   DXGI_ERROR_DEVICE_REMOVED);
    }
    return false;
  }

  // Check the fence - needed for all kinds of submissions (to reclaim transient
  // resources early) and specifically for frames (not to queue too many), and
  // await the availability of the current frame.
  uint64_t await_submission = is_opening_frame ? closed_frame_submissions_[frame_current_ % kQueueFrames] : 0;
  // --d3d12_queued_frames below 3: also wait for the frame that many frames ago.
  const uint64_t queued_frames = uint64_t(std::clamp(REXCVAR_GET(d3d12_queued_frames), 1, int32_t(kQueueFrames)));
  if (is_opening_frame && queued_frames < kQueueFrames && frame_current_ >= queued_frames) {
    await_submission =
        std::max(await_submission, closed_frame_submissions_[(frame_current_ - queued_frames) % kQueueFrames]);
  }
  CheckSubmissionFence(await_submission);
  // TODO(Triang3l): If failed to await (completed submission < awaited frame
  // submission), do something like dropping the draw command that wanted to
  // open the frame.
  if (is_opening_frame) {
    // Update the completed frame index, also obtaining the actual completed
    // frame number (since the CPU may be actually less than 3 frames behind)
    // before reclaiming resources tracked with the frame number.
    frame_completed_ = std::max(frame_current_, uint64_t(kQueueFrames)) - kQueueFrames;
    for (uint64_t frame = frame_completed_ + 1; frame < frame_current_; ++frame) {
      if (closed_frame_submissions_[frame % kQueueFrames] > submission_completed_) {
        break;
      }
      frame_completed_ = frame;
    }
  }

  if (!submission_open_) {
    submission_open_ = true;

    // Start a new deferred command list - will submit it to the real one in the
    // end of the submission (when async pipeline creation requests are
    // fulfilled).
    deferred_command_list_.Reset();
    GpuProfileBeginSubmission();

    // Reset cached state of the command list.
    ff_viewport_update_needed_ = true;
    ff_scissor_update_needed_ = true;
    ff_blend_factor_update_needed_ = true;
    ff_stencil_ref_update_needed_ = true;
    viewport_cache_valid_ = false;
    current_guest_pipeline_ = nullptr;
    current_external_pipeline_ = nullptr;
    current_graphics_root_signature_ = nullptr;
    tile_replay_root_signature_ = nullptr;
    current_graphics_root_up_to_date_ = 0;
    if (bindless_resources_used_) {
      deferred_command_list_.SetDescriptorHeaps(view_bindless_heap_,
                                                sampler_bindless_heap_current_);
    } else {
      view_bindful_heap_current_ = nullptr;
      sampler_bindful_heap_current_ = nullptr;
    }
    primitive_topology_ = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;

    render_target_cache_->BeginSubmission();

    primitive_processor_->BeginSubmission();

    texture_cache_->BeginSubmission(submission_current_);
  }

  if (is_opening_frame) {
    frame_open_ = true;

    // Reset bindings that depend on the data stored in the pools.
    std::memset(current_float_constant_map_vertex_, 0, sizeof(current_float_constant_map_vertex_));
    std::memset(current_float_constant_map_pixel_, 0, sizeof(current_float_constant_map_pixel_));
    cbuffer_binding_system_.up_to_date = false;
    cbuffer_binding_float_vertex_.up_to_date = false;
    cbuffer_binding_float_pixel_.up_to_date = false;
    cbuffer_binding_bool_loop_.up_to_date = false;
    cbuffer_binding_fetch_.up_to_date = false;
    current_shared_memory_binding_is_uav_.reset();
    if (bindless_resources_used_) {
      cbuffer_binding_descriptor_indices_vertex_.up_to_date = false;
      cbuffer_binding_descriptor_indices_pixel_.up_to_date = false;
    } else {
      draw_view_bindful_heap_index_ = ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid;
      draw_sampler_bindful_heap_index_ = ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid;
      bindful_textures_written_vertex_ = false;
      bindful_textures_written_pixel_ = false;
      bindful_samplers_written_vertex_ = false;
      bindful_samplers_written_pixel_ = false;
    }

    // Reclaim pool pages - no need to do this every small submission since some
    // may be reused.
    constant_buffer_pool_->Reclaim(frame_completed_);
    if (!bindless_resources_used_) {
      view_bindful_heap_pool_->Reclaim(frame_completed_);
      sampler_bindful_heap_pool_->Reclaim(frame_completed_);
    }
    EvictOldReadbackBuffers(readback_buffers_);
    EvictOldReadbackBuffers(memexport_readback_buffers_);

    primitive_processor_->BeginFrame();

    texture_cache_->BeginFrame();
  }

  return true;
}

bool D3D12CommandProcessor::EndSubmission(bool is_swap) {
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();

  // Make sure there is a command allocator to write commands to.
  if (submission_open_ && !command_allocator_writable_first_) {
    ID3D12CommandAllocator* command_allocator;
    if (FAILED(provider.GetDevice()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                            IID_PPV_ARGS(&command_allocator)))) {
      REXGPU_ERROR("Failed to create a command allocator");
      // Try to submit later. Completely dropping the submission is not
      // permitted because resources would be left in an undefined state.
      return false;
    }
    command_allocator_writable_first_ = new CommandAllocator;
    command_allocator_writable_first_->command_allocator = command_allocator;
    command_allocator_writable_first_->last_usage_submission = 0;
    command_allocator_writable_first_->next = nullptr;
    command_allocator_writable_last_ = command_allocator_writable_first_;
  }

  bool is_closing_frame = is_swap && frame_open_;

  if (is_closing_frame) {
    texture_cache_->EndFrame();

    primitive_processor_->EndFrame();
  }
  // Swaps: everything submitted before the presenter submits its own commands
  // (on returning, after this submission is queued below).
  struct AwaitOnSwap {
    D3D12CommandProcessor* self;
    bool is_swap;
    ~AwaitOnSwap() {
      if (is_swap && !self->presenter_tail_on_submission_thread_) self->AwaitSubmissionThread();
    }
  } await_on_swap{this, is_swap};

  if (submission_open_) {
    assert_false(scratch_buffer_used_);

    if (active_occlusion_query_.valid && occlusion_query_heap_) {
      deferred_command_list_.D3DEndQuery(occlusion_query_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION,
                                         active_occlusion_query_.host_index);
      active_occlusion_query_ = {};
    }

    pipeline_cache_->EndSubmission();

    // Submit barriers now because resources with the queued barriers may be
    // destroyed between frames.
    SubmitBarriers();

    GpuProfileEndSubmission();
    if (rex::perf::IsGpuProfiling()) {
      rex::perf::GetGpuProfileStats().cp_submissions.fetch_add(1, std::memory_order_relaxed);
    }

    ID3D12CommandQueue* direct_queue = provider.GetDirectQueue();

    // Submit the deferred command list.
    // Only one deferred command list must be executed in the same
    // ExecuteCommandLists - the boundaries of ExecuteCommandLists are a full
    // UAV and aliasing barrier, and subsystems of the emulator assume it
    // happens between Xenia submissions.
    ID3D12CommandAllocator* command_allocator =
        command_allocator_writable_first_->command_allocator;
    const bool threaded = UseSubmissionThread();
    if (threaded) {
      SubmissionJob job;
      job.command_allocator = command_allocator;
      job.fence_value = submission_current_;
      {
        std::lock_guard lock(submission_mutex_);
        if (!submission_free_streams_.empty()) {
          job.command_stream = std::move(submission_free_streams_.back());
          submission_free_streams_.pop_back();
        }
      }
      deferred_command_list_.SwapCommandStream(job.command_stream);
      deferred_command_list_.SwapExternalCallbacks(job.external_callbacks);
      {
        std::lock_guard lock(submission_mutex_);
        submission_jobs_.push_back(std::move(job));
      }
      submission_job_added_.notify_one();
    } else {
      command_allocator->Reset();
      command_list_->Reset(command_allocator, nullptr);
      deferred_command_list_.Execute(command_list_, command_list_1_);
      command_list_->Close();
      ID3D12CommandList* execute_command_lists[] = {command_list_};
      direct_queue->ExecuteCommandLists(1, execute_command_lists);
    }
    command_allocator_writable_first_->last_usage_submission = submission_current_;
    if (command_allocator_submitted_last_) {
      command_allocator_submitted_last_->next = command_allocator_writable_first_;
    } else {
      command_allocator_submitted_first_ = command_allocator_writable_first_;
    }
    command_allocator_submitted_last_ = command_allocator_writable_first_;
    command_allocator_writable_first_ = command_allocator_writable_first_->next;
    command_allocator_submitted_last_->next = nullptr;
    if (!command_allocator_writable_first_) {
      command_allocator_writable_last_ = nullptr;
    }

    if (threaded) {
      // Signaled by the submission thread after submitting.
      ++submission_current_;
    } else {
      direct_queue->Signal(submission_fence_, submission_current_++);
    }

    submission_open_ = false;

    // Queue operations done directly (like UpdateTileMappings) will be awaited
    // alongside the last submission if needed.
    queue_operations_done_since_submission_signal_ = false;
  }

  if (is_closing_frame) {
    if (REXCVAR_GET(clear_memory_page_state) && shared_memory_) {
      shared_memory_->SetSystemPageBlocksValidWithGpuDataWritten();
    }
    frame_open_ = false;
    // Submission already closed now, so minus 1.
    closed_frame_submissions_[(frame_current_++) % kQueueFrames] = submission_current_ - 1;
    if (REXCVAR_GET(d3d12_gpu_done_log)) LogGpuDone(submission_fence_, submission_current_ - 1);

    if (cache_clear_requested_ && AwaitAllQueueOperationsCompletion()) {
      cache_clear_requested_ = false;

      ClearCommandAllocatorCache();

      ui::d3d12::util::ReleaseAndNull(scratch_buffer_);
      scratch_buffer_size_ = 0;

      if (bindless_resources_used_) {
        texture_cache_bindless_sampler_map_.clear();
        for (const auto& sampler_bindless_heap_overflowed : sampler_bindless_heaps_overflowed_) {
          sampler_bindless_heap_overflowed.first->Release();
        }
        sampler_bindless_heaps_overflowed_.clear();
        sampler_bindless_heap_allocated_ = 0;
      } else {
        sampler_bindful_heap_pool_->ClearCache();
        view_bindful_heap_pool_->ClearCache();
      }
      constant_buffer_pool_->ClearCache();

      texture_cache_->ClearCache();

      // Not clearing the root signatures as they're referenced by pipelines,
      // which are not destroyed.

      primitive_processor_->ClearCache();

      render_target_cache_->ClearCache();

      shared_memory_->ClearCache();
    }
  }

  return true;
}

bool D3D12CommandProcessor::UploadFloatConstants(uint32_t stage, const uint64_t* float_bitmap,
                                                 uint32_t float_count, uint32_t first_register,
                                                 D3D12_GPU_VIRTUAL_ADDRESS& address_out) {
  const RegisterFile& regs = *register_file_;
  // Even if the shader doesn't need any float constants, a valid binding must
  // still be provided.
  const size_t size = sizeof(float) * 4 * std::max(float_count, uint32_t(1));
  // Gathered into cached memory first: compared with the last upload, and
  // copied to the (write-combined) upload buffer in one go.
  // Both buffers stay at the maximum size (no zero-filling as the size
  // changes), and the staging one becomes the last upload by a swap.
  constexpr size_t kFloatConstantsMaxSize = sizeof(float) * 4 * 256;
  std::vector<uint8_t>& staging = float_constants_staging_;
  if (staging.size() != kFloatConstantsMaxSize) {
    staging.resize(kFloatConstantsMaxSize);
  }
  uint8_t* gathered = staging.data();
  if (!float_count) {
    std::memset(gathered, 0, size);
  }
  for (uint32_t i = 0; i < 4; ++i) {
    uint64_t float_constant_map_entry = float_bitmap[i];
    uint32_t float_constant_index;
    while (rex::bit_scan_forward(float_constant_map_entry, &float_constant_index)) {
      // Contiguous runs at once.
      uint32_t run = uint32_t(std::countr_one(float_constant_map_entry >> float_constant_index));
      float_constant_map_entry &= ~(((run < 64 ? (1ull << run) : 0ull) - 1) << float_constant_index);
      std::memcpy(gathered, &regs[first_register + (i << 8) + (float_constant_index << 2)],
                  4 * sizeof(float) * run);
      gathered += 4 * sizeof(float) * run;
    }
  }
  FloatConstantUpload& last = float_constants_last_[stage];
  if (REXCVAR_GET(d3d12_reuse_float_constants) && last.frame == frame_current_ &&
      last.size == size && !std::memcmp(last.data.data(), staging.data(), size)) {
    address_out = last.address;
    ++float_constant_reuses_;
    return true;
  }
  uint8_t* float_constants = constant_buffer_pool_->Request(
      frame_current_, size, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, nullptr, nullptr,
      &address_out);
  if (float_constants == nullptr) {
    return false;
  }
  std::memcpy(float_constants, staging.data(), size);
  ++float_constant_uploads_;
  float_constant_upload_bytes_ += size;
  last.data.swap(staging);
  last.size = size;
  last.frame = frame_current_;
  last.address = address_out;
  return true;
}

bool D3D12CommandProcessor::UseSubmissionThread() {
  if (!submission_thread_state_) {
    submission_thread_state_ = REXCVAR_GET(d3d12_submission_thread) ? 1 : -1;
    if (submission_thread_state_ > 0) {
      submission_thread_exit_ = false;
      submission_thread_ = std::thread([this] { SubmissionThreadMain(); });
      REXGPU_INFO("D3D12: command lists are submitted on a separate thread");
    }
  }
  return submission_thread_state_ > 0;
}

void D3D12CommandProcessor::SubmissionThreadMain() {
#ifdef _WIN32
  SetThreadDescription(GetCurrentThread(), L"GPU Submission");
#endif
  DeferredCommandList list(*this);
  ID3D12CommandQueue* direct_queue = GetD3D12Provider().GetDirectQueue();
  std::unique_lock lock(submission_mutex_);
  while (true) {
    submission_job_added_.wait(
        lock, [this] { return submission_thread_exit_ || !submission_jobs_.empty(); });
    if (submission_jobs_.empty()) {
      break;  // Exiting with nothing left.
    }
    SubmissionJob job = std::move(submission_jobs_.front());
    submission_jobs_.pop_front();
    submission_thread_busy_ = true;
    lock.unlock();
    if (job.callback) {
      job.callback();
      job.callback = nullptr;
      lock.lock();
      submission_thread_busy_ = false;
      if (submission_jobs_.empty()) {
        submission_jobs_done_.notify_all();
      }
      continue;
    }
    job.command_allocator->Reset();
    command_list_->Reset(job.command_allocator, nullptr);
    list.SwapCommandStream(job.command_stream);
    list.SwapExternalCallbacks(job.external_callbacks);
    list.Execute(command_list_, command_list_1_);
    list.SwapCommandStream(job.command_stream);
    list.SwapExternalCallbacks(job.external_callbacks);
    command_list_->Close();
    ID3D12CommandList* execute_command_lists[] = {command_list_};
    direct_queue->ExecuteCommandLists(1, execute_command_lists);
    direct_queue->Signal(submission_fence_, job.fence_value);
    job.command_stream.clear();
    lock.lock();
    submission_free_streams_.push_back(std::move(job.command_stream));
    submission_thread_busy_ = false;
    if (submission_jobs_.empty()) {
      submission_jobs_done_.notify_all();
    }
  }
}

void D3D12CommandProcessor::EnqueueSubmissionCallback(std::function<void()> callback) {
  SubmissionJob job;
  job.command_allocator = nullptr;
  job.fence_value = 0;
  job.callback = std::move(callback);
  {
    std::lock_guard lock(submission_mutex_);
    submission_jobs_.push_back(std::move(job));
  }
  submission_job_added_.notify_one();
}

void D3D12CommandProcessor::AwaitSubmissionThread() {
  if (submission_thread_state_ <= 0) {
    return;
  }
  std::unique_lock lock(submission_mutex_);
  submission_jobs_done_.wait(
      lock, [this] { return submission_jobs_.empty() && !submission_thread_busy_; });
}

void D3D12CommandProcessor::ShutdownSubmissionThread() {
  if (submission_thread_state_ > 0 && submission_thread_.joinable()) {
    {
      std::lock_guard lock(submission_mutex_);
      submission_thread_exit_ = true;
    }
    submission_job_added_.notify_one();
    submission_thread_.join();
  }
  submission_thread_state_ = 0;
}

bool D3D12CommandProcessor::CanEndSubmissionImmediately() const {
  return !submission_open_ || !pipeline_cache_->IsCreatingPipelines();
}

void D3D12CommandProcessor::ClearCommandAllocatorCache() {
  while (command_allocator_submitted_first_) {
    auto next = command_allocator_submitted_first_->next;
    command_allocator_submitted_first_->command_allocator->Release();
    delete command_allocator_submitted_first_;
    command_allocator_submitted_first_ = next;
  }
  command_allocator_submitted_last_ = nullptr;
  while (command_allocator_writable_first_) {
    auto next = command_allocator_writable_first_->next;
    command_allocator_writable_first_->command_allocator->Release();
    delete command_allocator_writable_first_;
    command_allocator_writable_first_ = next;
  }
  command_allocator_writable_last_ = nullptr;
}

void D3D12CommandProcessor::UpdateFixedFunctionState(
    const draw_util::ViewportInfo& viewport_info, const draw_util::Scissor& scissor,
    bool primitive_polygonal, reg::RB_DEPTHCONTROL normalized_depth_control) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  // Viewport.
  D3D12_VIEWPORT viewport;
  viewport.TopLeftX = float(viewport_info.xy_offset[0]);
  viewport.TopLeftY = float(viewport_info.xy_offset[1]);
  viewport.Width = float(viewport_info.xy_extent[0]);
  viewport.Height = float(viewport_info.xy_extent[1]);
  viewport.MinDepth = viewport_info.z_min;
  viewport.MaxDepth = viewport_info.z_max;
  SetViewport(viewport);

  // Scissor.
  D3D12_RECT scissor_rect;
  scissor_rect.left = LONG(scissor.offset[0]);
  scissor_rect.top = LONG(scissor.offset[1]);
  scissor_rect.right = LONG(scissor.offset[0] + scissor.extent[0]);
  scissor_rect.bottom = LONG(scissor.offset[1] + scissor.extent[1]);
  SetScissorRect(scissor_rect);

  if (render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets) {
    const RegisterFile& regs = *register_file_;

    // Blend factor.
    float blend_factor[] = {
        regs.Get<float>(XE_GPU_REG_RB_BLEND_RED),
        regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN),
        regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE),
        regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA),
    };
    // std::memcmp instead of != so in case of NaN, every draw won't be
    // invalidating it.
    ff_blend_factor_update_needed_ |=
        std::memcmp(ff_blend_factor_, blend_factor, sizeof(float) * 4) != 0;
    if (ff_blend_factor_update_needed_) {
      std::memcpy(ff_blend_factor_, blend_factor, sizeof(float) * 4);
      deferred_command_list_.D3DOMSetBlendFactor(ff_blend_factor_);
      ff_blend_factor_update_needed_ = false;
    }

    // Stencil reference value. Per-face reference not supported by Direct3D 12,
    // choose the back face one only if drawing only back faces.
    Register stencil_ref_mask_reg;
    auto pa_su_sc_mode_cntl = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
    if (primitive_polygonal && normalized_depth_control.backface_enable &&
        pa_su_sc_mode_cntl.cull_front && !pa_su_sc_mode_cntl.cull_back) {
      stencil_ref_mask_reg = XE_GPU_REG_RB_STENCILREFMASK_BF;
    } else {
      stencil_ref_mask_reg = XE_GPU_REG_RB_STENCILREFMASK;
    }
    uint32_t stencil_ref = regs.Get<reg::RB_STENCILREFMASK>(stencil_ref_mask_reg).stencilref;
    ff_stencil_ref_update_needed_ |= ff_stencil_ref_ != stencil_ref;
    if (ff_stencil_ref_update_needed_) {
      ff_stencil_ref_ = stencil_ref;
      deferred_command_list_.D3DOMSetStencilRef(ff_stencil_ref_);
      ff_stencil_ref_update_needed_ = false;
    }
  }
}

void D3D12CommandProcessor::UpdateSystemConstantValues(
    bool shared_memory_is_uav, bool primitive_polygonal, uint32_t line_loop_closing_index,
    xenos::Endian index_endian, const draw_util::ViewportInfo& viewport_info,
    uint32_t used_texture_mask, reg::RB_DEPTHCONTROL normalized_depth_control,
    uint32_t normalized_color_mask) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  const RegisterFile& regs = *register_file_;
  auto pa_cl_clip_cntl = regs.Get<reg::PA_CL_CLIP_CNTL>();
  auto pa_cl_vte_cntl = regs.Get<reg::PA_CL_VTE_CNTL>();
  auto pa_su_sc_mode_cntl = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
  auto rb_alpha_ref = regs.Get<float>(XE_GPU_REG_RB_ALPHA_REF);
  auto rb_colorcontrol = regs.Get<reg::RB_COLORCONTROL>();
  auto rb_depth_info = regs.Get<reg::RB_DEPTH_INFO>();
  auto rb_stencilrefmask = regs.Get<reg::RB_STENCILREFMASK>();
  auto rb_stencilrefmask_bf = regs.Get<reg::RB_STENCILREFMASK>(XE_GPU_REG_RB_STENCILREFMASK_BF);
  auto rb_surface_info = regs.Get<reg::RB_SURFACE_INFO>();
  auto sq_context_misc = regs.Get<reg::SQ_CONTEXT_MISC>();
  auto sq_program_cntl = regs.Get<reg::SQ_PROGRAM_CNTL>();
  auto vgt_draw_initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();
  uint32_t vgt_indx_offset = regs.Get<reg::VGT_INDX_OFFSET>().indx_offset;
  uint32_t vgt_max_vtx_indx = regs.Get<reg::VGT_MAX_VTX_INDX>().max_indx;
  uint32_t vgt_min_vtx_indx = regs.Get<reg::VGT_MIN_VTX_INDX>().min_indx;

  bool edram_rov_used =
      render_target_cache_->GetPath() == RenderTargetCache::Path::kPixelShaderInterlock;
  uint32_t draw_resolution_scale_x = texture_cache_->draw_resolution_scale_x();
  uint32_t draw_resolution_scale_y = texture_cache_->draw_resolution_scale_y();

  // Get the color info register values for each render target. Also, for ROV,
  // exclude components that don't exist in the format from the write mask.
  // Don't exclude fully overlapping render targets, however - two render
  // targets with the same base address are used in the lighting pass of
  // 4D5307E6, for example, with the needed one picked with dynamic control
  // flow.
  reg::RB_COLOR_INFO color_infos[4];
  float rt_clamp[4][4];
  // Two UINT32_MAX if no components actually existing in the RT are written.
  uint32_t rt_keep_masks[4][2];
  for (uint32_t i = 0; i < 4; ++i) {
    auto color_info = regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[i]);
    color_infos[i] = color_info;
    if (edram_rov_used) {
      RenderTargetCache::GetPSIColorFormatInfo(
          color_info.color_format, (normalized_color_mask >> (i * 4)) & 0b1111, rt_clamp[i][0],
          rt_clamp[i][1], rt_clamp[i][2], rt_clamp[i][3], rt_keep_masks[i][0], rt_keep_masks[i][1]);
    }
  }

  // Disable depth and stencil if it aliases a color render target (for
  // instance, during the XBLA logo in 58410954, though depth writing is already
  // disabled there).
  bool depth_stencil_enabled =
      normalized_depth_control.stencil_enable || normalized_depth_control.z_enable;
  if (edram_rov_used && depth_stencil_enabled) {
    for (uint32_t i = 0; i < 4; ++i) {
      if (rb_depth_info.depth_base == color_infos[i].color_base &&
          (rt_keep_masks[i][0] != UINT32_MAX || rt_keep_masks[i][1] != UINT32_MAX)) {
        depth_stencil_enabled = false;
        break;
      }
    }
  }

  bool dirty = false;

  // Flags.
  uint32_t flags = 0;
  // Whether shared memory is an SRV or a UAV. Because a resource can't be in a
  // read-write (UAV) and a read-only (SRV, IBV) state at once, if any shader in
  // the pipeline uses memexport, the shared memory buffer must be a UAV.
  if (shared_memory_is_uav) {
    flags |= DxbcShaderTranslator::kSysFlag_SharedMemoryIsUAV;
  }
  // W0 division control.
  // http://www.x.org/docs/AMD/old/evergreen_3D_registers_v2.pdf
  // 8: VTX_XY_FMT = true: the incoming XY have already been multiplied by 1/W0.
  //               = false: multiply the X, Y coordinates by 1/W0.
  // 9: VTX_Z_FMT = true: the incoming Z has already been multiplied by 1/W0.
  //              = false: multiply the Z coordinate by 1/W0.
  // 10: VTX_W0_FMT = true: the incoming W0 is not 1/W0. Perform the reciprocal
  //                        to get 1/W0.
  if (pa_cl_vte_cntl.vtx_xy_fmt) {
    flags |= DxbcShaderTranslator::kSysFlag_XYDividedByW;
  }
  if (pa_cl_vte_cntl.vtx_z_fmt) {
    flags |= DxbcShaderTranslator::kSysFlag_ZDividedByW;
  }
  if (pa_cl_vte_cntl.vtx_w0_fmt) {
    flags |= DxbcShaderTranslator::kSysFlag_WNotReciprocal;
  }
  // Whether the primitive is polygonal and SV_IsFrontFace matters.
  if (primitive_polygonal) {
    flags |= DxbcShaderTranslator::kSysFlag_PrimitivePolygonal;
  }
  // Primitive type.
  if (draw_util::IsPrimitiveLine(regs)) {
    flags |= DxbcShaderTranslator::kSysFlag_PrimitiveLine;
  }
  // Depth format.
  if (rb_depth_info.depth_format == xenos::DepthRenderTargetFormat::kD24FS8) {
    flags |= DxbcShaderTranslator::kSysFlag_DepthFloat24;
  }
  // Alpha test.
  xenos::CompareFunction alpha_test_function = rb_colorcontrol.alpha_test_enable
                                                   ? rb_colorcontrol.alpha_func
                                                   : xenos::CompareFunction::kAlways;
  flags |= uint32_t(alpha_test_function) << DxbcShaderTranslator::kSysFlag_AlphaPassIfLess_Shift;
  // Gamma writing.
  if (!render_target_cache_->gamma_render_target_as_unorm16()) {
    for (uint32_t i = 0; i < 4; ++i) {
      if (color_infos[i].color_format == xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA) {
        flags |= DxbcShaderTranslator::kSysFlag_ConvertColor0ToGamma << i;
      }
    }
  }
  if (edram_rov_used && depth_stencil_enabled) {
    flags |= DxbcShaderTranslator::kSysFlag_ROVDepthStencil;
    if (normalized_depth_control.z_enable) {
      flags |= uint32_t(normalized_depth_control.zfunc)
               << DxbcShaderTranslator::kSysFlag_ROVDepthPassIfLess_Shift;
      if (normalized_depth_control.z_write_enable) {
        flags |= DxbcShaderTranslator::kSysFlag_ROVDepthWrite;
      }
    } else {
      // In case stencil is used without depth testing - always pass, and
      // don't modify the stored depth.
      flags |= DxbcShaderTranslator::kSysFlag_ROVDepthPassIfLess |
               DxbcShaderTranslator::kSysFlag_ROVDepthPassIfEqual |
               DxbcShaderTranslator::kSysFlag_ROVDepthPassIfGreater;
    }
    if (normalized_depth_control.stencil_enable) {
      flags |= DxbcShaderTranslator::kSysFlag_ROVStencilTest;
    }
    // Hint - if not applicable to the shader, will not have effect.
    if (alpha_test_function == xenos::CompareFunction::kAlways &&
        !rb_colorcontrol.alpha_to_mask_enable) {
      flags |= DxbcShaderTranslator::kSysFlag_ROVDepthStencilEarlyWrite;
    }
  }
  dirty |= system_constants_.flags != flags;
  system_constants_.flags = flags;

  // Tessellation factor range, plus 1.0 according to the images in
  // https://www.slideshare.net/blackdevilvikas/next-generation-graphics-programming-on-xbox-360
  float tessellation_factor_min = regs.Get<float>(XE_GPU_REG_VGT_HOS_MIN_TESS_LEVEL) + 1.0f;
  float tessellation_factor_max = regs.Get<float>(XE_GPU_REG_VGT_HOS_MAX_TESS_LEVEL) + 1.0f;
  dirty |= system_constants_.tessellation_factor_range_min != tessellation_factor_min;
  system_constants_.tessellation_factor_range_min = tessellation_factor_min;
  dirty |= system_constants_.tessellation_factor_range_max != tessellation_factor_max;
  system_constants_.tessellation_factor_range_max = tessellation_factor_max;

  // Line loop closing index (or 0 when drawing other primitives or using an
  // index buffer).
  dirty |= system_constants_.line_loop_closing_index != line_loop_closing_index;
  system_constants_.line_loop_closing_index = line_loop_closing_index;

  // Index or tessellation edge factor buffer endianness.
  dirty |= system_constants_.vertex_index_endian != index_endian;
  system_constants_.vertex_index_endian = index_endian;

  // Vertex index offset.
  dirty |= system_constants_.vertex_index_offset != vgt_indx_offset;
  system_constants_.vertex_index_offset = vgt_indx_offset;

  // Vertex index range.
  dirty |= system_constants_.vertex_index_min != vgt_min_vtx_indx;
  dirty |= system_constants_.vertex_index_max != vgt_max_vtx_indx;
  system_constants_.vertex_index_min = vgt_min_vtx_indx;
  system_constants_.vertex_index_max = vgt_max_vtx_indx;

  // User clip planes (UCP_ENA_#), when not CLIP_DISABLE.
  // The shader knows only the total count - tightly packing the user clip
  // planes that are actually used.
  if (!pa_cl_clip_cntl.clip_disable) {
    float* user_clip_plane_write_ptr = system_constants_.user_clip_planes[0];
    uint32_t user_clip_planes_remaining = pa_cl_clip_cntl.ucp_ena;
    uint32_t user_clip_plane_index;
    while (rex::bit_scan_forward(user_clip_planes_remaining, &user_clip_plane_index)) {
      user_clip_planes_remaining &= ~(UINT32_C(1) << user_clip_plane_index);
      const void* user_clip_plane_regs =
          &regs[XE_GPU_REG_PA_CL_UCP_0_X + user_clip_plane_index * 4];
      if (std::memcmp(user_clip_plane_write_ptr, user_clip_plane_regs, 4 * sizeof(float))) {
        dirty = true;
        std::memcpy(user_clip_plane_write_ptr, user_clip_plane_regs, 4 * sizeof(float));
      }
      user_clip_plane_write_ptr += 4;
    }
  }

  // Conversion to Direct3D 12 normalized device coordinates.
  // With DLSS, the scene draws (classified in IssueDraw) are shifted by the
  // frame's sub-pixel jitter - like a jittered projection, as the offset is
  // multiplied by W.
  float ndc_jitter[3] = {};
  if (dlss_ && dlss_scene_draw_ && (dlss_->jitter_x() != 0.0f || dlss_->jitter_y() != 0.0f) &&
      viewport_info.xy_extent[0] && viewport_info.xy_extent[1]) {
    ndc_jitter[0] = 2.0f * dlss_->jitter_x() / float(viewport_info.xy_extent[0]);
    ndc_jitter[1] = -2.0f * dlss_->jitter_y() / float(viewport_info.xy_extent[1]);
  }
  // Ultrawide experiment: every draw squeezed horizontally about the centre,
  // so a frame stretched to a wider window keeps its proportions (3D shows a
  // wider view; 2D keeps its shape).
  // Only the draws IssueDraw picked (ultrawide_draw_): off-screen passes and
  // full-screen copies keep their own layout.
  const float squeeze = ultrawide_draw_ ? float(REXCVAR_GET(ultrawide_squeeze)) : 1.0f;
  // The same for screens taller than 16:9 (16:10: the Steam Deck), vertically.
  const float squeeze_y = ultrawide_draw_ ? float(REXCVAR_GET(tall_squeeze)) : 1.0f;
  for (uint32_t i = 0; i < 3; ++i) {
    const float k = i == 0 && squeeze > 0.0f ? squeeze : i == 1 && squeeze_y > 0.0f ? squeeze_y : 1.0f;
    // --ui_scale first: about the guest viewport's centre (the frame's,
    // where guest clip space 0 lands: ndc_offset, which also carries the
    // predicated tile's window offset), then the squeeze.
    const float u = ui_scale_draw_ && i < 2 ? std::clamp(float(REXCVAR_GET(ui_scale)), 0.5f, 1.3f) : 1.0f;
    const float ui_offset = viewport_info.ndc_offset[i];
    const float ndc_scale = viewport_info.ndc_scale[i] * u * k;
    // A pinned sprite moves out by the bar the squeeze leaves on its side.
    const float pin = i == 0 ? ultrawide_pin_ * (1.0f - k) : 0.0f;
    const float ndc_offset = (ui_offset + ndc_jitter[i]) * k + pin;
    dirty |= system_constants_.ndc_scale[i] != ndc_scale;
    dirty |= system_constants_.ndc_offset[i] != ndc_offset;
    system_constants_.ndc_scale[i] = ndc_scale;
    system_constants_.ndc_offset[i] = ndc_offset;
  }

  // Point size.
  if (vgt_draw_initiator.prim_type == xenos::PrimitiveType::kPointList) {
    auto pa_su_point_minmax = regs.Get<reg::PA_SU_POINT_MINMAX>();
    auto pa_su_point_size = regs.Get<reg::PA_SU_POINT_SIZE>();
    float point_vertex_diameter_min = float(pa_su_point_minmax.min_size) * (2.0f / 16.0f);
    float point_vertex_diameter_max = float(pa_su_point_minmax.max_size) * (2.0f / 16.0f);
    float point_constant_diameter_x = float(pa_su_point_size.width) * (2.0f / 16.0f);
    float point_constant_diameter_y = float(pa_su_point_size.height) * (2.0f / 16.0f);
    dirty |= system_constants_.point_vertex_diameter_min != point_vertex_diameter_min;
    dirty |= system_constants_.point_vertex_diameter_max != point_vertex_diameter_max;
    dirty |= system_constants_.point_constant_diameter[0] != point_constant_diameter_x;
    dirty |= system_constants_.point_constant_diameter[1] != point_constant_diameter_y;
    system_constants_.point_vertex_diameter_min = point_vertex_diameter_min;
    system_constants_.point_vertex_diameter_max = point_vertex_diameter_max;
    system_constants_.point_constant_diameter[0] = point_constant_diameter_x;
    system_constants_.point_constant_diameter[1] = point_constant_diameter_y;
    // 2 because 1 in the NDC is half of the viewport's axis, 0.5 for diameter
    // to radius conversion to avoid multiplying the per-vertex diameter by an
    // additional constant in the shader.
    float point_screen_diameter_to_ndc_radius_x =
        (/* 0.5f * 2.0f * */ float(draw_resolution_scale_x)) /
        std::max(viewport_info.xy_extent[0], uint32_t(1));
    float point_screen_diameter_to_ndc_radius_y =
        (/* 0.5f * 2.0f * */ float(draw_resolution_scale_y)) /
        std::max(viewport_info.xy_extent[1], uint32_t(1));
    dirty |= system_constants_.point_screen_diameter_to_ndc_radius[0] !=
             point_screen_diameter_to_ndc_radius_x;
    dirty |= system_constants_.point_screen_diameter_to_ndc_radius[1] !=
             point_screen_diameter_to_ndc_radius_y;
    system_constants_.point_screen_diameter_to_ndc_radius[0] =
        point_screen_diameter_to_ndc_radius_x;
    system_constants_.point_screen_diameter_to_ndc_radius[1] =
        point_screen_diameter_to_ndc_radius_y;
  }

  // Texture signedness / gamma.
  uint32_t textures_resolution_scaled = 0;
  uint32_t textures_remaining = used_texture_mask;
  uint32_t texture_index;
  while (rex::bit_scan_forward(textures_remaining, &texture_index)) {
    textures_remaining &= ~(uint32_t(1) << texture_index);
    uint32_t& texture_signs_uint = system_constants_.texture_swizzled_signs[texture_index >> 2];
    uint32_t texture_signs_shift = (texture_index & 3) * 8;
    uint8_t texture_signs = texture_cache_->GetActiveTextureSwizzledSigns(texture_index);
    uint32_t texture_signs_shifted = uint32_t(texture_signs) << texture_signs_shift;
    uint32_t texture_signs_mask = uint32_t(0b11111111) << texture_signs_shift;
    dirty |= (texture_signs_uint & texture_signs_mask) != texture_signs_shifted;
    texture_signs_uint = (texture_signs_uint & ~texture_signs_mask) | texture_signs_shifted;
    textures_resolution_scaled |=
        uint32_t(texture_cache_->IsActiveTextureResolutionScaled(texture_index)) << texture_index;
  }
  dirty |= system_constants_.textures_resolution_scaled != textures_resolution_scaled;
  system_constants_.textures_resolution_scaled = textures_resolution_scaled;

  // Log2 of sample count, for alpha to mask and with ROV, for EDRAM address
  // calculation with MSAA.
  uint32_t sample_count_log2_x = rb_surface_info.msaa_samples >= xenos::MsaaSamples::k4X ? 1 : 0;
  uint32_t sample_count_log2_y = rb_surface_info.msaa_samples >= xenos::MsaaSamples::k2X ? 1 : 0;
  dirty |= system_constants_.sample_count_log2[0] != sample_count_log2_x;
  dirty |= system_constants_.sample_count_log2[1] != sample_count_log2_y;
  system_constants_.sample_count_log2[0] = sample_count_log2_x;
  system_constants_.sample_count_log2[1] = sample_count_log2_y;

  // Alpha test and alpha to coverage.
  dirty |= system_constants_.alpha_test_reference != rb_alpha_ref;
  system_constants_.alpha_test_reference = rb_alpha_ref;
  uint32_t alpha_to_mask =
      rb_colorcontrol.alpha_to_mask_enable ? (rb_colorcontrol.value >> 24) | (1 << 8) : 0;
  dirty |= system_constants_.alpha_to_mask != alpha_to_mask;
  system_constants_.alpha_to_mask = alpha_to_mask;

  uint32_t edram_tile_dwords_scaled = xenos::kEdramTileWidthSamples *
                                      xenos::kEdramTileHeightSamples *
                                      (draw_resolution_scale_x * draw_resolution_scale_y);

  // EDRAM pitch for ROV writing.
  if (edram_rov_used) {
    // Align, then multiply by 32bpp tile size in dwords.
    uint32_t edram_32bpp_tile_pitch_dwords_scaled =
        ((rb_surface_info.surface_pitch *
          (rb_surface_info.msaa_samples >= xenos::MsaaSamples::k4X ? 2 : 1)) +
         (xenos::kEdramTileWidthSamples - 1)) /
        xenos::kEdramTileWidthSamples * edram_tile_dwords_scaled;
    dirty |= system_constants_.edram_32bpp_tile_pitch_dwords_scaled !=
             edram_32bpp_tile_pitch_dwords_scaled;
    system_constants_.edram_32bpp_tile_pitch_dwords_scaled = edram_32bpp_tile_pitch_dwords_scaled;
  }

  // Color exponent bias and ROV render target writing.
  for (uint32_t i = 0; i < 4; ++i) {
    reg::RB_COLOR_INFO color_info = color_infos[i];
    // Exponent bias is in bits 20:25 of RB_COLOR_INFO.
    int32_t color_exp_bias = color_info.color_exp_bias;
    if (color_info.color_format == xenos::ColorRenderTargetFormat::k_16_16 ||
        color_info.color_format == xenos::ColorRenderTargetFormat::k_16_16_16_16) {
      if (render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets &&
          !render_target_cache_->IsFixed16TruncatedToMinus1To1()) {
        // Remap from -32...32 to -1...1 by dividing the output values by 32,
        // losing blending correctness, but getting the full range.
        color_exp_bias -= 5;
      }
    }
    auto color_exp_bias_scale =
        rex::memory::Reinterpret<float>(int32_t(0x3F800000 + (color_exp_bias << 23)));
    dirty |= system_constants_.color_exp_bias[i] != color_exp_bias_scale;
    system_constants_.color_exp_bias[i] = color_exp_bias_scale;
    if (edram_rov_used) {
      dirty |= system_constants_.edram_rt_keep_mask[i][0] != rt_keep_masks[i][0];
      system_constants_.edram_rt_keep_mask[i][0] = rt_keep_masks[i][0];
      dirty |= system_constants_.edram_rt_keep_mask[i][1] != rt_keep_masks[i][1];
      system_constants_.edram_rt_keep_mask[i][1] = rt_keep_masks[i][1];
      if (rt_keep_masks[i][0] != UINT32_MAX || rt_keep_masks[i][1] != UINT32_MAX) {
        uint32_t rt_base_dwords_scaled = color_info.color_base * edram_tile_dwords_scaled;
        dirty |= system_constants_.edram_rt_base_dwords_scaled[i] != rt_base_dwords_scaled;
        system_constants_.edram_rt_base_dwords_scaled[i] = rt_base_dwords_scaled;
        uint32_t format_flags = RenderTargetCache::AddPSIColorFormatFlags(color_info.color_format);
        dirty |= system_constants_.edram_rt_format_flags[i] != format_flags;
        system_constants_.edram_rt_format_flags[i] = format_flags;
        // Can't do float comparisons here because NaNs would result in always
        // setting the dirty flag.
        dirty |=
            std::memcmp(system_constants_.edram_rt_clamp[i], rt_clamp[i], 4 * sizeof(float)) != 0;
        std::memcpy(system_constants_.edram_rt_clamp[i], rt_clamp[i], 4 * sizeof(float));
        uint32_t blend_factors_ops =
            regs[reg::RB_BLENDCONTROL::rt_register_indices[i]] & 0x1FFF1FFF;
        dirty |= system_constants_.edram_rt_blend_factors_ops[i] != blend_factors_ops;
        system_constants_.edram_rt_blend_factors_ops[i] = blend_factors_ops;
      }
    }
  }

  if (edram_rov_used) {
    uint32_t depth_base_dwords_scaled = rb_depth_info.depth_base * edram_tile_dwords_scaled;
    dirty |= system_constants_.edram_depth_base_dwords_scaled != depth_base_dwords_scaled;
    system_constants_.edram_depth_base_dwords_scaled = depth_base_dwords_scaled;

    // For non-polygons, front polygon offset is used, and it's enabled if
    // POLY_OFFSET_PARA_ENABLED is set, for polygons, separate front and back
    // are used.
    float poly_offset_front_scale = 0.0f, poly_offset_front_offset = 0.0f;
    float poly_offset_back_scale = 0.0f, poly_offset_back_offset = 0.0f;
    if (primitive_polygonal) {
      if (pa_su_sc_mode_cntl.poly_offset_front_enable) {
        poly_offset_front_scale = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_SCALE);
        poly_offset_front_offset = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_OFFSET);
      }
      if (pa_su_sc_mode_cntl.poly_offset_back_enable) {
        poly_offset_back_scale = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_BACK_SCALE);
        poly_offset_back_offset = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_BACK_OFFSET);
      }
    } else {
      if (pa_su_sc_mode_cntl.poly_offset_para_enable) {
        poly_offset_front_scale = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_SCALE);
        poly_offset_front_offset = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_OFFSET);
        poly_offset_back_scale = poly_offset_front_scale;
        poly_offset_back_offset = poly_offset_front_offset;
      }
    }
    // With non-square resolution scaling, make sure the worst-case impact is
    // reverted (slope only along the scaled axis), thus max. More bias is
    // better than less bias, because less bias means Z fighting with the
    // background is more likely.
    float poly_offset_scale_factor = xenos::kPolygonOffsetScaleSubpixelUnit *
                                     std::max(draw_resolution_scale_x, draw_resolution_scale_y);
    poly_offset_front_scale *= poly_offset_scale_factor;
    poly_offset_back_scale *= poly_offset_scale_factor;
    dirty |= system_constants_.edram_poly_offset_front_scale != poly_offset_front_scale;
    system_constants_.edram_poly_offset_front_scale = poly_offset_front_scale;
    dirty |= system_constants_.edram_poly_offset_front_offset != poly_offset_front_offset;
    system_constants_.edram_poly_offset_front_offset = poly_offset_front_offset;
    dirty |= system_constants_.edram_poly_offset_back_scale != poly_offset_back_scale;
    system_constants_.edram_poly_offset_back_scale = poly_offset_back_scale;
    dirty |= system_constants_.edram_poly_offset_back_offset != poly_offset_back_offset;
    system_constants_.edram_poly_offset_back_offset = poly_offset_back_offset;

    if (depth_stencil_enabled && normalized_depth_control.stencil_enable) {
      dirty |= system_constants_.edram_stencil_front_reference != rb_stencilrefmask.stencilref;
      system_constants_.edram_stencil_front_reference = rb_stencilrefmask.stencilref;
      dirty |= system_constants_.edram_stencil_front_read_mask != rb_stencilrefmask.stencilmask;
      system_constants_.edram_stencil_front_read_mask = rb_stencilrefmask.stencilmask;
      dirty |=
          system_constants_.edram_stencil_front_write_mask != rb_stencilrefmask.stencilwritemask;
      system_constants_.edram_stencil_front_write_mask = rb_stencilrefmask.stencilwritemask;
      uint32_t stencil_func_ops = (normalized_depth_control.value >> 8) & ((1 << 12) - 1);
      dirty |= system_constants_.edram_stencil_front_func_ops != stencil_func_ops;
      system_constants_.edram_stencil_front_func_ops = stencil_func_ops;

      if (primitive_polygonal && normalized_depth_control.backface_enable) {
        dirty |= system_constants_.edram_stencil_back_reference != rb_stencilrefmask_bf.stencilref;
        system_constants_.edram_stencil_back_reference = rb_stencilrefmask_bf.stencilref;
        dirty |= system_constants_.edram_stencil_back_read_mask != rb_stencilrefmask_bf.stencilmask;
        system_constants_.edram_stencil_back_read_mask = rb_stencilrefmask_bf.stencilmask;
        dirty |= system_constants_.edram_stencil_back_write_mask !=
                 rb_stencilrefmask_bf.stencilwritemask;
        system_constants_.edram_stencil_back_write_mask = rb_stencilrefmask_bf.stencilwritemask;
        uint32_t stencil_func_ops_bf = (normalized_depth_control.value >> 20) & ((1 << 12) - 1);
        dirty |= system_constants_.edram_stencil_back_func_ops != stencil_func_ops_bf;
        system_constants_.edram_stencil_back_func_ops = stencil_func_ops_bf;
      } else {
        dirty |= std::memcmp(system_constants_.edram_stencil_back,
                             system_constants_.edram_stencil_front, 4 * sizeof(uint32_t)) != 0;
        std::memcpy(system_constants_.edram_stencil_back, system_constants_.edram_stencil_front,
                    4 * sizeof(uint32_t));
      }
    }

    dirty |= system_constants_.edram_blend_constant[0] != regs.Get<float>(XE_GPU_REG_RB_BLEND_RED);
    system_constants_.edram_blend_constant[0] = regs.Get<float>(XE_GPU_REG_RB_BLEND_RED);
    dirty |=
        system_constants_.edram_blend_constant[1] != regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN);
    system_constants_.edram_blend_constant[1] = regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN);
    dirty |= system_constants_.edram_blend_constant[2] != regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE);
    system_constants_.edram_blend_constant[2] = regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE);
    dirty |=
        system_constants_.edram_blend_constant[3] != regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA);
    system_constants_.edram_blend_constant[3] = regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA);
  }

  cbuffer_binding_system_.up_to_date &= !dirty;
}

bool D3D12CommandProcessor::UpdateBindings(const D3D12Shader* vertex_shader,
                                           const D3D12Shader* pixel_shader,
                                           ID3D12RootSignature* root_signature,
                                           bool shared_memory_is_uav) {
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  const RegisterFile& regs = *register_file_;

#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  // Set the new root signature.
  if (current_graphics_root_signature_ != root_signature) {
    current_graphics_root_signature_ = root_signature;
    tile_replay_root_signature_ = nullptr;
    if (!bindless_resources_used_) {
      GetRootBindfulExtraParameterIndices(vertex_shader, pixel_shader,
                                          current_graphics_root_bindful_extras_);
    }
    // Changing the root signature invalidates all bindings.
    current_graphics_root_up_to_date_ = 0;
    deferred_command_list_.D3DSetGraphicsRootSignature(root_signature);
  }

  // Select the root parameter indices depending on the used binding model.
  uint32_t root_parameter_fetch_constants = bindless_resources_used_
                                                ? kRootParameter_Bindless_FetchConstants
                                                : kRootParameter_Bindful_FetchConstants;
  uint32_t root_parameter_float_constants_vertex =
      bindless_resources_used_ ? kRootParameter_Bindless_FloatConstantsVertex
                               : kRootParameter_Bindful_FloatConstantsVertex;
  uint32_t root_parameter_float_constants_pixel = bindless_resources_used_
                                                      ? kRootParameter_Bindless_FloatConstantsPixel
                                                      : kRootParameter_Bindful_FloatConstantsPixel;
  uint32_t root_parameter_system_constants = bindless_resources_used_
                                                 ? kRootParameter_Bindless_SystemConstants
                                                 : kRootParameter_Bindful_SystemConstants;
  uint32_t root_parameter_bool_loop_constants = bindless_resources_used_
                                                    ? kRootParameter_Bindless_BoolLoopConstants
                                                    : kRootParameter_Bindful_BoolLoopConstants;
  uint32_t root_parameter_shared_memory_and_bindful_edram =
      bindless_resources_used_ ? kRootParameter_Bindless_SharedMemory
                               : kRootParameter_Bindful_SharedMemoryAndEdram;

  //
  // Update root constant buffers that are common for bindful and bindless.
  //

  // These are the constant base addresses/ranges for shaders.
  // We have these hardcoded right now cause nothing seems to differ on the Xbox
  // 360 (however, OpenGL ES on Adreno 200 on Android has different ranges).
  assert_true(regs[XE_GPU_REG_SQ_VS_CONST] == 0x000FF000 ||
              regs[XE_GPU_REG_SQ_VS_CONST] == 0x00000000);
  assert_true(regs[XE_GPU_REG_SQ_PS_CONST] == 0x000FF100 ||
              regs[XE_GPU_REG_SQ_PS_CONST] == 0x00000000);
  // Check if the float constant layout is still the same and get the counts.
  const Shader::ConstantRegisterMap& float_constant_map_vertex =
      vertex_shader->constant_register_map();
  uint32_t float_constant_count_vertex = float_constant_map_vertex.float_count;
  for (uint32_t i = 0; i < 4; ++i) {
    if (current_float_constant_map_vertex_[i] != float_constant_map_vertex.float_bitmap[i]) {
      current_float_constant_map_vertex_[i] = float_constant_map_vertex.float_bitmap[i];
      // If no float constants at all, we can reuse any buffer for them, so not
      // invalidating.
      if (float_constant_count_vertex) {
        cbuffer_binding_float_vertex_.up_to_date = false;
      }
    }
  }
  uint32_t float_constant_count_pixel = 0;
  if (pixel_shader != nullptr) {
    const Shader::ConstantRegisterMap& float_constant_map_pixel =
        pixel_shader->constant_register_map();
    float_constant_count_pixel = float_constant_map_pixel.float_count;
    for (uint32_t i = 0; i < 4; ++i) {
      if (current_float_constant_map_pixel_[i] != float_constant_map_pixel.float_bitmap[i]) {
        current_float_constant_map_pixel_[i] = float_constant_map_pixel.float_bitmap[i];
        if (float_constant_count_pixel) {
          cbuffer_binding_float_pixel_.up_to_date = false;
        }
      }
    }
  } else {
    std::memset(current_float_constant_map_pixel_, 0, sizeof(current_float_constant_map_pixel_));
  }

  // Write the constant buffer data.
  if (!cbuffer_binding_system_.up_to_date) {
    uint8_t* system_constants = constant_buffer_pool_->Request(
        frame_current_, sizeof(system_constants_), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
        nullptr, nullptr, &cbuffer_binding_system_.address);
    if (system_constants == nullptr) {
      return false;
    }
    std::memcpy(system_constants, &system_constants_, sizeof(system_constants_));
    cbuffer_binding_system_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << root_parameter_system_constants);
  }
  if (!cbuffer_binding_float_vertex_.up_to_date) {
    // Even if the shader doesn't need any float constants, a valid binding must
    // still be provided, so if the first draw in the frame with the current
    // root signature doesn't have float constants at all, still allocate an
    // empty buffer.
    if (!UploadFloatConstants(0, float_constant_map_vertex.float_bitmap, float_constant_count_vertex,
                              XE_GPU_REG_SHADER_CONSTANT_000_X,
                              cbuffer_binding_float_vertex_.address)) {
      return false;
    }
    cbuffer_binding_float_vertex_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << root_parameter_float_constants_vertex);
  }
  if (!cbuffer_binding_float_pixel_.up_to_date) {
    static const uint64_t kNoFloatConstants[4] = {};
    if (!UploadFloatConstants(
            1,
            pixel_shader ? pixel_shader->constant_register_map().float_bitmap : kNoFloatConstants,
            float_constant_count_pixel, XE_GPU_REG_SHADER_CONSTANT_256_X,
            cbuffer_binding_float_pixel_.address)) {
      return false;
    }
    cbuffer_binding_float_pixel_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << root_parameter_float_constants_pixel);
  }
  if (!cbuffer_binding_bool_loop_.up_to_date) {
    constexpr uint32_t kBoolLoopConstantsSize = (8 + 32) * sizeof(uint32_t);
    uint8_t* bool_loop_constants = constant_buffer_pool_->Request(
        frame_current_, kBoolLoopConstantsSize, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
        nullptr, nullptr, &cbuffer_binding_bool_loop_.address);
    if (bool_loop_constants == nullptr) {
      return false;
    }
    std::memcpy(bool_loop_constants, &regs[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031],
                kBoolLoopConstantsSize);
    cbuffer_binding_bool_loop_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << root_parameter_bool_loop_constants);
  }
  if (!cbuffer_binding_fetch_.up_to_date) {
    constexpr uint32_t kFetchConstantsSize = 32 * 6 * sizeof(uint32_t);
    uint8_t* fetch_constants = constant_buffer_pool_->Request(
        frame_current_, kFetchConstantsSize, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
        nullptr, nullptr, &cbuffer_binding_fetch_.address);
    if (fetch_constants == nullptr) {
      return false;
    }
    std::memcpy(fetch_constants, &regs[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0], kFetchConstantsSize);
    cbuffer_binding_fetch_.up_to_date = true;
    current_graphics_root_up_to_date_ &= ~(1u << root_parameter_fetch_constants);
  }

  //
  // Update descriptors.
  //

  if (!current_shared_memory_binding_is_uav_.has_value() ||
      current_shared_memory_binding_is_uav_.value() != shared_memory_is_uav) {
    current_shared_memory_binding_is_uav_ = shared_memory_is_uav;
    current_graphics_root_up_to_date_ &= ~(1u << root_parameter_shared_memory_and_bindful_edram);
  }

  // Get textures and samplers used by the vertex shader, check if the last used
  // samplers are compatible and update them.
  size_t texture_layout_uid_vertex = vertex_shader->GetTextureBindingLayoutUserUID();
  size_t sampler_layout_uid_vertex = vertex_shader->GetSamplerBindingLayoutUserUID();
  const std::vector<D3D12Shader::TextureBinding>& textures_vertex =
      vertex_shader->GetTextureBindingsAfterTranslation();
  const std::vector<D3D12Shader::SamplerBinding>& samplers_vertex =
      vertex_shader->GetSamplerBindingsAfterTranslation();
  size_t texture_count_vertex = textures_vertex.size();
  size_t sampler_count_vertex = samplers_vertex.size();
  if (sampler_count_vertex) {
    if (current_sampler_layout_uid_vertex_ != sampler_layout_uid_vertex) {
      current_sampler_layout_uid_vertex_ = sampler_layout_uid_vertex;
      cbuffer_binding_descriptor_indices_vertex_.up_to_date = false;
      bindful_samplers_written_vertex_ = false;
    }
    current_samplers_vertex_.resize(
        std::max(current_samplers_vertex_.size(), sampler_count_vertex));
    for (size_t i = 0; i < sampler_count_vertex; ++i) {
      D3D12TextureCache::SamplerParameters parameters =
          texture_cache_->GetSamplerParameters(samplers_vertex[i]);
      if (current_samplers_vertex_[i] != parameters) {
        cbuffer_binding_descriptor_indices_vertex_.up_to_date = false;
        bindful_samplers_written_vertex_ = false;
        current_samplers_vertex_[i] = parameters;
      }
    }
  }

  // Get textures and samplers used by the pixel shader, check if the last used
  // samplers are compatible and update them.
  size_t texture_layout_uid_pixel, sampler_layout_uid_pixel;
  const std::vector<D3D12Shader::TextureBinding>* textures_pixel;
  const std::vector<D3D12Shader::SamplerBinding>* samplers_pixel;
  size_t texture_count_pixel, sampler_count_pixel;
  if (pixel_shader != nullptr) {
    texture_layout_uid_pixel = pixel_shader->GetTextureBindingLayoutUserUID();
    sampler_layout_uid_pixel = pixel_shader->GetSamplerBindingLayoutUserUID();
    textures_pixel = &pixel_shader->GetTextureBindingsAfterTranslation();
    texture_count_pixel = textures_pixel->size();
    samplers_pixel = &pixel_shader->GetSamplerBindingsAfterTranslation();
    sampler_count_pixel = samplers_pixel->size();
    if (sampler_count_pixel) {
      if (current_sampler_layout_uid_pixel_ != sampler_layout_uid_pixel) {
        current_sampler_layout_uid_pixel_ = sampler_layout_uid_pixel;
        cbuffer_binding_descriptor_indices_pixel_.up_to_date = false;
        bindful_samplers_written_pixel_ = false;
      }
      current_samplers_pixel_.resize(
          std::max(current_samplers_pixel_.size(), size_t(sampler_count_pixel)));
      for (uint32_t i = 0; i < sampler_count_pixel; ++i) {
        D3D12TextureCache::SamplerParameters parameters =
            texture_cache_->GetSamplerParameters((*samplers_pixel)[i]);
        if (current_samplers_pixel_[i] != parameters) {
          current_samplers_pixel_[i] = parameters;
          cbuffer_binding_descriptor_indices_pixel_.up_to_date = false;
          bindful_samplers_written_pixel_ = false;
        }
      }
    }
  } else {
    texture_layout_uid_pixel = PipelineCache::kLayoutUIDEmpty;
    sampler_layout_uid_pixel = PipelineCache::kLayoutUIDEmpty;
    textures_pixel = nullptr;
    texture_count_pixel = 0;
    samplers_pixel = nullptr;
    sampler_count_pixel = 0;
  }

  assert_true(sampler_count_vertex + sampler_count_pixel <= kSamplerHeapSize);

  if (bindless_resources_used_) {
    //
    // Bindless descriptors path.
    //

    // Check if need to write new descriptor indices.
    // Samplers have already been checked.
    if (texture_count_vertex && cbuffer_binding_descriptor_indices_vertex_.up_to_date &&
        (current_texture_layout_uid_vertex_ != texture_layout_uid_vertex ||
         !texture_cache_->AreActiveTextureSRVKeysUpToDate(current_texture_srv_keys_vertex_.data(),
                                                          textures_vertex.data(),
                                                          texture_count_vertex))) {
      cbuffer_binding_descriptor_indices_vertex_.up_to_date = false;
    }
    if (texture_count_pixel && cbuffer_binding_descriptor_indices_pixel_.up_to_date &&
        (current_texture_layout_uid_pixel_ != texture_layout_uid_pixel ||
         !texture_cache_->AreActiveTextureSRVKeysUpToDate(current_texture_srv_keys_pixel_.data(),
                                                          textures_pixel->data(),
                                                          texture_count_pixel))) {
      cbuffer_binding_descriptor_indices_pixel_.up_to_date = false;
    }

    // Get sampler descriptor indices, write new samplers, and handle sampler
    // heap overflow if it happens.
    if ((sampler_count_vertex && !cbuffer_binding_descriptor_indices_vertex_.up_to_date) ||
        (sampler_count_pixel && !cbuffer_binding_descriptor_indices_pixel_.up_to_date)) {
      for (uint32_t i = 0; i < 2; ++i) {
        if (i) {
          // Overflow happened - invalidate sampler bindings because their
          // descriptor indices can't be used anymore (and even if heap creation
          // fails, because current_sampler_bindless_indices_#_ are in an
          // undefined state now) and switch to a new sampler heap.
          cbuffer_binding_descriptor_indices_vertex_.up_to_date = false;
          cbuffer_binding_descriptor_indices_pixel_.up_to_date = false;
          ID3D12DescriptorHeap* sampler_heap_new;
          if (!sampler_bindless_heaps_overflowed_.empty() &&
              sampler_bindless_heaps_overflowed_.front().second <= submission_completed_) {
            sampler_heap_new = sampler_bindless_heaps_overflowed_.front().first;
            sampler_bindless_heaps_overflowed_.pop_front();
          } else {
            D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_new_desc;
            sampler_heap_new_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
            sampler_heap_new_desc.NumDescriptors = kSamplerHeapSize;
            sampler_heap_new_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            sampler_heap_new_desc.NodeMask = 0;
            if (FAILED(device->CreateDescriptorHeap(&sampler_heap_new_desc,
                                                    IID_PPV_ARGS(&sampler_heap_new)))) {
              REXGPU_ERROR(
                  "Failed to create a new bindless sampler descriptor heap "
                  "after an overflow of the previous one");
              return false;
            }
          }
          // Only change the heap if a new heap was created successfully, not to
          // leave the values in an undefined state in case CreateDescriptorHeap
          // has failed.
          sampler_bindless_heaps_overflowed_.push_back(
              std::make_pair(sampler_bindless_heap_current_, submission_current_));
          sampler_bindless_heap_current_ = sampler_heap_new;
          sampler_bindless_heap_cpu_start_ =
              sampler_bindless_heap_current_->GetCPUDescriptorHandleForHeapStart();
          sampler_bindless_heap_gpu_start_ =
              sampler_bindless_heap_current_->GetGPUDescriptorHandleForHeapStart();
          sampler_bindless_heap_allocated_ = 0;
          // The only thing the heap is used for now is texture cache samplers -
          // invalidate all of them.
          texture_cache_bindless_sampler_map_.clear();
          ++tile_replay_sampler_heap_generation_;
          deferred_command_list_.SetDescriptorHeaps(view_bindless_heap_,
                                                    sampler_bindless_heap_current_);
          current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_SamplerHeap);
        }
        bool samplers_overflowed = false;
        if (sampler_count_vertex && !cbuffer_binding_descriptor_indices_vertex_.up_to_date) {
          current_sampler_bindless_indices_vertex_.resize(std::max(
              current_sampler_bindless_indices_vertex_.size(), size_t(sampler_count_vertex)));
          for (uint32_t j = 0; j < sampler_count_vertex; ++j) {
            D3D12TextureCache::SamplerParameters sampler_parameters = current_samplers_vertex_[j];
            uint32_t sampler_index;
            auto it = texture_cache_bindless_sampler_map_.find(sampler_parameters.value);
            if (it != texture_cache_bindless_sampler_map_.end()) {
              sampler_index = it->second;
            } else {
              if (sampler_bindless_heap_allocated_ >= kSamplerHeapSize) {
                samplers_overflowed = true;
                break;
              }
              sampler_index = sampler_bindless_heap_allocated_++;
              texture_cache_->WriteSampler(sampler_parameters,
                                           provider.OffsetSamplerDescriptor(
                                               sampler_bindless_heap_cpu_start_, sampler_index));
              texture_cache_bindless_sampler_map_.emplace(sampler_parameters.value, sampler_index);
            }
            current_sampler_bindless_indices_vertex_[j] = sampler_index;
          }
        }
        if (samplers_overflowed) {
          continue;
        }
        if (sampler_count_pixel && !cbuffer_binding_descriptor_indices_pixel_.up_to_date) {
          current_sampler_bindless_indices_pixel_.resize(std::max(
              current_sampler_bindless_indices_pixel_.size(), size_t(sampler_count_pixel)));
          for (uint32_t j = 0; j < sampler_count_pixel; ++j) {
            D3D12TextureCache::SamplerParameters sampler_parameters = current_samplers_pixel_[j];
            uint32_t sampler_index;
            auto it = texture_cache_bindless_sampler_map_.find(sampler_parameters.value);
            if (it != texture_cache_bindless_sampler_map_.end()) {
              sampler_index = it->second;
            } else {
              if (sampler_bindless_heap_allocated_ >= kSamplerHeapSize) {
                samplers_overflowed = true;
                break;
              }
              sampler_index = sampler_bindless_heap_allocated_++;
              texture_cache_->WriteSampler(sampler_parameters,
                                           provider.OffsetSamplerDescriptor(
                                               sampler_bindless_heap_cpu_start_, sampler_index));
              texture_cache_bindless_sampler_map_.emplace(sampler_parameters.value, sampler_index);
            }
            current_sampler_bindless_indices_pixel_[j] = sampler_index;
          }
        }
        if (!samplers_overflowed) {
          break;
        }
      }
    }

    if (!cbuffer_binding_descriptor_indices_vertex_.up_to_date) {
      uint32_t* descriptor_indices = reinterpret_cast<uint32_t*>(constant_buffer_pool_->Request(
          frame_current_,
          std::max(texture_count_vertex + sampler_count_vertex, size_t(1)) * sizeof(uint32_t),
          D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, nullptr, nullptr,
          &cbuffer_binding_descriptor_indices_vertex_.address));
      if (!descriptor_indices) {
        return false;
      }
      for (size_t i = 0; i < texture_count_vertex; ++i) {
        const D3D12Shader::TextureBinding& texture = textures_vertex[i];
        descriptor_indices[texture.bindless_descriptor_index] =
            texture_cache_->GetActiveTextureBindlessSRVIndex(texture) -
            uint32_t(SystemBindlessView::kUnboundedSRVsStart);
      }
      current_texture_layout_uid_vertex_ = texture_layout_uid_vertex;
      if (texture_count_vertex) {
        current_texture_srv_keys_vertex_.resize(
            std::max(current_texture_srv_keys_vertex_.size(), size_t(texture_count_vertex)));
        texture_cache_->WriteActiveTextureSRVKeys(current_texture_srv_keys_vertex_.data(),
                                                  textures_vertex.data(), texture_count_vertex);
      }
      // Current samplers have already been updated.
      for (size_t i = 0; i < sampler_count_vertex; ++i) {
        descriptor_indices[samplers_vertex[i].bindless_descriptor_index] =
            current_sampler_bindless_indices_vertex_[i];
      }
      cbuffer_binding_descriptor_indices_vertex_.up_to_date = true;
      current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_DescriptorIndicesVertex);
    }

    if (!cbuffer_binding_descriptor_indices_pixel_.up_to_date) {
      uint32_t* descriptor_indices = reinterpret_cast<uint32_t*>(constant_buffer_pool_->Request(
          frame_current_,
          std::max(texture_count_pixel + sampler_count_pixel, size_t(1)) * sizeof(uint32_t),
          D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, nullptr, nullptr,
          &cbuffer_binding_descriptor_indices_pixel_.address));
      if (!descriptor_indices) {
        return false;
      }
      for (size_t i = 0; i < texture_count_pixel; ++i) {
        const D3D12Shader::TextureBinding& texture = (*textures_pixel)[i];
        descriptor_indices[texture.bindless_descriptor_index] =
            texture_cache_->GetActiveTextureBindlessSRVIndex(texture) -
            uint32_t(SystemBindlessView::kUnboundedSRVsStart);
      }
      current_texture_layout_uid_pixel_ = texture_layout_uid_pixel;
      if (texture_count_pixel) {
        current_texture_srv_keys_pixel_.resize(
            std::max(current_texture_srv_keys_pixel_.size(), size_t(texture_count_pixel)));
        texture_cache_->WriteActiveTextureSRVKeys(current_texture_srv_keys_pixel_.data(),
                                                  textures_pixel->data(), texture_count_pixel);
      }
      // Current samplers have already been updated.
      for (size_t i = 0; i < sampler_count_pixel; ++i) {
        descriptor_indices[(*samplers_pixel)[i].bindless_descriptor_index] =
            current_sampler_bindless_indices_pixel_[i];
      }
      cbuffer_binding_descriptor_indices_pixel_.up_to_date = true;
      current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindless_DescriptorIndicesPixel);
    }
  } else {
    //
    // Bindful descriptors path.
    //

    // See what descriptors need to be updated.
    // Samplers have already been checked.
    bool write_textures_vertex =
        texture_count_vertex && (!bindful_textures_written_vertex_ ||
                                 current_texture_layout_uid_vertex_ != texture_layout_uid_vertex ||
                                 !texture_cache_->AreActiveTextureSRVKeysUpToDate(
                                     current_texture_srv_keys_vertex_.data(),
                                     textures_vertex.data(), texture_count_vertex));
    bool write_textures_pixel =
        texture_count_pixel &&
        (!bindful_textures_written_pixel_ ||
         current_texture_layout_uid_pixel_ != texture_layout_uid_pixel ||
         !texture_cache_->AreActiveTextureSRVKeysUpToDate(
             current_texture_srv_keys_pixel_.data(), textures_pixel->data(), texture_count_pixel));
    bool write_samplers_vertex = sampler_count_vertex && !bindful_samplers_written_vertex_;
    bool write_samplers_pixel = sampler_count_pixel && !bindful_samplers_written_pixel_;
    bool edram_rov_used =
        render_target_cache_->GetPath() == RenderTargetCache::Path::kPixelShaderInterlock;

    // Allocate the descriptors.
    size_t view_count_partial_update = 0;
    if (write_textures_vertex) {
      view_count_partial_update += texture_count_vertex;
    }
    if (write_textures_pixel) {
      view_count_partial_update += texture_count_pixel;
    }
    // Shared memory SRV and null UAV + null SRV and shared memory UAV +
    // textures.
    size_t view_count_full_update = 4 + texture_count_vertex + texture_count_pixel;
    if (edram_rov_used) {
      // + EDRAM UAV in two tables (with the shared memory SRV and with the
      // shared memory UAV).
      view_count_full_update += 2;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE view_cpu_handle;
    D3D12_GPU_DESCRIPTOR_HANDLE view_gpu_handle;
    uint32_t descriptor_size_view = provider.GetViewDescriptorSize();
    uint64_t view_heap_index = RequestViewBindfulDescriptors(
        draw_view_bindful_heap_index_, uint32_t(view_count_partial_update),
        uint32_t(view_count_full_update), view_cpu_handle, view_gpu_handle);
    if (view_heap_index == ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid) {
      REXGPU_ERROR("Failed to allocate view descriptors");
      return false;
    }
    size_t sampler_count_partial_update = 0;
    if (write_samplers_vertex) {
      sampler_count_partial_update += sampler_count_vertex;
    }
    if (write_samplers_pixel) {
      sampler_count_partial_update += sampler_count_pixel;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE sampler_cpu_handle = {};
    D3D12_GPU_DESCRIPTOR_HANDLE sampler_gpu_handle = {};
    uint32_t descriptor_size_sampler = provider.GetSamplerDescriptorSize();
    uint64_t sampler_heap_index = ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid;
    if (sampler_count_vertex != 0 || sampler_count_pixel != 0) {
      sampler_heap_index = RequestSamplerBindfulDescriptors(
          draw_sampler_bindful_heap_index_, uint32_t(sampler_count_partial_update),
          uint32_t(sampler_count_vertex + sampler_count_pixel), sampler_cpu_handle,
          sampler_gpu_handle);
      if (sampler_heap_index == ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid) {
        REXGPU_ERROR("Failed to allocate sampler descriptors");
        return false;
      }
    }
    if (draw_view_bindful_heap_index_ != view_heap_index) {
      // Need to update all view descriptors.
      write_textures_vertex = texture_count_vertex != 0;
      write_textures_pixel = texture_count_pixel != 0;
      bindful_textures_written_vertex_ = false;
      bindful_textures_written_pixel_ = false;
      // If updating fully, write the shared memory SRV and UAV descriptors and,
      // if needed, the EDRAM descriptor.
      // SRV + null UAV + EDRAM.
      gpu_handle_shared_memory_srv_and_edram_ = view_gpu_handle;
      shared_memory_->WriteRawSRVDescriptor(view_cpu_handle);
      view_cpu_handle.ptr += descriptor_size_view;
      view_gpu_handle.ptr += descriptor_size_view;
      ui::d3d12::util::CreateBufferRawUAV(device, view_cpu_handle, nullptr, 0);
      view_cpu_handle.ptr += descriptor_size_view;
      view_gpu_handle.ptr += descriptor_size_view;
      if (edram_rov_used) {
        render_target_cache_->WriteEdramUintPow2UAVDescriptor(view_cpu_handle, 2);
        view_cpu_handle.ptr += descriptor_size_view;
        view_gpu_handle.ptr += descriptor_size_view;
      }
      // Null SRV + UAV + EDRAM.
      gpu_handle_shared_memory_uav_and_edram_ = view_gpu_handle;
      ui::d3d12::util::CreateBufferRawSRV(device, view_cpu_handle, nullptr, 0);
      view_cpu_handle.ptr += descriptor_size_view;
      view_gpu_handle.ptr += descriptor_size_view;
      shared_memory_->WriteRawUAVDescriptor(view_cpu_handle);
      view_cpu_handle.ptr += descriptor_size_view;
      view_gpu_handle.ptr += descriptor_size_view;
      if (edram_rov_used) {
        render_target_cache_->WriteEdramUintPow2UAVDescriptor(view_cpu_handle, 2);
        view_cpu_handle.ptr += descriptor_size_view;
        view_gpu_handle.ptr += descriptor_size_view;
      }
      current_graphics_root_up_to_date_ &= ~(1u << kRootParameter_Bindful_SharedMemoryAndEdram);
    }
    if (sampler_heap_index != ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid &&
        draw_sampler_bindful_heap_index_ != sampler_heap_index) {
      write_samplers_vertex = sampler_count_vertex != 0;
      write_samplers_pixel = sampler_count_pixel != 0;
      bindful_samplers_written_vertex_ = false;
      bindful_samplers_written_pixel_ = false;
    }

    // Write the descriptors.
    if (write_textures_vertex) {
      assert_true(current_graphics_root_bindful_extras_.textures_vertex !=
                  RootBindfulExtraParameterIndices::kUnavailable);
      gpu_handle_textures_vertex_ = view_gpu_handle;
      for (size_t i = 0; i < texture_count_vertex; ++i) {
        texture_cache_->WriteActiveTextureBindfulSRV(textures_vertex[i], view_cpu_handle);
        view_cpu_handle.ptr += descriptor_size_view;
        view_gpu_handle.ptr += descriptor_size_view;
      }
      current_texture_layout_uid_vertex_ = texture_layout_uid_vertex;
      current_texture_srv_keys_vertex_.resize(
          std::max(current_texture_srv_keys_vertex_.size(), size_t(texture_count_vertex)));
      texture_cache_->WriteActiveTextureSRVKeys(current_texture_srv_keys_vertex_.data(),
                                                textures_vertex.data(), texture_count_vertex);
      bindful_textures_written_vertex_ = true;
      current_graphics_root_up_to_date_ &=
          ~(1u << current_graphics_root_bindful_extras_.textures_vertex);
    }
    if (write_textures_pixel) {
      assert_true(current_graphics_root_bindful_extras_.textures_pixel !=
                  RootBindfulExtraParameterIndices::kUnavailable);
      gpu_handle_textures_pixel_ = view_gpu_handle;
      for (size_t i = 0; i < texture_count_pixel; ++i) {
        texture_cache_->WriteActiveTextureBindfulSRV((*textures_pixel)[i], view_cpu_handle);
        view_cpu_handle.ptr += descriptor_size_view;
        view_gpu_handle.ptr += descriptor_size_view;
      }
      current_texture_layout_uid_pixel_ = texture_layout_uid_pixel;
      current_texture_srv_keys_pixel_.resize(
          std::max(current_texture_srv_keys_pixel_.size(), size_t(texture_count_pixel)));
      texture_cache_->WriteActiveTextureSRVKeys(current_texture_srv_keys_pixel_.data(),
                                                textures_pixel->data(), texture_count_pixel);
      bindful_textures_written_pixel_ = true;
      current_graphics_root_up_to_date_ &=
          ~(1u << current_graphics_root_bindful_extras_.textures_pixel);
    }
    if (write_samplers_vertex) {
      assert_true(current_graphics_root_bindful_extras_.samplers_vertex !=
                  RootBindfulExtraParameterIndices::kUnavailable);
      gpu_handle_samplers_vertex_ = sampler_gpu_handle;
      for (size_t i = 0; i < sampler_count_vertex; ++i) {
        texture_cache_->WriteSampler(current_samplers_vertex_[i], sampler_cpu_handle);
        sampler_cpu_handle.ptr += descriptor_size_sampler;
        sampler_gpu_handle.ptr += descriptor_size_sampler;
      }
      // Current samplers have already been updated.
      bindful_samplers_written_vertex_ = true;
      current_graphics_root_up_to_date_ &=
          ~(1u << current_graphics_root_bindful_extras_.samplers_vertex);
    }
    if (write_samplers_pixel) {
      assert_true(current_graphics_root_bindful_extras_.samplers_pixel !=
                  RootBindfulExtraParameterIndices::kUnavailable);
      gpu_handle_samplers_pixel_ = sampler_gpu_handle;
      for (size_t i = 0; i < sampler_count_pixel; ++i) {
        texture_cache_->WriteSampler(current_samplers_pixel_[i], sampler_cpu_handle);
        sampler_cpu_handle.ptr += descriptor_size_sampler;
        sampler_gpu_handle.ptr += descriptor_size_sampler;
      }
      // Current samplers have already been updated.
      bindful_samplers_written_pixel_ = true;
      current_graphics_root_up_to_date_ &=
          ~(1u << current_graphics_root_bindful_extras_.samplers_pixel);
    }

    // Wrote new descriptors on the current page.
    draw_view_bindful_heap_index_ = view_heap_index;
    if (sampler_heap_index != ui::d3d12::D3D12DescriptorHeapPool::kHeapIndexInvalid) {
      draw_sampler_bindful_heap_index_ = sampler_heap_index;
    }
  }

  // Update the root parameters.
  if (!(current_graphics_root_up_to_date_ & (1u << root_parameter_fetch_constants))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(root_parameter_fetch_constants,
                                                                cbuffer_binding_fetch_.address);
    current_graphics_root_up_to_date_ |= 1u << root_parameter_fetch_constants;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << root_parameter_float_constants_vertex))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
        root_parameter_float_constants_vertex, cbuffer_binding_float_vertex_.address);
    current_graphics_root_up_to_date_ |= 1u << root_parameter_float_constants_vertex;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << root_parameter_float_constants_pixel))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
        root_parameter_float_constants_pixel, cbuffer_binding_float_pixel_.address);
    current_graphics_root_up_to_date_ |= 1u << root_parameter_float_constants_pixel;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << root_parameter_system_constants))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(root_parameter_system_constants,
                                                                cbuffer_binding_system_.address);
    current_graphics_root_up_to_date_ |= 1u << root_parameter_system_constants;
  }
  if (!(current_graphics_root_up_to_date_ & (1u << root_parameter_bool_loop_constants))) {
    deferred_command_list_.D3DSetGraphicsRootConstantBufferView(root_parameter_bool_loop_constants,
                                                                cbuffer_binding_bool_loop_.address);
    current_graphics_root_up_to_date_ |= 1u << root_parameter_bool_loop_constants;
  }
  if (!(current_graphics_root_up_to_date_ &
        (1u << root_parameter_shared_memory_and_bindful_edram))) {
    assert_true(current_shared_memory_binding_is_uav_.has_value());
    D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle_shared_memory_and_bindful_edram;
    if (bindless_resources_used_) {
      gpu_handle_shared_memory_and_bindful_edram = provider.OffsetViewDescriptor(
          view_bindless_heap_gpu_start_,
          uint32_t(current_shared_memory_binding_is_uav_.value()
                       ? SystemBindlessView ::kNullRawSRVAndSharedMemoryRawUAVStart
                       : SystemBindlessView ::kSharedMemoryRawSRVAndNullRawUAVStart));
    } else {
      gpu_handle_shared_memory_and_bindful_edram = current_shared_memory_binding_is_uav_.value()
                                                       ? gpu_handle_shared_memory_uav_and_edram_
                                                       : gpu_handle_shared_memory_srv_and_edram_;
    }
    deferred_command_list_.D3DSetGraphicsRootDescriptorTable(
        root_parameter_shared_memory_and_bindful_edram, gpu_handle_shared_memory_and_bindful_edram);
    current_graphics_root_up_to_date_ |= 1u << root_parameter_shared_memory_and_bindful_edram;
  }
  uint32_t root_parameter_screen_extents = bindless_resources_used_
                                               ? kRootParameter_Bindless_ScreenExtents
                                               : kRootParameter_Bindful_ScreenExtents;
  if (!(current_graphics_root_up_to_date_ & (1u << root_parameter_screen_extents))) {
    deferred_command_list_.D3DSetGraphicsRootUnorderedAccessView(
        root_parameter_screen_extents, screen_extents_buffer_->GetGPUVirtualAddress() +
                                           uint64_t(screen_extents_root_slot_) * 16);
    current_graphics_root_up_to_date_ |= 1u << root_parameter_screen_extents;
  }
  if (bindless_resources_used_) {
    if (!(current_graphics_root_up_to_date_ &
          (1u << kRootParameter_Bindless_DescriptorIndicesPixel))) {
      deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
          kRootParameter_Bindless_DescriptorIndicesPixel,
          cbuffer_binding_descriptor_indices_pixel_.address);
      current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_DescriptorIndicesPixel;
    }
    if (!(current_graphics_root_up_to_date_ &
          (1u << kRootParameter_Bindless_DescriptorIndicesVertex))) {
      deferred_command_list_.D3DSetGraphicsRootConstantBufferView(
          kRootParameter_Bindless_DescriptorIndicesVertex,
          cbuffer_binding_descriptor_indices_vertex_.address);
      current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_DescriptorIndicesVertex;
    }
    if (!(current_graphics_root_up_to_date_ & (1u << kRootParameter_Bindless_SamplerHeap))) {
      deferred_command_list_.D3DSetGraphicsRootDescriptorTable(kRootParameter_Bindless_SamplerHeap,
                                                               sampler_bindless_heap_gpu_start_);
      current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_SamplerHeap;
    }
    if (!(current_graphics_root_up_to_date_ & (1u << kRootParameter_Bindless_ViewHeap))) {
      deferred_command_list_.D3DSetGraphicsRootDescriptorTable(kRootParameter_Bindless_ViewHeap,
                                                               view_bindless_heap_gpu_start_);
      current_graphics_root_up_to_date_ |= 1u << kRootParameter_Bindless_ViewHeap;
    }
  } else {
    uint32_t extra_index;
    extra_index = current_graphics_root_bindful_extras_.textures_pixel;
    if (extra_index != RootBindfulExtraParameterIndices::kUnavailable &&
        !(current_graphics_root_up_to_date_ & (1u << extra_index))) {
      deferred_command_list_.D3DSetGraphicsRootDescriptorTable(extra_index,
                                                               gpu_handle_textures_pixel_);
      current_graphics_root_up_to_date_ |= 1u << extra_index;
    }
    extra_index = current_graphics_root_bindful_extras_.samplers_pixel;
    if (extra_index != RootBindfulExtraParameterIndices::kUnavailable &&
        !(current_graphics_root_up_to_date_ & (1u << extra_index))) {
      deferred_command_list_.D3DSetGraphicsRootDescriptorTable(extra_index,
                                                               gpu_handle_samplers_pixel_);
      current_graphics_root_up_to_date_ |= 1u << extra_index;
    }
    extra_index = current_graphics_root_bindful_extras_.textures_vertex;
    if (extra_index != RootBindfulExtraParameterIndices::kUnavailable &&
        !(current_graphics_root_up_to_date_ & (1u << extra_index))) {
      deferred_command_list_.D3DSetGraphicsRootDescriptorTable(extra_index,
                                                               gpu_handle_textures_vertex_);
      current_graphics_root_up_to_date_ |= 1u << extra_index;
    }
    extra_index = current_graphics_root_bindful_extras_.samplers_vertex;
    if (extra_index != RootBindfulExtraParameterIndices::kUnavailable &&
        !(current_graphics_root_up_to_date_ & (1u << extra_index))) {
      deferred_command_list_.D3DSetGraphicsRootDescriptorTable(extra_index,
                                                               gpu_handle_samplers_vertex_);
      current_graphics_root_up_to_date_ |= 1u << extra_index;
    }
  }

  return true;
}

void D3D12CommandProcessor::EvictOldReadbackBuffers(
    std::unordered_map<uint64_t, ReadbackBuffer>& buffer_map) {
  if (buffer_map.empty()) {
    return;
  }
  const uint64_t eviction_frame_floor = (frame_current_ > kReadbackBufferEvictionAgeFrames)
                                            ? (frame_current_ - kReadbackBufferEvictionAgeFrames)
                                            : 0;
  for (auto it = buffer_map.begin(); it != buffer_map.end();) {
    ReadbackBuffer& readback = it->second;
    bool evict =
        buffer_map.size() > kMaxReadbackBuffers || readback.last_used_frame < eviction_frame_floor;
    if (!evict) {
      ++it;
      continue;
    }
    for (uint32_t i = 0; i < 2; ++i) {
      if (readback.buffers[i]) {
        if (readback.mapped_data[i]) {
          readback.buffers[i]->Unmap(0, nullptr);
        }
        readback.buffers[i]->Release();
      }
      readback.buffers[i] = nullptr;
      readback.mapped_data[i] = nullptr;
      readback.sizes[i] = 0;
      readback.submission_written[i] = 0;
      readback.written_size[i] = 0;
    }
    it = buffer_map.erase(it);
  }
}

ID3D12Resource* D3D12CommandProcessor::RequestReadbackBuffer(uint32_t size) {
  if (size == 0) {
    return nullptr;
  }
  size = rex::align(size, kReadbackBufferSizeIncrement);
  if (size > readback_buffer_size_) {
    const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
    ID3D12Device* device = provider.GetDevice();
    D3D12_RESOURCE_DESC buffer_desc;
    ui::d3d12::util::FillBufferResourceDesc(buffer_desc, size, D3D12_RESOURCE_FLAG_NONE);
    ID3D12Resource* buffer;
    if (FAILED(device->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesReadback, provider.GetHeapFlagCreateNotZeroed(),
            &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&buffer)))) {
      REXGPU_ERROR("Failed to create a {} MB readback buffer", size >> 20);
      return nullptr;
    }
    if (readback_buffer_ != nullptr) {
      readback_buffer_->Release();
    }
    readback_buffer_ = buffer;
    readback_buffer_size_ = size;
  }
  return readback_buffer_;
}

bool D3D12CommandProcessor::InitializeOcclusionQueryResources() {
  active_occlusion_query_ = {};
  occlusion_query_cursor_ = 0;
  occlusion_query_resources_available_ = false;
  occlusion_query_heap_.Reset();
  occlusion_query_readback_.Reset();
  occlusion_query_readback_mapping_ = nullptr;

  ID3D12Device* device = GetD3D12Provider().GetDevice();
  if (!device) {
    return false;
  }

  D3D12_QUERY_HEAP_DESC heap_desc;
  heap_desc.Type = D3D12_QUERY_HEAP_TYPE_OCCLUSION;
  heap_desc.Count = kMaxOcclusionQueries;
  heap_desc.NodeMask = 0;
  if (FAILED(device->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&occlusion_query_heap_)))) {
    REXGPU_WARN(
        "D3D12CommandProcessor: Failed to create occlusion query heap, using fake sample counts");
    return false;
  }

  D3D12_RESOURCE_DESC buffer_desc;
  ui::d3d12::util::FillBufferResourceDesc(buffer_desc, sizeof(uint64_t) * kMaxOcclusionQueries,
                                          D3D12_RESOURCE_FLAG_NONE);
  if (FAILED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesReadback,
                                             GetD3D12Provider().GetHeapFlagCreateNotZeroed(),
                                             &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                             IID_PPV_ARGS(&occlusion_query_readback_)))) {
    REXGPU_WARN(
        "D3D12CommandProcessor: Failed to allocate occlusion query readback buffer, using fake "
        "sample counts");
    occlusion_query_heap_.Reset();
    return false;
  }

  D3D12_RANGE read_range = {0, sizeof(uint64_t) * kMaxOcclusionQueries};
  void* mapping = nullptr;
  if (FAILED(occlusion_query_readback_->Map(0, &read_range, &mapping))) {
    REXGPU_WARN(
        "D3D12CommandProcessor: Failed to map occlusion query readback buffer, using fake sample "
        "counts");
    occlusion_query_readback_.Reset();
    occlusion_query_heap_.Reset();
    return false;
  }

  occlusion_query_readback_mapping_ = reinterpret_cast<uint64_t*>(mapping);
  occlusion_query_resources_available_ = true;
  return true;
}

void D3D12CommandProcessor::ShutdownOcclusionQueryResources() {
  DisableHostOcclusionQueries();

  if (occlusion_query_readback_ && occlusion_query_readback_mapping_) {
    occlusion_query_readback_->Unmap(0, nullptr);
  }
  occlusion_query_readback_mapping_ = nullptr;
  occlusion_query_readback_.Reset();
  occlusion_query_heap_.Reset();
}

bool D3D12CommandProcessor::AcquireOcclusionQueryIndex(uint32_t& host_index_out) {
  if (occlusion_query_cursor_ >= kMaxOcclusionQueries) {
    occlusion_query_cursor_ = 0;
  }
  host_index_out = occlusion_query_cursor_++;
  return true;
}

void D3D12CommandProcessor::DisableHostOcclusionQueries() {
  if (active_occlusion_query_.valid && occlusion_query_heap_) {
    uint32_t host_index = active_occlusion_query_.host_index;
    // Clear before EndSubmission to prevent the EndSubmission safety net from issuing a second
    // EndQuery for the same index.
    active_occlusion_query_ = {};
    if (BeginSubmission(true)) {
      deferred_command_list_.D3DEndQuery(occlusion_query_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION,
                                         host_index);
      EndSubmission(false);
    }
  } else {
    active_occlusion_query_ = {};
  }
  occlusion_query_cursor_ = 0;
  occlusion_query_resources_available_ = false;
}

bool D3D12CommandProcessor::BeginGuestOcclusionQuery(uint32_t sample_count_address) {
  if (!REXCVAR_GET(occlusion_query_enable) || !occlusion_query_resources_available_) {
    return false;
  }
  if (active_occlusion_query_.valid) {
    REXGPU_WARN(
        "D3D12CommandProcessor: Occlusion query begin issued while another query is active");
    DisableHostOcclusionQueries();
    return false;
  }

  uint32_t host_index = 0;
  if (!AcquireOcclusionQueryIndex(host_index)) {
    return false;
  }
  if (!BeginSubmission(true)) {
    return false;
  }

  deferred_command_list_.D3DBeginQuery(occlusion_query_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION,
                                       host_index);
  active_occlusion_query_.sample_count_address = sample_count_address;
  active_occlusion_query_.host_index = host_index;
  active_occlusion_query_.valid = true;
  return true;
}

bool D3D12CommandProcessor::EndGuestOcclusionQuery(
    uint32_t sample_count_address, xenos::xe_gpu_depth_sample_counts* sample_counts) {
  if (!REXCVAR_GET(occlusion_query_enable) || !occlusion_query_resources_available_ ||
      !active_occlusion_query_.valid || !occlusion_query_heap_ || !occlusion_query_readback_) {
    return false;
  }

  uint32_t host_index = active_occlusion_query_.host_index;
  active_occlusion_query_ = {};
  if (rex::perf::IsGpuProfiling()) {
    rex::perf::GetGpuProfileStats().occlusion_queries.fetch_add(1, std::memory_order_relaxed);
  }

  if (!BeginSubmission(true)) {
    return false;
  }

  deferred_command_list_.D3DEndQuery(occlusion_query_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION,
                                     host_index);
  deferred_command_list_.D3DResolveQueryData(
      occlusion_query_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION, host_index, 1,
      occlusion_query_readback_.Get(), sizeof(uint64_t) * host_index);

  if (!EndSubmission(false)) {
    return false;
  }

  uint64_t query_submission = submission_current_ ? submission_current_ - 1 : 0;
  CheckSubmissionFence(query_submission);
  if (submission_completed_ < query_submission) {
    return false;
  }
  if (!occlusion_query_readback_mapping_) {
    return false;
  }

  uint64_t samples = occlusion_query_readback_mapping_[host_index];
  samples = NormalizeOcclusionSamples(samples);
  WriteGuestOcclusionResult(sample_counts, samples);
  return true;
}

namespace {

// Captures the position and interpolator 0 a vertex shader exports.
class ClearDrawExportSink : public ShaderInterpreter::ExportSink {
 public:
  void Export(ucode::ExportRegister export_register, const float* value,
              uint32_t value_mask) override {
    float* dest = nullptr;
    uint32_t* written = nullptr;
    if (export_register == ucode::ExportRegister::kVSPosition) {
      dest = position;
      written = &position_written;
    } else if (export_register == ucode::ExportRegister::kVSInterpolator0) {
      dest = color;
      written = &color_written;
    } else if (export_register == ucode::ExportRegister::kVSPointSizeEdgeFlagKillVertex) {
      other_written = true;
      return;
    } else {
      return;
    }
    for (uint32_t i = 0; i < 4; ++i) {
      if (value_mask & (uint32_t(1) << i)) {
        dest[i] = value[i];
      }
    }
    *written |= value_mask;
  }
  void Reset() {
    position_written = 0;
    color_written = 0;
    other_written = false;
  }
  float position[4] = {};
  float color[4] = {};
  uint32_t position_written = 0;
  uint32_t color_written = 0;
  bool other_written = false;
};

// Pixel shaders of clears done by drawing: oC0 = interpolator 0.
// 2E372EA28CC404B7: the Xbox 360 Direct3D 9 Clear (exec: max oC0, r0, r0).
// 2B4F1D2C5EDCC366: the same in launch titles (Amped 3), other exec control words.
constexpr uint64_t kClearDrawPixelShaders[] = {0x2E372EA28CC404B7ull, 0x2B4F1D2C5EDCC366ull};

}  // namespace

bool D3D12CommandProcessor::TryClearDrawInPlace(const Shader& vertex_shader,
                                                const Shader* pixel_shader,
                                                reg::RB_DEPTHCONTROL normalized_depth_control) {
  const RegisterFile& regs = *register_file_;
  // A screen-space rectangle (no clipping, no viewport transform), 3 vertices.
  const auto vgt_draw_initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();
  if (vgt_draw_initiator.prim_type != xenos::PrimitiveType::kRectangleList ||
      vgt_draw_initiator.source_select != xenos::SourceSelect::kAutoIndex ||
      vgt_draw_initiator.num_indices != 3 ||
      (xenos::IsMajorModeExplicit(vgt_draw_initiator.major_mode, vgt_draw_initiator.prim_type) &&
       regs.Get<reg::VGT_OUTPUT_PATH_CNTL>().path_select ==
           xenos::VGTOutputPath::kTessellationEnable)) {
    return false;
  }
  const auto vte = regs.Get<reg::PA_CL_VTE_CNTL>();
  if ((vte.value & 0x3F) || !vte.vtx_xy_fmt || !vte.vtx_z_fmt ||
      !regs.Get<reg::PA_CL_CLIP_CNTL>().clip_disable) {
    return false;
  }
  const xenos::EdramMode edram_mode = regs.Get<reg::RB_MODECONTROL>().edram_mode;
  if (edram_mode != xenos::EdramMode::kColorDepth && edram_mode != xenos::EdramMode::kDepthOnly) {
    return false;
  }
  const auto rb_surface_info = regs.Get<reg::RB_SURFACE_INFO>();
  if (!rb_surface_info.surface_pitch || rb_surface_info.msaa_samples > xenos::MsaaSamples::k4X) {
    return false;
  }
  // Every sample written.
  if (rb_surface_info.msaa_samples != xenos::MsaaSamples::k1X &&
      (regs[XE_GPU_REG_PA_SC_AA_MASK] & 0xFFFF) != 0xFFFF) {
    return false;
  }

  // Depth / stencil: always passing and replacing everything, or not used.
  const reg::RB_DEPTHCONTROL depth_control = normalized_depth_control;
  bool depth = false, stencil = false;
  uint32_t stencil_value = 0;
  if (depth_control.z_enable || depth_control.stencil_enable) {
    if (!depth_control.z_enable || !depth_control.z_write_enable ||
        depth_control.zfunc != xenos::CompareFunction::kAlways) {
      return false;
    }
    depth = true;
    if (depth_control.stencil_enable) {
      const auto ref_mask = regs.Get<reg::RB_STENCILREFMASK>();
      if (depth_control.stencilfunc != xenos::CompareFunction::kAlways ||
          (depth_control.stencilzpass != xenos::StencilOp::kReplace &&
           depth_control.stencilzpass != xenos::StencilOp::kZero) ||
          ref_mask.stencilwritemask != 0xFF || depth_control.backface_enable) {
        return false;
      }
      stencil = true;
      stencil_value =
          depth_control.stencilzpass == xenos::StencilOp::kReplace ? ref_mask.stencilref : 0;
    }
  }

  // Color: only render target 0, all components, no blending, the clear
  // pixel shader.
  bool color = false;
  if (edram_mode == xenos::EdramMode::kColorDepth && pixel_shader) {
    const uint64_t ps_hash = pixel_shader->ucode_data_hash();
    if (std::find(std::begin(kClearDrawPixelShaders), std::end(kClearDrawPixelShaders), ps_hash) ==
        std::end(kClearDrawPixelShaders)) {
      return false;
    }
    const uint32_t color_mask =
        draw_util::GetNormalizedColorMask(regs, pixel_shader->writes_color_targets());
    if (color_mask & ~uint32_t(0xF)) {
      return false;
    }
    if (color_mask) {
      if (color_mask != 0xF) {
        return false;
      }
      const auto blend = regs.Get<reg::RB_BLENDCONTROL>(reg::RB_BLENDCONTROL::rt_register_indices[0]);
      const auto color_control = regs.Get<reg::RB_COLORCONTROL>();
      if (blend.color_srcblend != xenos::BlendFactor::kOne ||
          blend.color_destblend != xenos::BlendFactor::kZero ||
          blend.color_comb_fcn != xenos::BlendOp::kAdd ||
          blend.alpha_srcblend != xenos::BlendFactor::kOne ||
          blend.alpha_destblend != xenos::BlendFactor::kZero ||
          blend.alpha_comb_fcn != xenos::BlendOp::kAdd ||
          (color_control.alpha_test_enable &&
           color_control.alpha_func != xenos::CompareFunction::kAlways) ||
          color_control.alpha_to_mask_enable) {
        return false;
      }
      const auto color_info = regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[0]);
      if (color_info.color_exp_bias) {
        return false;
      }
      color = true;
    }
  } else if (pixel_shader) {
    return false;
  }
  if (!depth && !color) {
    return false;
  }

  // The vertices, through the vertex shader on the CPU.
  if (!ShaderInterpreter::CanInterpretShader(vertex_shader)) {
    return false;
  }
  if (!clear_draw_interpreter_) {
    clear_draw_interpreter_ = std::make_unique<ShaderInterpreter>(regs, *memory_);
  }
  ShaderInterpreter& interpreter = *clear_draw_interpreter_;
  ClearDrawExportSink sink;
  interpreter.SetShader(vertex_shader);
  interpreter.SetExportSink(&sink);
  const uint32_t index_offset = regs.Get<reg::VGT_INDX_OFFSET>().indx_offset;
  const uint32_t min_index = regs.Get<reg::VGT_MIN_VTX_INDX>().min_indx;
  const uint32_t max_index = regs.Get<reg::VGT_MAX_VTX_INDX>().max_indx;
  float x[3], y[3], z[3], rgba[4] = {};
  bool vertices_valid = true;
  for (uint32_t i = 0; i < 3 && vertices_valid; ++i) {
    const uint32_t vertex_index = std::min(max_index, std::max(min_index, (i + index_offset) & 0xFFFFFF));
    sink.Reset();
    std::memset(interpreter.temp_registers(), 0, sizeof(float) * 4);
    interpreter.temp_registers()[0] = float(vertex_index);
    interpreter.Execute();
    if (sink.other_written || (sink.position_written & 0b0111) != 0b0111 ||
        (color && (sink.color_written & 0xF) != 0xF)) {
      vertices_valid = false;
      break;
    }
    x[i] = sink.position[0];
    y[i] = sink.position[1];
    z[i] = sink.position[2];
    if (color) {
      if (!i) {
        std::memcpy(rgba, sink.color, sizeof(rgba));
      } else if (std::memcmp(rgba, sink.color, sizeof(rgba))) {
        vertices_valid = false;
      }
    }
    if (i && z[i] != z[0]) {
      vertices_valid = false;
    }
  }
  interpreter.SetExportSink(nullptr);
  if (!vertices_valid) {
    return false;
  }
  // An axis-aligned rectangle with edges between pixels.
  float x_min = std::min({x[0], x[1], x[2]}), x_max = std::max({x[0], x[1], x[2]});
  float y_min = std::min({y[0], y[1], y[2]}), y_max = std::max({y[0], y[1], y[2]});
  for (uint32_t i = 0; i < 3; ++i) {
    if ((x[i] != x_min && x[i] != x_max) || (y[i] != y_min && y[i] != y_max)) {
      return false;
    }
  }
  // Pixel centers at integers with kD3DZero, at .5 otherwise.
  const float center_offset =
      regs.Get<reg::PA_SU_VTX_CNTL>().pix_center == xenos::PixelCenter::kD3DZero ? 0.5f : 0.0f;
  x_min += center_offset;
  x_max += center_offset;
  y_min += center_offset;
  y_max += center_offset;
  if (!(x_min >= -8192.0f && x_max <= 16384.0f && y_min >= -8192.0f && y_max <= 16384.0f) ||
      x_min != std::floor(x_min) || x_max != std::floor(x_max) || y_min != std::floor(y_min) ||
      y_max != std::floor(y_max)) {
    return false;
  }
  int32_t rect_x0 = int32_t(x_min), rect_x1 = int32_t(x_max);
  int32_t rect_y0 = int32_t(y_min), rect_y1 = int32_t(y_max);
  if (regs.Get<reg::PA_SU_SC_MODE_CNTL>().vtx_window_offset_enable) {
    const auto window_offset = regs.Get<reg::PA_SC_WINDOW_OFFSET>();
    rect_x0 += window_offset.window_x_offset;
    rect_x1 += window_offset.window_x_offset;
    rect_y0 += window_offset.window_y_offset;
    rect_y1 += window_offset.window_y_offset;
  }
  draw_util::Scissor scissor;
  draw_util::GetScissor(regs, scissor);
  rect_x0 = std::max(rect_x0, int32_t(scissor.offset[0]));
  rect_y0 = std::max(rect_y0, int32_t(scissor.offset[1]));
  rect_x1 = std::min(rect_x1, int32_t(scissor.offset[0] + scissor.extent[0]));
  rect_y1 = std::min(rect_y1, int32_t(scissor.offset[1] + scissor.extent[1]));
  rect_x1 = std::min(rect_x1, int32_t(rb_surface_info.surface_pitch));
  if (rect_x1 <= rect_x0 || rect_y1 <= rect_y0) {
    // Nothing drawn - leave it to the normal path.
    return false;
  }

  D3D12RenderTargetCache::DrawnClear clear;
  const uint32_t msaa_x_log2 = uint32_t(rb_surface_info.msaa_samples >= xenos::MsaaSamples::k4X);
  clear.pitch_tiles_at_32bpp =
      ((rb_surface_info.surface_pitch << msaa_x_log2) + (xenos::kEdramTileWidthSamples - 1)) /
      xenos::kEdramTileWidthSamples;
  clear.msaa_samples = rb_surface_info.msaa_samples;
  clear.x0 = uint32_t(rect_x0);
  clear.y0 = uint32_t(rect_y0);
  clear.x1 = uint32_t(rect_x1);
  clear.y1 = uint32_t(rect_y1);
  if (depth) {
    const auto depth_info = regs.Get<reg::RB_DEPTH_INFO>();
    clear.depth = true;
    clear.depth_base = depth_info.depth_base;
    clear.depth_format = depth_info.depth_format;
    clear.depth_value = z[0];
    clear.stencil = stencil;
    clear.stencil_value = stencil_value;
    if (!(z[0] >= 0.0f && z[0] <= 1.0f)) {
      // Clamped on the host, but keep the exact path for unusual values -
      // except tiny negative ones (0 written by the guest's math).
      if (z[0] < 0.0f && z[0] > -1.0e-6f) {
        clear.depth_value = 0.0f;
      } else {
        return false;
      }
    }
  }
  if (color) {
    const auto color_info = regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[0]);
    clear.color = true;
    clear.color_base = color_info.color_base;
    clear.color_format = color_info.color_format;
    std::memcpy(clear.color_value, rgba, sizeof(rgba));
  }
  GpuProfilePassScope pass_scope(*this, rex::perf::GpuPass::kResolveClear);
  if (!render_target_cache_->ClearDrawnInPlace(clear)) {
    return false;
  }
  if (rex::perf::IsGpuProfiling()) {
    rex::perf::GetGpuProfileStats().clears_in_place.fetch_add(1, std::memory_order_relaxed);
  }
  return true;
}

void D3D12CommandProcessor::GpuProfileBeginSubmission() {
  if (!rex::perf::IsGpuProfiling() || gpu_profile_failed_) {
    return;
  }
  const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
  if (!gpu_profile_query_heap_) {
    ID3D12Device* device = provider.GetDevice();
    UINT64 frequency = 0;
    if (FAILED(provider.GetDirectQueue()->GetTimestampFrequency(&frequency)) || !frequency) {
      gpu_profile_failed_ = true;
      return;
    }
    gpu_profile_ns_per_tick_ = 1.0e9 / double(frequency);
    D3D12_QUERY_HEAP_DESC heap_desc = {};
    heap_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    heap_desc.Count = kGpuProfileQueryCount;
    D3D12_RESOURCE_DESC buffer_desc;
    ui::d3d12::util::FillBufferResourceDesc(buffer_desc, sizeof(uint64_t) * kGpuProfileQueryCount,
                                            D3D12_RESOURCE_FLAG_NONE);
    void* mapping = nullptr;
    D3D12_RANGE read_range = {0, sizeof(uint64_t) * kGpuProfileQueryCount};
    if (FAILED(device->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&gpu_profile_query_heap_))) ||
        FAILED(device->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesReadback, provider.GetHeapFlagCreateNotZeroed(),
            &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&gpu_profile_readback_))) ||
        FAILED(gpu_profile_readback_->Map(0, &read_range, &mapping))) {
      REXGPU_WARN("gpu_profile: couldn't create the timestamp query heap");
      gpu_profile_query_heap_.Reset();
      gpu_profile_readback_.Reset();
      gpu_profile_failed_ = true;
      return;
    }
    gpu_profile_readback_mapping_ = static_cast<const uint64_t*>(mapping);
    REXGPU_INFO("gpu_profile: timestamps at {} Hz", frequency);
  }
  // A range of the ring not used by submissions still in flight.
  uint32_t first = gpu_profile_query_cursor_;
  if (first + kGpuProfileQueriesPerSubmission > kGpuProfileQueryCount) {
    first = 0;
  }
  for (const GpuProfileSubmission& pending : gpu_profile_submissions_) {
    if (first < pending.first_query + pending.query_count &&
        pending.first_query < first + kGpuProfileQueriesPerSubmission) {
      rex::perf::GetGpuProfileStats().submissions_unprofiled.fetch_add(1,
                                                                       std::memory_order_relaxed);
      return;
    }
  }
  deferred_command_list_.BeginProfiling(gpu_profile_query_heap_.Get(), first,
                                        kGpuProfileQueriesPerSubmission);
}

void D3D12CommandProcessor::GpuProfileEndSubmission() {
  const uint32_t query_count = deferred_command_list_.EndProfiling();
  if (query_count < 2) {
    return;
  }
  GpuProfileSubmission record;
  record.submission = submission_current_;
  record.passes = deferred_command_list_.profile_passes();
  record.query_count = query_count;
  record.first_query = 0;
  // The first query is where BeginProfiling put it: the cursor or 0.
  record.first_query = gpu_profile_query_cursor_;
  if (record.first_query + kGpuProfileQueriesPerSubmission > kGpuProfileQueryCount) {
    record.first_query = 0;
  }
  deferred_command_list_.D3DResolveQueryData(
      gpu_profile_query_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, record.first_query, query_count,
      gpu_profile_readback_.Get(), sizeof(uint64_t) * record.first_query);
  gpu_profile_query_cursor_ = record.first_query + query_count;
  gpu_profile_submissions_.push_back(std::move(record));
}

void D3D12CommandProcessor::GpuProfileReadCompleted() {
  rex::perf::GpuProfileStats& stats = rex::perf::GetGpuProfileStats();
  while (!gpu_profile_submissions_.empty() &&
         gpu_profile_submissions_.front().submission <= submission_completed_) {
    const GpuProfileSubmission& record = gpu_profile_submissions_.front();
    const uint64_t* ticks = gpu_profile_readback_mapping_ + record.first_query;
    if (stats.enabled.load(std::memory_order_relaxed)) {
      for (uint32_t i = 0; i + 1 < record.query_count; ++i) {
        const uint32_t pass = record.passes[i];
        if (pass >= uint32_t(rex::perf::GpuPass::kCount) || ticks[i + 1] < ticks[i]) {
          continue;
        }
        stats.pass_ns[pass].fetch_add(
            uint64_t(double(ticks[i + 1] - ticks[i]) * gpu_profile_ns_per_tick_),
            std::memory_order_relaxed);
        stats.pass_count[pass].fetch_add(1, std::memory_order_relaxed);
      }
      const uint64_t start = ticks[0], end = ticks[record.query_count - 1];
      if (end >= start) {
        stats.submission_ns.fetch_add(uint64_t(double(end - start) * gpu_profile_ns_per_tick_),
                                      std::memory_order_relaxed);
        stats.submissions.fetch_add(1, std::memory_order_relaxed);
      }
      if (gpu_profile_last_end_tick_ && start > gpu_profile_last_end_tick_) {
        const uint64_t gap =
            uint64_t(double(start - gpu_profile_last_end_tick_) * gpu_profile_ns_per_tick_);
        if (gap < 100000000) {
          stats.gap_ns.fetch_add(gap, std::memory_order_relaxed);
        }
      }
      gpu_profile_last_end_tick_ = end;
    }
    gpu_profile_submissions_.pop_front();
  }
  if (!stats.enabled.load(std::memory_order_relaxed)) {
    gpu_profile_last_end_tick_ = 0;
  }
}

uint64_t D3D12CommandProcessor::NormalizeOcclusionSamples(uint64_t samples) const {
  if (samples == 0 || !texture_cache_) {
    return samples;
  }
  uint64_t scale_x = texture_cache_->draw_resolution_scale_x();
  uint64_t scale_y = texture_cache_->draw_resolution_scale_y();
  uint64_t scale = scale_x * scale_y;
  if (scale <= 1) {
    return samples;
  }
  return (samples + (scale >> 1)) / scale;
}

void D3D12CommandProcessor::WriteGuestOcclusionResult(
    xenos::xe_gpu_depth_sample_counts* sample_counts, uint64_t samples) {
  if (!sample_counts) {
    return;
  }
  uint32_t clamped = samples > uint64_t(UINT32_MAX) ? UINT32_MAX : uint32_t(samples);
  sample_counts->Total_A = clamped;
  sample_counts->Total_B = 0;
  sample_counts->ZPass_A = clamped;
  sample_counts->ZPass_B = 0;
  sample_counts->ZFail_A = 0;
  sample_counts->ZFail_B = 0;
  sample_counts->StencilFail_A = 0;
  sample_counts->StencilFail_B = 0;
}

void D3D12CommandProcessor::WriteGammaRampSRV(bool is_pwl,
                                              D3D12_CPU_DESCRIPTOR_HANDLE handle) const {
  ID3D12Device* device = GetD3D12Provider().GetDevice();
  D3D12_SHADER_RESOURCE_VIEW_DESC desc;
  desc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
  desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  desc.Buffer.StructureByteStride = 0;
  desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
  if (is_pwl) {
    desc.Format = DXGI_FORMAT_R16G16_UINT;
    desc.Buffer.FirstElement = 256 * 4 / 4;
    desc.Buffer.NumElements = 128 * 3;
  } else {
    desc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
    desc.Buffer.FirstElement = 0;
    desc.Buffer.NumElements = 256;
  }
  device->CreateShaderResourceView(gamma_ramp_buffer_.Get(), &desc, handle);
}

}  // namespace rex::graphics::d3d12
