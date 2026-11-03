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

#include <rex/graphics/graphics_system.h>

#include <algorithm>

#include <rex/perf/frame_rate.h>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <functional>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <utility>

#include <rex/cvar.h>
#include <rex/graphics/command_processor.h>
#include <rex/graphics/flags.h>
#include <rex/kernel/xboxkrnl/video.h>
#include <rex/logging.h>
#include <rex/stream.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xthread.h>
#include <rex/ui/graphics_provider.h>
#include <rex/ui/window.h>
#include <rex/ui/windowed_app_context.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

REXCVAR_DEFINE_STRING(swap_post_effect, "none", "GPU",
                      "Swap post effect: none, fxaa, fxaa_extreme, smaa")
    .allowed({"none", "fxaa", "fxaa_extreme", "smaa"})
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_BOOL(vblank_lock_display, false, "GPU",
                    "Lock the guest's vblank timer to the display's refreshes in rate and phase (needs "
                    "--d3d12_present_vsync and --d3d12_flip_watch, which time them): the game's frames then "
                    "reach the screen evenly instead of beating against its refresh (Proton / gamescope)");
REXCVAR_DEFINE_INT32(vblank_lock_offset_us, 0, "GPU",
                     "With --vblank_lock_display: where the guest vblank sits after a display refresh");
REXCVAR_DEFINE_INT32(vblank_spin_us, 2000, "GPU",
                     "Wake this many microseconds before each guest vblank and spin to it "
                     "(frames that flip on vblanks are only as even as the vblanks)");
REXCVAR_DEFINE_BOOL(store_shaders, true, "GPU",
                    "Store shaders persistently and load them when loading games to avoid "
                    "runtime spikes and freezes when playing the game not for the first time.");

namespace {

rex::graphics::CommandProcessor::SwapPostEffect ParseSwapPostEffect(
    const std::string& effect_name) {
  std::string lowered = effect_name;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
    c = static_cast<unsigned char>(std::tolower(c));
    return c == '-' ? '_' : char(c);
  });
  if (lowered == "fxaa") {
    return rex::graphics::CommandProcessor::SwapPostEffect::kFxaa;
  }
  if (lowered == "fxaa_extreme" || lowered == "extreme") {
    return rex::graphics::CommandProcessor::SwapPostEffect::kFxaaExtreme;
  }
  if (lowered == "smaa") {
    return rex::graphics::CommandProcessor::SwapPostEffect::kSmaa;
  }
  return rex::graphics::CommandProcessor::SwapPostEffect::kNone;
}
}  // namespace

namespace rex::graphics {

// Guest tick counts of the last and next guest vblank (command_processor.cpp).
extern std::atomic<uint64_t> g_last_vblank_tick, g_next_vblank_tick, g_vblank_mark_tick,
    g_vblank_done_tick;

// The running graphics system (for ReXSetSwapPostEffect).
GraphicsSystem* g_live_graphics_system = nullptr;

// Nvidia Optimus/AMD PowerXpress support.
// These exports force the process to trigger the discrete GPU in multi-GPU
// systems.
// https://developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/OptimusRenderingPolicies.pdf
// https://stackoverflow.com/questions/17458803/amd-equivalent-to-nvoptimusenablement
#if REX_PLATFORM_WIN32
extern "C" {
__declspec(dllexport) uint32_t NvOptimusEnablement = 0x00000001;
__declspec(dllexport) uint32_t AmdPowerXpressRequestHighPerformance = 1;
}  // extern "C"
#endif  // REX_PLATFORM_WIN32

GraphicsSystem::GraphicsSystem() : vsync_worker_running_(false) {}

GraphicsSystem::~GraphicsSystem() {
  if (g_live_graphics_system == this) g_live_graphics_system = nullptr;
}

X_STATUS GraphicsSystem::SetupPresentation(ui::WindowedAppContext* app_context) {
  if (presenter_) {
    return X_STATUS_SUCCESS;
  }

  if (!provider_) {
    CreateProvider(true);
    if (!provider_) {
      REXGPU_ERROR("Unable to create graphics provider");
      return X_STATUS_UNSUCCESSFUL;
    }
    provider_supports_presentation_ = true;
  } else if (!provider_supports_presentation_) {
    // A prior SetupGuestGpu built a headless provider; backends like Vulkan
    // need swapchain support baked in at provider creation time.
    REXGPU_ERROR("SetupPresentation called after headless SetupGuestGpu; call order is reversed");
    return X_STATUS_UNSUCCESSFUL;
  }

  app_context_ = app_context;
  auto loss_cb = [this](bool is_responsible, bool statically_from_ui_thread) {
    OnHostGpuLossFromAnyThread(is_responsible);
  };
  if (app_context_) {
    // Presenter creation must happen on the UI thread.
    app_context_->CallInUIThreadSynchronous(
        [this, loss_cb]() { presenter_ = provider_->CreatePresenter(loss_cb); });
  } else {
    // Offscreen path (e.g. capturing guest output without a window).
    presenter_ = provider_->CreatePresenter(loss_cb);
  }

  if (!presenter_) {
    REXGPU_ERROR("Unable to create presenter");
    return X_STATUS_UNSUCCESSFUL;
  }
  return X_STATUS_SUCCESS;
}

X_STATUS GraphicsSystem::SetupGuestGpu(runtime::FunctionDispatcher* function_dispatcher,
                                       system::KernelState* kernel_state) {
  memory_ = function_dispatcher->memory();
  function_dispatcher_ = function_dispatcher;
  kernel_state_ = kernel_state;

  // Headless path: no one set up presentation, so build a no-presentation
  // provider just for the command processor.
  if (!provider_) {
    CreateProvider(false);
    provider_supports_presentation_ = false;
  }

  // Create command processor. This will spin up a thread to process all
  // incoming ringbuffer packets.
  command_processor_ = CreateCommandProcessor();
  if (!command_processor_->Initialize()) {
    REXGPU_ERROR("Unable to initialize command processor");
    return X_STATUS_UNSUCCESSFUL;
  }
  command_processor_->SetDesiredSwapPostEffect(ParseSwapPostEffect(REXCVAR_GET(swap_post_effect)));
  g_live_graphics_system = this;

  // Register GPU MMIO handlers
  // GPU registers are at 0x7FC80000-0x7FCFFFFF
  memory_->AddVirtualMappedRange(0x7FC80000,  // base address
                                 0xFFFF0000,  // mask
                                 0x0000FFFF,  // size (64KB)
                                 this,        // context (GraphicsSystem*)
                                 reinterpret_cast<runtime::MMIOReadCallback>(ReadRegisterThunk),
                                 reinterpret_cast<runtime::MMIOWriteCallback>(WriteRegisterThunk));

  // Guest vblank timer based on the configured guest video mode.
  vsync_worker_running_ = true;
  vsync_worker_thread_ = system::object_ref<system::XHostThread>(
      new system::XHostThread(kernel_state_, 128 * 1024, 0, [this]() {
        system::X_VIDEO_MODE video_mode;
        kernel::xboxkrnl::VdQueryVideoMode(&video_mode);
        double refresh_rate_hz = std::max(1.0, double(float(video_mode.refresh_rate)));
        uint64_t guest_tick_frequency = chrono::Clock::guest_tick_frequency();
        uint64_t vsync_interval_ticks =
            std::max(uint64_t(1), uint64_t(double(guest_tick_frequency) / refresh_rate_hz));
        uint64_t no_vsync_interval_ticks = std::max(uint64_t(1), guest_tick_frequency / 1000);
        uint64_t last_frame_time = chrono::Clock::QueryGuestTickCount();
        const uint64_t nominal_interval_ticks = vsync_interval_ticks;
        const int64_t nominal_ns = int64_t(1e9 / refresh_rate_hz);
        struct {
          uint64_t last_stamp = 0;
          int64_t gaps[64] = {};
          int count = 0;
          int64_t abs_error_sum = 0;
          uint64_t logged = 0;
        } lock;
#ifdef _WIN32
        HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr,
                                              CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                              TIMER_ALL_ACCESS);
#endif
        while (vsync_worker_running_) {
          uint64_t current_time = chrono::Clock::QueryGuestTickCount();
          uint64_t interval_ticks =
              REXCVAR_GET(vsync) ? vsync_interval_ticks : no_vsync_interval_ticks;
          // (Signed: the display lock's nudge can put the last vblank a little
          // after `now`; unsigned, that wrapped into an endless catch-up.)
          while (int64_t(current_time - last_frame_time) >= int64_t(interval_ticks)) {
            g_last_vblank_tick.store(last_frame_time + interval_ticks, std::memory_order_relaxed);
            MarkVblank();
            last_frame_time += interval_ticks;
          }
          // Locked to the display (--vblank_lock_display, with vsync presents
          // timing the flips): the vblanks run at the display's measured
          // refresh period (the median of the last 64 flip gaps, a 33 ms gap
          // counting as two) and, once per flip, move a sixteenth of the way
          // (at most 0.2 ms) towards its phase - after the catch-up above, so
          // a nudge never makes two vblanks back to back. Without flips for
          // 0.5 s: back to the guest's own rate.
          if (REXCVAR_GET(vblank_lock_display) && REXCVAR_GET(vsync)) {
            const uint64_t stamp = rex::perf::LastDisplayFlip();
            const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now().time_since_epoch())
                                       .count();
            if (!stamp || now_ns - int64_t(stamp) > 500000000) {
              vsync_interval_ticks = nominal_interval_ticks;
              lock.count = 0;
              lock.last_stamp = 0;
            } else if (stamp != lock.last_stamp) {
              if (lock.last_stamp) {
                const int64_t gap = int64_t(stamp - lock.last_stamp);
                if (gap > nominal_ns / 2) {  // (the waitable's first signals come at once)
                  const int64_t k = (gap + nominal_ns / 2) / nominal_ns;
                  if (k >= 1 && k <= 4) lock.gaps[lock.count++ % 64] = gap / k;
                }
              }
              lock.last_stamp = stamp;
              int64_t period_ns = nominal_ns;
              if (lock.count >= 16) {
                const int n = std::min(lock.count, 64);
                int64_t sorted[64];
                std::copy(lock.gaps, lock.gaps + n, sorted);
                std::nth_element(sorted, sorted + n / 2, sorted + n);
                period_ns = std::clamp(sorted[n / 2], nominal_ns * 99 / 100, nominal_ns * 101 / 100);
                vsync_interval_ticks = uint64_t(double(period_ns) * double(guest_tick_frequency) / 1e9);
              }
              const uint64_t now_ticks = chrono::Clock::QueryGuestTickCount();
              const uint64_t flip =
                  now_ticks - uint64_t(double(now_ns - int64_t(stamp)) * double(guest_tick_frequency) / 1e9);
              const int64_t period = int64_t(vsync_interval_ticks);
              const int64_t offset =
                  int64_t(guest_tick_frequency) * REXCVAR_GET(vblank_lock_offset_us) / 1000000;
              int64_t error = (int64_t(last_frame_time) - int64_t(flip) - offset) % period;
              if (error > period / 2) error -= period;
              if (error < -period / 2) error += period;
              const int64_t limit = int64_t(guest_tick_frequency) / 5000;  // 0.2 ms
              last_frame_time -= uint64_t(std::clamp(error / 16, -limit, limit));
              // Every 600 flips (10 s): the display's measured rate and the phase error.
              const double us = 1e6 / double(guest_tick_frequency);
              lock.abs_error_sum += error < 0 ? -error : error;
              if (++lock.logged % 600 == 0) {
                REXGPU_INFO("vblank lock: display {:.4f} Hz (median of {} gaps), phase error avg {:.0f} us",
                            1e9 / double(period_ns), std::min(lock.count, 64), lock.abs_error_sum / 600 * us);
                lock.abs_error_sum = 0;
              }
            }
          }
          g_next_vblank_tick.store(REXCVAR_GET(vsync) ? last_frame_time + interval_ticks : 0,
                                   std::memory_order_relaxed);
          const uint64_t remaining =
              uint64_t(std::max<int64_t>(0, int64_t(last_frame_time + interval_ticks - current_time)));
          // Spin only for real vblanks; with vsync off they come every 1 ms and
          // spinning half of each would keep a core busy.
          // Wake this long before a vblank and spin the rest, for timer slack.
          // Spinning with pause, not yields: a yield on busy cores (the Deck)
          // can hand the core away past the vblank.
          const uint64_t spin_ticks =
              guest_tick_frequency * uint64_t(std::max(0, REXCVAR_GET(vblank_spin_us))) / 1000000;
          const uint64_t spin = interval_ticks >= guest_tick_frequency / 200 ? spin_ticks : 0;
          if (remaining > spin) {
#ifdef _WIN32
            LARGE_INTEGER due;
            due.QuadPart = -int64_t((remaining - spin) * 10000000 / guest_tick_frequency);
            if (timer && SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
              WaitForSingleObject(timer, INFINITE);
            } else {
              rex::thread::Sleep(std::chrono::milliseconds(1));
            }
#else
            rex::thread::Sleep(std::chrono::microseconds(
                (remaining - spin) * 1000000 / guest_tick_frequency));
#endif
          } else if (spin) {
            _mm_pause();
          } else {
            rex::thread::MaybeYield();
          }
        }
#ifdef _WIN32
        if (timer) {
          CloseHandle(timer);
        }
#endif
        return 0;
      }));
  // TODO: set_can_debugger_suspend not yet ported
  // vsync_worker_thread_->set_can_debugger_suspend(true);
  vsync_worker_thread_->set_name("GPU VSync");
  vsync_worker_thread_->Create();

  return X_STATUS_SUCCESS;
}

void GraphicsSystem::Shutdown() {
  if (command_processor_) {
    command_processor_->Shutdown();
    command_processor_.reset();
  }

  if (vsync_worker_thread_) {
    vsync_worker_running_ = false;
    vsync_worker_thread_->Wait(0, 0, 0, nullptr);
    vsync_worker_thread_.reset();
  }

  if (presenter_) {
    if (app_context_) {
      app_context_->CallInUIThreadSynchronous([this]() { presenter_.reset(); });
    }
    // If there's no app context (thus the presenter is owned by the thread that
    // initialized the GraphicsSystem) or can't be queueing UI thread calls
    // anymore, shutdown anyway.
    presenter_.reset();
  }

  provider_.reset();
}

void GraphicsSystem::OnHostGpuLossFromAnyThread([[maybe_unused]] bool is_responsible) {
  // TODO(Triang3l): Somehow gain exclusive ownership of the Provider (may be
  // used by the command processor, the presenter, and possibly anything else,
  // it's considered free-threaded, except for lifetime management which will be
  // involved in this case) and reset it so a new host GPU API device is
  // created. Then ask the command processor to reset itself in its thread, and
  // ask the UI thread to reset the Presenter (the UI thread manages its
  // lifetime - but if there's no WindowedAppContext, either don't reset it as
  // in this case there's no user who needs uninterrupted gameplay, or somehow
  // protect it with a mutex so any thread can be considered a UI thread and
  // reset).
  if (host_gpu_loss_reported_.test_and_set(std::memory_order_relaxed)) {
    return;
  }
  rex::FatalError("Graphics device lost (probably due to an internal error)");
}

uint32_t GraphicsSystem::ReadRegisterThunk(void* ppc_context, GraphicsSystem* gs, uint32_t addr) {
  return gs->ReadRegister(addr);
}

void GraphicsSystem::WriteRegisterThunk(void* ppc_context, GraphicsSystem* gs, uint32_t addr,
                                        uint32_t value) {
  gs->WriteRegister(addr, value);
}

uint32_t GraphicsSystem::ReadRegister(uint32_t addr) {
  uint32_t r = (addr & 0xFFFF) / 4;

  switch (r) {
    case 0x0F00:  // RB_EDRAM_TIMING
      return 0x08100748;
    case 0x0F01:  // RB_BC_CONTROL
      return 0x0000200E;
    case 0x194C: {  // R500_D1MODE_V_COUNTER
      system::X_VIDEO_MODE video_mode;
      kernel::xboxkrnl::VdQueryVideoMode(&video_mode);
      return std::min(uint32_t(video_mode.display_height), uint32_t(0x0FFF));
    }
    case 0x1951:    // interrupt status
      return 1;     // vblank
    case 0x1961: {  // AVIVO_D1MODE_VIEWPORT_SIZE
      // Maximum [width(0x0FFF), height(0x0FFF)].
      system::X_VIDEO_MODE video_mode;
      kernel::xboxkrnl::VdQueryVideoMode(&video_mode);
      uint32_t viewport_width = std::min(uint32_t(video_mode.display_width), uint32_t(0x0FFF));
      uint32_t viewport_height = std::min(uint32_t(video_mode.display_height), uint32_t(0x0FFF));
      return (viewport_width << 16) | viewport_height;
    }
    default:
      if (!register_file_.GetRegisterInfo(r)) {
        REXGPU_DEBUG("GPU: Read from unknown register ({:04X})", r);
      }
  }

  assert_true(r < RegisterFile::kRegisterCount);
  return register_file_.values[r];
}

void GraphicsSystem::WriteRegister(uint32_t addr, uint32_t value) {
  uint32_t r = (addr & 0xFFFF) / 4;

  switch (r) {
    case 0x01C5:  // CP_RB_WPTR
      command_processor_->UpdateWritePointer(value);
      break;
    case 0x1844:  // AVIVO_D1GRPH_PRIMARY_SURFACE_ADDRESS
      break;
    default:
      REXGPU_WARN("Unknown GPU register {:04X} write: {:08X}", r, value);
      break;
  }

  assert_true(r < RegisterFile::kRegisterCount);
  register_file_.values[r] = value;
}

void GraphicsSystem::InitializeRingBuffer(uint32_t ptr, uint32_t size_log2) {
  command_processor_->InitializeRingBuffer(ptr, size_log2);
}

void GraphicsSystem::EnableReadPointerWriteBack(uint32_t ptr, uint32_t block_size_log2) {
  command_processor_->EnableReadPointerWriteBack(ptr, block_size_log2);
}

void GraphicsSystem::SetInterruptCallback(uint32_t callback, uint32_t user_data) {
  interrupt_callback_ = callback;
  interrupt_callback_data_ = user_data;
  REXGPU_INFO("SetInterruptCallback({:08X}, {:08X})", callback, user_data);
}

void GraphicsSystem::DispatchInterruptCallback(uint32_t source, uint32_t cpu) {
  if (!interrupt_callback_) {
    return;
  }

  auto thread = system::XThread::GetCurrentThread();
  assert_not_null(thread);

  // Pick a CPU, if needed. We're going to guess 2. Because.
  if (cpu == 0xFFFFFFFF) {
    cpu = 2;
  }
  thread->SetActiveCpu(cpu);

  // REXGPU_INFO("Dispatching GPU interrupt at {:08X} w/ mode {} on cpu {}",
  //          interrupt_callback_, source, cpu);

  uint64_t args[] = {source, interrupt_callback_data_};
  function_dispatcher_->ExecuteInterrupt(thread->thread_state(), interrupt_callback_, args,
                                         rex::countof(args));
}

rex::thread::Event* VblankWakeEvent();  // command_processor.cpp

void GraphicsSystem::MarkVblank() {
  // TODO: Enable profiling once ported
  // SCOPE_profile_cpu_f("gpu");

  // Increment vblank counter (so the game sees us making progress).
  g_vblank_mark_tick.store(chrono::Clock::QueryGuestTickCount(), std::memory_order_relaxed);
  if (command_processor_) {
    command_processor_->increment_counter();
  }

  // TODO(benvanik): we shouldn't need to do the dispatch here, but there's
  //     something wrong and the CP will block waiting for code that
  //     needs to be run in the interrupt.
  DispatchInterruptCallback(0, 2);
  g_vblank_done_tick.store(chrono::Clock::QueryGuestTickCount(), std::memory_order_relaxed);
  VblankWakeEvent()->Set();
}

void GraphicsSystem::ClearCaches() {
  command_processor_->CallInThread([&]() { command_processor_->ClearCaches(); });
}

void GraphicsSystem::InvalidateGpuMemory() {
  command_processor_->CallInThread([&]() { command_processor_->InvalidateGpuMemory(); });
}

void GraphicsSystem::InitializeShaderStorage(const std::filesystem::path& cache_root,
                                             uint32_t title_id, bool blocking) {
  if (!REXCVAR_GET(store_shaders)) {
    return;
  }
  if (blocking) {
    if (command_processor_->is_paused()) {
      // Safe to run on any thread while the command processor is paused, no
      // race condition.
      command_processor_->InitializeShaderStorage(cache_root, title_id, true);
    } else {
      rex::thread::Fence fence;
      command_processor_->CallInThread([this, cache_root, title_id, &fence]() {
        command_processor_->InitializeShaderStorage(cache_root, title_id, true);
        fence.Signal();
      });
      fence.Wait();
    }
  } else {
    command_processor_->CallInThread([this, cache_root, title_id]() {
      command_processor_->InitializeShaderStorage(cache_root, title_id, false);
    });
  }
}

void GraphicsSystem::Pause() {
  paused_ = true;
  command_processor_->Pause();
}

void GraphicsSystem::Resume() {
  paused_ = false;
  command_processor_->Resume();
}

bool GraphicsSystem::Save(::rex::stream::ByteStream* stream) {
  stream->Write<uint32_t>(interrupt_callback_);
  stream->Write<uint32_t>(interrupt_callback_data_);
  return command_processor_->Save(stream);
}

bool GraphicsSystem::Restore(::rex::stream::ByteStream* stream) {
  interrupt_callback_ = stream->Read<uint32_t>();
  interrupt_callback_data_ = stream->Read<uint32_t>();
  return command_processor_->Restore(stream);
}

}  // namespace rex::graphics

// The swap post effect (none, fxaa, fxaa_extreme, smaa) changed while running,
// as a plain C export for callers built against an SDK whose headers predate
// it (they look it up in the DLL by name).
extern "C" __declspec(dllexport) void ReXSetSwapPostEffect(const char* name) {
  auto* system = rex::graphics::g_live_graphics_system;
  if (!system || !name) return;
  REXCVAR_SET(swap_post_effect, std::string(name));
  if (auto* command_processor = system->command_processor()) {
    command_processor->SetDesiredSwapPostEffect(ParseSwapPostEffect(name));
  }
}
