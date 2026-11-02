#include <intrin.h>
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
#include <array>
#include <atomic>
#include <deque>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string_view>

#include <fmt/format.h>

#include <rex/hash.h>
#include <rex/cvar.h>
#include <rex/platform.h>
#if REX_PLATFORM_WIN32
#include <windows.h>
#endif
#include <rex/dbg.h>
#include <rex/perf/counter.h>
#include <rex/chrono/clock.h>
#include <rex/graphics/command_processor.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/graphics_system.h>
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/sampler_info.h>
#include <rex/graphics/xenos.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/memory.h>
#include <rex/memory/ring_buffer.h>
#include <rex/stream.h>
#include <rex/system/kernel_state.h>
#include <rex/system/user_module.h>

REXCVAR_DEFINE_BOOL(gpu_read_pointer_progress, true, "GPU",
                    "Write the ring buffer read pointer back after every primary ring packet "
                    "(like the hardware), not only when the batch is done");
REXCVAR_DEFINE_BOOL(vsync, true, "GPU", "Enable vertical sync");
REXCVAR_DEFINE_INT32(gpu_cp_split, -1, "GPU",
                     "Run the command processor in two threads: one reads the command buffers "
                     "and register writes, the other executes the draws: -1 auto (on under "
                     "Proton/Wine, measured on the Steam Deck; off on Windows: on a Ryzen 9 "
                     "3900X it raised the uncapped average ~3% but lowered the 1% lows), 0 off, 1 on");
REXCVAR_DEFINE_INT32(gpu_screen_extents, 0, "GPU",
                     "Test: the screen extents EVENT_WRITE_EXT reports for the previous draw: "
                     "0 the whole screen (Xenia's), 1 a single 8x8 tile at the origin, 2 the "
                     "draw's vertices' bounding box from its vertex shader run on the CPU (slow), 3 "
                     "the same measured by the GPU (Direct3D 12): the first tile's draws "
                     "accumulate it in their vertex shaders, read back when the guest needs it");
REXCVAR_DEFINE_INT32(gpu_screen_extents_delay, 0, "GPU",
                     "Test (gpu_screen_extents=2): write the measured extents this many frames "
                     "late (the whole screen until then), as a GPU measurement would arrive; -1: at "
                     "the next EVENT_WRITE_SHD or swap (the end of the frame's GPU work); -2: at the "
                     "next INTERRUPT (the game's callback that patches the bin masks of tiles 2 "
                     "and 3 from tile 1's extents)");
REXCVAR_DEFINE_BOOL(gpu_tile_collapse, false, "GPU",
                    "amped3: draw the game's 3 predicated tiles (vertical columns of a 2x MSAA "
                    "surface) as one full-width pass without MSAA in tile 1, skipping tiles 2 "
                    "and 3 (needs the 4096-tile EDRAM)");
REXCVAR_DEFINE_BOOL(gpu_tile_collapse_msaa, false, "GPU",
                    "gpu_tile_collapse: keep 2x MSAA (experimental: the gameplay scene comes out wrong; menus 4x "
                    "becomes 2x) instead of none");
REXCVAR_DEFINE_INT32(gpu_cp_split_spin, 200000, "GPU",
                     "gpu_cp_split: pause instructions the draw thread spins for before sleeping "
                     "when it runs out of work (waking it again takes a system call)");
REXCVAR_DEFINE_BOOL(vblank_spin_wait, true, "GPU",
                    "Poll instead of sleeping in GPU waits when a guest vblank is due within "
                    "1.5 ms, so frames flip on time");
REXCVAR_DEFINE_BOOL(swap_times, false, "GPU",
                    "Debug: write the time of every guest swap to swaptimes.csv");

REXCVAR_DEFINE_BOOL(clear_memory_page_state, false, "GPU",
                    "At the end of every frame, forget which pages the GPU has from the CPU, so "
                    "everything used is uploaded again (a coherency safety net for titles writing "
                    "memory unnoticed; CPU writes are normally caught by write watches). Costly: "
                    "about 10 MB of uploads a frame in Burst Limit.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(occlusion_query_enable, true, "GPU", "Enable host occlusion query handling")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_STRING(readback_resolve, "none", "GPU",
                      "Controls CPU readback of render-to-texture resolve results.\n"
                      " none: Disable readback (default)\n"
                      " fast: Read previous frame (delayed, copy every frame)\n"
                      " some: Read previous frame (delayed, copy on cache miss)\n"
                      " full: Immediate sync readback (accurate but stalls)")
    .allowed({"none", "fast", "some", "full"})
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(readback_resolve_half_pixel_offset, false, "GPU",
                    "When draw resolution scaling is active, sample from the center of each "
                    "scaled block during resolve readback downscale")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(readback_memexport, true, "GPU",
                    "Enable CPU readback of shader memexport writes for guest memory "
                    "coherency (can reduce correctness issues, but may add GPU/CPU sync cost)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(readback_memexport_fast, true, "GPU",
                    "Use fast double-buffered memexport readback when possible, with "
                    "automatic fallback to full synchronous readback")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(query_occlusion_fake_sample_count, 1000, "GPU",
                     "Fake sample count for occlusion queries")
    .range(1, 100000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(async_shader_compilation, true, "GPU",
                    "Compile shaders and create pipelines asynchronously in background "
                    "threads. This reduces stutter but may cause brief visual artifacts while "
                    "pipelines are being prepared.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

namespace rex::graphics {

// Guest tick counts of the last guest vblank and the next one (0 with vsync
// off), set by the GPU VSync thread.
std::atomic<uint64_t> g_last_vblank_tick{0}, g_next_vblank_tick{0};
// When the last vblank was marked and when its interrupt callback returned
// (for swaptimes.csv).
std::atomic<uint64_t> g_vblank_mark_tick{0}, g_vblank_done_tick{0};

using namespace rex::graphics::xenos;

namespace {

ReadbackResolveMode ParseReadbackResolveMode(std::string_view value) {
  if (value == "fast") {
    return ReadbackResolveMode::kFast;
  }
  if (value == "some") {
    return ReadbackResolveMode::kSome;
  }
  if (value == "full") {
    return ReadbackResolveMode::kFull;
  }
  return ReadbackResolveMode::kDisabled;
}

}  // namespace

// amped3 cp perf (gpu_cp_split): the command processor in two threads. The
// "GPU Commands" thread (the parser) walks the ring and the indirect buffers,
// takes the register writes off the packets, handles the packets that don't
// depend on drawing (bin masks, the per-draw events), and queues the rest with
// the register writes, in order, for the "GPU Draws" thread (the executor),
// which runs them exactly as the single-threaded command processor would. Like
// the hardware's prefetch parser, the parser reports read pointer progress for
// the packets it has taken: they are copied into the queue.
struct CommandProcessor::Amped3Split {
  enum Op : uint32_t {
    kOpRegisters = 1,  // start index, big-endian values (WriteRegistersFromMem)
    kOpPacket,         // a whole big-endian packet (ExecutePacket)
    kOpPrimaryEnd,     // OnPrimaryBufferEnd
    kOpWrap,           // the rest of the buffer is unused, continue at its start
    kOpDraw,           // VGT_DRAW_INITIATOR, VGT_DMA_BASE, VGT_DMA_SIZE (Amped3ExecuteDraw),
                       // amped3_draw_address_
    kOpShader,         // shader type, Shader* (low, high) loaded by the parser
    kOpExtentsPass,    // amped3_extents_pass_, amped3_tile_
  };
  static constexpr uint32_t kCapacity = uint32_t(1) << 22;  // dwords
  std::unique_ptr<uint32_t[]> buffer{new uint32_t[kCapacity]};
  // Positions in dwords since the start, the buffer offset modulo kCapacity.
  alignas(64) std::atomic<uint64_t> write_pos{0};
  std::atomic<bool> executor_sleeping{false};
  alignas(64) std::atomic<uint64_t> read_pos{0};
  std::atomic<bool> parser_waiting{false};
  alignas(64) uint64_t local_write = 0;  // The parser's, published by Publish.
  uint64_t cached_read = 0;              // The parser's last look at read_pos.
  std::unique_ptr<rex::thread::Event> data_event{rex::thread::Event::CreateAutoResetEvent(false)};
  std::unique_ptr<rex::thread::Event> space_event{rex::thread::Event::CreateAutoResetEvent(false)};
  std::atomic<bool> running{true};
  system::object_ref<system::XHostThread> thread;
};

CommandProcessor::CommandProcessor(GraphicsSystem* graphics_system,
                                   system::KernelState* kernel_state)
    : memory_(graphics_system->memory()),
      kernel_state_(kernel_state),
      graphics_system_(graphics_system),
      register_file_(graphics_system_->register_file()),
      worker_running_(true),
      write_ptr_index_event_(rex::thread::Event::CreateAutoResetEvent(false)),
      write_ptr_index_(0) {
  assert_not_null(write_ptr_index_event_);
}

CommandProcessor::~CommandProcessor() = default;

bool CommandProcessor::Initialize() {
  // Initialize the gamma ramps to their default (linear) values - taken from
  // what games set when starting with the sRGB (return value 1)
  // VdGetCurrentDisplayGamma.
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t value = i * 0x3FF / 0xFF;
    reg::DC_LUT_30_COLOR& gamma_ramp_entry = gamma_ramp_256_entry_table_[i];
    gamma_ramp_entry.color_10_blue = value;
    gamma_ramp_entry.color_10_green = value;
    gamma_ramp_entry.color_10_red = value;
  }
  for (uint32_t i = 0; i < 128; ++i) {
    reg::DC_LUT_PWL_DATA gamma_ramp_entry = {};
    gamma_ramp_entry.base = (i * 0xFFFF / 0x7F) & ~UINT32_C(0x3F);
    gamma_ramp_entry.delta = i < 0x7F ? 0x200 : 0;
    for (uint32_t j = 0; j < 3; ++j) {
      gamma_ramp_pwl_rgb_[i][j] = gamma_ramp_entry;
    }
  }

  worker_running_ = true;
  worker_thread_ = system::object_ref<system::XHostThread>(
      new system::XHostThread(kernel_state_, 128 * 1024, 0, [this]() {
        WorkerThreadMain();
        return 0;
      }));
  worker_thread_->set_name("GPU Commands");
  worker_thread_->Create();

  bool split = REXCVAR_GET(gpu_cp_split) > 0;
  if (REXCVAR_GET(gpu_cp_split) < 0) {
#if REX_PLATFORM_WIN32
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    split = ntdll && GetProcAddress(ntdll, "wine_get_version");
#endif
  }
  if (split) {
    amped3_split_ = std::make_unique<Amped3Split>();
    amped3_split_->thread = system::object_ref<system::XHostThread>(
        new system::XHostThread(kernel_state_, 128 * 1024, 0, [this]() {
          Amped3SplitExecutorMain();
          return 0;
        }));
    amped3_split_->thread->set_name("GPU Draws");
    amped3_split_->thread->Create();
    REXGPU_INFO("gpu_cp_split: command processor in two threads");
  }

  return true;
}

void CommandProcessor::Shutdown() {
  worker_running_ = false;
  write_ptr_index_event_->Set();
  worker_thread_->Wait(0, 0, 0, nullptr);
  worker_thread_.reset();
}

void CommandProcessor::InitializeShaderStorage(const std::filesystem::path& cache_root,
                                               uint32_t title_id, bool blocking) {}

void CommandProcessor::CallInThread(std::function<void()> fn) {
  if (pending_fns_.empty() && system::XThread::IsInThread(worker_thread_.get())) {
    fn();
  } else {
    pending_fns_.push(std::move(fn));
  }
}

void CommandProcessor::ClearCaches() {}

void CommandProcessor::InvalidateGpuMemory() {}

ReadbackResolveMode CommandProcessor::GetReadbackResolveMode(
    bool legacy_readback_resolve_enabled) const {
  ReadbackResolveMode shared_mode = ParseReadbackResolveMode(REXCVAR_GET(readback_resolve));
  // Looked up by name (a search through all cvars) - once, not on every resolve.
  static const bool readback_resolve_set = rex::cvar::HasNonDefaultValue("readback_resolve");
  bool shared_mode_overrides_legacy =
      shared_mode != ReadbackResolveMode::kDisabled || readback_resolve_set;
  if (shared_mode_overrides_legacy) {
    return shared_mode;
  }
  return legacy_readback_resolve_enabled ? ReadbackResolveMode::kFast
                                         : ReadbackResolveMode::kDisabled;
}

bool CommandProcessor::IsReadbackMemexportEnabled(bool legacy_backend_flag) const {
  if (legacy_readback_memexport_cvar_name_ &&
      rex::cvar::HasNonDefaultValue(legacy_readback_memexport_cvar_name_)) {
    return legacy_backend_flag;
  }
  return REXCVAR_GET(readback_memexport);
}

void CommandProcessor::SetDesiredSwapPostEffect(SwapPostEffect swap_post_effect) {
  if (swap_post_effect_desired_ == swap_post_effect) {
    return;
  }
  swap_post_effect_desired_ = swap_post_effect;
  CallInThread([this, swap_post_effect]() { swap_post_effect_actual_ = swap_post_effect; });
}

void CommandProcessor::WorkerThreadMain() {
  if (!SetupContext()) {
    rex::FatalError("Unable to setup command processor internal state");
    return;
  }

  while (worker_running_) {
    if (!pending_fns_.empty() && amped3_split_) {
      Amped3SplitDrain();
    }
    while (!pending_fns_.empty()) {
      auto fn = std::move(pending_fns_.front());
      pending_fns_.pop();
      fn();
    }

    uint32_t write_ptr_index = write_ptr_index_.load();
    if (write_ptr_index == 0xBAADF00D || read_ptr_index_ == write_ptr_index) {
      SCOPE_profile_cpu_i("gpu", "rex::graphics::CommandProcessor::Stall");
      // We've run out of commands to execute.
      // We spin here waiting for new ones, as the overhead of waiting on our
      // event is too high.
      PrepareForWait();
      uint32_t loop_count = 0;
      do {
        // If we spin around too much, revert to a "low-power" state.
        if (loop_count > 500) {
          const int wait_time_ms = 5;
          rex::thread::Wait(write_ptr_index_event_.get(), true,
                            std::chrono::milliseconds(wait_time_ms));
        }

        rex::thread::MaybeYield();
        loop_count++;
        write_ptr_index = write_ptr_index_.load();
      } while (worker_running_ && pending_fns_.empty() &&
               (write_ptr_index == 0xBAADF00D || read_ptr_index_ == write_ptr_index));
      ReturnFromWait();
      if (!worker_running_ || !pending_fns_.empty()) {
        continue;
      }
    }
    assert_true(read_ptr_index_ != write_ptr_index);

    // Execute. Note that we handle wraparound transparently.
    read_ptr_index_ = ExecutePrimaryBuffer(read_ptr_index_, write_ptr_index);

    // TODO(benvanik): use reader->Read_update_freq_ and only issue after moving
    //     that many indices.
    if (read_ptr_writeback_ptr_) {
      memory::store_and_swap<uint32_t>(memory_->TranslatePhysical(read_ptr_writeback_ptr_),
                                       read_ptr_index_);
    }

    // FIXME: We're supposed to process the WAIT_UNTIL register at this point,
    // but no games seem to actually use it.
  }

  if (amped3_split_) {
    Amped3SplitDrain();
    amped3_split_->running = false;
    amped3_split_->data_event->Set();
    amped3_split_->thread->Wait(0, 0, 0, nullptr);
    amped3_split_->thread.reset();
  }
  ShutdownContext();
}

void CommandProcessor::Pause() {
  if (paused_) {
    return;
  }
  paused_ = true;

  thread::Fence fence;
  CallInThread([&fence]() {
    fence.Signal();
    thread::Thread::GetCurrentThread()->Suspend();
  });

  fence.Wait();
}

void CommandProcessor::Resume() {
  if (!paused_) {
    return;
  }
  paused_ = false;

  worker_thread_->thread()->Resume();
}

bool CommandProcessor::Save(::rex::stream::ByteStream* stream) {
  assert_true(paused_);

  stream->Write<uint32_t>(primary_buffer_ptr_);
  stream->Write<uint32_t>(primary_buffer_size_);
  stream->Write<uint32_t>(read_ptr_index_);
  stream->Write<uint32_t>(read_ptr_update_freq_);
  stream->Write<uint32_t>(read_ptr_writeback_ptr_);
  stream->Write<uint32_t>(write_ptr_index_.load());

  return true;
}

bool CommandProcessor::Restore(::rex::stream::ByteStream* stream) {
  assert_true(paused_);

  primary_buffer_ptr_ = stream->Read<uint32_t>();
  primary_buffer_size_ = stream->Read<uint32_t>();
  read_ptr_index_ = stream->Read<uint32_t>();
  read_ptr_update_freq_ = stream->Read<uint32_t>();
  read_ptr_writeback_ptr_ = stream->Read<uint32_t>();
  write_ptr_index_.store(stream->Read<uint32_t>());

  return true;
}

bool CommandProcessor::SetupContext() {
  return true;
}

void CommandProcessor::ShutdownContext() {}

void CommandProcessor::InitializeRingBuffer(uint32_t ptr, uint32_t size_log2) {
  read_ptr_index_ = 0;
  primary_buffer_ptr_ = ptr;
  primary_buffer_size_ = uint32_t(1) << (size_log2 + 3);
}

void CommandProcessor::EnableReadPointerWriteBack(uint32_t ptr, uint32_t block_size_log2) {
  // CP_RB_RPTR_ADDR Ring Buffer Read Pointer Address 0x70C
  // ptr = RB_RPTR_ADDR, pointer to write back the address to.
  read_ptr_writeback_ptr_ = ptr;
  // CP_RB_CNTL Ring Buffer Control 0x704
  // block_size = RB_BLKSZ, log2 of number of quadwords read between updates of
  //              the read pointer.
  read_ptr_update_freq_ = uint32_t(1) << block_size_log2 >> 2;
}

void CommandProcessor::UpdateWritePointer(uint32_t value) {
  write_ptr_index_ = value;
  write_ptr_index_event_->Set();
}

uint32_t CommandProcessor::ReadRegisterValue(uint32_t index) const {
  if (index < RegisterFile::kRegisterCount) {
    return register_file_->values[index];
  }
  auto it = extended_register_values_.find(index);
  return it != extended_register_values_.end() ? it->second : 0;
}

namespace {
// Registers with RegisterFile::GetRegisterInfo, for the unknown write log.
const std::array<uint64_t, (RegisterFile::kRegisterCount + 63) / 64> kKnownRegisters = [] {
  std::array<uint64_t, (RegisterFile::kRegisterCount + 63) / 64> known{};
  for (uint32_t i = 0; i < RegisterFile::kRegisterCount; ++i) {
    if (RegisterFile::GetRegisterInfo(i)) {
      known[i >> 6] |= uint64_t(1) << (i & 63);
    }
  }
  return known;
}();

// amped3 cp perf (after Project8Recomp's gpu_cp_fastpath, BSD-3): registers
// whose write has no effect beyond storing the value - known, and handled by
// no branch of CommandProcessor::WriteRegister (scratch writeback, coherency,
// gamma ramp) or D3D12CommandProcessor::WriteRegister (shader constants and
// fetch constants). Runs of these are stored in bulk by WriteRegistersFromMem.
// Over-excluding is always safe (the register takes the WriteRegister path).
const std::array<uint64_t, (RegisterFile::kRegisterCount + 63) / 64> kPlainRegisters = [] {
  auto plain = kKnownRegisters;
  auto exclude = [&plain](uint32_t first, uint32_t last) {
    for (uint32_t i = first; i <= last && i < RegisterFile::kRegisterCount; ++i) {
      plain[i >> 6] &= ~(uint64_t(1) << (i & 63));
    }
  };
  exclude(XE_GPU_REG_SCRATCH_REG0, XE_GPU_REG_SCRATCH_REG7);
  exclude(XE_GPU_REG_COHER_STATUS_HOST, XE_GPU_REG_COHER_STATUS_HOST);
  exclude(XE_GPU_REG_DC_LUT_RW_INDEX, XE_GPU_REG_DC_LUT_RW_INDEX);
  exclude(XE_GPU_REG_DC_LUT_SEQ_COLOR, XE_GPU_REG_DC_LUT_SEQ_COLOR);
  exclude(XE_GPU_REG_DC_LUT_PWL_DATA, XE_GPU_REG_DC_LUT_PWL_DATA);
  exclude(XE_GPU_REG_DC_LUT_30_COLOR, XE_GPU_REG_DC_LUT_30_COLOR);
  exclude(XE_GPU_REG_SHADER_CONSTANT_000_X, XE_GPU_REG_SHADER_CONSTANT_511_W);
  exclude(XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031, XE_GPU_REG_SHADER_CONSTANT_LOOP_31);
  exclude(XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0, XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5);
  return plain;
}();
}  // namespace

void CommandProcessor::WriteRegister(uint32_t index, uint32_t value) {
  RegisterFile& regs = *register_file_;
  if (index >= RegisterFile::kRegisterCount) {
    auto [it, inserted] = extended_register_values_.insert_or_assign(index, value);
    (void)it;
    if (inserted) {
      REXGPU_WARN(
          "CommandProcessor::WriteRegister index out of bounds: {} (stored as extended register)",
          index);
    }
    return;
  }

  // Volatile for the WAIT_REG_MEM loop.
  const_cast<volatile uint32_t&>(regs.values[index]) = value;
  // amped3 cp perf: known registers from a bitmap built at load, not the switch
  // (and no function-local static guard on every write).
  if (!(kKnownRegisters[index >> 6] & (uint64_t(1) << (index & 63)))) {
    REXGPU_DEBUG("GPU: Write to unknown register ({:04X} = {:08X})", index, value);
  }

  // Scratch register writeback.
  if (index >= XE_GPU_REG_SCRATCH_REG0 && index <= XE_GPU_REG_SCRATCH_REG7) {
    uint32_t scratch_reg = index - XE_GPU_REG_SCRATCH_REG0;
    if ((1 << scratch_reg) & regs.values[XE_GPU_REG_SCRATCH_UMSK]) {
      // Enabled - write to address.
      uint32_t scratch_addr = regs.values[XE_GPU_REG_SCRATCH_ADDR];
      uint32_t mem_addr = scratch_addr + (scratch_reg * 4);
      memory::store_and_swap<uint32_t>(memory_->TranslatePhysical(mem_addr), value);
    }
  } else {
    switch (index) {
      // If this is a COHER register, set the dirty flag.
      // This will block the command processor the next time it WAIT_REG_MEMs
      // and allow us to synchronize the memory.
      case XE_GPU_REG_COHER_STATUS_HOST: {
        const_cast<volatile uint32_t&>(regs.values[index]) |= UINT32_C(0x80000000);
      } break;

      case XE_GPU_REG_DC_LUT_RW_INDEX: {
        // Reset the sequential read / write component index (see the M56
        // DC_LUT_SEQ_COLOR documentation).
        gamma_ramp_rw_component_ = 0;
      } break;

      case XE_GPU_REG_DC_LUT_SEQ_COLOR: {
        // Should be in the 256-entry table writing mode.
        assert_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        // DC_LUT_SEQ_COLOR is in the red, green, blue order, but the write
        // enable mask is blue, green, red.
        bool write_gamma_ramp_component = (regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] &
                                           (UINT32_C(1) << (2 - gamma_ramp_rw_component_))) != 0;
        if (write_gamma_ramp_component) {
          reg::DC_LUT_30_COLOR& gamma_ramp_entry =
              gamma_ramp_256_entry_table_[gamma_ramp_rw_index.rw_index];
          // Bits 0:5 are hardwired to zero.
          uint32_t gamma_ramp_seq_color = regs.Get<reg::DC_LUT_SEQ_COLOR>().seq_color >> 6;
          switch (gamma_ramp_rw_component_) {
            case 0:
              gamma_ramp_entry.color_10_red = gamma_ramp_seq_color;
              break;
            case 1:
              gamma_ramp_entry.color_10_green = gamma_ramp_seq_color;
              break;
            case 2:
              gamma_ramp_entry.color_10_blue = gamma_ramp_seq_color;
              break;
          }
        }
        if (++gamma_ramp_rw_component_ >= 3) {
          gamma_ramp_rw_component_ = 0;
          reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
          ++new_gamma_ramp_rw_index.rw_index;
          WriteRegister(XE_GPU_REG_DC_LUT_RW_INDEX,
                        rex::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        }
        if (write_gamma_ramp_component) {
          OnGammaRamp256EntryTableValueWritten();
        }
      } break;

      case XE_GPU_REG_DC_LUT_PWL_DATA: {
        // Should be in the PWL writing mode.
        assert_not_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        // Bit 7 of the index is ignored for PWL.
        uint32_t gamma_ramp_rw_index_pwl = gamma_ramp_rw_index.rw_index & 0x7F;
        // DC_LUT_PWL_DATA is likely in the red, green, blue order because
        // DC_LUT_SEQ_COLOR is, but the write enable mask is blue, green, red.
        bool write_gamma_ramp_component = (regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] &
                                           (UINT32_C(1) << (2 - gamma_ramp_rw_component_))) != 0;
        if (write_gamma_ramp_component) {
          reg::DC_LUT_PWL_DATA& gamma_ramp_entry =
              gamma_ramp_pwl_rgb_[gamma_ramp_rw_index_pwl][gamma_ramp_rw_component_];
          auto gamma_ramp_value = regs.Get<reg::DC_LUT_PWL_DATA>();
          // Bits 0:5 are hardwired to zero.
          gamma_ramp_entry.base = gamma_ramp_value.base & ~UINT32_C(0x3F);
          gamma_ramp_entry.delta = gamma_ramp_value.delta & ~UINT32_C(0x3F);
        }
        if (++gamma_ramp_rw_component_ >= 3) {
          gamma_ramp_rw_component_ = 0;
          reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
          // TODO(Triang3l): Should this increase beyond 7 bits for PWL?
          // Direct3D 9 explicitly sets rw_index to 0x80 after writing the last
          // PWL entry. However, the DC_LUT_RW_INDEX documentation says that for
          // PWL, the bit 7 is ignored.
          new_gamma_ramp_rw_index.rw_index = (gamma_ramp_rw_index.rw_index & ~UINT32_C(0x7F)) |
                                             ((gamma_ramp_rw_index_pwl + 1) & 0x7F);
          WriteRegister(XE_GPU_REG_DC_LUT_RW_INDEX,
                        rex::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        }
        if (write_gamma_ramp_component) {
          OnGammaRampPWLValueWritten();
        }
      } break;

      case XE_GPU_REG_DC_LUT_30_COLOR: {
        // Should be in the 256-entry table writing mode.
        assert_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        uint32_t gamma_ramp_write_enable_mask = regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] & 0b111;
        if (gamma_ramp_write_enable_mask) {
          reg::DC_LUT_30_COLOR& gamma_ramp_entry =
              gamma_ramp_256_entry_table_[gamma_ramp_rw_index.rw_index];
          auto gamma_ramp_value = regs.Get<reg::DC_LUT_30_COLOR>();
          if (gamma_ramp_write_enable_mask & 0b001) {
            gamma_ramp_entry.color_10_blue = gamma_ramp_value.color_10_blue;
          }
          if (gamma_ramp_write_enable_mask & 0b010) {
            gamma_ramp_entry.color_10_green = gamma_ramp_value.color_10_green;
          }
          if (gamma_ramp_write_enable_mask & 0b100) {
            gamma_ramp_entry.color_10_red = gamma_ramp_value.color_10_red;
          }
        }
        // TODO(Triang3l): Should this reset the component write index? If this
        // increase is assumed to behave like a full DC_LUT_RW_INDEX write, it
        // probably should. Currently this also calls WriteRegister for
        // DC_LUT_RW_INDEX, which resets gamma_ramp_rw_component_ as well.
        gamma_ramp_rw_component_ = 0;
        reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
        ++new_gamma_ramp_rw_index.rw_index;
        WriteRegister(XE_GPU_REG_DC_LUT_RW_INDEX,
                      rex::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        if (gamma_ramp_write_enable_mask) {
          OnGammaRamp256EntryTableValueWritten();
        }
      } break;
    }
  }
}

void CommandProcessor::WriteRegistersFromMem(uint32_t start_index, uint32_t* base,
                                             uint32_t num_registers) {
  if (uint64_t(start_index) + num_registers > RegisterFile::kRegisterCount) {
    for (uint32_t i = 0; i < num_registers; ++i) {
      WriteRegister(start_index + i, memory::load_and_swap<uint32_t>(base + i));
    }
    return;
  }
  // amped3 cp perf: store runs of plain registers directly and only call the
  // (virtual) WriteRegister for registers with side effects, in order, so each
  // sees the registers before it already stored and the ones after it not yet.
  uint32_t* values = register_file_->values;
  uint32_t i = 0;
  while (i < num_registers) {
    uint32_t run = 0;
    for (uint32_t index = start_index + i;
         i + run < num_registers && (kPlainRegisters[index >> 6] >> (index & 63)) & 1;
         ++index) {
      ++run;
    }
    if (!run) {
      WriteRegister(start_index + i, memory::load_and_swap<uint32_t>(base + i));
      ++i;
      continue;
    }
    if (run >= 4) {
      memory::copy_and_swap(values + start_index + i, base + i, run);
    } else {
      for (uint32_t j = 0; j < run; ++j) {
        values[start_index + i + j] = memory::load_and_swap<uint32_t>(base + i + j);
      }
    }
    i += run;
  }
}

void CommandProcessor::WriteRegisterRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                                  uint32_t num_registers) {
  if (!num_registers) {
    return;
  }
  memory::RingBuffer::ReadRange range = ring->BeginRead(size_t(num_registers) * sizeof(uint32_t));
  if (range.first_length != 0) {
    uint32_t first_count = uint32_t(range.first_length / sizeof(uint32_t));
    WriteRegistersFromMem(base, reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(range.first)),
                          first_count);
    base += first_count;
  }
  if (range.second_length != 0) {
    WriteRegistersFromMem(base, reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(range.second)),
                          uint32_t(range.second_length / sizeof(uint32_t)));
  }
  ring->EndRead(range);
}

void CommandProcessor::WriteALURangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                             uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x4000, num_registers);
}

void CommandProcessor::WriteFetchRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                               uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x4800, num_registers);
}

void CommandProcessor::WriteBoolRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                              uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x4900, num_registers);
}

void CommandProcessor::WriteLoopRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                              uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x4908, num_registers);
}

void CommandProcessor::WriteREGISTERSRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                                   uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x2000, num_registers);
}

void CommandProcessor::WriteALURangeFromMem(uint32_t start_index, uint32_t* base,
                                            uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4000, base, num_registers);
}

void CommandProcessor::WriteFetchRangeFromMem(uint32_t start_index, uint32_t* base,
                                              uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4800, base, num_registers);
}

void CommandProcessor::WriteBoolRangeFromMem(uint32_t start_index, uint32_t* base,
                                             uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4900, base, num_registers);
}

void CommandProcessor::WriteLoopRangeFromMem(uint32_t start_index, uint32_t* base,
                                             uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4908, base, num_registers);
}

void CommandProcessor::WriteREGISTERSRangeFromMem(uint32_t start_index, uint32_t* base,
                                                  uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x2000, base, num_registers);
}

void CommandProcessor::MakeCoherent() {
  SCOPE_profile_cpu_f("gpu");

  // Status host often has 0x01000000 or 0x03000000.
  // This is likely toggling VC (vertex cache) or TC (texture cache).
  // Or, it also has a direction in here maybe - there is probably
  // some way to check for dest coherency (what all the COHER_DEST_BASE_*
  // registers are for).
  // Best docs I've found on this are here:
  // https://web.archive.org/web/20160711162346/https://amd-dev.wpengine.netdna-cdn.com/wordpress/media/2013/10/R6xx_R7xx_3D.pdf
  // https://cgit.freedesktop.org/xorg/driver/xf86-video-radeonhd/tree/src/r6xx_accel.c?id=3f8b6eccd9dba116cc4801e7f80ce21a879c67d2#n454

  // Volatile because this may be called from the WAIT_REG_MEM loop.
  volatile uint32_t* regs_volatile = register_file_->values;
  auto status_host = rex::memory::Reinterpret<reg::COHER_STATUS_HOST>(
      uint32_t(regs_volatile[XE_GPU_REG_COHER_STATUS_HOST]));
  uint32_t base_host = regs_volatile[XE_GPU_REG_COHER_BASE_HOST];
  uint32_t size_host = regs_volatile[XE_GPU_REG_COHER_SIZE_HOST];

  if (!status_host.status) {
    return;
  }

  const char* action = "N/A";
  if (status_host.vc_action_ena && status_host.tc_action_ena) {
    action = "VC | TC";
  } else if (status_host.tc_action_ena) {
    action = "TC";
  } else if (status_host.vc_action_ena) {
    action = "VC";
  }

  // TODO(benvanik): notify resource cache of base->size and type.
  REXGPU_TRACE("Make {:08X} -> {:08X} ({}b) coherent, action = {}", base_host,
               base_host + size_host, size_host, action);

  // Mark coherent.
  regs_volatile[XE_GPU_REG_COHER_STATUS_HOST] = 0;
}

void CommandProcessor::PrepareForWait() {}

void CommandProcessor::ReturnFromWait() {}

// amped3 debug (--d3d12_draw_timing): the command processor's busy time
// (primary buffer execution), read by the D3D12 per-frame log.
uint64_t g_amped3_cp_busy_tsc;
// amped3 debug (d3d12_draw_timing): type-3 packets per opcode, and type-0
// packets and their register count, for sizing a command processor split.
bool g_amped3_count_packets;
uint64_t g_amped3_opcode_counts[128];
uint64_t g_amped3_type0_packets, g_amped3_type0_registers;
// amped3 debug: one frame of type-3 packets to tiling_trace.txt, started by
// creating trace_tiling.txt next to the exe (single-threaded command processor).
std::FILE* g_amped3_tiling_trace;
struct Amped3DelayedExtents {
  uint32_t frame;
  uint32_t address;
  uint16_t extents[6];
};
// gpu_screen_extents_delay: measured extents waiting to be written.
std::deque<Amped3DelayedExtents> g_amped3_delayed_extents;
uint32_t g_amped3_extents_frame;
// gpu_cp_split: the executor's busy time (g_amped3_cp_busy_tsc is the parser's).
uint64_t g_amped3_cp_exec_busy_tsc;

uint32_t CommandProcessor::ExecutePrimaryBuffer(uint32_t read_index, uint32_t write_index) {
  SCOPE_profile_cpu_f("gpu");
  struct BusyTimer {
    uint64_t start = __rdtsc();
    ~BusyTimer() { g_amped3_cp_busy_tsc += __rdtsc() - start; }
  } busy_timer;

  // Execute commands!
  memory::RingBuffer reader(memory_->TranslatePhysical(primary_buffer_ptr_), primary_buffer_size_);
  reader.set_read_offset(read_index * sizeof(uint32_t));
  reader.set_write_offset(write_index * sizeof(uint32_t));
  if (amped3_split_) {
    do {
      if (!Amped3SplitParsePacket(&reader)) {
        REXGPU_ERROR("**** PRIMARY RINGBUFFER: Failed to execute packet.");
        assert_always();
        break;
      }
      if (read_ptr_writeback_ptr_ && REXCVAR_GET(gpu_read_pointer_progress)) {
        memory::store_and_swap<uint32_t>(memory_->TranslatePhysical(read_ptr_writeback_ptr_),
                                         uint32_t(reader.read_offset() / sizeof(uint32_t)));
      }
    } while (reader.read_count());
    Amped3SplitReserve(Amped3Split::kOpPrimaryEnd, 0);
    Amped3SplitPublish();
    return write_index;
  }
  do {
    if (!ExecutePacket(&reader)) {
      // This probably should be fatal - but we're going to continue anyways.
      REXGPU_ERROR("**** PRIMARY RINGBUFFER: Failed to execute packet.");
      assert_always();
      break;
    }
    // Like the hardware (every RB_BLKSZ), report the progress while going: a
    // guest waiting for ring space (Amped 3 spins on it) continues as soon as
    // the packets before it are done, not after the whole batch. They are
    // mostly indirect buffers, executed completely by now.
    if (read_ptr_writeback_ptr_ && REXCVAR_GET(gpu_read_pointer_progress)) {
      memory::store_and_swap<uint32_t>(memory_->TranslatePhysical(read_ptr_writeback_ptr_),
                                       uint32_t(reader.read_offset() / sizeof(uint32_t)));
    }
  } while (reader.read_count());

  OnPrimaryBufferEnd();

  return write_index;
}

void CommandProcessor::ExecuteIndirectBuffer(uint32_t ptr, uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // Execute commands!
  memory::RingBuffer reader(memory_->TranslatePhysical(ptr), count * sizeof(uint32_t));
  reader.set_write_offset(count * sizeof(uint32_t));
  do {
    if (!ExecutePacket(&reader)) {
      // Return up a level if we encounter a bad packet.
      REXGPU_ERROR("**** INDIRECT RINGBUFFER: Failed to execute packet.");
      assert_always();
      break;
    }
  } while (reader.read_count());
}

void CommandProcessor::ExecutePacket(uint32_t ptr, uint32_t count) {
  // Execute commands!
  memory::RingBuffer reader(memory_->TranslatePhysical(ptr), count * sizeof(uint32_t));
  reader.set_write_offset(count * sizeof(uint32_t));
  do {
    if (!ExecutePacket(&reader)) {
      REXGPU_ERROR("**** ExecutePacket: Failed to execute packet.");
      assert_always();
      break;
    }
  } while (reader.read_count());
}

bool CommandProcessor::ExecutePacket(memory::RingBuffer* reader) {
  const uint32_t packet = reader->ReadAndSwap<uint32_t>();
  const uint32_t packet_type = packet >> 30;
  if (packet == 0) {
    return true;
  }

  if (packet == 0xCDCDCDCD) {
    REXGPU_WARN("GPU packet is CDCDCDCD - probably read uninitialized memory!");
  }

  switch (packet_type) {
    case 0x00:
      return ExecutePacketType0(reader, packet);
    case 0x01:
      return ExecutePacketType1(reader, packet);
    case 0x02:
      return ExecutePacketType2(reader, packet);
    case 0x03:
      return ExecutePacketType3(reader, packet);
    default:
      assert_unhandled_case(packet_type);
      return false;
  }
}

uint32_t* CommandProcessor::Amped3SplitReserve(uint32_t op, uint32_t payload_dwords) {
  Amped3Split& split = *amped3_split_;
  const uint32_t total = 1 + payload_dwords;
  assert_true(total <= Amped3Split::kCapacity / 2);
  const uint32_t offset = uint32_t(split.local_write % Amped3Split::kCapacity);
  const bool wrap = offset + total > Amped3Split::kCapacity;
  const uint64_t start = wrap ? split.local_write + (Amped3Split::kCapacity - offset)
                              : split.local_write;
  if (start + total - split.cached_read > Amped3Split::kCapacity) {
    // Full: let the executor see everything so far, and wait for space.
    Amped3SplitPublish();
    for (uint32_t spins = 0;; ++spins) {
      split.cached_read = split.read_pos.load(std::memory_order_acquire);
      if (start + total - split.cached_read <= Amped3Split::kCapacity) {
        break;
      }
      if (spins < 1000) {
        _mm_pause();
        continue;
      }
      split.parser_waiting.store(true);
      split.cached_read = split.read_pos.load();
      if (start + total - split.cached_read > Amped3Split::kCapacity) {
        rex::thread::Wait(split.space_event.get(), false, std::chrono::milliseconds(1));
      }
      split.parser_waiting.store(false);
    }
  }
  if (wrap) {
    split.buffer[offset] = uint32_t(Amped3Split::kOpWrap) << 24;
  }
  uint32_t* header = &split.buffer[start % Amped3Split::kCapacity];
  header[0] = (op << 24) | payload_dwords;
  split.local_write = start + total;
  return header + 1;
}

void CommandProcessor::Amped3SplitPublish() {
  Amped3Split& split = *amped3_split_;
  split.write_pos.store(split.local_write);
  if (split.executor_sleeping.load()) {
    split.data_event->Set();
  }
}

void CommandProcessor::Amped3SplitDrain() {
  Amped3Split& split = *amped3_split_;
  Amped3SplitPublish();
  for (uint32_t spins = 0;; ++spins) {
    split.cached_read = split.read_pos.load(std::memory_order_acquire);
    if (split.cached_read == split.local_write) {
      return;
    }
    if (spins < 1000) {
      _mm_pause();
      continue;
    }
    split.parser_waiting.store(true);
    if (split.read_pos.load() != split.local_write) {
      rex::thread::Wait(split.space_event.get(), false, std::chrono::milliseconds(1));
    }
    split.parser_waiting.store(false);
  }
}

void CommandProcessor::Amped3SplitQueueRegisters(uint32_t start_index, memory::RingBuffer* reader,
                                                 uint32_t count) {
  uint32_t* payload = Amped3SplitReserve(Amped3Split::kOpRegisters, 1 + count);
  payload[0] = start_index;
  memory::RingBuffer::ReadRange range = reader->BeginRead(size_t(count) * sizeof(uint32_t));
  std::memcpy(payload + 1, range.first, range.first_length);
  if (range.second_length) {
    std::memcpy(reinterpret_cast<uint8_t*>(payload + 1) + range.first_length, range.second,
                range.second_length);
  }
  reader->EndRead(range);
}

void CommandProcessor::Amped3SplitForward(uint32_t packet, memory::RingBuffer* reader,
                                          uint32_t count) {
  uint32_t* payload = Amped3SplitReserve(Amped3Split::kOpPacket, 1 + count);
  payload[0] = rex::byte_swap(packet);
  memory::RingBuffer::ReadRange range = reader->BeginRead(size_t(count) * sizeof(uint32_t));
  std::memcpy(payload + 1, range.first, range.first_length);
  if (range.second_length) {
    std::memcpy(reinterpret_cast<uint8_t*>(payload + 1) + range.first_length, range.second,
                range.second_length);
  }
  reader->EndRead(range);
  Amped3SplitPublish();
}

void CommandProcessor::Amped3SplitParseIndirectBuffer(uint32_t ptr, uint32_t count) {
  memory::RingBuffer reader(memory_->TranslatePhysical(ptr), count * sizeof(uint32_t));
  reader.set_write_offset(count * sizeof(uint32_t));
  do {
    if (!Amped3SplitParsePacket(&reader)) {
      REXGPU_ERROR("**** INDIRECT RINGBUFFER: Failed to execute packet.");
      assert_always();
      break;
    }
  } while (reader.read_count());
}

// The parser's part of ExecutePacket: what it doesn't handle itself goes to the
// executor (in order with the register writes), with the same validation as
// ExecutePacketType0/3 before it's queued.
bool CommandProcessor::Amped3SplitParsePacket(memory::RingBuffer* reader) {
  const uint32_t packet = reader->ReadAndSwap<uint32_t>();
  if (packet == 0) {
    return true;
  }
  if (packet == 0xCDCDCDCD) {
    REXGPU_WARN("GPU packet is CDCDCDCD - probably read uninitialized memory!");
  }
  switch (packet >> 30) {
    case 0x00: {
      uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
      if (reader->read_count() < count * sizeof(uint32_t)) {
        REXGPU_ERROR("ExecutePacketType0 overflow (read count {:08X}, packet count {:08X})",
                     reader->read_count(), count * sizeof(uint32_t));
        return false;
      }
      if (!((packet >> 15) & 0x1)) {
        Amped3SplitQueueRegisters(packet & 0x7FFF, reader, count);
      } else {
        Amped3SplitForward(packet, reader, count);
      }
      return true;
    }
    case 0x01:
      Amped3SplitForward(packet, reader, 2);
      return true;
    case 0x02:
      return true;
    case 0x03:
      break;
  }

  uint32_t opcode = (packet >> 8) & 0x7F;
  uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
  if (reader->read_count() < count * sizeof(uint32_t)) {
    REXGPU_ERROR("ExecutePacketType3 overflow (read count {:08X}, packet count {:08X})",
                 reader->read_count(), count * sizeof(uint32_t));
    return false;
  }
  // The predicate (bin) check here, with the bin registers the parser keeps.
  if (packet & 1) {
    bool any_pass = (bin_select_ & bin_mask_) != 0;
    if (!any_pass || opcode == PM4_XE_SWAP) {
      reader->AdvanceRead(count * sizeof(uint32_t));
      return true;
    }
  }
  switch (opcode) {
    case PM4_INDIRECT_BUFFER:
    case PM4_INDIRECT_BUFFER_PFD: {
      uint32_t list_ptr = CpuToGpu(reader->ReadAndSwap<uint32_t>());
      uint32_t list_length = reader->ReadAndSwap<uint32_t>() & 0xFFFFF;
      reader->AdvanceRead((count - 2) * sizeof(uint32_t));
      if (Amped3CollapseSkipsIndirectBuffer()) {
        return true;
      }
      Amped3SplitParseIndirectBuffer(GpuToCpu(list_ptr), list_length);
      return true;
    }
    case PM4_NOP:
    case PM4_INVALIDATE_STATE:
      // (INVALIDATE_STATE does nothing in ExecutePacketType3_INVALIDATE_STATE.)
      reader->AdvanceRead(count * sizeof(uint32_t));
      return true;
    case PM4_DRAW_INDX:
    case PM4_DRAW_INDX_2: {
      uint32_t count_remaining = count;
      const char* opcode_name = opcode == PM4_DRAW_INDX ? "PM4_DRAW_INDX" : "PM4_DRAW_INDX_2";
      uint32_t draw_address =
          uint32_t(reader->read_ptr() - uintptr_t(memory_->TranslatePhysical(0)));
      if (opcode == PM4_DRAW_INDX) {
        // The viz query token (unused, as in ExecutePacketType3Draw).
        reader->ReadAndSwap<uint32_t>();
        --count_remaining;
      }
      uint32_t words[3];
      if (Amped3ReadDraw(reader, opcode_name, count_remaining, words)) {
        uint32_t* payload = Amped3SplitReserve(Amped3Split::kOpDraw, 4);
        std::memcpy(payload, words, sizeof(words));
        payload[3] = draw_address;
        Amped3SplitPublish();
      }
      return true;
    }
    case PM4_IM_LOAD:
    case PM4_IM_LOAD_IMMEDIATE: {
      // The shader lookup (hashing its microcode) here: only this thread
      // loads shaders while the command processor runs.
      uint32_t dword0 = reader->ReadAndSwap<uint32_t>();
      uint32_t start_size = reader->ReadAndSwap<uint32_t>();
      uint32_t size_dwords = start_size & 0xFFFF;
      assert_true((start_size >> 16) == 0);
      Shader* shader;
      xenos::ShaderType shader_type;
      if (opcode == PM4_IM_LOAD) {
        shader_type = static_cast<xenos::ShaderType>(dword0 & 0x3);
        uint32_t addr = dword0 & ~uint32_t(0x3);
        shader = LoadShader(shader_type, addr, memory_->TranslatePhysical<uint32_t*>(addr),
                            size_dwords);
        reader->AdvanceRead((count - 2) * sizeof(uint32_t));
      } else {
        shader_type = static_cast<xenos::ShaderType>(dword0);
        assert_true(count - 2 >= size_dwords);
        shader = LoadShader(shader_type, uint32_t(reader->read_ptr()),
                            reinterpret_cast<uint32_t*>(reader->read_ptr()), size_dwords);
        reader->AdvanceRead((count - 2) * sizeof(uint32_t));
      }
      uint32_t* payload = Amped3SplitReserve(Amped3Split::kOpShader, 3);
      payload[0] = uint32_t(shader_type);
      payload[1] = uint32_t(reinterpret_cast<uintptr_t>(shader));
      payload[2] = uint32_t(uint64_t(reinterpret_cast<uintptr_t>(shader)) >> 32);
      return true;
    }
    case PM4_SET_CONSTANT: {
      uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
      uint32_t index = offset_type & 0x7FF;
      static const uint32_t kConstantBases[] = {0x4000, 0x4800, 0x4900, 0x4908, 0x2000};
      uint32_t type = (offset_type >> 16) & 0xFF;
      if (type >= rex::countof(kConstantBases)) {
        reader->AdvanceRead((count - 1) * sizeof(uint32_t));
        return true;
      }
      Amped3SplitQueueRegisters(index + kConstantBases[type], reader, count - 1);
      return true;
    }
    case PM4_SET_CONSTANT2:
    case PM4_SET_SHADER_CONSTANTS: {
      uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
      Amped3SplitQueueRegisters(offset_type & 0xFFFF, reader, count - 1);
      return true;
    }
    case PM4_EVENT_WRITE:
      if (count == 1) {
        uint32_t* payload = Amped3SplitReserve(Amped3Split::kOpRegisters, 2);
        payload[0] = XE_GPU_REG_VGT_EVENT_INITIATOR;
        payload[1] = rex::byte_swap(reader->ReadAndSwap<uint32_t>() & 0x3F);
        return true;
      }
      break;
    case PM4_EVENT_WRITE_EXT:
      // (gpu_screen_extents=2 measures the previous draw: on the executor.)
      if (count == 2 && REXCVAR_GET(gpu_screen_extents) < 2) {
        // Fake screen extents (as ExecutePacketType3_EVENT_WRITE_EXT): not
        // dependent on drawing.
        uint32_t initiator = reader->ReadAndSwap<uint32_t>();
        uint32_t address = reader->ReadAndSwap<uint32_t>() & ~uint32_t(0x3);
        uint32_t* payload = Amped3SplitReserve(Amped3Split::kOpRegisters, 2);
        payload[0] = XE_GPU_REG_VGT_EVENT_INITIATOR;
        payload[1] = rex::byte_swap(initiator & 0x3F);
        uint16_t extents[] = {
            0 >> 3, xenos::kTexture2DCubeMaxWidthHeight >> 3,
            0 >> 3, xenos::kTexture2DCubeMaxWidthHeight >> 3,
            0,      1,
        };
        Amped3FillScreenExtents(extents, address);
        memory::copy_and_swap_16_unaligned(memory_->TranslatePhysical(address), extents,
                                           rex::countof(extents));
        return true;
      }
      break;
    case PM4_SET_BIN_MASK_LO:
    case PM4_SET_BIN_MASK_HI:
    case PM4_SET_BIN_SELECT_LO:
    case PM4_SET_BIN_SELECT_HI:
    case PM4_SET_BIN_MASK:
    case PM4_SET_BIN_SELECT: {
      // The parser's own state (the predicate check above); reuse the
      // single-threaded handling on a copy of the packet.
      uint32_t words[3] = {};
      for (uint32_t i = 0; i < count && i < 2; ++i) {
        words[i] = reader->ReadAndSwap<uint32_t>();
      }
      if (count > 2) {
        reader->AdvanceRead((count - 2) * sizeof(uint32_t));
      }
      switch (opcode) {
        case PM4_SET_BIN_MASK_LO:
          bin_mask_ = (bin_mask_ & 0xFFFFFFFF00000000ull) | words[0];
          break;
        case PM4_SET_BIN_MASK_HI:
          bin_mask_ = (bin_mask_ & 0xFFFFFFFFull) | (uint64_t(words[0]) << 32);
          break;
        case PM4_SET_BIN_SELECT_LO:
          bin_select_ = (bin_select_ & 0xFFFFFFFF00000000ull) | words[0];
          break;
        case PM4_SET_BIN_SELECT_HI:
          bin_select_ = (bin_select_ & 0xFFFFFFFFull) | (uint64_t(words[0]) << 32);
          break;
        case PM4_SET_BIN_MASK:
          bin_mask_ = (uint64_t(words[0]) << 32) | words[1];
          break;
        case PM4_SET_BIN_SELECT:
          bin_select_ = (uint64_t(words[0]) << 32) | words[1];
          break;
      }
      if (Amped3IsExtentsPass(bin_select_) != amped3_split_extents_pass_ ||
          Amped3TileOf(bin_select_) != amped3_split_tile_) {
        amped3_split_extents_pass_ = Amped3IsExtentsPass(bin_select_);
        amped3_split_tile_ = Amped3TileOf(bin_select_);
        uint32_t* payload = Amped3SplitReserve(Amped3Split::kOpExtentsPass, 2);
        payload[0] = uint32_t(amped3_split_extents_pass_);
        payload[1] = amped3_split_tile_;
      }
      return true;
    }
    default:
      break;
  }
  // Everything else runs on the executor, in order. Without the predicate bit:
  // it has been checked here.
  Amped3SplitForward(packet & ~uint32_t(1), reader, count);
  return true;
}

void CommandProcessor::Amped3SplitExecutorMain() {
  Amped3Split& split = *amped3_split_;
  uint64_t read = 0, write = 0, signaled_read = 0;
  uint64_t busy_start = __rdtsc();
  const uint32_t spin_limit = uint32_t(std::max(REXCVAR_GET(gpu_cp_split_spin), 0));
  while (true) {
    if (read == write) {
      write = split.write_pos.load(std::memory_order_acquire);
      if (read == write) {
        split.read_pos.store(read, std::memory_order_release);
        if (split.parser_waiting.load()) {
          split.space_event->Set();
        }
        g_amped3_cp_exec_busy_tsc += __rdtsc() - busy_start;
        for (uint32_t spins = 0; read == write; ++spins) {
          if (!split.running.load(std::memory_order_relaxed)) {
            return;
          }
          if (spins < spin_limit) {
            _mm_pause();
          } else {
            split.executor_sleeping.store(true);
            if (split.write_pos.load() == read) {
              rex::thread::Wait(split.data_event.get(), false, std::chrono::milliseconds(5));
            }
            split.executor_sleeping.store(false);
          }
          write = split.write_pos.load(std::memory_order_acquire);
        }
        busy_start = __rdtsc();
      }
    }
    uint32_t* header = &split.buffer[read % Amped3Split::kCapacity];
    uint32_t op = header[0] >> 24;
    uint32_t payload_dwords = header[0] & 0xFFFFFF;
    switch (op) {
      case Amped3Split::kOpRegisters:
        WriteRegistersFromMem(header[1], header + 2, payload_dwords - 1);
        break;
      case Amped3Split::kOpPacket: {
        memory::RingBuffer reader(reinterpret_cast<uint8_t*>(header + 1),
                                  payload_dwords * sizeof(uint32_t));
        reader.set_write_offset(payload_dwords * sizeof(uint32_t));
        if (!ExecutePacket(&reader)) {
          REXGPU_ERROR("gpu_cp_split: failed to execute a packet");
        }
      } break;
      case Amped3Split::kOpPrimaryEnd:
        OnPrimaryBufferEnd();
        break;
      case Amped3Split::kOpDraw:
        amped3_draw_address_ = header[4];
        Amped3ExecuteDraw(header + 1, "PM4_DRAW_INDX");
        break;
      case Amped3Split::kOpExtentsPass:
        Amped3SetBinSelectState(header[1] != 0, header[2]);
        break;
      case Amped3Split::kOpShader: {
        auto shader =
            reinterpret_cast<Shader*>(uintptr_t(header[2]) | (uintptr_t(header[3]) << 32));
        switch (static_cast<xenos::ShaderType>(header[1])) {
          case xenos::ShaderType::kVertex:
            active_vertex_shader_ = shader;
            break;
          case xenos::ShaderType::kPixel:
            active_pixel_shader_ = shader;
            break;
          default:
            assert_always();
            break;
        }
      } break;
      case Amped3Split::kOpWrap:
        read += Amped3Split::kCapacity - read % Amped3Split::kCapacity;
        continue;
      default:
        assert_always();
        break;
    }
    read += 1 + payload_dwords;
    split.read_pos.store(read, std::memory_order_release);
    // A parser waiting for space is woken after an eighth of the buffer (one
    // waiting for everything to be done, by the empty queue above): setting
    // the event is a system call.
    if (read - signaled_read >= Amped3Split::kCapacity / 8 &&
        split.parser_waiting.load(std::memory_order_relaxed)) {
      signaled_read = read;
      split.space_event->Set();
    }
  }
}

bool CommandProcessor::ExecutePacketType0(memory::RingBuffer* reader, uint32_t packet) {
  // Type-0 packet.
  // Write count registers in sequence to the registers starting at
  // (base_index << 2).

  uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
  if (reader->read_count() < count * sizeof(uint32_t)) {
    REXGPU_ERROR("ExecutePacketType0 overflow (read count {:08X}, packet count {:08X})",
                 reader->read_count(), count * sizeof(uint32_t));
    return false;
  }

  uint32_t base_index = (packet & 0x7FFF);
  uint32_t write_one_reg = (packet >> 15) & 0x1;
  if (g_amped3_count_packets) {
    ++g_amped3_type0_packets;
    g_amped3_type0_registers += count;
  }
  if (!write_one_reg) {
    // amped3 cp perf: consecutive registers through the range path, which
    // copies whole constant ranges at once (as in Xenia Canary).
    WriteRegisterRangeFromRing(reader, base_index, count);
    return true;
  }
  for (uint32_t m = 0; m < count; m++) {
    WriteRegister(base_index, reader->ReadAndSwap<uint32_t>());
  }

  return true;
}

bool CommandProcessor::ExecutePacketType1(memory::RingBuffer* reader, uint32_t packet) {
  // Type-1 packet.
  // Contains two registers of data. Type-0 should be more common.
  uint32_t reg_index_1 = packet & 0x7FF;
  uint32_t reg_index_2 = (packet >> 11) & 0x7FF;
  uint32_t reg_data_1 = reader->ReadAndSwap<uint32_t>();
  uint32_t reg_data_2 = reader->ReadAndSwap<uint32_t>();
  WriteRegister(reg_index_1, reg_data_1);
  WriteRegister(reg_index_2, reg_data_2);
  return true;
}

bool CommandProcessor::ExecutePacketType2(memory::RingBuffer* reader, uint32_t packet) {
  // Type-2 packet.
  // No-op. Do nothing.
  return true;
}

bool CommandProcessor::ExecutePacketType3(memory::RingBuffer* reader, uint32_t packet) {
  // Type-3 packet.
  uint32_t opcode = (packet >> 8) & 0x7F;
  uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
  auto data_start_offset = reader->read_offset();
  if (g_amped3_count_packets) {
    ++g_amped3_opcode_counts[opcode];
  }
  if (g_amped3_tiling_trace) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(reader->read_ptr());
    uint32_t d[3] = {};
    for (uint32_t i = 0; i < 3 && i < count && (i + 1) * 4 <= reader->read_count(); ++i) {
      d[i] = rex::byte_swap(reinterpret_cast<const uint32_t*>(data)[i]);
    }
    std::fprintf(g_amped3_tiling_trace, "%02X n=%u p=%u sel=%llX mask=%llX at=%08X %08X %08X %08X\n",
                 opcode, count, packet & 1, (unsigned long long)bin_select_,
                 (unsigned long long)bin_mask_, uint32_t(reader->read_ptr() - uintptr_t(memory_->TranslatePhysical(0))), d[0], d[1], d[2]);
  }

  if (reader->read_count() < count * sizeof(uint32_t)) {
    REXGPU_ERROR("ExecutePacketType3 overflow (read count {:08X}, packet count {:08X})",
                 reader->read_count(), count * sizeof(uint32_t));
    return false;
  }

  // & 1 == predicate - when set, we do bin check to see if we should execute
  // the packet. Only type 3 packets are affected.
  // We also skip predicated swaps, as they are never valid (probably?).
  if (packet & 1) {
    bool any_pass = (bin_select_ & bin_mask_) != 0;
    if (!any_pass || opcode == PM4_XE_SWAP) {
      reader->AdvanceRead(count * sizeof(uint32_t));
      return true;
    }
  }

  bool result = false;
  switch (opcode) {
    case PM4_ME_INIT:
      result = ExecutePacketType3_ME_INIT(reader, packet, count);
      break;
    case PM4_NOP:
      result = ExecutePacketType3_NOP(reader, packet, count);
      break;
    case PM4_INTERRUPT:
      result = ExecutePacketType3_INTERRUPT(reader, packet, count);
      break;
    case PM4_XE_SWAP:
      result = ExecutePacketType3_XE_SWAP(reader, packet, count);
      break;
    case PM4_INDIRECT_BUFFER:
    case PM4_INDIRECT_BUFFER_PFD:
      result = ExecutePacketType3_INDIRECT_BUFFER(reader, packet, count);
      break;
    case PM4_WAIT_REG_MEM:
      result = ExecutePacketType3_WAIT_REG_MEM(reader, packet, count);
      break;
    case PM4_REG_RMW:
      result = ExecutePacketType3_REG_RMW(reader, packet, count);
      break;
    case PM4_REG_TO_MEM:
      result = ExecutePacketType3_REG_TO_MEM(reader, packet, count);
      break;
    case PM4_MEM_WRITE:
      result = ExecutePacketType3_MEM_WRITE(reader, packet, count);
      break;
    case PM4_COND_WRITE:
      result = ExecutePacketType3_COND_WRITE(reader, packet, count);
      break;
    case PM4_EVENT_WRITE:
      result = ExecutePacketType3_EVENT_WRITE(reader, packet, count);
      break;
    case PM4_EVENT_WRITE_SHD:
      result = ExecutePacketType3_EVENT_WRITE_SHD(reader, packet, count);
      break;
    case PM4_EVENT_WRITE_EXT:
      result = ExecutePacketType3_EVENT_WRITE_EXT(reader, packet, count);
      break;
    case PM4_EVENT_WRITE_ZPD:
      result = ExecutePacketType3_EVENT_WRITE_ZPD(reader, packet, count);
      break;
    case PM4_DRAW_INDX:
      result = ExecutePacketType3_DRAW_INDX(reader, packet, count);
      break;
    case PM4_DRAW_INDX_2:
      result = ExecutePacketType3_DRAW_INDX_2(reader, packet, count);
      break;
    case PM4_SET_CONSTANT:
      result = ExecutePacketType3_SET_CONSTANT(reader, packet, count);
      break;
    case PM4_SET_CONSTANT2:
      result = ExecutePacketType3_SET_CONSTANT2(reader, packet, count);
      break;
    case PM4_LOAD_ALU_CONSTANT:
      result = ExecutePacketType3_LOAD_ALU_CONSTANT(reader, packet, count);
      break;
    case PM4_SET_SHADER_CONSTANTS:
      result = ExecutePacketType3_SET_SHADER_CONSTANTS(reader, packet, count);
      break;
    case PM4_IM_LOAD:
      result = ExecutePacketType3_IM_LOAD(reader, packet, count);
      break;
    case PM4_IM_LOAD_IMMEDIATE:
      result = ExecutePacketType3_IM_LOAD_IMMEDIATE(reader, packet, count);
      break;
    case PM4_INVALIDATE_STATE:
      result = ExecutePacketType3_INVALIDATE_STATE(reader, packet, count);
      break;
    case PM4_VIZ_QUERY:
      result = ExecutePacketType3_VIZ_QUERY(reader, packet, count);
      break;

    case PM4_SET_BIN_MASK_LO: {
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      bin_mask_ = (bin_mask_ & 0xFFFFFFFF00000000ull) | value;
      result = true;
    } break;
    case PM4_SET_BIN_MASK_HI: {
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      bin_mask_ = (bin_mask_ & 0xFFFFFFFFull) | (static_cast<uint64_t>(value) << 32);
      result = true;
    } break;
    case PM4_SET_BIN_SELECT_LO: {
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      bin_select_ = (bin_select_ & 0xFFFFFFFF00000000ull) | value;
      Amped3SetBinSelectState(Amped3IsExtentsPass(bin_select_), Amped3TileOf(bin_select_));
      result = true;
    } break;
    case PM4_SET_BIN_SELECT_HI: {
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      bin_select_ = (bin_select_ & 0xFFFFFFFFull) | (static_cast<uint64_t>(value) << 32);
      Amped3SetBinSelectState(Amped3IsExtentsPass(bin_select_), Amped3TileOf(bin_select_));
      result = true;
    } break;
    case PM4_SET_BIN_MASK: {
      assert_true(count == 2);
      uint64_t val_hi = reader->ReadAndSwap<uint32_t>();
      uint64_t val_lo = reader->ReadAndSwap<uint32_t>();
      bin_mask_ = (val_hi << 32) | val_lo;
      result = true;
    } break;
    case PM4_SET_BIN_SELECT: {
      assert_true(count == 2);
      uint64_t val_hi = reader->ReadAndSwap<uint32_t>();
      uint64_t val_lo = reader->ReadAndSwap<uint32_t>();
      bin_select_ = (val_hi << 32) | val_lo;
      Amped3SetBinSelectState(Amped3IsExtentsPass(bin_select_), Amped3TileOf(bin_select_));
      result = true;
    } break;
    case PM4_CONTEXT_UPDATE: {
      assert_true(count == 1);
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      REXGPU_INFO("GPU context update = {:08X}", value);
      assert_true(value == 0);
      result = true;
      break;
    }
    case PM4_WAIT_FOR_IDLE: {
      // This opcode is used by 5454084E while going / being ingame.
      assert_true(count == 1);
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      REXGPU_INFO("GPU wait for idle = {:08X}", value);
      result = true;
      break;
    }

    default:
      REXGPU_INFO("Unimplemented GPU OPCODE: 0x{:02X}\t\tCOUNT: {}\n", opcode, count);
      assert_always();
      reader->AdvanceRead(count * sizeof(uint32_t));
      break;
  }

  assert_true(reader->read_offset() ==
              (data_start_offset + (count * sizeof(uint32_t))) % reader->capacity());
  return result;
}

bool CommandProcessor::ExecutePacketType3_ME_INIT(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count) {
  // initialize CP's micro-engine
  me_bin_.clear();
  for (uint32_t i = 0; i < count; i++) {
    me_bin_.push_back(reader->ReadAndSwap<uint32_t>());
  }

  return true;
}

bool CommandProcessor::ExecutePacketType3_NOP(memory::RingBuffer* reader, uint32_t packet,
                                              uint32_t count) {
  // skip N 32-bit words to get to the next packet
  // No-op, ignore some data.
  reader->AdvanceRead(count * sizeof(uint32_t));
  return true;
}

bool CommandProcessor::ExecutePacketType3_INTERRUPT(memory::RingBuffer* reader, uint32_t packet,
                                                    uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // generate interrupt from the command stream
  uint32_t cpu_mask = reader->ReadAndSwap<uint32_t>();
  Amped3ResolveScreenExtents();
  if (REXCVAR_GET(gpu_screen_extents_delay) == -2) {
    for (const Amped3DelayedExtents& delayed : g_amped3_delayed_extents) {
      memory::copy_and_swap_16_unaligned(memory_->TranslatePhysical(delayed.address),
                                         delayed.extents, 6);
    }
    g_amped3_delayed_extents.clear();
  }
  for (int n = 0; n < 6; n++) {
    if (cpu_mask & (1 << n)) {
      if (graphics_system_) {
        graphics_system_->DispatchInterruptCallback(1, n);
      }
    }
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_XE_SWAP(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

#ifdef REXGLUE_ENABLE_PERF_COUNTERS
  {
    static uint64_t last_frame_tick = 0;
    uint64_t now = rex::chrono::Clock::QueryHostTickCount();
    if (last_frame_tick) {
      uint64_t freq = rex::chrono::Clock::QueryHostTickFrequency();
      int64_t dt_us = static_cast<int64_t>((now - last_frame_tick) * 1000000 / freq);
      PROFILE_FRAME_TIME_US(dt_us);
      PROFILE_FPS(freq / (now - last_frame_tick));
    }
    last_frame_tick = now;
  }
#endif
  rex::perf::Profiler::Flip();
  Amped3ResolveScreenExtents();
  ++g_amped3_extents_frame;
  while (!g_amped3_delayed_extents.empty() &&
         (REXCVAR_GET(gpu_screen_extents_delay) < 0 ||
          g_amped3_extents_frame - g_amped3_delayed_extents.front().frame >=
              uint32_t(REXCVAR_GET(gpu_screen_extents_delay)))) {
    const Amped3DelayedExtents& delayed = g_amped3_delayed_extents.front();
    memory::copy_and_swap_16_unaligned(memory_->TranslatePhysical(delayed.address),
                                       delayed.extents, 6);
    g_amped3_delayed_extents.pop_front();
  }
  // (Several frames: trace_tiling.txt may hold the frame count, default 1.)
  static int tiling_trace_frames = 0;
  if (g_amped3_tiling_trace) {
    std::fprintf(g_amped3_tiling_trace, "SWAP\n");
    if (--tiling_trace_frames <= 0) {
      std::fclose(g_amped3_tiling_trace);
      g_amped3_tiling_trace = nullptr;
    }
  } else if (std::FILE* trigger = std::fopen("trace_tiling.txt", "r")) {
    if (std::fscanf(trigger, "%d", &tiling_trace_frames) != 1) {
      tiling_trace_frames = 1;
    }
    std::fclose(trigger);
    std::remove("trace_tiling.txt");
    g_amped3_tiling_trace = std::fopen("tiling_trace.txt", "w");
  }

  // Xenia-specific VdSwap hook.
  // VdSwap will post this to tell us we need to swap the screen/fire an
  // interrupt.
  // 63 words here, but only the first has any data.
  uint32_t magic = reader->ReadAndSwap<memory::fourcc_t>();
  assert_true(magic == kSwapSignature);

  // TODO(benvanik): only swap frontbuffer ptr.
  uint32_t frontbuffer_ptr = reader->ReadAndSwap<uint32_t>();
  uint32_t frontbuffer_width = reader->ReadAndSwap<uint32_t>();
  uint32_t frontbuffer_height = reader->ReadAndSwap<uint32_t>();
  reader->AdvanceRead((count - 4) * sizeof(uint32_t));

  if (REXCVAR_GET(swap_times)) {
    static std::FILE* csv = nullptr;
    static uint64_t first = 0, last = 0, n = 0;
    const uint64_t now = rex::chrono::Clock::QueryHostTickCount();
    const double freq = double(rex::chrono::Clock::QueryHostTickFrequency());
    if (!csv && (csv = std::fopen("swaptimes.csv", "w"))) {
      std::fputs("time_s,frame_ms,since_vblank_ms,mark_ms,interrupt_done_ms\n", csv);
      first = last = now;
    }
    if (csv) {
      const uint64_t vblank = g_last_vblank_tick.load(std::memory_order_relaxed);
      const uint64_t guest_now = rex::chrono::Clock::QueryGuestTickCount();
      const double guest_ms = 1000.0 / rex::chrono::Clock::guest_tick_frequency();
      const auto since = [&](uint64_t t) { return t ? (int64_t(t) - int64_t(vblank)) * guest_ms : -1.0; };
      std::fprintf(csv, "%.5f,%.3f,%.3f,%.3f,%.3f\n", (now - first) / freq,
                   (now - last) * 1000.0 / freq, since(guest_now),
                   since(g_vblank_mark_tick.load(std::memory_order_relaxed)),
                   since(g_vblank_done_tick.load(std::memory_order_relaxed)));
      last = now;
      if (++n % 120 == 0) std::fflush(csv);
    }
  }
  IssueSwap(frontbuffer_ptr, frontbuffer_width, frontbuffer_height);

  ++counter_;
  return true;
}

bool CommandProcessor::ExecutePacketType3_INDIRECT_BUFFER(memory::RingBuffer* reader,
                                                          uint32_t packet, uint32_t count) {
  // indirect buffer dispatch
  uint32_t list_ptr = CpuToGpu(reader->ReadAndSwap<uint32_t>());
  uint32_t list_length = reader->ReadAndSwap<uint32_t>();
  assert_zero(list_length & ~0xFFFFF);
  list_length &= 0xFFFFF;
  if (Amped3CollapseSkipsIndirectBuffer()) {
    return true;
  }
  ExecuteIndirectBuffer(GpuToCpu(list_ptr), list_length);
  return true;
}

// Signaled by GraphicsSystem::MarkVblank after the vblank interrupt has run.
rex::thread::Event* VblankWakeEvent() {
  static auto event = rex::thread::Event::CreateAutoResetEvent(false);
  return event.get();
}

bool CommandProcessor::ExecutePacketType3_WAIT_REG_MEM(memory::RingBuffer* reader, uint32_t packet,
                                                       uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // wait until a register or memory location is a specific value

  uint32_t wait_info = reader->ReadAndSwap<uint32_t>();
  uint32_t poll_reg_addr = reader->ReadAndSwap<uint32_t>();
  uint32_t ref = reader->ReadAndSwap<uint32_t>();
  uint32_t mask = reader->ReadAndSwap<uint32_t>();
  uint32_t wait = reader->ReadAndSwap<uint32_t>();

  bool is_memory = (wait_info & 0x10) != 0;

  bool matched = false;
  do {
    uint32_t value = 0;
    if (is_memory) {
      value =
          *reinterpret_cast<uint32_t*>(memory_->TranslatePhysical(poll_reg_addr & ~uint32_t(0x3)));
      value = xenos::GpuSwap(value, static_cast<xenos::Endian>(poll_reg_addr & 0x3));
    } else {
      value = ReadRegisterValue(poll_reg_addr);
      if (poll_reg_addr == XE_GPU_REG_COHER_STATUS_HOST) {
        MakeCoherent();
        value = ReadRegisterValue(poll_reg_addr);
      }
    }
    switch (wait_info & 0x7) {
      case 0x0:  // Never.
        matched = false;
        break;
      case 0x1:  // Less than reference.
        matched = (value & mask) < ref;
        break;
      case 0x2:  // Less than or equal to reference.
        matched = (value & mask) <= ref;
        break;
      case 0x3:  // Equal to reference.
        matched = (value & mask) == ref;
        break;
      case 0x4:  // Not equal to reference.
        matched = (value & mask) != ref;
        break;
      case 0x5:  // Greater than or equal to reference.
        matched = (value & mask) >= ref;
        break;
      case 0x6:  // Greater than reference.
        matched = (value & mask) > ref;
        break;
      case 0x7:  // Always
        matched = true;
        break;
    }
    if (!matched) {
      // Wait.
      if (wait >= 0x100) {
        PrepareForWait();
        const uint64_t next_vblank = g_next_vblank_tick.load(std::memory_order_relaxed);
        const uint64_t now = rex::chrono::Clock::QueryGuestTickCount();
        if (!REXCVAR_GET(vsync)) {
          // User wants it fast and dangerous.
          rex::thread::MaybeYield();
        } else if (REXCVAR_GET(vblank_spin_wait) && next_vblank &&
                   now + rex::chrono::Clock::guest_tick_frequency() * 3 / 2000 >= next_vblank) {
          // A vblank is due within 1.5 ms: poll for it instead of sleeping on
          // the wake event, whose wake-up can come ~0.5 ms late (Wine), and the
          // swap after this wait with it.
          for (int i = 0; i < 64; ++i) {
            _mm_pause();
          }
        } else {
          rex::thread::Wait(VblankWakeEvent(), false,
                            std::chrono::milliseconds(wait / 0x100));
        }
        rex::thread::SyncMemory();
        ReturnFromWait();

        if (!worker_running_) {
          // Short-circuited exit.
          return false;
        }
      } else {
        rex::thread::MaybeYield();
      }
    }
  } while (!matched);

  return true;
}

bool CommandProcessor::ExecutePacketType3_REG_RMW(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count) {
  // register read/modify/write
  // ? (used during shader upload and edram setup)
  uint32_t rmw_info = reader->ReadAndSwap<uint32_t>();
  uint32_t and_mask = reader->ReadAndSwap<uint32_t>();
  uint32_t or_mask = reader->ReadAndSwap<uint32_t>();
  uint32_t value = register_file_->values[rmw_info & 0x1FFF];
  if ((rmw_info >> 31) & 0x1) {
    // & reg
    value &= register_file_->values[and_mask & 0x1FFF];
  } else {
    // & imm
    value &= and_mask;
  }
  if ((rmw_info >> 30) & 0x1) {
    // | reg
    value |= register_file_->values[or_mask & 0x1FFF];
  } else {
    // | imm
    value |= or_mask;
  }
  WriteRegister(rmw_info & 0x1FFF, value);
  return true;
}

bool CommandProcessor::ExecutePacketType3_REG_TO_MEM(memory::RingBuffer* reader, uint32_t packet,
                                                     uint32_t count) {
  // Copy Register to Memory (?)
  // Count is 2, assuming a Register Addr and a Memory Addr.

  uint32_t reg_addr = reader->ReadAndSwap<uint32_t>();
  uint32_t mem_addr = reader->ReadAndSwap<uint32_t>();

  uint32_t reg_val = ReadRegisterValue(reg_addr);

  auto endianness = static_cast<xenos::Endian>(mem_addr & 0x3);
  mem_addr &= ~0x3;
  reg_val = GpuSwap(reg_val, endianness);
  memory::store(memory_->TranslatePhysical(mem_addr), reg_val);

  return true;
}

bool CommandProcessor::ExecutePacketType3_MEM_WRITE(memory::RingBuffer* reader, uint32_t packet,
                                                    uint32_t count) {
  uint32_t write_addr = reader->ReadAndSwap<uint32_t>();
  for (uint32_t i = 0; i < count - 1; i++) {
    uint32_t write_data = reader->ReadAndSwap<uint32_t>();

    auto endianness = static_cast<xenos::Endian>(write_addr & 0x3);
    auto addr = write_addr & ~0x3;
    write_data = GpuSwap(write_data, endianness);
    memory::store(memory_->TranslatePhysical(addr), write_data);
    write_addr += 4;
  }

  return true;
}

bool CommandProcessor::ExecutePacketType3_COND_WRITE(memory::RingBuffer* reader, uint32_t packet,
                                                     uint32_t count) {
  // conditional write to memory or register
  uint32_t wait_info = reader->ReadAndSwap<uint32_t>();
  uint32_t poll_reg_addr = reader->ReadAndSwap<uint32_t>();
  uint32_t ref = reader->ReadAndSwap<uint32_t>();
  uint32_t mask = reader->ReadAndSwap<uint32_t>();
  uint32_t write_reg_addr = reader->ReadAndSwap<uint32_t>();
  uint32_t write_data = reader->ReadAndSwap<uint32_t>();
  uint32_t value;
  if (wait_info & 0x10) {
    // Memory.
    auto endianness = static_cast<xenos::Endian>(poll_reg_addr & 0x3);
    poll_reg_addr &= ~0x3;
    value = memory::load<uint32_t>(memory_->TranslatePhysical(poll_reg_addr));
    value = GpuSwap(value, endianness);
  } else {
    // Register.
    value = ReadRegisterValue(poll_reg_addr);
  }
  bool matched = false;
  switch (wait_info & 0x7) {
    case 0x0:  // Never.
      matched = false;
      break;
    case 0x1:  // Less than reference.
      matched = (value & mask) < ref;
      break;
    case 0x2:  // Less than or equal to reference.
      matched = (value & mask) <= ref;
      break;
    case 0x3:  // Equal to reference.
      matched = (value & mask) == ref;
      break;
    case 0x4:  // Not equal to reference.
      matched = (value & mask) != ref;
      break;
    case 0x5:  // Greater than or equal to reference.
      matched = (value & mask) >= ref;
      break;
    case 0x6:  // Greater than reference.
      matched = (value & mask) > ref;
      break;
    case 0x7:  // Always
      matched = true;
      break;
  }
  if (matched) {
    // Write.
    if (wait_info & 0x100) {
      // Memory.
      auto endianness = static_cast<xenos::Endian>(write_reg_addr & 0x3);
      write_reg_addr &= ~0x3;
      write_data = GpuSwap(write_data, endianness);
      memory::store(memory_->TranslatePhysical(write_reg_addr), write_data);
    } else {
      // Register.
      WriteRegister(write_reg_addr, write_data);
    }
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_EVENT_WRITE(memory::RingBuffer* reader, uint32_t packet,
                                                      uint32_t count) {
  // generate an event that creates a write to memory when completed
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  // Writeback initiator.
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);
  if (count == 1) {
    // Just an event flag? Where does this write?
  } else {
    // Write to an address.
    assert_always();
    reader->AdvanceRead((count - 1) * sizeof(uint32_t));
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_EVENT_WRITE_SHD(memory::RingBuffer* reader,
                                                          uint32_t packet, uint32_t count) {
  // generate a VS|PS_done event
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  uint32_t address = reader->ReadAndSwap<uint32_t>();
  uint32_t value = reader->ReadAndSwap<uint32_t>();

  // Writeback initiator.
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);
  if (REXCVAR_GET(gpu_screen_extents_delay) == -1) {
    for (const Amped3DelayedExtents& delayed : g_amped3_delayed_extents) {
      memory::copy_and_swap_16_unaligned(memory_->TranslatePhysical(delayed.address),
                                         delayed.extents, 6);
    }
    g_amped3_delayed_extents.clear();
  }
  uint32_t data_value;
  if ((initiator >> 31) & 0x1) {
    // Write counter (GPU vblank counter?).
    data_value = counter_;
  } else {
    // Write value.
    data_value = value;
  }
  auto endianness = static_cast<xenos::Endian>(address & 0x3);
  address &= ~0x3;
  data_value = GpuSwap(data_value, endianness);
  memory::store(memory_->TranslatePhysical(address), data_value);
  return true;
}

bool CommandProcessor::ExecutePacketType3_EVENT_WRITE_EXT(memory::RingBuffer* reader,
                                                          uint32_t packet, uint32_t count) {
  // generate a screen extent event
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  uint32_t address = reader->ReadAndSwap<uint32_t>();
  // Writeback initiator.
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);
  auto endianness = static_cast<xenos::Endian>(address & 0x3);
  address &= ~0x3;

  // Let us hope we can fake this.
  // This callback tells the driver the xy coordinates affected by a previous
  // drawcall.
  // https://www.google.com/patents/US20060055701
  uint16_t extents[] = {
      0 >> 3,                                    // min x
      xenos::kTexture2DCubeMaxWidthHeight >> 3,  // max x
      0 >> 3,                                    // min y
      xenos::kTexture2DCubeMaxWidthHeight >> 3,  // max y
      0,                                         // min z
      1,                                         // max z
  };
  Amped3FillScreenExtents(extents, address);
  assert_true(endianness == xenos::Endian::k8in16);
  memory::copy_and_swap_16_unaligned(memory_->TranslatePhysical(address), extents,
                                     rex::countof(extents));
  if (REXCVAR_GET(gpu_screen_extents) >= 3) {
    Amped3RecordScreenExtents(address);
  }
  return true;
}

namespace draw_util {
extern bool amped3_collapse_tile_1;
extern uint32_t amped3_collapse_tile_right;
extern uint32_t amped3_collapse_tile_bottom;
}  // namespace draw_util

bool CommandProcessor::Amped3CollapseSkipsIndirectBuffer() const {
  // gpu_tile_collapse: tiles 2 and 3 run the indirect buffers tile 1 already
  // ran (drawn full-size there); skipping them leaves the registers as tile 1
  // left them plus the outer stream's per-tile writes, as running them would.
  return Amped3TileOf(bin_select_) >= 2 && xenos::kEdramTileCount >= 4096 &&
         REXCVAR_GET(gpu_tile_collapse);
}

void CommandProcessor::Amped3CollapseTile1Registers() {
  // gpu_tile_collapse: the game draws its 720p scene in 3 predicated tiles
  // (vertical columns of a 2x MSAA surface in gameplay, horizontal bands of a
  // 4x MSAA one in menus), each fitting the 10 MB EDRAM. Tile 1 draws
  // everything, so drawn full-size (1280x720, no MSAA, the targets in
  // 720-tile EDRAM slots in their order) it is the whole picture.
  constexpr uint32_t kWidth = 1280, kHeight = 720;
  // 1280x720 at 1x is 16 x 45 tiles; 2x MSAA has twice the rows.
  // (2x MSAA needs 3 * 1440 tiles: only with an 8192-tile EDRAM build.)
  const bool keep_msaa = REXCVAR_GET(gpu_tile_collapse_msaa) && xenos::kEdramTileCount >= 8192;
  const uint32_t kSlotTiles = keep_msaa ? 1440 : 720;
  RegisterFile& regs = *register_file_;
  Amped3Collapse& c = amped3_collapse_;
  uint32_t& scissor_tl = regs.values[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL];
  uint32_t& scissor_br = regs.values[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR];
  uint32_t right = scissor_br & 0x7FFF, bottom = (scissor_br >> 16) & 0x7FFF;
  if (right < kWidth || bottom < kHeight) {
    // Tile 1's own rectangle (resolves of it are widened too).
    c.tile_width = right - (scissor_tl & 0x7FFF);
    c.tile_height = bottom - ((scissor_tl >> 16) & 0x7FFF);
    draw_util::amped3_collapse_tile_right = right < kWidth ? right : 0;
    draw_util::amped3_collapse_tile_bottom = bottom < kHeight ? bottom : 0;
    scissor_br = (scissor_br & ~uint32_t(0x7FFF7FFF)) | std::max(right, kWidth) |
                 (std::max(bottom, kHeight) << 16);
  }
  uint32_t& surface = regs.values[XE_GPU_REG_RB_SURFACE_INFO];
  uint32_t pitch = surface & 0x3FFF;
  auto msaa = xenos::MsaaSamples((surface >> 16) & 3);
  if (surface != c.surface && (msaa != xenos::MsaaSamples::k1X || pitch < kWidth) && c.tile_height) {
    // The EDRAM tiles each target of the tiled surface takes.
    uint32_t pitch_tiles = xenos::GetSurfacePitchTiles(pitch, msaa, false);
    uint32_t rows = (c.tile_height << uint32_t(msaa >= xenos::MsaaSamples::k2X)) / 16;
    c.slot_tiles = std::max(pitch_tiles * rows, uint32_t(1));
    uint32_t new_msaa = keep_msaa ? std::min(uint32_t(msaa), uint32_t(xenos::MsaaSamples::k2X)) : 0;
    surface = (surface & ~uint32_t(0x33FFF)) | kWidth | (new_msaa << 16);
    c.surface = surface;
  } else if (surface != c.surface) {
    return;
  }
  // With MSAA, from slot 1 up: the game's full-size 1x surfaces (the final
  // composite) use the first 1440 tiles, and overlapping them with a 2x target
  // would reinterpret one's samples as the other's.
  const uint32_t first_slot = keep_msaa ? 1 : 0;
  auto remap = [&c, kSlotTiles, first_slot](uint32_t& info) {
    uint32_t base = info & 0x7FF;
    uint32_t slot = (base + c.slot_tiles / 2) / c.slot_tiles + first_slot;
    constexpr uint32_t kBaseMask = (uint32_t(1) << xenos::kEdramBaseTilesBits) - 1;
    info = (info & ~kBaseMask) | std::min(slot * kSlotTiles, kBaseMask);
  };
  remap(regs.values[XE_GPU_REG_RB_COLOR_INFO]);
  remap(regs.values[XE_GPU_REG_RB_COLOR1_INFO]);
  remap(regs.values[XE_GPU_REG_RB_COLOR2_INFO]);
  remap(regs.values[XE_GPU_REG_RB_COLOR3_INFO]);
  remap(regs.values[XE_GPU_REG_RB_DEPTH_INFO]);
}

void CommandProcessor::Amped3FillScreenExtents(uint16_t extents[6], uint32_t address) {
  switch (REXCVAR_GET(gpu_screen_extents)) {
    case 1:
      extents[1] = extents[3] = 0;
      break;
    case 2: {
      float e[4];
      if (Amped3ScreenExtents(e)) {
        auto to_tiles = [](float value, bool max) {
          float clamped = std::min(std::max(max ? std::ceil(value) : std::floor(value), 0.0f),
                                   float(xenos::kTexture2DCubeMaxWidthHeight - 1));
          return uint16_t(uint32_t(clamped) >> 3);
        };
        uint16_t measured[6] = {to_tiles(e[0], false), to_tiles(e[1], true),
                                to_tiles(e[2], false), to_tiles(e[3], true),
                                extents[4],            extents[5]};
        if (REXCVAR_GET(gpu_screen_extents_delay) != 0) {
          Amped3DelayedExtents delayed;
          delayed.frame = g_amped3_extents_frame;
          delayed.address = address;
          std::memcpy(delayed.extents, measured, sizeof(measured));
          g_amped3_delayed_extents.push_back(delayed);
        } else {
          std::memcpy(extents, measured, sizeof(measured));
        }
      }
    } break;
  }
}

bool CommandProcessor::ExecutePacketType3_EVENT_WRITE_ZPD(memory::RingBuffer* reader,
                                                          uint32_t packet, uint32_t count) {
  // Set by D3D as BE but struct ABI is LE
  const uint32_t kQueryFinished = rex::byte_swap(0xFFFFFEED);
  assert_true(count == 1);
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  // Writeback initiator.
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);

  // Occlusion queries:
  // This command is send on query begin and end.
  // As a workaround report some fixed amount of passed samples.
  auto fake_sample_count = REXCVAR_GET(query_occlusion_fake_sample_count);
  if (fake_sample_count >= 0) {
    auto* pSampleCounts = memory_->TranslatePhysical<xe_gpu_depth_sample_counts*>(
        register_file_->values[XE_GPU_REG_RB_SAMPLE_COUNT_ADDR]);
    if (!pSampleCounts) {
      return true;
    }
    // 0xFFFFFEED is written to this two locations by D3D only on D3DISSUE_END
    // and used to detect a finished query.
    bool is_end_via_z_pass =
        pSampleCounts->ZPass_A == kQueryFinished || pSampleCounts->ZPass_B == kQueryFinished;
    // Older versions of D3D also checks for ZFail (4D5307D5).
    bool is_end_via_z_fail =
        pSampleCounts->ZFail_A == kQueryFinished || pSampleCounts->ZFail_B == kQueryFinished;
    std::memset(pSampleCounts, 0, sizeof(xe_gpu_depth_sample_counts));
    if (is_end_via_z_pass || is_end_via_z_fail) {
      pSampleCounts->ZPass_A = fake_sample_count;
      pSampleCounts->Total_A = fake_sample_count;
    }
  }

  return true;
}

bool CommandProcessor::ExecutePacketType3Draw(memory::RingBuffer* reader, uint32_t packet,
                                              const char* opcode_name, uint32_t viz_query_condition,
                                              uint32_t count_remaining) {
  // if viz_query_condition != 0, this is a conditional draw based on viz query.
  // This ID matches the one issued in PM4_VIZ_QUERY
  // uint32_t viz_id = viz_query_condition & 0x3F;
  // when true, render conditionally based on query result
  // uint32_t viz_use = viz_query_condition & 0x100;
  uint32_t words[3];
  amped3_draw_address_ = uint32_t(reader->read_ptr() - uintptr_t(memory_->TranslatePhysical(0)));
  if (!Amped3ReadDraw(reader, opcode_name, count_remaining, words)) {
    return false;
  }
  return Amped3ExecuteDraw(words, opcode_name);
}

// amped3: the reading part of ExecutePacketType3Draw (also gpu_cp_split's
// parser): VGT_DRAW_INITIATOR and, for indexed draws, VGT_DMA_BASE and
// VGT_DMA_SIZE, then skips the rest of the packet.
bool CommandProcessor::Amped3ReadDraw(memory::RingBuffer* reader, const char* opcode_name,
                                      uint32_t count_remaining, uint32_t words[3]) {
  assert_not_zero(count_remaining);
  if (!count_remaining) {
    REXGPU_ERROR("{}: Packet too small, can't read VGT_DRAW_INITIATOR", opcode_name);
    return false;
  }
  reg::VGT_DRAW_INITIATOR vgt_draw_initiator;
  vgt_draw_initiator.value = reader->ReadAndSwap<uint32_t>();
  --count_remaining;
  words[0] = vgt_draw_initiator.value;
  words[1] = words[2] = 0;
  if (vgt_draw_initiator.source_select == xenos::SourceSelect::kDMA) {
    // Two separate bounds checks so if there's only one missing register
    // value out of two, one uint32_t will be skipped in the command buffer,
    // not two.
    assert_not_zero(count_remaining);
    if (!count_remaining) {
      REXGPU_ERROR("{}: Packet too small, can't read VGT_DMA_BASE", opcode_name);
      return false;
    }
    words[1] = reader->ReadAndSwap<uint32_t>();
    --count_remaining;
    assert_not_zero(count_remaining);
    if (!count_remaining) {
      REXGPU_ERROR("{}: Packet too small, can't read VGT_DMA_SIZE", opcode_name);
      return false;
    }
    words[2] = reader->ReadAndSwap<uint32_t>();
    --count_remaining;
  }
  // Skip to the next command, for example, if there are immediate indexes that
  // we don't support yet.
  reader->AdvanceRead(count_remaining * sizeof(uint32_t));
  return true;
}

bool CommandProcessor::Amped3ExecuteDraw(const uint32_t words[3], const char* opcode_name) {
  reg::VGT_DRAW_INITIATOR vgt_draw_initiator;
  vgt_draw_initiator.value = words[0];
  WriteRegister(XE_GPU_REG_VGT_DRAW_INITIATOR, vgt_draw_initiator.value);

  bool draw_succeeded = true;
  // TODO(Triang3l): Remove IndexBufferInfo and replace handling of all this
  // with PrimitiveProcessor when the old Vulkan renderer is removed.
  bool is_indexed = false;
  IndexBufferInfo index_buffer_info;
  switch (vgt_draw_initiator.source_select) {
    case xenos::SourceSelect::kDMA: {
      // Indexed draw.
      is_indexed = true;
      uint32_t vgt_dma_base = words[1];
      WriteRegister(XE_GPU_REG_VGT_DMA_BASE, vgt_dma_base);
      reg::VGT_DMA_SIZE vgt_dma_size;
      vgt_dma_size.value = words[2];
      WriteRegister(XE_GPU_REG_VGT_DMA_SIZE, vgt_dma_size.value);

      uint32_t index_size_bytes = vgt_draw_initiator.index_size == xenos::IndexFormat::kInt16
                                      ? sizeof(uint16_t)
                                      : sizeof(uint32_t);
      // The base address must already be word-aligned according to the R6xx
      // documentation, but for safety.
      index_buffer_info.guest_base = vgt_dma_base & ~(index_size_bytes - 1);
      index_buffer_info.endianness = vgt_dma_size.swap_mode;
      index_buffer_info.format = vgt_draw_initiator.index_size;
      index_buffer_info.length = vgt_dma_size.num_words * index_size_bytes;
      index_buffer_info.count = vgt_draw_initiator.num_indices;
    } break;
    case xenos::SourceSelect::kImmediate: {
      // TODO(Triang3l): VGT_IMMED_DATA.
      REXGPU_ERROR(
          "{}: Using immediate vertex indices, which are not supported yet. "
          "Report the game to Xenia developers!",
          opcode_name, uint32_t(vgt_draw_initiator.source_select));
      draw_succeeded = false;
      assert_always();
    } break;
    case xenos::SourceSelect::kAutoIndex: {
      // Auto draw.
      index_buffer_info.guest_base = 0;
      index_buffer_info.length = 0;
    } break;
    default: {
      // Invalid source selection.
      draw_succeeded = false;
      assert_unhandled_case(vgt_draw_initiator.source_select);
    } break;
  }

  if (draw_succeeded) {
    auto viz_query = register_file_->Get<reg::PA_SC_VIZ_QUERY>();
    if (!(viz_query.viz_query_ena && viz_query.kill_pix_post_hi_z)) {
      // TODO(Triang3l): Don't drop the draw call completely if the vertex
      // shader has memexport.
      // TODO(Triang3l || JoelLinn): Handle this properly in the render
      // backends.

      bool major_mode_explicit =
          xenos::IsMajorModeExplicit(vgt_draw_initiator.major_mode, vgt_draw_initiator.prim_type);
      // (Only with the 4096-tile EDRAM the full-size targets need.)
      const bool collapse = REXCVAR_GET(gpu_tile_collapse) && xenos::kEdramTileCount >= 4096;
      draw_util::amped3_collapse_tile_1 = amped3_tile_ == 1 && collapse;
      if (amped3_tile_ && collapse) {
        if (amped3_tile_ >= 2) {
          // Tile 1 drew (and resolved) everything.
          return true;
        }
        Amped3CollapseTile1Registers();
      }
      if (g_amped3_tiling_trace) {
        // amped3 debug: the state of each draw / copy, to compare the tiles:
        // key registers and XXH3 of the register file in 256-register chunks.
        const RegisterFile& r = *register_file_;
        std::fprintf(g_amped3_tiling_trace,
                     "D mode=%u prim=%u n=%u wo=%08X ws=%08X,%08X surf=%08X c0=%08X c1=%08X "
                     "dep=%08X cpc=%08X cpd=%08X cpp=%08X cpi=%08X ss=%08X vp=%08X,%08X,%08X,%08X "
                     "h=",
                     uint32_t(r.Get<reg::RB_MODECONTROL>().edram_mode),
                     uint32_t(vgt_draw_initiator.prim_type), vgt_draw_initiator.num_indices,
                     r[XE_GPU_REG_PA_SC_WINDOW_OFFSET], r[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL],
                     r[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR], r[XE_GPU_REG_RB_SURFACE_INFO],
                     r[XE_GPU_REG_RB_COLOR_INFO], r[XE_GPU_REG_RB_COLOR1_INFO],
                     r[XE_GPU_REG_RB_DEPTH_INFO], r[XE_GPU_REG_RB_COPY_CONTROL],
                     r[XE_GPU_REG_RB_COPY_DEST_BASE], r[XE_GPU_REG_RB_COPY_DEST_PITCH],
                     r[XE_GPU_REG_RB_COPY_DEST_INFO], r[XE_GPU_REG_PA_SC_SCREEN_SCISSOR_BR],
                     r[XE_GPU_REG_PA_CL_VPORT_XSCALE], r[XE_GPU_REG_PA_CL_VPORT_XOFFSET],
                     r[XE_GPU_REG_PA_CL_VPORT_YSCALE], r[XE_GPU_REG_PA_CL_VPORT_YOFFSET]);
        for (size_t i = 0; i < RegisterFile::kRegisterCount; i += 256) {
          size_t n = std::min(size_t(256), RegisterFile::kRegisterCount - i);
          std::fprintf(g_amped3_tiling_trace, "%08X",
                       uint32_t(XXH3_64bits(&r.values[i], n * sizeof(uint32_t))));
        }
        // The register groups that differ between the tiles, in full.
        for (uint32_t chunk : {64u, 67u, 68u, 71u, 72u}) {
          std::fprintf(g_amped3_tiling_trace, " c%u:", chunk);
          for (uint32_t i = chunk * 256; i < chunk * 256 + 256; ++i) {
            std::fprintf(g_amped3_tiling_trace, "%X,", r.values[i]);
          }
        }
        std::fputc('\n', g_amped3_tiling_trace);
      }
      draw_succeeded = IssueDraw(vgt_draw_initiator.prim_type, vgt_draw_initiator.num_indices,
                                 is_indexed ? &index_buffer_info : nullptr, major_mode_explicit);
      if (!draw_succeeded) {
        auto vgt_output_path_cntl = register_file_->Get<reg::VGT_OUTPUT_PATH_CNTL>();
        auto vgt_hos_cntl = register_file_->Get<reg::VGT_HOS_CNTL>();
        auto rb_modecontrol = register_file_->Get<reg::RB_MODECONTROL>();
        REXGPU_ERROR(
            "{}({}, {}, {}): Failed in backend "
            "(major_mode={}, explicit_major={}, path_select={}, tess_mode={}, edram_mode={})",
            opcode_name, static_cast<uint32_t>(vgt_draw_initiator.num_indices),
            uint32_t(vgt_draw_initiator.prim_type), uint32_t(vgt_draw_initiator.source_select),
            uint32_t(vgt_draw_initiator.major_mode), uint32_t(major_mode_explicit),
            uint32_t(vgt_output_path_cntl.path_select), uint32_t(vgt_hos_cntl.tess_mode),
            uint32_t(rb_modecontrol.edram_mode));
      }
    }
  }

  // If read the packed correctly, but merely couldn't execute it (because of,
  // for instance, features not supported by the host), don't terminate command
  // buffer processing as that would leave rendering in a way more inconsistent
  // state than just a single dropped draw command.
  return true;
}

bool CommandProcessor::ExecutePacketType3_DRAW_INDX(memory::RingBuffer* reader, uint32_t packet,
                                                    uint32_t count) {
  // "initiate fetch of index buffer and draw"
  // Generally used by Xbox 360 Direct3D 9 for kDMA and kAutoIndex sources.
  // With a viz query token as the first one.
  uint32_t count_remaining = count;
  assert_not_zero(count_remaining);
  if (!count_remaining) {
    REXGPU_ERROR("PM4_DRAW_INDX: Packet too small, can't read the viz query token");
    return false;
  }
  uint32_t viz_query_condition = reader->ReadAndSwap<uint32_t>();
  --count_remaining;
  return ExecutePacketType3Draw(reader, packet, "PM4_DRAW_INDX", viz_query_condition,
                                count_remaining);
}

bool CommandProcessor::ExecutePacketType3_DRAW_INDX_2(memory::RingBuffer* reader, uint32_t packet,
                                                      uint32_t count) {
  // "draw using supplied indices in packet"
  // Generally used by Xbox 360 Direct3D 9 for kAutoIndex source.
  // No viz query token.
  return ExecutePacketType3Draw(reader, packet, "PM4_DRAW_INDX_2", 0, count);
}

bool CommandProcessor::ExecutePacketType3_SET_CONSTANT(memory::RingBuffer* reader, uint32_t packet,
                                                       uint32_t count) {
  // load constant into chip and to memory
  // PM4_REG(reg) ((0x4 << 16) | (GSL_HAL_SUBBLOCK_OFFSET(reg)))
  //                                     reg - 0x2000
  uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0x7FF;
  uint32_t type = (offset_type >> 16) & 0xFF;
  uint32_t count_registers = count - 1;
  switch (type) {
    case 0:  // ALU
      WriteALURangeFromRing(reader, index, count_registers);
      break;
    case 1:  // FETCH
      WriteFetchRangeFromRing(reader, index, count_registers);
      break;
    case 2:  // BOOL
      WriteBoolRangeFromRing(reader, index, count_registers);
      break;
    case 3:  // LOOP
      WriteLoopRangeFromRing(reader, index, count_registers);
      break;
    case 4:  // REGISTERS
      WriteREGISTERSRangeFromRing(reader, index, count_registers);
      break;
    default:
      assert_always();
      reader->AdvanceRead((count - 1) * sizeof(uint32_t));
      return true;
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_SET_CONSTANT2(memory::RingBuffer* reader, uint32_t packet,
                                                        uint32_t count) {
  uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0xFFFF;
  WriteRegisterRangeFromRing(reader, index, count - 1);
  return true;
}

bool CommandProcessor::ExecutePacketType3_LOAD_ALU_CONSTANT(memory::RingBuffer* reader,
                                                            uint32_t packet, uint32_t count) {
  // load constants from memory
  uint32_t address = reader->ReadAndSwap<uint32_t>();
  address &= 0x3FFFFFFF;
  uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0x7FF;
  uint32_t size_dwords = reader->ReadAndSwap<uint32_t>();
  size_dwords &= 0xFFF;
  uint32_t type = (offset_type >> 16) & 0xFF;
  uint32_t* xlat_address = memory_->TranslatePhysical<uint32_t*>(address);
  switch (type) {
    case 0:  // ALU
      WriteALURangeFromMem(index, xlat_address, size_dwords);
      break;
    case 1:  // FETCH
      WriteFetchRangeFromMem(index, xlat_address, size_dwords);
      break;
    case 2:  // BOOL
      WriteBoolRangeFromMem(index, xlat_address, size_dwords);
      break;
    case 3:  // LOOP
      WriteLoopRangeFromMem(index, xlat_address, size_dwords);
      break;
    case 4:  // REGISTERS
      WriteREGISTERSRangeFromMem(index, xlat_address, size_dwords);
      break;
    default:
      assert_always();
      return true;
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_SET_SHADER_CONSTANTS(memory::RingBuffer* reader,
                                                               uint32_t packet, uint32_t count) {
  uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0xFFFF;
  WriteRegisterRangeFromRing(reader, index, count - 1);
  return true;
}

bool CommandProcessor::ExecutePacketType3_IM_LOAD(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // load sequencer instruction memory (pointer-based)
  uint32_t addr_type = reader->ReadAndSwap<uint32_t>();
  auto shader_type = static_cast<xenos::ShaderType>(addr_type & 0x3);
  uint32_t addr = addr_type & ~0x3;
  uint32_t start_size = reader->ReadAndSwap<uint32_t>();
  uint32_t start = start_size >> 16;
  uint32_t size_dwords = start_size & 0xFFFF;  // dwords
  assert_true(start == 0);

  auto shader =
      LoadShader(shader_type, addr, memory_->TranslatePhysical<uint32_t*>(addr), size_dwords);
  switch (shader_type) {
    case xenos::ShaderType::kVertex:
      active_vertex_shader_ = shader;
      break;
    case xenos::ShaderType::kPixel:
      active_pixel_shader_ = shader;
      break;
    default:
      assert_unhandled_case(shader_type);
      return false;
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_IM_LOAD_IMMEDIATE(memory::RingBuffer* reader,
                                                            uint32_t packet, uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // load sequencer instruction memory (code embedded in packet)
  uint32_t dword0 = reader->ReadAndSwap<uint32_t>();
  uint32_t dword1 = reader->ReadAndSwap<uint32_t>();
  auto shader_type = static_cast<xenos::ShaderType>(dword0);
  uint32_t start_size = dword1;
  uint32_t start = start_size >> 16;
  uint32_t size_dwords = start_size & 0xFFFF;  // dwords
  assert_true(start == 0);
  assert_true(reader->read_count() >= size_dwords * 4);
  assert_true(count - 2 >= size_dwords);
  auto shader = LoadShader(shader_type, uint32_t(reader->read_ptr()),
                           reinterpret_cast<uint32_t*>(reader->read_ptr()), size_dwords);
  switch (shader_type) {
    case xenos::ShaderType::kVertex:
      active_vertex_shader_ = shader;
      break;
    case xenos::ShaderType::kPixel:
      active_pixel_shader_ = shader;
      break;
    default:
      assert_unhandled_case(shader_type);
      return false;
  }
  reader->AdvanceRead(size_dwords * sizeof(uint32_t));
  return true;
}

bool CommandProcessor::ExecutePacketType3_INVALIDATE_STATE(memory::RingBuffer* reader,
                                                           uint32_t packet, uint32_t count) {
  // selective invalidation of state pointers
  /*uint32_t mask =*/reader->ReadAndSwap<uint32_t>();
  // driver_->InvalidateState(mask);
  return true;
}

bool CommandProcessor::ExecutePacketType3_VIZ_QUERY(memory::RingBuffer* reader, uint32_t packet,
                                                    uint32_t count) {
  // begin/end initiator for viz query extent processing
  // https://www.google.com/patents/US20050195186
  assert_true(count == 1);

  uint32_t dword0 = reader->ReadAndSwap<uint32_t>();

  uint32_t id = dword0 & 0x3F;
  uint32_t end = dword0 & 0x100;
  if (!end) {
    // begin a new viz query @ id
    // On hardware this clears the internal state of the scan converter (which
    // is different to the register)
    WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, VIZQUERY_START);
    REXGPU_INFO("Begin viz query ID {:02X}", id);
  } else {
    // end the viz query
    WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, VIZQUERY_END);
    REXGPU_INFO("End viz query ID {:02X}", id);
    // The scan converter writes the internal result back to the register here.
    // We just fake it and say it was visible in case it is read back.
    if (id < 32) {
      register_file_->values[XE_GPU_REG_PA_SC_VIZ_QUERY_STATUS_0] |= uint32_t(1) << id;
    } else {
      register_file_->values[XE_GPU_REG_PA_SC_VIZ_QUERY_STATUS_1] |= uint32_t(1) << (id - 32);
    }
  }

  return true;
}

}  // namespace rex::graphics
