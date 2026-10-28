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

// Disable warnings about unused parameters for kernel functions
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <cstring>
#include <string>
#include <mutex>
#include <map>
#include <chrono>

#include <rex/assert.h>
#include <rex/audio/audio_system.h>
#include <rex/audio/xma/decoder.h>
#include <rex/kernel/xboxkrnl/private.h>
#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/hook.h>
#include <rex/types.h>
#include <rex/runtime.h>
#include <rex/system/kernel_state.h>
#include <rex/system/thread_state.h>
#include <rex/system/xthread.h>
#include <rex/system/xtypes.h>

REXCVAR_DEFINE_BOOL(xma_trace, false, "Audio", "Debug: log XMA context calls");
#define XMA_TRACE(...) \
  do { if (REXCVAR_GET(xma_trace)) REXKRNL_INFO(__VA_ARGS__); } while (0)

namespace rex::kernel::xboxkrnl {

// --xma_trace: FNV-1a of an input buffer's packets (2048 bytes each, physical
// address), to tell whether a stream replays the same data (loops).
uint32_t BufferHash(uint32_t physical, uint32_t packets) {
  const auto* p = REX_KERNEL_MEMORY()->TranslatePhysical<const uint8_t*>(physical);
  uint32_t h = 2166136261u;
  for (uint32_t i = 0; i < packets * 2048; ++i) {
    h = (h ^ p[i]) * 16777619u;
  }
  return h;
}

// --xma_trace: the calling guest thread's return addresses (lr, then the saved
// lr of each frame up the back chain), to find the game code behind a call.
std::string GuestStack() {
  auto* thread = rex::system::XThread::GetCurrentThread();
  if (!thread || !thread->thread_state() || !thread->thread_state()->context()) {
    return "?";
  }
  const auto* ctx = thread->thread_state()->context();
  auto* memory = REX_KERNEL_MEMORY();
  auto load = [memory](uint32_t address) {
    const auto* p = memory->TranslateVirtual<const uint8_t*>(address);
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
  };
  std::string out = fmt::format("{:08X}", uint32_t(ctx->lr));
  uint32_t sp = ctx->r1.u32;
  for (int i = 0; i < 12 && sp >= 0x70000000 && sp < 0x80000000; ++i) {
    const uint32_t caller_sp = load(sp);
    if (caller_sp <= sp || caller_sp >= 0x80000000) {
      break;
    }
    out += fmt::format(">{:08X}", load(caller_sp - 8));
    sp = caller_sp;
  }
  return out;
}

// --xma_trace: most XMA calls come every audio frame, too often to log one by
// one; count them per context and function and log the counts once a second.
void XmaTraceCount(uint32_t context, const char* function) {
  static std::mutex mutex;
  static std::map<std::pair<uint32_t, std::string>, uint32_t> counts;
  static auto window_start = std::chrono::steady_clock::now();
  std::lock_guard lock(mutex);
  ++counts[{context, function}];
  const auto now = std::chrono::steady_clock::now();
  if (now - window_start < std::chrono::seconds(1)) {
    return;
  }
  std::string out;
  for (const auto& [key, n] : counts) {
    out += fmt::format(" {:08X}:{}={}", key.first, key.second, n);
  }
  REXKRNL_INFO("XMA calls per second:{}", out);
  counts.clear();
  window_start = now;
}
using namespace rex::system;

using rex::audio::XMA_CONTEXT_DATA;

// See audio_system.cc for implementation details.
//
// XMA details:
// https://devel.nuclex.org/external/svn/directx/trunk/include/xma2defs.h
// https://github.com/gdawg/fsbext/blob/master/src/xma_header.h
//
// XMA is undocumented, but the methods are pretty simple.
// Games do this sequence to decode (now):
//   (not sure we are setting buffer validity/offsets right)
// d> XMACreateContext(20656800)
// d> XMAIsInputBuffer0Valid(000103E0)
// d> XMAIsInputBuffer1Valid(000103E0)
// d> XMADisableContext(000103E0, 0)
// d> XMABlockWhileInUse(000103E0)
// d> XMAInitializeContext(000103E0, 20008810)
// d> XMASetOutputBufferValid(000103E0)
// d> XMASetInputBuffer0Valid(000103E0)
// d> XMAEnableContext(000103E0)
// d> XMAGetOutputBufferWriteOffset(000103E0)
// d> XMAGetOutputBufferReadOffset(000103E0)
// d> XMAIsOutputBufferValid(000103E0)
// d> XMAGetOutputBufferReadOffset(000103E0)
// d> XMAGetOutputBufferWriteOffset(000103E0)
// d> XMAIsInputBuffer0Valid(000103E0)
// d> XMAIsInputBuffer1Valid(000103E0)
// d> XMAIsInputBuffer0Valid(000103E0)
// d> XMAIsInputBuffer1Valid(000103E0)
// d> XMAReleaseContext(000103E0)
//
// XAudio2 uses XMA under the covers, and seems to map with the same
// restrictions of frame/subframe/etc:
// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.xaudio2.xaudio2_buffer(v=vs.85).aspx

u32 XMACreateContext_entry(mapped_u32 context_out_ptr) {
  REXKRNL_NOISY_DEBUG("XMACreateContext called!");
  auto xma_decoder =
      static_cast<audio::AudioSystem*>(REX_KERNEL_STATE()->emulator()->audio_system())
          ->xma_decoder();
  uint32_t context_ptr = xma_decoder->AllocateContext();
  *context_out_ptr = context_ptr;
  XMA_TRACE("XMA create {:08X}", context_ptr);
  if (!context_ptr) {
    return X_STATUS_NO_MEMORY;
  }
  return X_STATUS_SUCCESS;
}

u32 XMAReleaseContext_entry(mapped_void context_ptr) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "ReleaseContext");
  auto xma_decoder =
      static_cast<audio::AudioSystem*>(REX_KERNEL_STATE()->emulator()->audio_system())
          ->xma_decoder();
  XMA_TRACE("XMA release {:08X} stack={}", context_ptr.guest_address(), GuestStack());
  xma_decoder->ReleaseContext(context_ptr.guest_address());
  return 0;
}

void StoreXmaContextIndexedRegister(system::KernelState* kernel_state, uint32_t base_reg,
                                    uint32_t context_ptr) {
  uint32_t context_physical_address = REX_KERNEL_MEMORY()->GetPhysicalAddress(context_ptr);
  assert_true(context_physical_address != UINT32_MAX);
  auto xma_decoder =
      static_cast<audio::AudioSystem*>(kernel_state->emulator()->audio_system())->xma_decoder();
  uint32_t hw_index =
      (context_physical_address - xma_decoder->context_array_ptr()) / sizeof(XMA_CONTEXT_DATA);
  uint32_t reg_num = base_reg + (hw_index >> 5) * 4;
  uint32_t reg_value = 1 << (hw_index & 0x1F);
  xma_decoder->WriteRegister(reg_num, rex::byte_swap(reg_value));
}

struct XMA_LOOP_DATA {
  rex::be<uint32_t> loop_start;
  rex::be<uint32_t> loop_end;
  uint8_t loop_count;
  uint8_t loop_subframe_end;
  uint8_t loop_subframe_skip;
};
static_assert_size(XMA_LOOP_DATA, 12);

struct XMA_CONTEXT_INIT {
  rex::be<uint32_t> input_buffer_0_ptr;
  rex::be<uint32_t> input_buffer_0_packet_count;
  rex::be<uint32_t> input_buffer_1_ptr;
  rex::be<uint32_t> input_buffer_1_packet_count;
  rex::be<uint32_t> input_buffer_read_offset;
  rex::be<uint32_t> output_buffer_ptr;
  rex::be<uint32_t> output_buffer_block_count;
  rex::be<uint32_t> work_buffer;
  rex::be<uint32_t> subframe_decode_count;
  rex::be<uint32_t> channel_count;
  rex::be<uint32_t> sample_rate;
  XMA_LOOP_DATA loop_data;
};
static_assert_size(XMA_CONTEXT_INIT, 56);

u32 XMAInitializeContext_entry(mapped_void context_ptr, ppc_ptr_t<XMA_CONTEXT_INIT> context_init) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "InitializeContext");
  // Input buffers may be null (buffer 1 in 415607D4).
  // Convert to host endianness.
  uint32_t input_buffer_0_guest_ptr = context_init->input_buffer_0_ptr;
  uint32_t input_buffer_0_physical_address = 0;
  if (input_buffer_0_guest_ptr) {
    input_buffer_0_physical_address =
        REX_KERNEL_MEMORY()->GetPhysicalAddress(input_buffer_0_guest_ptr);
    // Xenia-specific safety check.
    assert_true(input_buffer_0_physical_address != UINT32_MAX);
    if (input_buffer_0_physical_address == UINT32_MAX) {
      REXKRNL_ERROR("XMAInitializeContext: Invalid input buffer 0 virtual address {:08X}",
                    input_buffer_0_guest_ptr);
      return X_E_FALSE;
    }
  }
  uint32_t input_buffer_1_guest_ptr = context_init->input_buffer_1_ptr;
  uint32_t input_buffer_1_physical_address = 0;
  if (input_buffer_1_guest_ptr) {
    input_buffer_1_physical_address =
        REX_KERNEL_MEMORY()->GetPhysicalAddress(input_buffer_1_guest_ptr);
    assert_true(input_buffer_1_physical_address != UINT32_MAX);
    if (input_buffer_1_physical_address == UINT32_MAX) {
      REXKRNL_ERROR("XMAInitializeContext: Invalid input buffer 1 virtual address {:08X}",
                    input_buffer_1_guest_ptr);
      return X_E_FALSE;
    }
  }
  uint32_t output_buffer_guest_ptr = context_init->output_buffer_ptr;
  assert_not_zero(output_buffer_guest_ptr);
  uint32_t output_buffer_physical_address =
      REX_KERNEL_MEMORY()->GetPhysicalAddress(output_buffer_guest_ptr);
  assert_true(output_buffer_physical_address != UINT32_MAX);
  if (output_buffer_physical_address == UINT32_MAX) {
    REXKRNL_ERROR("XMAInitializeContext: Invalid output buffer virtual address {:08X}",
                  output_buffer_guest_ptr);
    return X_E_FALSE;
  }

  XMA_TRACE("XMA init {:08X} in0={:08X}x{} in1={:08X}x{} out={:08X}x{} ch={} rate={} loop={:X}-{:X} n={}",
            context_ptr.guest_address(), uint32_t(context_init->input_buffer_0_ptr),
            uint32_t(context_init->input_buffer_0_packet_count),
            uint32_t(context_init->input_buffer_1_ptr),
            uint32_t(context_init->input_buffer_1_packet_count), output_buffer_guest_ptr,
            uint32_t(context_init->output_buffer_block_count), uint32_t(context_init->channel_count),
            uint32_t(context_init->sample_rate), uint32_t(context_init->loop_data.loop_start),
            uint32_t(context_init->loop_data.loop_end), context_init->loop_data.loop_count);
  std::memset(context_ptr, 0, sizeof(XMA_CONTEXT_DATA));

  XMA_CONTEXT_DATA context(context_ptr);

  context.input_buffer_0_ptr = input_buffer_0_physical_address;
  context.input_buffer_0_packet_count = context_init->input_buffer_0_packet_count;
  context.input_buffer_1_ptr = input_buffer_1_physical_address;
  context.input_buffer_1_packet_count = context_init->input_buffer_1_packet_count;
  context.input_buffer_read_offset = context_init->input_buffer_read_offset;
  context.output_buffer_ptr = output_buffer_physical_address;
  context.output_buffer_block_count = context_init->output_buffer_block_count;

  // context.work_buffer = context_init->work_buffer;  // ?
  context.subframe_decode_count = context_init->subframe_decode_count;
  context.is_stereo = context_init->channel_count >= 1;
  context.sample_rate = context_init->sample_rate;

  context.loop_start = context_init->loop_data.loop_start;
  context.loop_end = context_init->loop_data.loop_end;
  context.loop_count = context_init->loop_data.loop_count;
  context.loop_subframe_end = context_init->loop_data.loop_subframe_end;
  context.loop_subframe_skip = context_init->loop_data.loop_subframe_skip;

  context.Store(context_ptr);

  StoreXmaContextIndexedRegister(REX_KERNEL_STATE(), 0x1A80, context_ptr.guest_address());

  return 0;
}

u32 XMASetLoopData_entry(mapped_void context_ptr, ppc_ptr_t<XMA_CONTEXT_DATA> loop_data) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "SetLoopData");
  XMA_CONTEXT_DATA context(context_ptr);
  XMA_TRACE("XMA loop {:08X} {:X}-{:X} n={}", context_ptr.guest_address(),
            uint32_t(loop_data->loop_start), uint32_t(loop_data->loop_end),
            uint32_t(loop_data->loop_count));

  context.loop_start = loop_data->loop_start;
  context.loop_end = loop_data->loop_end;
  context.loop_count = loop_data->loop_count;
  context.loop_subframe_end = loop_data->loop_subframe_end;
  context.loop_subframe_skip = loop_data->loop_subframe_skip;

  context.Store(context_ptr);

  return 0;
}

u32 XMAGetInputBufferReadOffset_entry(mapped_void context_ptr) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "GetInputBufferReadOffset");
  XMA_CONTEXT_DATA context(context_ptr);
  return context.input_buffer_read_offset;
}

u32 XMASetInputBufferReadOffset_entry(mapped_void context_ptr, u32 value) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "SetInputBufferReadOffset");
  XMA_CONTEXT_DATA context(context_ptr);
  context.input_buffer_read_offset = value;
  context.Store(context_ptr);

  return 0;
}

u32 XMASetInputBuffer0_entry(mapped_void context_ptr, mapped_void buffer, u32 packet_count) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "SetInputBuffer0");
  uint32_t buffer_physical_address =
      REX_KERNEL_MEMORY()->GetPhysicalAddress(buffer.guest_address());
  assert_true(buffer_physical_address != UINT32_MAX);
  if (buffer_physical_address == UINT32_MAX) {
    // Xenia-specific safety check.
    REXKRNL_ERROR("XMASetInputBuffer0: Invalid buffer virtual address {:08X}",
                  buffer.guest_address());
    return X_E_FALSE;
  }

  XMA_CONTEXT_DATA context(context_ptr);

  context.input_buffer_0_ptr = buffer_physical_address;
  context.input_buffer_0_packet_count = packet_count;

  context.Store(context_ptr);

  return 0;
}

u32 XMAIsInputBuffer0Valid_entry(mapped_void context_ptr) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "IsInputBuffer0Valid");
  XMA_CONTEXT_DATA context(context_ptr);
  return context.input_buffer_0_valid;
}

u32 XMASetInputBuffer0Valid_entry(mapped_void context_ptr) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "SetInputBuffer0Valid");
  XMA_CONTEXT_DATA context(context_ptr);
  XMA_TRACE("XMA feed0 {:08X} buf={:08X}x{} rd={:X} hash={:08X} stack={}", context_ptr.guest_address(),
            uint32_t(context.input_buffer_0_ptr), uint32_t(context.input_buffer_0_packet_count),
            uint32_t(context.input_buffer_read_offset),
            BufferHash(context.input_buffer_0_ptr, context.input_buffer_0_packet_count), GuestStack());
  context.input_buffer_0_valid = 1;
  context.Store(context_ptr);

  return 0;
}

u32 XMASetInputBuffer1_entry(mapped_void context_ptr, mapped_void buffer, u32 packet_count) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "SetInputBuffer1");
  uint32_t buffer_physical_address =
      REX_KERNEL_MEMORY()->GetPhysicalAddress(buffer.guest_address());
  assert_true(buffer_physical_address != UINT32_MAX);
  if (buffer_physical_address == UINT32_MAX) {
    // Xenia-specific safety check.
    REXKRNL_ERROR("XMASetInputBuffer1: Invalid buffer virtual address {:08X}",
                  buffer.guest_address());
    return X_E_FALSE;
  }

  XMA_CONTEXT_DATA context(context_ptr);

  context.input_buffer_1_ptr = buffer_physical_address;
  context.input_buffer_1_packet_count = packet_count;

  context.Store(context_ptr);

  return 0;
}

u32 XMAIsInputBuffer1Valid_entry(mapped_void context_ptr) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "IsInputBuffer1Valid");
  XMA_CONTEXT_DATA context(context_ptr);
  return context.input_buffer_1_valid;
}

u32 XMASetInputBuffer1Valid_entry(mapped_void context_ptr) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "SetInputBuffer1Valid");
  XMA_CONTEXT_DATA context(context_ptr);
  XMA_TRACE("XMA feed1 {:08X} buf={:08X}x{} rd={:X} hash={:08X} stack={}", context_ptr.guest_address(),
            uint32_t(context.input_buffer_1_ptr), uint32_t(context.input_buffer_1_packet_count),
            uint32_t(context.input_buffer_read_offset),
            BufferHash(context.input_buffer_1_ptr, context.input_buffer_1_packet_count), GuestStack());
  context.input_buffer_1_valid = 1;
  context.Store(context_ptr);

  return 0;
}

u32 XMAIsOutputBufferValid_entry(mapped_void context_ptr) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "IsOutputBufferValid");
  XMA_CONTEXT_DATA context(context_ptr);
  return context.output_buffer_valid;
}

u32 XMASetOutputBufferValid_entry(mapped_void context_ptr) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "SetOutputBufferValid");
  XMA_CONTEXT_DATA context(context_ptr);
  context.output_buffer_valid = 1;
  context.Store(context_ptr);

  return 0;
}

u32 XMAGetOutputBufferReadOffset_entry(mapped_void context_ptr) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "GetOutputBufferReadOffset");
  XMA_CONTEXT_DATA context(context_ptr);
  return context.output_buffer_read_offset;
}

u32 XMASetOutputBufferReadOffset_entry(mapped_void context_ptr, u32 value) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "SetOutputBufferReadOffset");
  XMA_CONTEXT_DATA context(context_ptr);
  context.output_buffer_read_offset = value;
  context.Store(context_ptr);

  return 0;
}

u32 XMAGetOutputBufferWriteOffset_entry(mapped_void context_ptr) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "GetOutputBufferWriteOffset");
  XMA_CONTEXT_DATA context(context_ptr);
  return context.output_buffer_write_offset;
}

u32 XMAGetPacketMetadata_entry(mapped_void context_ptr) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "GetPacketMetadata");
  XMA_CONTEXT_DATA context(context_ptr);
  return context.packet_metadata;
}

u32 XMAEnableContext_entry(mapped_void context_ptr) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "EnableContext");
  StoreXmaContextIndexedRegister(REX_KERNEL_STATE(), 0x1940, context_ptr.guest_address());
  return 0;
}

u32 XMADisableContext_entry(mapped_void context_ptr, u32 wait) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "DisableContext");
  X_HRESULT result = X_E_SUCCESS;
  StoreXmaContextIndexedRegister(REX_KERNEL_STATE(), 0x1A40, context_ptr.guest_address());
  if (!static_cast<audio::AudioSystem*>(REX_KERNEL_STATE()->emulator()->audio_system())
           ->xma_decoder()
           ->BlockOnContext(context_ptr.guest_address(), !wait)) {
    result = X_E_FALSE;
  }
  return result;
}

u32 XMABlockWhileInUse_entry(mapped_void context_ptr) {
  if (REXCVAR_GET(xma_trace)) XmaTraceCount(context_ptr.guest_address(), "BlockWhileInUse");
  do {
    XMA_CONTEXT_DATA context(context_ptr);
    if (!context.input_buffer_0_valid && !context.input_buffer_1_valid) {
      break;
    }
    if (!context.work_buffer_ptr) {
      break;
    }
    rex::thread::Sleep(std::chrono::milliseconds(1));
  } while (true);
  return 0;
}

}  // namespace rex::kernel::xboxkrnl

REX_EXPORT(__imp__XMACreateContext, rex::kernel::xboxkrnl::XMACreateContext_entry)
REX_EXPORT(__imp__XMAReleaseContext, rex::kernel::xboxkrnl::XMAReleaseContext_entry)
REX_EXPORT(__imp__XMAInitializeContext, rex::kernel::xboxkrnl::XMAInitializeContext_entry)
REX_EXPORT(__imp__XMASetLoopData, rex::kernel::xboxkrnl::XMASetLoopData_entry)
REX_EXPORT(__imp__XMAGetInputBufferReadOffset,
           rex::kernel::xboxkrnl::XMAGetInputBufferReadOffset_entry)
REX_EXPORT(__imp__XMASetInputBufferReadOffset,
           rex::kernel::xboxkrnl::XMASetInputBufferReadOffset_entry)
REX_EXPORT(__imp__XMASetInputBuffer0, rex::kernel::xboxkrnl::XMASetInputBuffer0_entry)
REX_EXPORT(__imp__XMAIsInputBuffer0Valid, rex::kernel::xboxkrnl::XMAIsInputBuffer0Valid_entry)
REX_EXPORT(__imp__XMASetInputBuffer0Valid, rex::kernel::xboxkrnl::XMASetInputBuffer0Valid_entry)
REX_EXPORT(__imp__XMASetInputBuffer1, rex::kernel::xboxkrnl::XMASetInputBuffer1_entry)
REX_EXPORT(__imp__XMAIsInputBuffer1Valid, rex::kernel::xboxkrnl::XMAIsInputBuffer1Valid_entry)
REX_EXPORT(__imp__XMASetInputBuffer1Valid, rex::kernel::xboxkrnl::XMASetInputBuffer1Valid_entry)
REX_EXPORT(__imp__XMAIsOutputBufferValid, rex::kernel::xboxkrnl::XMAIsOutputBufferValid_entry)
REX_EXPORT(__imp__XMASetOutputBufferValid, rex::kernel::xboxkrnl::XMASetOutputBufferValid_entry)
REX_EXPORT(__imp__XMAGetOutputBufferReadOffset,
           rex::kernel::xboxkrnl::XMAGetOutputBufferReadOffset_entry)
REX_EXPORT(__imp__XMASetOutputBufferReadOffset,
           rex::kernel::xboxkrnl::XMASetOutputBufferReadOffset_entry)
REX_EXPORT(__imp__XMAGetOutputBufferWriteOffset,
           rex::kernel::xboxkrnl::XMAGetOutputBufferWriteOffset_entry)
REX_EXPORT(__imp__XMAGetPacketMetadata, rex::kernel::xboxkrnl::XMAGetPacketMetadata_entry)
REX_EXPORT(__imp__XMAEnableContext, rex::kernel::xboxkrnl::XMAEnableContext_entry)
REX_EXPORT(__imp__XMADisableContext, rex::kernel::xboxkrnl::XMADisableContext_entry)
REX_EXPORT(__imp__XMABlockWhileInUse, rex::kernel::xboxkrnl::XMABlockWhileInUse_entry)
