/**
 * @file        graphics/d3d12/dlss.cpp
 * @brief       NVIDIA DLAA for the 3D scene, run before the HUD is drawn
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include <rex/graphics/d3d12/dlss.h>

#include <algorithm>
#include <mutex>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/graphics/d3d12/command_processor.h>
#include <rex/logging.h>
#include <rex/ui/d3d12/d3d12_util.h>

#if defined(REX_HAS_DLSS)
#include <nvsdk_ngx_helpers.h>
#endif

REXCVAR_DEFINE_STRING(dlss_mode, "off", "GPU",
                      "NVIDIA DLSS for the 3D scene (RTX GPUs): off, or dlaa - anti-aliasing at "
                      "the render resolution")
    .allowed({"off", "dlaa"});

REXCVAR_DEFINE_STRING(dlss_preset, "k", "GPU",
                      "DLSS model (NVIDIA's render presets): k (DLAA's default), l, or m - "
                      "sharper and more stable than k, at about the same cost on RTX 40 GPUs")
    .allowed({"k", "l", "m"});

REXCVAR_DEFINE_INT32(dlss_debug_view, 0, "GPU/Debug",
                     "Show DLSS's inputs instead of its output: 0 = off, 1 = motion vectors, "
                     "2 = depth")
    .range(0, 2);

REXCVAR_DEFINE_BOOL(dlss_jitter, true, "GPU/Debug",
                    "Jitter the scene for DLSS (off = no temporal anti-aliasing, to compare)");

REXCVAR_DECLARE(bool, dlss_log);
REXCVAR_DEFINE_INT32(dlss_jitter_sign, 0, "GPU/Debug",
                     "Negate the jitter given to DLSS: bit 0 = X, bit 1 = Y (to check its "
                     "convention)")
    .range(0, 3);

namespace rex::graphics::d3d12 {

namespace {
// AMD FSR 3 frame generation (the presenter, in another module) asks for the
// scene's depth and camera motion vectors, made at the scene hand-off as for
// DLSS, on any GPU.
std::atomic<bool> g_frame_generation_inputs_requested{false};
// NGX for the game's menu (ReXDlssState): 0 not tried yet, 1 available, 2 not.
std::atomic<int32_t> g_ngx_state{0};
struct FrameGenerationInputs {
  ID3D12Resource* depth = nullptr;
  ID3D12Resource* motion = nullptr;
  uint32_t width = 0, height = 0;
  bool reset = true;
  uint64_t serial = 0;
};
std::mutex g_frame_generation_inputs_mutex;
FrameGenerationInputs g_frame_generation_inputs;
}  // namespace

namespace shaders {
#include "../shaders/bytecode/d3d12_5_1/dlss_debug_cs.h"
#include "../shaders/bytecode/d3d12_5_1/dlss_inputs_cs.h"
}  // namespace shaders

namespace {

// A project ID of our own - NGX only wants a GUID-like string for custom
// engines.
constexpr char kNgxProjectId[] = "6c1f3b2e-8d4a-4f7b-9a35-2e81c0d47b19";

// Halton (2, 3) points, as recommended for DLSS - at least 16 phases for DLAA.
constexpr uint32_t kJitterPhases = 32;

float Halton(uint32_t index, uint32_t base) {
  float result = 0.0f;
  float fraction = 1.0f;
  while (index) {
    fraction /= float(base);
    result += fraction * float(index % base);
    index /= base;
  }
  return result;
}

// Inverse of a row-major 4x4 matrix (Gauss-Jordan with partial pivoting).
bool Invert4x4(const double* matrix, double* inverse) {
  double a[4][8];
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      a[row][column] = matrix[row * 4 + column];
      a[row][4 + column] = row == column ? 1.0 : 0.0;
    }
  }
  for (int column = 0; column < 4; ++column) {
    int pivot = column;
    for (int row = column + 1; row < 4; ++row) {
      if (std::abs(a[row][column]) > std::abs(a[pivot][column])) {
        pivot = row;
      }
    }
    if (std::abs(a[pivot][column]) < 1.0e-12) {
      return false;
    }
    if (pivot != column) {
      for (int i = 0; i < 8; ++i) {
        std::swap(a[pivot][i], a[column][i]);
      }
    }
    double scale = 1.0 / a[column][column];
    for (int i = 0; i < 8; ++i) {
      a[column][i] *= scale;
    }
    for (int row = 0; row < 4; ++row) {
      if (row != column) {
        double factor = a[row][column];
        for (int i = 0; i < 8; ++i) {
          a[row][i] -= factor * a[column][i];
        }
      }
    }
  }
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      inverse[row * 4 + column] = a[row][4 + column];
    }
  }
  return true;
}

struct InputsConstants {
  uint32_t size[2];
  uint32_t has_motion;
  uint32_t padding;
  float reprojection[16];
};

struct DebugConstants {
  uint32_t size[2];
  uint32_t view;
  uint32_t padding;
};

// Root signature: 32-bit constants at b0, then one descriptor table per SRV
// (t0...) and per UAV (u0...) - one-use descriptors aren't contiguous with
// bindless resources.
ID3D12RootSignature* CreateComputeRootSignature(const ui::d3d12::D3D12Provider& provider,
                                                uint32_t constant_count, uint32_t srv_count,
                                                uint32_t uav_count) {
  std::vector<D3D12_DESCRIPTOR_RANGE> ranges(srv_count + uav_count);
  std::vector<D3D12_ROOT_PARAMETER> parameters(1 + srv_count + uav_count);
  D3D12_ROOT_PARAMETER& constants = parameters[0];
  constants.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  constants.Constants.ShaderRegister = 0;
  constants.Constants.RegisterSpace = 0;
  constants.Constants.Num32BitValues = constant_count;
  constants.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  for (uint32_t i = 0; i < srv_count + uav_count; ++i) {
    bool uav = i >= srv_count;
    D3D12_DESCRIPTOR_RANGE& range = ranges[i];
    range.RangeType = uav ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV : D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range.NumDescriptors = 1;
    range.BaseShaderRegister = uav ? i - srv_count : i;
    range.RegisterSpace = 0;
    range.OffsetInDescriptorsFromTableStart = 0;
    D3D12_ROOT_PARAMETER& table = parameters[1 + i];
    table.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    table.DescriptorTable.NumDescriptorRanges = 1;
    table.DescriptorTable.pDescriptorRanges = &range;
    table.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }
  D3D12_ROOT_SIGNATURE_DESC desc;
  desc.NumParameters = UINT(parameters.size());
  desc.pParameters = parameters.data();
  desc.NumStaticSamplers = 0;
  desc.pStaticSamplers = nullptr;
  desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
  return ui::d3d12::util::CreateRootSignature(provider, desc);
}

void CreateTexture2DSrv(ID3D12Device* device, ID3D12Resource* resource, DXGI_FORMAT format,
                        D3D12_CPU_DESCRIPTOR_HANDLE handle) {
  D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
  desc.Format = format;
  desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  desc.Texture2D.MipLevels = 1;
  device->CreateShaderResourceView(resource, &desc, handle);
}

void CreateTexture2DUav(ID3D12Device* device, ID3D12Resource* resource, DXGI_FORMAT format,
                        D3D12_CPU_DESCRIPTOR_HANDLE handle) {
  D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
  desc.Format = format;
  desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  device->CreateUnorderedAccessView(resource, nullptr, &desc, handle);
}

}  // namespace

D3D12Dlss::D3D12Dlss(D3D12CommandProcessor& command_processor)
    : command_processor_(command_processor) {}

D3D12Dlss::~D3D12Dlss() {
  Shutdown();
}

bool D3D12Dlss::Initialize() {
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  *(inputs_root_signature_.ReleaseAndGetAddressOf()) =
      CreateComputeRootSignature(provider, sizeof(InputsConstants) / sizeof(uint32_t), 1, 2);
  *(debug_root_signature_.ReleaseAndGetAddressOf()) =
      CreateComputeRootSignature(provider, sizeof(DebugConstants) / sizeof(uint32_t), 2, 1);
  if (!inputs_root_signature_ || !debug_root_signature_) {
    REXGPU_ERROR("DLSS: failed to create the root signatures");
    return false;
  }
  *(inputs_pipeline_.ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
      device, shaders::dlss_inputs_cs, sizeof(shaders::dlss_inputs_cs),
      inputs_root_signature_.Get());
  *(debug_pipeline_.ReleaseAndGetAddressOf()) = ui::d3d12::util::CreateComputePipeline(
      device, shaders::dlss_debug_cs, sizeof(shaders::dlss_debug_cs), debug_root_signature_.Get());
  if (!inputs_pipeline_ || !debug_pipeline_) {
    REXGPU_ERROR("DLSS: failed to create the compute pipelines");
    return false;
  }
  return true;
}

void D3D12Dlss::Shutdown() {
  {
    std::lock_guard lock(g_frame_generation_inputs_mutex);
    g_frame_generation_inputs = {};
  }
  ShutdownNgx();
  resources_to_release_.clear();
  depth_ = {};
  motion_ = {};
  output_ = {};
  debug_pipeline_.Reset();
  debug_root_signature_.Reset();
  inputs_pipeline_.Reset();
  inputs_root_signature_.Reset();
}

bool D3D12Dlss::IsEnabled() const {
#if defined(REX_HAS_DLSS)
  return inputs_pipeline_ &&
         ((ngx_state_ != NgxState::kUnavailable && !feature_failed_ && REXCVAR_GET(dlss_mode) == "dlaa") ||
          g_frame_generation_inputs_requested.load());
#else
  return false;
#endif
}

bool D3D12Dlss::InitializeNgx() {
#if defined(REX_HAS_DLSS)
  if (ngx_state_ != NgxState::kNotInitialized) {
    return ngx_state_ == NgxState::kAvailable;
  }
  ngx_state_ = NgxState::kUnavailable;
  g_ngx_state.store(2);
  ID3D12Device* device = command_processor_.GetD3D12Provider().GetDevice();

  // nvngx_dlss.dll: next to the executable, or in vendor\dlss there (where the
  // launcher puts it when the player fetches it from NVIDIA).
  std::wstring dll_folder = rex::filesystem::GetExecutableFolder().wstring();
  std::wstring vendor_folder = (rex::filesystem::GetExecutableFolder() / "vendor" / "dlss").wstring();
  const wchar_t* dll_folders[] = {dll_folder.c_str(), vendor_folder.c_str()};
  NVSDK_NGX_FeatureCommonInfo feature_info = {};
  feature_info.PathListInfo.Path = dll_folders;
  feature_info.PathListInfo.Length = 2;
  // NGX wants a writable folder for its logs.
  std::error_code error;
  std::filesystem::path data_folder = std::filesystem::temp_directory_path(error) / "rexglue_ngx";
  std::filesystem::create_directories(data_folder, error);

  NVSDK_NGX_Result result =
      NVSDK_NGX_D3D12_Init_with_ProjectID(kNgxProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0",
                                          data_folder.c_str(), device, &feature_info);
  if (NVSDK_NGX_FAILED(result)) {
    REXGPU_WARN("DLSS: not available - NGX initialization failed ({:08X}); it needs an NVIDIA "
                "RTX GPU",
                uint32_t(result));
    return false;
  }
  result = NVSDK_NGX_D3D12_GetCapabilityParameters(&ngx_parameters_);
  if (NVSDK_NGX_FAILED(result) || !ngx_parameters_) {
    REXGPU_WARN("DLSS: not available - no NGX capability parameters ({:08X})", uint32_t(result));
    ngx_parameters_ = nullptr;
    NVSDK_NGX_D3D12_Shutdown1(device);
    return false;
  }
  int needs_updated_driver = 0;
  int available = 0;
  NVSDK_NGX_Parameter_GetI(ngx_parameters_, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver,
                           &needs_updated_driver);
  NVSDK_NGX_Parameter_GetI(ngx_parameters_, NVSDK_NGX_Parameter_SuperSampling_Available,
                           &available);
  if (!available || needs_updated_driver) {
    int init_result = 0;
    NVSDK_NGX_Parameter_GetI(ngx_parameters_, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult,
                             &init_result);
    REXGPU_WARN("DLSS: not available (available {}, needs a newer driver {}, init result {:08X})",
                available, needs_updated_driver, uint32_t(init_result));
    NVSDK_NGX_D3D12_DestroyParameters(ngx_parameters_);
    ngx_parameters_ = nullptr;
    NVSDK_NGX_D3D12_Shutdown1(device);
    return false;
  }
  ngx_state_ = NgxState::kAvailable;
  g_ngx_state.store(1);
  REXGPU_INFO("DLSS: NGX initialized, DLSS is available");
  return true;
#else
  ngx_state_ = NgxState::kUnavailable;
  return false;
#endif
}

void D3D12Dlss::ShutdownNgx() {
#if defined(REX_HAS_DLSS)
  for (auto& feature : features_to_release_) {
    NVSDK_NGX_D3D12_ReleaseFeature(feature.second);
  }
  features_to_release_.clear();
  if (feature_) {
    NVSDK_NGX_D3D12_ReleaseFeature(feature_);
    feature_ = nullptr;
  }
  feature_scheduled_ = false;
  feature_width_ = 0;
  feature_height_ = 0;
  if (ngx_state_ == NgxState::kAvailable) {
    if (ngx_parameters_) {
      NVSDK_NGX_D3D12_DestroyParameters(ngx_parameters_);
      ngx_parameters_ = nullptr;
    }
    NVSDK_NGX_D3D12_Shutdown1(command_processor_.GetD3D12Provider().GetDevice());
  }
#endif
  ngx_state_ = NgxState::kNotInitialized;
}

bool D3D12Dlss::EnsureTexture(Texture& texture, DXGI_FORMAT format, uint32_t width,
                              uint32_t height, const char* name) {
  if (texture.resource) {
    D3D12_RESOURCE_DESC desc = texture.resource->GetDesc();
    if (desc.Format == format && desc.Width == width && desc.Height == height) {
      return true;
    }
    resources_to_release_.emplace_back(command_processor_.GetCurrentSubmission(),
                                       std::move(texture.resource));
    texture = {};
  }
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  D3D12_RESOURCE_DESC desc = {};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  desc.Width = width;
  desc.Height = height;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.Format = format;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  texture.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  if (FAILED(provider.GetDevice()->CreateCommittedResource(
          &ui::d3d12::util::kHeapPropertiesDefault, provider.GetHeapFlagCreateNotZeroed(), &desc,
          texture.state, nullptr, IID_PPV_ARGS(&texture.resource)))) {
    REXGPU_ERROR("DLSS: failed to create the {} texture ({}x{})", name, width, height);
    texture = {};
    return false;
  }
  return true;
}

void D3D12Dlss::Transition(Texture& texture, D3D12_RESOURCE_STATES state) {
  if (texture.state != state) {
    command_processor_.PushTransitionBarrier(texture.resource.Get(), texture.state, state);
    texture.state = state;
  }
}

namespace {
// clip = M * position (each row of M a dp4).
void Multiply4x4(const double a[16], const double b[16], double out[16]) {
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      double sum = 0.0;
      for (int i = 0; i < 4; ++i) {
        sum += a[row * 4 + i] * b[i * 4 + column];
      }
      out[row * 4 + column] = sum;
    }
  }
}

// Where some points of this frame's screen (NDC x, y at two depths) were in
// the previous one, under a reprojection (row vectors); false if behind.
bool MovedPoints(const double reprojection[16], double moved_out[36]) {
  int n = 0;
  for (double depth_value : {0.5, 0.99}) {
    for (double y : {-0.8, 0.0, 0.8}) {
      for (double x : {-0.8, 0.0, 0.8}) {
        const double point[4] = {x, y, depth_value, 1.0};
        double moved[4];
        for (int column = 0; column < 4; ++column) {
          moved[column] = point[0] * reprojection[column] + point[1] * reprojection[4 + column] +
                          point[2] * reprojection[8 + column] + reprojection[12 + column];
        }
        if (moved[3] <= 1.0e-6) {
          return false;
        }
        moved_out[n++] = moved[0] / moved[3];
        moved_out[n++] = moved[1] / moved[3];
      }
    }
  }
  return true;
}
}  // namespace

bool D3D12Dlss::CameraReprojection(double reprojection_out[16]) const {
  // Objects drawn once in both frames: previous * current^-1 maps this
  // frame's clip space to the last one's.
  struct Candidate {
    double reprojection[16];
    double moved[36];
  };
  std::vector<Candidate> candidates;
  for (const ObjectMatrix& current : frame_objects_) {
    if (current.count != 1 || candidates.size() >= 96) {
      continue;
    }
    const auto previous = std::find_if(previous_objects_.begin(), previous_objects_.end(),
                                       [&](const ObjectMatrix& p) { return p.object == current.object; });
    if (previous == previous_objects_.end() || previous->count != 1) {
      continue;
    }
    double current_matrix[16], previous_matrix[16], current_inverse[16], column_form[16];
    for (int i = 0; i < 16; ++i) {
      current_matrix[i] = current.matrix[i];
      previous_matrix[i] = previous->matrix[i];
    }
    if (!Invert4x4(current_matrix, current_inverse)) {
      continue;
    }
    Multiply4x4(previous_matrix, current_inverse, column_form);
    Candidate candidate;
    // As row vectors for the shader: transposed.
    for (int row = 0; row < 4; ++row) {
      for (int column = 0; column < 4; ++column) {
        candidate.reprojection[row * 4 + column] = column_form[column * 4 + row];
      }
    }
    if (MovedPoints(candidate.reprojection, candidate.moved)) {
      candidates.push_back(candidate);
    }
  }
  // The one the most others agree with (to 1/200 of the screen), if at least
  // three objects and a quarter of them do.
  size_t best = SIZE_MAX, best_agree = 0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    size_t agree = 0;
    for (const Candidate& other : candidates) {
      double difference = 0.0;
      for (int k = 0; k < 36; ++k) {
        difference = std::max(difference, std::abs(candidates[i].moved[k] - other.moved[k]));
      }
      agree += difference < 0.01;
    }
    if (agree > best_agree) {
      best_agree = agree;
      best = i;
    }
  }
  if (best == SIZE_MAX || best_agree < 3 || best_agree * 4 < candidates.size()) {
    return false;
  }
  std::memcpy(reprojection_out, candidates[best].reprojection, sizeof(double) * 16);
  return true;
}

void D3D12Dlss::OnSceneDraw(const float* constants_c0_c3, uint64_t object) {
  if (frame_scene_taken_ || !IsEnabled()) {
    return;
  }
  frame_scene_drawn_ = true;
  for (auto& drawn : frame_objects_) {
    if (drawn.object == object) {
      if (std::memcmp(drawn.matrix.data(), constants_c0_c3, sizeof(float) * 16)) {
        ++drawn.count;
      }
      return;
    }
  }
  if (frame_objects_.size() < 512) {
    auto& drawn = frame_objects_.emplace_back();
    drawn.object = object;
    std::memcpy(drawn.matrix.data(), constants_c0_c3, sizeof(float) * 16);
    drawn.count = 1;
  }
}

bool D3D12Dlss::WantsScene() const {
  return frame_scene_drawn_ && !frame_scene_taken_ && IsEnabled();
}

D3D12_RESOURCE_STATES D3D12Dlss::ProcessScene(ID3D12Resource* color,
                                              D3D12_CPU_DESCRIPTOR_HANDLE depth_srv,
                                              uint32_t width, uint32_t height) {
  D3D12_RESOURCE_STATES color_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  frame_scene_taken_ = true;
#if defined(REX_HAS_DLSS)
  // DLAA (NVIDIA, NGX up) and/or only the inputs, for frame generation.
  const bool dlaa = width && height && REXCVAR_GET(dlss_mode) == "dlaa" && !feature_failed_ &&
                    InitializeNgx();
  frame_dlaa_ = dlaa;
  if (!width || !height || (!dlaa && !g_frame_generation_inputs_requested.load())) {
    return color_state;
  }
  D3D12_RESOURCE_DESC color_desc = color->GetDesc();
  if (color_desc.SampleDesc.Count != 1 || width > color_desc.Width ||
      height > color_desc.Height) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      REXGPU_WARN("DLSS: the scene's color target ({}x{}, {} samples, format {}) doesn't fit {}x{}",
                  color_desc.Width, color_desc.Height, color_desc.SampleDesc.Count,
                  uint32_t(color_desc.Format), width, height);
    }
    return color_state;
  }
  DXGI_FORMAT output_format = color_desc.Format;
  switch (dlaa ? output_format : DXGI_FORMAT_R8G8B8A8_UNORM) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
      break;
    default: {
      static bool format_logged = false;
      if (!format_logged) {
        format_logged = true;
        REXGPU_WARN("DLSS: the scene's color format {} isn't supported", int(output_format));
      }
      return color_state;
    }
  }
  if (!EnsureTexture(depth_, DXGI_FORMAT_R32_FLOAT, width, height, "depth") ||
      !EnsureTexture(motion_, DXGI_FORMAT_R16G16_FLOAT, width, height, "motion vector") ||
      (dlaa && !EnsureTexture(output_, output_format, width, height, "output"))) {
    return color_state;
  }

  // Camera motion: this frame's clip space -> the previous frame's.
  bool reset = !previous_processed_ || width != previous_width_ || height != previous_height_;
  InputsConstants inputs_constants = {};
  inputs_constants.size[0] = width;
  inputs_constants.size[1] = height;
  double reprojection[16];
  const bool camera = !frame_objects_.empty();
  frame_camera_valid_ = camera;
  if (camera && previous_camera_valid_ && CameraReprojection(reprojection)) {
    // A camera cut: points of the screen moving by more than a quarter of it
    // at once.
    double moved[36];
    bool cut = !MovedPoints(reprojection, moved);
    for (int k = 0; !cut && k < 36; ++k) {
      static const double kPoints[18] = {-0.8, -0.8, 0.0, -0.8, 0.8, -0.8, -0.8, 0.0, 0.0,
                                         0.0,  0.8,  0.0, -0.8, 0.8, 0.0,  0.8,  0.8, 0.8};
      cut = std::abs(moved[k] - kPoints[k % 18]) > 0.5;
    }
    if (cut) {
      reset = true;
    } else {
      inputs_constants.has_motion = 1;
      for (int i = 0; i < 16; ++i) {
        inputs_constants.reprojection[i] = float(reprojection[i]);
      }
    }
  }
  if (!inputs_constants.has_motion) {
    reset = true;
  }
  // Why frame generation's inputs reset (every ~2 s with --dlss_log).
  if (REXCVAR_GET(dlss_log)) {
    static uint32_t frames = 0, no_camera = 0, no_previous = 0, cuts = 0, resets = 0;
    ++frames;
    no_camera += !camera;
    no_previous += camera && !previous_camera_valid_;
    cuts += camera && previous_camera_valid_ && !inputs_constants.has_motion;
    resets += reset;
    if (frames >= 120) {
      REXGPU_INFO("DLSS inputs: {} frames, {} reset (no scene {}, no previous frame {}, no agreed "
                  "camera or a cut {})",
                  frames, resets, no_camera, no_previous, cuts);
      frames = no_camera = no_previous = cuts = resets = 0;
    }
  }

  D3D12CommandProcessor& command_processor = command_processor_;
  ID3D12Device* device = command_processor.GetD3D12Provider().GetDevice();
  DeferredCommandList& command_list = command_processor.GetDeferredCommandList();
  uint32_t group_count_x = (width + 7) / 8;
  uint32_t group_count_y = (height + 7) / 8;

  // Depth and motion vectors.
  ui::d3d12::util::DescriptorCpuGpuHandlePair inputs_descriptors[3];
  if (!command_processor.RequestOneUseSingleViewDescriptors(3, inputs_descriptors)) {
    return color_state;
  }
  device->CopyDescriptorsSimple(1, inputs_descriptors[0].first, depth_srv,
                                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  CreateTexture2DUav(device, depth_.resource.Get(), DXGI_FORMAT_R32_FLOAT,
                     inputs_descriptors[1].first);
  CreateTexture2DUav(device, motion_.resource.Get(), DXGI_FORMAT_R16G16_FLOAT,
                     inputs_descriptors[2].first);
  Transition(depth_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  Transition(motion_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  command_processor.SubmitBarriers();
  command_processor.SetExternalPipeline(inputs_pipeline_.Get());
  command_list.D3DSetComputeRootSignature(inputs_root_signature_.Get());
  command_list.D3DSetComputeRoot32BitConstants(0, sizeof(inputs_constants) / sizeof(uint32_t),
                                               &inputs_constants, 0);
  command_list.D3DSetComputeRootDescriptorTable(1, inputs_descriptors[0].second);
  command_list.D3DSetComputeRootDescriptorTable(2, inputs_descriptors[1].second);
  command_list.D3DSetComputeRootDescriptorTable(3, inputs_descriptors[2].second);
  command_list.D3DDispatch(group_count_x, group_count_y, 1);
  Transition(depth_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  Transition(motion_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  {
    std::lock_guard lock(g_frame_generation_inputs_mutex);
    g_frame_generation_inputs.depth = depth_.resource.Get();
    g_frame_generation_inputs.motion = motion_.resource.Get();
    g_frame_generation_inputs.width = width;
    g_frame_generation_inputs.height = height;
    g_frame_generation_inputs.reset = reset;
    ++g_frame_generation_inputs.serial;
  }
  if (!dlaa) {
    // Inputs only (frame generation): the scene stays as it is.
    command_processor.SubmitBarriers();
    frame_processed_ = true;
    previous_width_ = width;
    previous_height_ = height;
    return color_state;
  }
  Transition(output_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  command_processor.SubmitBarriers();

  int32_t debug_view = REXCVAR_GET(dlss_debug_view);
  if (debug_view) {
    ui::d3d12::util::DescriptorCpuGpuHandlePair debug_descriptors[3];
    if (!command_processor.RequestOneUseSingleViewDescriptors(3, debug_descriptors)) {
      return color_state;
    }
    CreateTexture2DSrv(device, motion_.resource.Get(), DXGI_FORMAT_R16G16_FLOAT,
                       debug_descriptors[0].first);
    CreateTexture2DSrv(device, depth_.resource.Get(), DXGI_FORMAT_R32_FLOAT,
                       debug_descriptors[1].first);
    CreateTexture2DUav(device, output_.resource.Get(), output_format, debug_descriptors[2].first);
    DebugConstants debug_constants = {};
    debug_constants.size[0] = width;
    debug_constants.size[1] = height;
    debug_constants.view = uint32_t(debug_view);
    command_processor.SetExternalPipeline(debug_pipeline_.Get());
    command_list.D3DSetComputeRootSignature(debug_root_signature_.Get());
    command_list.D3DSetComputeRoot32BitConstants(0, sizeof(debug_constants) / sizeof(uint32_t),
                                                 &debug_constants, 0);
    command_list.D3DSetComputeRootDescriptorTable(1, debug_descriptors[0].second);
    command_list.D3DSetComputeRootDescriptorTable(2, debug_descriptors[1].second);
    command_list.D3DSetComputeRootDescriptorTable(3, debug_descriptors[2].second);
    command_list.D3DDispatch(group_count_x, group_count_y, 1);
  } else {
    const std::string& preset_name = REXCVAR_GET(dlss_preset);
    uint32_t preset = preset_name == "k"   ? uint32_t(NVSDK_NGX_DLSS_Hint_Render_Preset_K)
                      : preset_name == "l" ? uint32_t(NVSDK_NGX_DLSS_Hint_Render_Preset_L)
                                           : uint32_t(NVSDK_NGX_DLSS_Hint_Render_Preset_M);
    bool create = !feature_scheduled_ || feature_width_ != width || feature_height_ != height ||
                  feature_output_format_ != output_format || feature_preset_ != preset;
    const uint64_t release_after = command_processor.GetCurrentSubmission();
    if (create) {
      feature_scheduled_ = true;
      feature_width_ = width;
      feature_height_ = height;
      feature_output_format_ = output_format;
      feature_preset_ = preset;
      reset = true;
    }
    int32_t jitter_sign = REXCVAR_GET(dlss_jitter_sign);
    float jitter_x = (jitter_sign & 1) ? -jitter_[0] : jitter_[0];
    float jitter_y = (jitter_sign & 2) ? -jitter_[1] : jitter_[1];
    ID3D12Resource* depth_resource = depth_.resource.Get();
    ID3D12Resource* motion_resource = motion_.resource.Get();
    ID3D12Resource* output_resource = output_.resource.Get();
    command_list.ExternalCallback([this, create, release_after, preset, color, depth_resource, motion_resource,
                                   output_resource, width, height, jitter_x, jitter_y,
                                   reset](ID3D12GraphicsCommandList* d3d_command_list) {
      if (create) {
        // The previous feature, released once the GPU is done with it.
        if (feature_) {
          std::lock_guard lock(features_to_release_mutex_);
          features_to_release_.emplace_back(release_after, feature_);
          feature_ = nullptr;
        }
        NVSDK_NGX_Parameter_SetUI(ngx_parameters_, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,
                                  preset);
        NVSDK_NGX_DLSS_Create_Params create_params = {};
        create_params.Feature.InWidth = width;
        create_params.Feature.InHeight = height;
        create_params.Feature.InTargetWidth = width;
        create_params.Feature.InTargetHeight = height;
        create_params.Feature.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_DLAA;
        // Motion vectors at the render resolution, without the jitter; LDR
        // color; depth 0 = near.
        create_params.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
        NVSDK_NGX_Result result = NGX_D3D12_CREATE_DLSS_EXT(d3d_command_list, 1, 1, &feature_,
                                                            ngx_parameters_, &create_params);
        if (NVSDK_NGX_FAILED(result) || !feature_) {
          REXGPU_ERROR("DLSS: failed to create the DLAA feature ({}x{}, {:08X})", width, height,
                       uint32_t(result));
          feature_ = nullptr;
          feature_failed_ = true;
          return;
        }
        REXGPU_INFO("DLSS: DLAA at {}x{}, preset {}", width, height, char('A' + preset - 1));
      }
      if (!feature_) {
        return;
      }
      NVSDK_NGX_D3D12_DLSS_Eval_Params eval_params = {};
      eval_params.Feature.pInColor = color;
      eval_params.Feature.pInOutput = output_resource;
      eval_params.pInDepth = depth_resource;
      eval_params.pInMotionVectors = motion_resource;
      eval_params.InJitterOffsetX = jitter_x;
      eval_params.InJitterOffsetY = jitter_y;
      eval_params.InRenderSubrectDimensions.Width = width;
      eval_params.InRenderSubrectDimensions.Height = height;
      eval_params.InReset = reset ? 1 : 0;
      eval_params.InMVScaleX = 1.0f;
      eval_params.InMVScaleY = 1.0f;
      NVSDK_NGX_Result result =
          NGX_D3D12_EVALUATE_DLSS_EXT(d3d_command_list, feature_, ngx_parameters_, &eval_params);
      if (NVSDK_NGX_FAILED(result)) {
        REXGPU_ERROR("DLSS: evaluation failed ({:08X})", uint32_t(result));
        feature_failed_ = true;
      }
    });
    command_processor.InvalidateStateAfterExternalCommands();
  }

  // The output replaces the scene in the color render target.
  Transition(output_, D3D12_RESOURCE_STATE_COPY_SOURCE);
  command_processor.PushTransitionBarrier(color, color_state, D3D12_RESOURCE_STATE_COPY_DEST);
  color_state = D3D12_RESOURCE_STATE_COPY_DEST;
  command_processor.SubmitBarriers();
  D3D12_TEXTURE_COPY_LOCATION copy_dest;
  copy_dest.pResource = color;
  copy_dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  copy_dest.SubresourceIndex = 0;
  D3D12_TEXTURE_COPY_LOCATION copy_source;
  copy_source.pResource = output_.resource.Get();
  copy_source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  copy_source.SubresourceIndex = 0;
  D3D12_BOX copy_box = {0, 0, 0, width, height, 1};
  command_list.D3DCopyTextureRegion(&copy_dest, 0, 0, 0, &copy_source, &copy_box);

  frame_processed_ = true;
  previous_width_ = width;
  previous_height_ = height;
  static uint32_t processed_log = 0;
  if (processed_log++ % 600 == 0) {
    REXGPU_INFO("DLSS: scene processed ({}x{}, color {}x{} format {}, camera {})", width, height,
                color_desc.Width, color_desc.Height, int(output_format), camera ? "found" : "none");
  }
#endif
  return color_state;
}

void D3D12Dlss::EndFrame() {
  uint64_t completed_submission = command_processor_.GetCompletedSubmission();
  std::erase_if(resources_to_release_, [completed_submission](const auto& resource) {
    return resource.first <= completed_submission;
  });
#if defined(REX_HAS_DLSS)
  std::lock_guard features_lock(features_to_release_mutex_);
  std::erase_if(features_to_release_, [completed_submission](const auto& feature) {
    if (feature.first > completed_submission) {
      return false;
    }
    NVSDK_NGX_D3D12_ReleaseFeature(feature.second);
    return true;
  });
#endif

  previous_processed_ = frame_processed_;
  previous_objects_ = std::move(frame_objects_);
  previous_camera_valid_ = frame_processed_ && frame_camera_valid_;

  // Only jitter the next frame if DLSS ran in this one - in frames it doesn't
  // see (menus without a 3D scene, the HUD hidden) the jitter would just make
  // the picture wobble.
  if (frame_processed_ && frame_dlaa_ && REXCVAR_GET(dlss_jitter)) {
    jitter_index_ = (jitter_index_ + 1) % kJitterPhases;
    jitter_[0] = Halton(jitter_index_ + 1, 2) - 0.5f;
    jitter_[1] = Halton(jitter_index_ + 1, 3) - 0.5f;
  } else {
    jitter_[0] = 0.0f;
    jitter_[1] = 0.0f;
  }

  frame_scene_drawn_ = false;
  frame_scene_taken_ = false;
  frame_processed_ = false;
  frame_objects_.clear();
  frame_camera_valid_ = false;
}

}  // namespace rex::graphics::d3d12

// For the presenter's frame generation (looked up by name: rexruntime doesn't
// link this plugin).
// The game's menu: DLAA on ("dlaa") or off ("off") while running, and whether
// NGX works (0 not tried yet, 1 yes, 2 no: not an RTX GPU or no nvngx_dlss.dll).
extern "C" __declspec(dllexport) void ReXSetDlssMode(const char* mode) {
  rex::cvar::SetFlagByName("dlss_mode", mode);
}
extern "C" __declspec(dllexport) int32_t ReXDlssState() {
  return rex::graphics::d3d12::g_ngx_state.load();
}

extern "C" __declspec(dllexport) void ReXRequestFrameGenerationInputs(int32_t requested) {
  rex::graphics::d3d12::g_frame_generation_inputs_requested.store(requested != 0);
}

// The latest scene depth (R32_FLOAT, 0 = near) and motion vectors (R16G16_FLOAT,
// render-resolution pixels, current to previous), both left in
// NON_PIXEL_SHADER_RESOURCE; false if none yet.
extern "C" __declspec(dllexport) int32_t ReXGetFrameGenerationInputs(ID3D12Resource** depth,
                                                                     ID3D12Resource** motion,
                                                                     uint32_t* width, uint32_t* height,
                                                                     int32_t* reset, uint64_t* serial) {
  using namespace rex::graphics::d3d12;
  std::lock_guard lock(g_frame_generation_inputs_mutex);
  if (!g_frame_generation_inputs.depth || !g_frame_generation_inputs.motion) {
    return 0;
  }
  *depth = g_frame_generation_inputs.depth;
  *motion = g_frame_generation_inputs.motion;
  *width = g_frame_generation_inputs.width;
  *height = g_frame_generation_inputs.height;
  *reset = g_frame_generation_inputs.reset ? 1 : 0;
  *serial = g_frame_generation_inputs.serial;
  return 1;
}
