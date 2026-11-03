/**
 * Presenting through Vulkan under vkd3d-proton (Proton / the Steam Deck):
 * --d3d12_present_vulkan. Step 1: a probe of what vkd3d's Vulkan device
 * offers for our own swap chain (logged; nothing is presented this way yet).
 *
 * vkd3d-proton's ID3D12DXVKInteropDevice1 (include/vkd3d_device_vkd3d_ext.idl)
 * gives the VkInstance / VkDevice behind the D3D12 device, the VkImage behind
 * a resource and the VkQueue behind a command queue. Vulkan functions come
 * from vulkan-1.dll at run time (winevulkan under Proton), so native Windows,
 * without vkd3d, is untouched: the interface just isn't there.
 */

#include <rex/ui/d3d12/vk_interop_present.h>

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <rex/logging.h>
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES  // everything comes from vulkan-1.dll at run time
#endif
#include <vulkan/vulkan.h>
#include <wrl/client.h>

namespace rex::ui::d3d12::vk_interop {

// vkd3d-proton's interop interfaces (method order as in the IDL).
MIDL_INTERFACE("39da4e09-bd1c-4198-9fae-86bbe3be41fd")
ID3D12DXVKInteropDevice : public IUnknown {
  virtual HRESULT STDMETHODCALLTYPE GetDXGIAdapter(REFIID iid, void** object) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetInstanceExtensions(UINT* extension_count, const char** extensions) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetDeviceExtensions(UINT* extension_count, const char** extensions) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetDeviceFeatures(const VkPhysicalDeviceFeatures2** features) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetVulkanHandles(VkInstance* vk_instance, VkPhysicalDevice* vk_physical_device,
                                                     VkDevice* vk_device) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetVulkanQueueInfo(ID3D12CommandQueue* queue, VkQueue* vk_queue,
                                                       UINT32* vk_queue_family) = 0;
  virtual void STDMETHODCALLTYPE GetVulkanImageLayout(ID3D12Resource* resource, D3D12_RESOURCE_STATES state,
                                                       VkImageLayout* vk_layout) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetVulkanResourceInfo(ID3D12Resource* resource, UINT64* vk_handle,
                                                          UINT64* buffer_offset) = 0;
  virtual HRESULT STDMETHODCALLTYPE LockCommandQueue(ID3D12CommandQueue* queue) = 0;
  virtual HRESULT STDMETHODCALLTYPE UnlockCommandQueue(ID3D12CommandQueue* queue) = 0;
};

MIDL_INTERFACE("902d8115-59eb-4406-9518-fe00f991ee65")
ID3D12DXVKInteropDevice1 : public ID3D12DXVKInteropDevice {
  virtual HRESULT STDMETHODCALLTYPE GetVulkanResourceInfo1(ID3D12Resource* resource, UINT64* vk_handle,
                                                           UINT64* buffer_offset, VkFormat* format) = 0;
  virtual HRESULT STDMETHODCALLTYPE CreateInteropCommandQueue(const D3D12_COMMAND_QUEUE_DESC* desc,
                                                              UINT32 vk_queue_family_index,
                                                              ID3D12CommandQueue** queue) = 0;
  virtual HRESULT STDMETHODCALLTYPE CreateInteropCommandAllocator(D3D12_COMMAND_LIST_TYPE type,
                                                                  UINT32 vk_queue_family_index,
                                                                  ID3D12CommandAllocator** allocator) = 0;
  virtual HRESULT STDMETHODCALLTYPE BeginVkCommandBufferInterop(ID3D12CommandList* command_list,
                                                                VkCommandBuffer* command_buffer) = 0;
  virtual HRESULT STDMETHODCALLTYPE EndVkCommandBufferInterop(ID3D12CommandList* command_list) = 0;
};

namespace {

bool HasExtension(const std::vector<std::string>& list, const char* name) {
  for (const std::string& e : list) {
    if (e == name) return true;
  }
  return false;
}

std::vector<std::string> Extensions(ID3D12DXVKInteropDevice* interop, bool device) {
  UINT count = 0;
  if (FAILED(device ? interop->GetDeviceExtensions(&count, nullptr) : interop->GetInstanceExtensions(&count, nullptr))) {
    return {};
  }
  std::vector<const char*> names(count);
  if (FAILED(device ? interop->GetDeviceExtensions(&count, names.data())
                    : interop->GetInstanceExtensions(&count, names.data()))) {
    return {};
  }
  return std::vector<std::string>(names.begin(), names.begin() + count);
}

}  // namespace

// ---- Presenting -----------------------------------------------------------

namespace {

struct Fns {
  PFN_vkGetDeviceProcAddr get_device_proc = nullptr;
  PFN_vkCreateWin32SurfaceKHR create_surface = nullptr;
  PFN_vkDestroySurfaceKHR destroy_surface = nullptr;
  PFN_vkGetPhysicalDeviceSurfaceSupportKHR surface_support = nullptr;
  PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR surface_caps = nullptr;
  PFN_vkGetPhysicalDeviceQueueFamilyProperties families = nullptr;
  PFN_vkCreateSwapchainKHR create_swapchain = nullptr;
  PFN_vkDestroySwapchainKHR destroy_swapchain = nullptr;
  PFN_vkGetSwapchainImagesKHR swapchain_images = nullptr;
  PFN_vkAcquireNextImageKHR acquire = nullptr;
  PFN_vkQueuePresentKHR present = nullptr;
  PFN_vkWaitForPresentKHR wait_for_present = nullptr;
  PFN_vkCreateFence create_fence = nullptr;
  PFN_vkDestroyFence destroy_fence = nullptr;
  PFN_vkWaitForFences wait_fences = nullptr;
  PFN_vkResetFences reset_fences = nullptr;
  PFN_vkCmdPipelineBarrier cmd_barrier = nullptr;
  PFN_vkCmdCopyImage cmd_copy = nullptr;
  PFN_vkDeviceWaitIdle device_wait_idle = nullptr;
};

}  // namespace

struct Presenter::State {
  Microsoft::WRL::ComPtr<ID3D12DXVKInteropDevice1> interop;
  Microsoft::WRL::ComPtr<ID3D12Device> device;
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice vk_device = VK_NULL_HANDLE;
  Fns fn;
  HWND hwnd = nullptr;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  uint32_t family = ~0u;
  Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;  // ours, on `family`
  VkQueue vk_queue = VK_NULL_HANDLE;
  static constexpr uint32_t kLists = 3;
  Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocators[kLists];
  Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> lists[kLists];
  uint64_t list_index = 0;
  Microsoft::WRL::ComPtr<ID3D12Fence> paint_fence;  // signalled on the game's queue after the paint
  uint64_t paint_value = 0;
  Microsoft::WRL::ComPtr<ID3D12Fence> copy_fence;  // signalled on ours after the copy
  uint64_t copy_value = 0;
  HANDLE copy_event = nullptr;
  VkFence acquire_fence = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  uint32_t width = 0, height = 0;
  std::vector<VkImage> images;
  bool present_wait = false;
  uint64_t present_id = 0;
  // Display flips: a thread waits for each present id.
  std::thread flip_thread;
  std::mutex flip_mutex;
  std::condition_variable flip_cv;
  std::deque<uint64_t> flip_ids;
  bool flip_stop = false;
};

Presenter::Presenter() = default;
Presenter::~Presenter() { Shutdown(); }

bool Presenter::Initialize(ID3D12Device* device, HWND hwnd) {
  auto st = std::make_unique<State>();
  if (FAILED(device->QueryInterface(__uuidof(ID3D12DXVKInteropDevice1),
                                    reinterpret_cast<void**>(st->interop.GetAddressOf())))) {
    return false;
  }
  st->device = device;
  st->hwnd = hwnd;
  if (FAILED(st->interop->GetVulkanHandles(&st->instance, &st->physical_device, &st->vk_device))) return false;
  const std::vector<std::string> device_ext = Extensions(st->interop.Get(), true);
  st->present_wait = HasExtension(device_ext, "VK_KHR_present_id") && HasExtension(device_ext, "VK_KHR_present_wait");
  HMODULE vulkan = LoadLibraryW(L"vulkan-1.dll");
  auto gip = vulkan ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(vulkan, "vkGetInstanceProcAddr"))
                    : nullptr;
  if (!gip) return false;
  Fns& f = st->fn;
#define REX_VKI(member, name) f.member = reinterpret_cast<PFN_##name>(gip(st->instance, #name))
  REX_VKI(get_device_proc, vkGetDeviceProcAddr);
  REX_VKI(create_surface, vkCreateWin32SurfaceKHR);
  REX_VKI(destroy_surface, vkDestroySurfaceKHR);
  REX_VKI(surface_support, vkGetPhysicalDeviceSurfaceSupportKHR);
  REX_VKI(surface_caps, vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
  REX_VKI(families, vkGetPhysicalDeviceQueueFamilyProperties);
#undef REX_VKI
  if (!f.get_device_proc) return false;
#define REX_VKD(member, name) f.member = reinterpret_cast<PFN_##name>(f.get_device_proc(st->vk_device, #name))
  REX_VKD(create_swapchain, vkCreateSwapchainKHR);
  REX_VKD(destroy_swapchain, vkDestroySwapchainKHR);
  REX_VKD(swapchain_images, vkGetSwapchainImagesKHR);
  REX_VKD(acquire, vkAcquireNextImageKHR);
  REX_VKD(present, vkQueuePresentKHR);
  REX_VKD(wait_for_present, vkWaitForPresentKHR);
  REX_VKD(create_fence, vkCreateFence);
  REX_VKD(destroy_fence, vkDestroyFence);
  REX_VKD(wait_fences, vkWaitForFences);
  REX_VKD(reset_fences, vkResetFences);
  REX_VKD(cmd_barrier, vkCmdPipelineBarrier);
  REX_VKD(cmd_copy, vkCmdCopyImage);
  REX_VKD(device_wait_idle, vkDeviceWaitIdle);
#undef REX_VKD
  if (!f.create_surface || !f.create_swapchain || !f.acquire || !f.present || !f.cmd_copy || !f.cmd_barrier) {
    REXLOG_WARN("Vulkan present: functions missing");
    return false;
  }
  if (!f.wait_for_present) st->present_wait = false;

  VkWin32SurfaceCreateInfoKHR surface_info = {VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
  surface_info.hinstance = GetModuleHandleW(nullptr);
  surface_info.hwnd = hwnd;
  REXLOG_INFO("Vulkan present: functions loaded");
  if (f.create_surface(st->instance, &surface_info, nullptr, &st->surface) != VK_SUCCESS) {
    REXLOG_WARN("Vulkan present: no surface for the window");
    return false;
  }
  // A compute family that can present (its own hardware queue: the copy never
  // waits behind the game's graphics work), else any that can.
  uint32_t family_count = 0;
  f.families(st->physical_device, &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  f.families(st->physical_device, &family_count, families.data());
  uint32_t fallback = ~0u;
  for (uint32_t i = 0; i < family_count; ++i) {
    VkBool32 can_present = VK_FALSE;
    f.surface_support(st->physical_device, i, st->surface, &can_present);
    if (!can_present) continue;
    if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
      st->family = i;
      break;
    }
    if (fallback == ~0u) fallback = i;
  }
  if (st->family == ~0u) st->family = fallback;
  if (st->family == ~0u) {
    REXLOG_WARN("Vulkan present: no queue family can present");
    f.destroy_surface(st->instance, st->surface, nullptr);
    return false;
  }
  const bool compute = !(families[st->family].queueFlags & VK_QUEUE_GRAPHICS_BIT);
  const D3D12_COMMAND_LIST_TYPE list_type = compute ? D3D12_COMMAND_LIST_TYPE_COMPUTE : D3D12_COMMAND_LIST_TYPE_DIRECT;
  REXLOG_INFO("Vulkan present: surface made, presenting from queue family {}", st->family);
  UINT32 queue_family_check = 0;
  D3D12_COMMAND_QUEUE_DESC queue_desc = {};
  queue_desc.Type = list_type;
  if (FAILED(st->interop->CreateInteropCommandQueue(&queue_desc, st->family, &st->queue)) ||
      FAILED(st->interop->GetVulkanQueueInfo(st->queue.Get(), &st->vk_queue, &queue_family_check))) {
    REXLOG_WARN("Vulkan present: no queue on family {}", st->family);
    f.destroy_surface(st->instance, st->surface, nullptr);
    return false;
  }
  REXLOG_INFO("Vulkan present: our queue made (family {})", queue_family_check);
  for (uint32_t i = 0; i < State::kLists; ++i) {
    if (FAILED(st->interop->CreateInteropCommandAllocator(list_type, st->family, &st->allocators[i])) ||
        FAILED(device->CreateCommandList(0, list_type, st->allocators[i].Get(), nullptr,
                                         IID_PPV_ARGS(&st->lists[i])))) {
      REXLOG_WARN("Vulkan present: no command lists");
      f.destroy_surface(st->instance, st->surface, nullptr);
      return false;
    }
    st->lists[i]->Close();
  }
  REXLOG_INFO("Vulkan present: command lists made");
  device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&st->paint_fence));
  device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&st->copy_fence));
  st->copy_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  f.create_fence(st->vk_device, &fence_info, nullptr, &st->acquire_fence);
  if (!st->paint_fence || !st->copy_fence || !st->copy_event || !st->acquire_fence) return false;
  REXLOG_INFO("Vulkan present: own swap chain on queue family {} ({}), present wait {}", st->family,
              compute ? "compute" : "graphics", st->present_wait);
  state_ = std::move(st);
  return true;
}

void Presenter::DestroySwapchain() {
  State& st = *state_;
  if (st.flip_thread.joinable()) {
    {
      std::lock_guard<std::mutex> lock(st.flip_mutex);
      st.flip_stop = true;
    }
    st.flip_cv.notify_all();
    st.flip_thread.join();
    st.flip_stop = false;
    st.flip_ids.clear();
  }
  if (st.swapchain) {
    st.interop->LockCommandQueue(st.queue.Get());
    st.fn.device_wait_idle(st.vk_device);
    st.interop->UnlockCommandQueue(st.queue.Get());
    st.fn.destroy_swapchain(st.vk_device, st.swapchain, nullptr);
    st.swapchain = VK_NULL_HANDLE;
    st.images.clear();
  }
}

bool Presenter::CreateSwapchain(uint32_t width, uint32_t height) {
  State& st = *state_;
  DestroySwapchain();
  VkSurfaceCapabilitiesKHR caps = {};
  st.fn.surface_caps(st.physical_device, st.surface, &caps);
  VkSwapchainCreateInfoKHR info = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  info.surface = st.surface;
  info.minImageCount = std::max(caps.minImageCount, 3u);
  if (caps.maxImageCount) info.minImageCount = std::min(info.minImageCount, caps.maxImageCount);
  info.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;  // the DXGI swap chain's (kSwapChainFormat)
  info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  info.imageExtent = {width, height};
  info.imageArrayLayers = 1;
  info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  info.clipped = VK_TRUE;
  if (st.fn.create_swapchain(st.vk_device, &info, nullptr, &st.swapchain) != VK_SUCCESS) {
    REXLOG_WARN("Vulkan present: vkCreateSwapchainKHR failed ({}x{})", width, height);
    st.swapchain = VK_NULL_HANDLE;
    return false;
  }
  uint32_t count = 0;
  st.fn.swapchain_images(st.vk_device, st.swapchain, &count, nullptr);
  st.images.resize(count);
  st.fn.swapchain_images(st.vk_device, st.swapchain, &count, st.images.data());
  st.width = width;
  st.height = height;
  if (st.present_wait) {
    st.flip_thread = std::thread([&st, this] {
      for (;;) {
        uint64_t id;
        {
          std::unique_lock<std::mutex> lock(st.flip_mutex);
          st.flip_cv.wait(lock, [&st] { return st.flip_stop || !st.flip_ids.empty(); });
          if (st.flip_stop) return;
          id = st.flip_ids.front();
          st.flip_ids.pop_front();
        }
        if (st.fn.wait_for_present(st.vk_device, st.swapchain, id, 200000000) == VK_SUCCESS && on_flip_) {
          on_flip_();
        }
      }
    });
  }
  REXLOG_INFO("Vulkan present: swap chain {}x{}, {} images, FIFO", width, height, count);
  return true;
}

void Presenter::SetFlipCallback(void (*callback)()) { on_flip_ = callback; }

bool Presenter::Present(ID3D12Resource* source, ID3D12CommandQueue* paint_queue) {
  if (!state_) return false;
  State& st = *state_;
  const D3D12_RESOURCE_DESC desc = source->GetDesc();
  if (!st.swapchain || st.width != uint32_t(desc.Width) || st.height != desc.Height) {
    if (!CreateSwapchain(uint32_t(desc.Width), desc.Height)) return false;
  }
  // Our queue waits for the paint on the game's queue.
  paint_queue->Signal(st.paint_fence.Get(), ++st.paint_value);
  st.queue->Wait(st.paint_fence.Get(), st.paint_value);
  // A free swap chain image.
  uint32_t image_index = 0;
  VkResult r = st.fn.acquire(st.vk_device, st.swapchain, UINT64_MAX, VK_NULL_HANDLE, st.acquire_fence, &image_index);
  if (r == VK_ERROR_OUT_OF_DATE_KHR) {
    CreateSwapchain(st.width, st.height);
    return true;  // this frame is skipped
  }
  if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) return false;
  st.fn.wait_fences(st.vk_device, 1, &st.acquire_fence, VK_TRUE, UINT64_MAX);
  st.fn.reset_fences(st.vk_device, 1, &st.acquire_fence);
  // The copy, in Vulkan commands inside one of our D3D12 command lists.
  const uint32_t li = uint32_t(st.list_index++ % State::kLists);
  if (st.copy_fence->GetCompletedValue() + State::kLists <= st.copy_value) {
    st.copy_fence->SetEventOnCompletion(st.copy_value + 1 - State::kLists, st.copy_event);
    WaitForSingleObject(st.copy_event, INFINITE);
  }
  st.allocators[li]->Reset();
  st.lists[li]->Reset(st.allocators[li].Get(), nullptr);
  VkCommandBuffer cb = VK_NULL_HANDLE;
  UINT64 source_handle = 0, offset = 0;
  VkImageLayout source_layout = VK_IMAGE_LAYOUT_GENERAL;
  st.interop->GetVulkanResourceInfo(source, &source_handle, &offset);
  st.interop->GetVulkanImageLayout(source, D3D12_RESOURCE_STATE_PRESENT, &source_layout);
  if (FAILED(st.interop->BeginVkCommandBufferInterop(st.lists[li].Get(), &cb)) || !cb || !source_handle) {
    st.lists[li]->Close();
    return false;
  }
  VkImage dst = st.images[image_index];
  VkImageMemoryBarrier to_dst = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  to_dst.srcAccessMask = 0;
  to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_dst.srcQueueFamilyIndex = to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_dst.image = dst;
  to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  st.fn.cmd_barrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                    1, &to_dst);
  VkImageCopy region = {};
  region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.extent = {st.width, st.height, 1};
  st.fn.cmd_copy(cb, reinterpret_cast<VkImage>(source_handle), source_layout, dst,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  VkImageMemoryBarrier to_present = to_dst;
  to_present.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_present.dstAccessMask = 0;
  to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  st.fn.cmd_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                    nullptr, 1, &to_present);
  st.interop->EndVkCommandBufferInterop(st.lists[li].Get());
  st.lists[li]->Close();
  ID3D12CommandList* lists[] = {st.lists[li].Get()};
  st.queue->ExecuteCommandLists(1, lists);
  st.queue->Signal(st.copy_fence.Get(), ++st.copy_value);
  // The copy done before the present (no semaphores between D3D12 and our
  // raw present: a CPU wait, ~1 ms).
  st.copy_fence->SetEventOnCompletion(st.copy_value, st.copy_event);
  WaitForSingleObject(st.copy_event, INFINITE);
  VkPresentIdKHR present_id = {VK_STRUCTURE_TYPE_PRESENT_ID_KHR};
  const uint64_t id = ++st.present_id;
  present_id.swapchainCount = 1;
  present_id.pPresentIds = &id;
  VkPresentInfoKHR present = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  present.pNext = st.present_wait ? &present_id : nullptr;
  present.swapchainCount = 1;
  present.pSwapchains = &st.swapchain;
  present.pImageIndices = &image_index;
  st.interop->LockCommandQueue(st.queue.Get());
  r = st.fn.present(st.vk_queue, &present);
  st.interop->UnlockCommandQueue(st.queue.Get());
  if (st.present_wait && (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR)) {
    {
      std::lock_guard<std::mutex> lock(st.flip_mutex);
      st.flip_ids.push_back(id);
    }
    st.flip_cv.notify_one();
  }
  if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) CreateSwapchain(st.width, st.height);
  return r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR || r == VK_ERROR_OUT_OF_DATE_KHR;
}

void Presenter::Shutdown() {
  if (!state_) return;
  DestroySwapchain();
  State& st = *state_;
  if (st.acquire_fence) st.fn.destroy_fence(st.vk_device, st.acquire_fence, nullptr);
  if (st.surface) st.fn.destroy_surface(st.instance, st.surface, nullptr);
  if (st.copy_event) CloseHandle(st.copy_event);
  state_.reset();
}

bool Probe(ID3D12Device* device, ID3D12CommandQueue* direct_queue, HWND hwnd) {
  Microsoft::WRL::ComPtr<ID3D12DXVKInteropDevice1> interop;
  if (FAILED(device->QueryInterface(__uuidof(ID3D12DXVKInteropDevice1), reinterpret_cast<void**>(interop.GetAddressOf())))) {
    REXLOG_INFO("Vulkan present: no vkd3d-proton interop (not running under vkd3d)");
    return false;
  }
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice vk_device = VK_NULL_HANDLE;
  if (FAILED(interop->GetVulkanHandles(&instance, &physical_device, &vk_device))) {
    REXLOG_WARN("Vulkan present: GetVulkanHandles failed");
    return false;
  }
  const std::vector<std::string> instance_ext = Extensions(interop.Get(), false);
  const std::vector<std::string> device_ext = Extensions(interop.Get(), true);
  REXLOG_INFO("Vulkan present: vkd3d instance has {} extensions (win32_surface {}), device {} (swapchain {}, present_id {}, "
              "present_wait {}, swapchain_maintenance1 {})",
              instance_ext.size(), HasExtension(instance_ext, "VK_KHR_win32_surface"), device_ext.size(),
              HasExtension(device_ext, "VK_KHR_swapchain"), HasExtension(device_ext, "VK_KHR_present_id"),
              HasExtension(device_ext, "VK_KHR_present_wait"),
              HasExtension(device_ext, "VK_EXT_swapchain_maintenance1"));

  HMODULE vulkan = LoadLibraryW(L"vulkan-1.dll");
  auto get_instance_proc = vulkan ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(vulkan, "vkGetInstanceProcAddr"))
                                  : nullptr;
  if (!get_instance_proc) {
    REXLOG_WARN("Vulkan present: vulkan-1.dll / vkGetInstanceProcAddr not found");
    return false;
  }
  auto get_families = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
      get_instance_proc(instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
  auto create_surface =
      reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(get_instance_proc(instance, "vkCreateWin32SurfaceKHR"));
  auto destroy_surface = reinterpret_cast<PFN_vkDestroySurfaceKHR>(get_instance_proc(instance, "vkDestroySurfaceKHR"));
  auto surface_support = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
      get_instance_proc(instance, "vkGetPhysicalDeviceSurfaceSupportKHR"));
  auto surface_modes = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfacePresentModesKHR>(
      get_instance_proc(instance, "vkGetPhysicalDeviceSurfacePresentModesKHR"));
  if (!get_families || !create_surface || !destroy_surface || !surface_support || !surface_modes) {
    REXLOG_WARN("Vulkan present: instance functions missing");
    return false;
  }

  // Which queue the game's D3D12 direct queue is.
  VkQueue direct_vk = VK_NULL_HANDLE;
  UINT32 direct_family = ~0u;
  interop->GetVulkanQueueInfo(direct_queue, &direct_vk, &direct_family);

  VkWin32SurfaceCreateInfoKHR surface_info = {VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
  surface_info.hinstance = GetModuleHandleW(nullptr);
  surface_info.hwnd = hwnd;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  const VkResult surface_result = create_surface(instance, &surface_info, nullptr, &surface);
  uint32_t family_count = 0;
  get_families(physical_device, &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  get_families(physical_device, &family_count, families.data());
  for (uint32_t i = 0; i < family_count; ++i) {
    VkBool32 present = VK_FALSE;
    if (surface) surface_support(physical_device, i, surface, &present);
    REXLOG_INFO("Vulkan present: queue family {}: flags {:#x} x{}, present {}{}", i, uint32_t(families[i].queueFlags),
                families[i].queueCount, bool(present), i == direct_family ? " (the game's direct queue)" : "");
  }
  if (surface) {
    uint32_t mode_count = 0;
    surface_modes(physical_device, surface, &mode_count, nullptr);
    std::vector<VkPresentModeKHR> modes(mode_count);
    surface_modes(physical_device, surface, &mode_count, modes.data());
    std::string list;
    for (VkPresentModeKHR m : modes) list += " " + std::to_string(int(m));
    REXLOG_INFO("Vulkan present: surface ok, present modes:{} (0 immediate, 1 mailbox, 2 fifo, 3 fifo_relaxed)", list);
    destroy_surface(instance, surface, nullptr);
  } else {
    REXLOG_WARN("Vulkan present: vkCreateWin32SurfaceKHR failed ({})", int(surface_result));
  }
  return surface_result == VK_SUCCESS;
}

}  // namespace rex::ui::d3d12::vk_interop
