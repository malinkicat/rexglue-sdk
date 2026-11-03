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

#include <string>
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
