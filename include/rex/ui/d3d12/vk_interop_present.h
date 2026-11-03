#pragma once

// Presenting through Vulkan under vkd3d-proton (--d3d12_present_vulkan); see
// src/ui/d3d12/vk_interop_present.cpp.

#include <memory>

#include <rex/ui/d3d12/d3d12_api.h>

namespace rex::ui::d3d12::vk_interop {

// Logs what vkd3d's Vulkan device offers for presenting to `hwnd` ourselves
// (extensions, queue families that can present, present modes). False when
// not running under vkd3d-proton or a surface can't be made.
bool Probe(ID3D12Device* device, ID3D12CommandQueue* direct_queue, HWND hwnd);

// Our own FIFO Vulkan swap chain on vkd3d's device for `hwnd`: Present copies
// a painted D3D12 image (B8G8R8A8, in the PRESENT state) into it on a queue of
// its own (a compute family when one can present), then presents with a
// present id; the flip callback runs when each frame is on the display.
class Presenter {
 public:
  Presenter();
  ~Presenter();
  bool Initialize(ID3D12Device* device, HWND hwnd);  // false: not under vkd3d, or unusable
  bool Active() const { return state_ != nullptr; }
  bool Present(ID3D12Resource* source, ID3D12CommandQueue* paint_queue);
  void SetFlipCallback(void (*callback)());
  void Shutdown();

 private:
  struct State;
  bool CreateSwapchain(uint32_t width, uint32_t height);
  void DestroySwapchain();
  std::unique_ptr<State> state_;
  void (*on_flip_)() = nullptr;
};

}  // namespace rex::ui::d3d12::vk_interop
