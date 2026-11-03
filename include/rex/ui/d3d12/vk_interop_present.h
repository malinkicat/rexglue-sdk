#pragma once

// Presenting through Vulkan under vkd3d-proton (--d3d12_present_vulkan); see
// src/ui/d3d12/vk_interop_present.cpp.

#include <rex/ui/d3d12/d3d12_api.h>

namespace rex::ui::d3d12::vk_interop {

// Logs what vkd3d's Vulkan device offers for presenting to `hwnd` ourselves
// (extensions, queue families that can present, present modes). False when
// not running under vkd3d-proton or a surface can't be made.
bool Probe(ID3D12Device* device, ID3D12CommandQueue* direct_queue, HWND hwnd);

}  // namespace rex::ui::d3d12::vk_interop
