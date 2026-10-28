// Direct resolve (gpu_direct_resolve): a k_8_8_8_8 host render target copied
// straight into the R8G8B8A8 host texture the game samples the resolved data
// with, instead of render target -> EDRAM buffer (dump) -> guest memory (resolve
// copy) -> texture (load). The texture gets exactly what that chain gives:
// the render target value packed like the dump (x * 255 + 0.5 to integer),
// red and blue swapped if the resolve swaps them (fast 32bpp resolve), and the
// endian swaps of the resolve and of the texture load cancelling each other
// (only used when they're the same).
//
// Constants (root constants of the texture load root signature):
//   xe_direct_origin - the render target pixel of the texture's (0, 0)
//   xe_direct_size - texture size in texels
//   xe_direct_swap_red_blue
//   xe_direct_fill - with resolution scaling, the left columns / top rows of
//     the resolve area taking the first surely covered column / row (the
//     resolve_resolution_scale_fill_half_pixel_offset hack of the resolve
//     copies, draw_resolution_scale >> 1 when it's used, otherwise 0)
//
//   xe_direct_dest_offset - texel of the texture written first (x | y << 16),
//     for resolves of a part of the texture (predicated tiling)
//   xe_direct_samples - with XE_DIRECT_MSAA, the host samples averaged
//     (a | b << 8, the same twice for one sample)
//
// Build: fxc /nologo /T cs_5_1 /E main /O3 /Vn resolve_rt_to_texture_8888_cs
//            /Fh bytecode/d3d12_5_1/resolve_rt_to_texture_8888_cs.h
//            resolve_rt_to_texture_8888.cs.hlsl
//        and with /D XE_DIRECT_MSAA /Vn resolve_rt_to_texture_8888_ms_cs
//            /Fh bytecode/d3d12_5_1/resolve_rt_to_texture_8888_ms_cs.h

cbuffer xe_direct_resolve_constants : register(b0) {
  uint2 xe_direct_origin;
  uint2 xe_direct_size;
  uint xe_direct_swap_red_blue;
  uint2 xe_direct_fill;
  uint xe_direct_dest_offset;
  uint xe_direct_samples;
};

#ifdef XE_DIRECT_MSAA
Texture2DMS<float4> xe_direct_source : register(t0);
#else
Texture2D<float4> xe_direct_source : register(t0);
#endif
RWTexture2D<uint4> xe_direct_dest : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 xe_thread_id : SV_DispatchThreadID) {
  if (any(xe_thread_id.xy >= xe_direct_size)) {
    return;
  }
  int2 source = int2(xe_direct_origin + max(xe_thread_id.xy, xe_direct_fill));
#ifdef XE_DIRECT_MSAA
  uint sample_a = xe_direct_samples & 0xFFu, sample_b = (xe_direct_samples >> 8u) & 0xFFu;
  float4 color = xe_direct_source.Load(source, sample_a);
  if (sample_b != sample_a) {
    color = (color + xe_direct_source.Load(source, sample_b)) * 0.5f;
  }
#else
  float4 color = xe_direct_source.Load(int3(source, 0));
#endif
  uint4 bytes = uint4(color * 255.0f + 0.5f);
  if (xe_direct_swap_red_blue != 0u) {
    bytes.xz = bytes.zx;
  }
  xe_direct_dest[xe_thread_id.xy + uint2(xe_direct_dest_offset & 0xFFFFu,
                                         xe_direct_dest_offset >> 16u)] = bytes;
}
