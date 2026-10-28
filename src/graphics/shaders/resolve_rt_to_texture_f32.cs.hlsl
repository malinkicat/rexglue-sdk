// Direct resolve (gpu_direct_resolve) of a k_32_FLOAT render target into the
// k_32_FLOAT (R32_FLOAT) texture that samples the resolved data: the render
// target's bits (read through its R32_UINT view) as they are, like the dump,
// the fast 32bpp resolve copy and the texture load do (the endian swaps of the
// resolve and of the load cancelling each other - only used when they're the
// same). See resolve_rt_to_texture_8888.cs.hlsl for the constants.
//
// Build: fxc /nologo /T cs_5_1 /E main /O3 /Vn resolve_rt_to_texture_f32_cs
//            /Fh bytecode/d3d12_5_1/resolve_rt_to_texture_f32_cs.h
//            resolve_rt_to_texture_f32.cs.hlsl

cbuffer xe_direct_resolve_constants : register(b0) {
  uint2 xe_direct_origin;
  uint2 xe_direct_size;
  uint xe_direct_swap_red_blue;
  uint2 xe_direct_fill;
};

Texture2D<uint> xe_direct_source : register(t0);
RWTexture2D<float> xe_direct_dest : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 xe_thread_id : SV_DispatchThreadID) {
  if (any(xe_thread_id.xy >= xe_direct_size)) {
    return;
  }
  xe_direct_dest[xe_thread_id.xy] = asfloat(xe_direct_source.Load(
      int3(xe_direct_origin + max(xe_thread_id.xy, xe_direct_fill), 0)));
}
