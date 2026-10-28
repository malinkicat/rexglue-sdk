// Direct resolve (gpu_direct_resolve) of a kD24S8 depth render target into the
// k_24_8 (R32_FLOAT) texture that samples the resolved depth: exactly what the
// depth dump (depth * 0xFFFFFF rounded to the nearest even integer), the fast
// 32bpp resolve copy and the depth texture load ((d + (d >> 23)) * 2^-24) give.
// The stencil goes to a separate R8_UINT texture, for writing the resolved
// words (depth << 8 | stencil) to memory later if something else reads them
// (texture_store_32bpb_tiled with XE_STORE_DEPTH24).
// See resolve_rt_to_texture_8888.cs.hlsl for the constants.
//
// XE_DIRECT_MSAA: a multisampled render target, sample xe_direct_samples & 0xFF.
// XE_DIRECT_FLOAT24: kD24FS8 - the host depth is the float24 value * 0.5, the
// dump's 20e4 bits are what the k_24_8 texture gets (truncated, or rounded to
// the nearest even with xe_direct_samples bit 16, like depth_float24_round).
// xe_direct_samples bit 17: a k_24_8_FLOAT texture - the 24 bits as 20e4, as
// the depth float texture load converts them (exactly invertible), rather than
// as unorm.
//
// Build: fxc /nologo /T cs_5_1 /E main /O3 /Vn resolve_depth_to_texture_cs
//            /Fh bytecode/d3d12_5_1/resolve_depth_to_texture_cs.h
//            resolve_depth_to_texture.cs.hlsl
//        and the variants: /D XE_DIRECT_MSAA (_ms), /D XE_DIRECT_FLOAT24 (_f24),
//        both (_f24_ms).

cbuffer xe_direct_resolve_constants : register(b0) {
  uint2 xe_direct_origin;
  uint2 xe_direct_size;
  uint xe_direct_swap_red_blue;
  uint2 xe_direct_fill;
  uint xe_direct_dest_offset;
  uint xe_direct_samples;
};

#ifdef XE_DIRECT_MSAA
Texture2DMS<float> xe_direct_depth : register(t0);
Texture2DMS<uint2> xe_direct_stencil : register(t1);
#else
Texture2D<float> xe_direct_depth : register(t0);
Texture2D<uint2> xe_direct_stencil : register(t1);
#endif
RWTexture2D<float> xe_direct_dest : register(u0);
RWTexture2D<uint> xe_direct_stencil_dest : register(u1);

// xenos::Float32To20e4.
uint XeFloat32To20e4(float f32, bool round_to_nearest_even) {
  if (!(f32 > 0.0f)) {
    return 0u;
  }
  uint bits = asuint(f32);
  if (bits >= 0x3FFFFFF8u) {
    return 0xFFFFFFu;
  }
  if (bits < 0x38800000u) {
    uint shift = min(113u - (bits >> 23u), 24u);
    bits = (0x800000u | (bits & 0x7FFFFFu)) >> shift;
  } else {
    bits += 0xC8000000u;
  }
  if (round_to_nearest_even) {
    bits += 3u + ((bits >> 3u) & 1u);
  }
  return (bits >> 3u) & 0xFFFFFFu;
}

// xenos::Float20e4To32.
float XeFloat20e4To32(uint f24) {
  uint mantissa = f24 & 0xFFFFFu;
  uint exponent = f24 >> 20u;
  if (exponent == 0u) {
    // Normalize (lzcnt(mantissa) - 11 = 20 - firstbithigh(mantissa)).
    uint shift = mantissa != 0u ? 20u - firstbithigh(mantissa) : 21u;
    exponent = 1u - shift;
    mantissa = (mantissa << shift) & 0xFFFFFu;
  }
  return f24 != 0u ? asfloat(((exponent + 112u) << 23u) | (mantissa << 3u)) : 0.0f;
}

[numthreads(8, 8, 1)]
void main(uint3 xe_thread_id : SV_DispatchThreadID) {
  if (any(xe_thread_id.xy >= xe_direct_size)) {
    return;
  }
  int2 source = int2(xe_direct_origin + max(xe_thread_id.xy, xe_direct_fill));
#ifdef XE_DIRECT_MSAA
  uint sample_index = xe_direct_samples & 0xFFu;
  float host_depth = xe_direct_depth.Load(source, sample_index);
  uint stencil = xe_direct_stencil.Load(source, sample_index).y;
#else
  float host_depth = xe_direct_depth.Load(int3(source, 0));
  uint stencil = xe_direct_stencil.Load(int3(source, 0)).y;
#endif
#ifdef XE_DIRECT_FLOAT24
  uint depth = XeFloat32To20e4(host_depth * 2.0f, (xe_direct_samples & 0x10000u) != 0u);
#else
  uint depth = uint(round(host_depth * 16777215.0f));
#endif
  uint2 dest = xe_thread_id.xy + uint2(xe_direct_dest_offset & 0xFFFFu, xe_direct_dest_offset >> 16u);
  xe_direct_dest[dest] = (xe_direct_samples & 0x20000u) != 0u
                             ? XeFloat20e4To32(depth)
                             : float(depth + (depth >> 23u)) * 5.9604644775390625e-8f;
  xe_direct_stencil_dest[dest] = stencil & 0xFFu;
}
