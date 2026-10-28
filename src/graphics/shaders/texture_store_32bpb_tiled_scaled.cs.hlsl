// texture_store_32bpb_tiled with resolution scaling: writes an R8G8B8A8
// texture of a resolution-scaled resolve (read as R8G8B8A8_UINT, the bits
// unchanged) to the scaled resolve memory in the layout texture_load_32bpb_scaled
// reads - every 16 guest bytes (4 blocks along X of the tiled guest layout)
// become scale.x * scale.y runs of 16 bytes, run (sub.x * scale.y + sub.y)
// holding the 4 host texels of host column group sub.x and host row sub.y.
// Gives the scaled resolve memory the data of a direct resolve
// (gpu_direct_resolve) when something other than that texture needs it.
//
// Constants: the layout of TextureCache::LoadConstants, the guest offset in
// bytes of the scaled resolve range bound as the destination.
// One thread = 4 host texels along X (16 contiguous bytes), 8 x 16 threads per
// group. The guest width must be a multiple of 4.
//
// Build: fxc /nologo /T cs_5_1 /E main /O3 /Vn texture_store_32bpb_tiled_scaled_cs
//            /Fh bytecode/d3d12_5_1/texture_store_32bpb_tiled_scaled_cs.h
//            texture_store_32bpb_tiled_scaled.cs.hlsl
//        with /D XE_STORE_DEPTH24 /Vn ..._depth24_... for k_24_8,
//        and with /D XE_STORE_FLOAT32 /Vn texture_store_f32_tiled_scaled_cs
//            /Fh bytecode/d3d12_5_1/texture_store_f32_tiled_scaled_cs.h

cbuffer push_consts_xe : register(b0) {
  uint xe_texture_load_is_tiled_3d_endian_scale;
  uint xe_texture_load_guest_offset;
  // In blocks, aligned to 32.
  uint xe_texture_load_guest_pitch_aligned;
  uint xe_texture_load_guest_z_stride_block_rows_aligned;
  // Host texels.
  uint3 xe_texture_load_size_blocks;
  uint xe_texture_load_host_offset;
  uint xe_texture_load_host_pitch;
  uint xe_texture_load_height_texels;
};

// XE_STORE_FLOAT32: an R32_FLOAT texture (k_32_FLOAT), its bits as they are.
// XE_STORE_DEPTH24: a k_24_8 texture (R32_FLOAT, the 24-bit depth as the
// texture load converts it - exactly invertible) and the stencil of the resolve
// kept next to it (R8_UINT), packed like the depth dump: depth << 8 | stencil.
#if defined(XE_STORE_DEPTH24)
Texture2D<float> xe_texture_store_source : register(t0);
Texture2D<uint> xe_texture_store_stencil : register(t1);
#elif defined(XE_STORE_FLOAT32)
Texture2D<float> xe_texture_store_source : register(t0);
#else
Texture2D<uint4> xe_texture_store_source : register(t0);
#endif
RWBuffer<uint4> xe_texture_store_dest : register(u0);

// xenos::Float32To20e4, truncating (the values are exact 20e4 here).
uint XeFloat32To20e4(float f32) {
  if (!(f32 > 0.0f)) {
    return 0u;
  }
  uint bits = asuint(f32);
  if (bits >= 0x3FFFFFF8u) {
    return 0xFFFFFFu;
  }
  if (bits < 0x38800000u) {
    bits = (0x800000u | (bits & 0x7FFFFFu)) >> min(113u - (bits >> 23u), 24u);
  } else {
    bits += 0xC8000000u;
  }
  return (bits >> 3u) & 0xFFFFFFu;
}

uint4 XeEndianSwap32(uint4 value, uint endian) {
  if (endian == 1u || endian == 2u) {
    value = ((value << 8u) & 0xFF00FF00u) | ((value >> 8u) & 0x00FF00FFu);
  }
  if (endian == 2u || endian == 3u) {
    value = (value << 16u) | (value >> 16u);
  }
  return value;
}

uint XeTextureTiledOffset2D(uint2 p, uint pitch_aligned, uint bpb_log2) {
  uint macro_offset = ((p.x >> 5u) + (p.y >> 5u) * (pitch_aligned >> 5u)) << (bpb_log2 + 7u);
  uint micro_offset = ((p.x & 7u) + ((p.y & 0xEu) << 2u)) << bpb_log2;
  uint offset = macro_offset + ((micro_offset & ~0xFu) << 1u) + (micro_offset & 0xFu) +
                ((p.y & 1u) << 4u);
  return ((offset & ~0x1FFu) << 3u) + ((p.y & 16u) << 7u) + ((offset & 0x1C0u) << 2u) +
         (((((p.y & 8u) >> 2u) + (p.x >> 3u)) & 3u) << 6u) + (offset & 0x3Fu);
}

[numthreads(8, 16, 1)]
void main(uint3 xe_thread_id : SV_DispatchThreadID) {
  uint2 host_block_index = uint2(xe_thread_id.x << 2u, xe_thread_id.y);
  if (any(host_block_index >= xe_texture_load_size_blocks.xy)) {
    return;
  }
  uint2 scale = (xe_texture_load_is_tiled_3d_endian_scale >> uint2(4u, 7u)) & 7u;
  // Host group of 4 texels along X, and the guest group of 4 blocks with it.
  uint2 host_group = uint2(xe_thread_id.x, xe_thread_id.y);
  uint2 guest_group = host_group / scale;
  uint2 sub = host_group - guest_group * scale;
  uint4 blocks;
  [unroll] for (uint i = 0u; i < 4u; ++i) {
#if defined(XE_STORE_DEPTH24)
    {
      int3 xe_texel = int3(host_block_index.x + i + (xe_texture_load_host_offset & 0xFFFFu), host_block_index.y + (xe_texture_load_host_offset >> 16u), 0);
      // (d + (d >> 23)) * 2^-24 back to d, or k_24_8_FLOAT (bit 10) 20e4 -> 32
      // back to 20e4: exact.
      float xe_depth_value = xe_texture_store_source.Load(xe_texel);
      uint xe_depth_scaled = uint(xe_depth_value * 16777216.0f);
      uint xe_depth = (xe_texture_load_is_tiled_3d_endian_scale & (1u << 10u)) != 0u
                          ? XeFloat32To20e4(xe_depth_value)
                          : xe_depth_scaled - uint(xe_depth_scaled > 0x800000u);
      blocks[i] = (xe_depth << 8u) | (xe_texture_store_stencil.Load(xe_texel) & 0xFFu);
    }
#elif defined(XE_STORE_FLOAT32)
    blocks[i] =
        asuint(xe_texture_store_source.Load(int3(host_block_index.x + i + (xe_texture_load_host_offset & 0xFFFFu), host_block_index.y + (xe_texture_load_host_offset >> 16u), 0)));
#else
    uint4 texel =
        xe_texture_store_source.Load(int3(host_block_index.x + i + (xe_texture_load_host_offset & 0xFFFFu), host_block_index.y + (xe_texture_load_host_offset >> 16u), 0));
    blocks[i] = texel.x | (texel.y << 8u) | (texel.z << 16u) | (texel.w << 24u);
#endif
  }
  uint guest_offset =
      XeTextureTiledOffset2D(uint2(guest_group.x << 2u, guest_group.y),
                             xe_texture_load_guest_pitch_aligned, 2u) *
          (scale.x * scale.y) +
      ((sub.x * scale.y + sub.y) << 4u) + xe_texture_load_guest_offset;
  xe_texture_store_dest[guest_offset >> 4u] =
      XeEndianSwap32(blocks, (xe_texture_load_is_tiled_3d_endian_scale >> 2u) & 3u);
}
