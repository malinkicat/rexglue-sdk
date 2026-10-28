// The reverse of texture_load_32bpb_to_texture: writes an R8G8B8A8 texture
// (read as R8G8B8A8_UINT, the bits unchanged) back to tiled guest memory with
// the endian swap. Gives the guest memory the data of a direct resolve
// (gpu_direct_resolve) when something other than that texture needs it.
//
// Constants: the layout of TextureCache::LoadConstants.
// One thread = 4 blocks along X (16 contiguous bytes), 8 x 16 threads per
// group. The width must be a multiple of 4.
//
// Build: fxc /nologo /T cs_5_1 /E main /O3 /Vn texture_store_32bpb_tiled_cs
//            /Fh bytecode/d3d12_5_1/texture_store_32bpb_tiled_cs.h
//            texture_store_32bpb_tiled.cs.hlsl
//        with /D XE_STORE_DEPTH24 /Vn ..._depth24_... for k_24_8,
//        and with /D XE_STORE_FLOAT32 /Vn texture_store_f32_tiled_cs
//            /Fh bytecode/d3d12_5_1/texture_store_f32_tiled_cs.h

cbuffer push_consts_xe : register(b0) {
  uint xe_texture_load_is_tiled_3d_endian_scale;
  uint xe_texture_load_guest_offset;
  // In blocks, aligned to 32.
  uint xe_texture_load_guest_pitch_aligned;
  uint xe_texture_load_guest_z_stride_block_rows_aligned;
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
  uint2 block_index = uint2(xe_thread_id.x << 2u, xe_thread_id.y);
  if (any(block_index >= xe_texture_load_size_blocks.xy)) {
    return;
  }
  uint4 blocks;
  [unroll] for (uint i = 0u; i < 4u; ++i) {
#if defined(XE_STORE_DEPTH24)
    {
      int3 xe_texel = int3(block_index.x + i + (xe_texture_load_host_offset & 0xFFFFu), block_index.y + (xe_texture_load_host_offset >> 16u), 0);
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
    blocks[i] = asuint(xe_texture_store_source.Load(int3(block_index.x + i + (xe_texture_load_host_offset & 0xFFFFu), block_index.y + (xe_texture_load_host_offset >> 16u), 0)));
#else
    uint4 texel = xe_texture_store_source.Load(int3(block_index.x + i + (xe_texture_load_host_offset & 0xFFFFu), block_index.y + (xe_texture_load_host_offset >> 16u), 0));
    blocks[i] = texel.x | (texel.y << 8u) | (texel.z << 16u) | (texel.w << 24u);
#endif
  }
  uint guest_offset =
      xe_texture_load_guest_offset +
      XeTextureTiledOffset2D(block_index, xe_texture_load_guest_pitch_aligned, 2u);
  xe_texture_store_dest[guest_offset >> 4u] =
      XeEndianSwap32(blocks, (xe_texture_load_is_tiled_3d_endian_scale >> 2u) & 3u);
}
