// Texture loading straight into the host texture, for 2D tiled single-level
// 32 bits per block textures stored as R8G8B8A8 (the render targets a game
// copies to memory and reads back every frame): untiles and endian-swaps the
// guest data like texture_load_32bpb, but writes the texture through an
// R8G8B8A8_UINT unordered access view (the bits unchanged) instead of to a
// buffer that then has to be copied into the texture.
//
// Constants: the layout of TextureCache::LoadConstants (texture_load_32bpb).
// One thread = 4 blocks along X, 8 x 16 threads = 32 x 16 blocks per group.
//
// Build: fxc /nologo /T cs_5_1 /E main /O3 /Vn texture_load_32bpb_to_texture_cs
//            /Fh bytecode/d3d12_5_1/texture_load_32bpb_to_texture_cs.h
//            texture_load_32bpb_to_texture.cs.hlsl

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

Buffer<uint4> xe_texture_load_source : register(t0);
RWTexture2D<uint4> xe_texture_load_dest : register(u0);

uint4 XeEndianSwap32(uint4 value, uint endian) {
  // k8in16 or k8in32.
  if (endian == 1u || endian == 2u) {
    value = ((value << 8u) & 0xFF00FF00u) | ((value >> 8u) & 0x00FF00FFu);
  }
  // k8in32 or k16in32.
  if (endian == 2u || endian == 3u) {
    value = (value << 16u) | (value >> 16u);
  }
  return value;
}

// Xenos 2D tiling (XGAddress2DTiledOffset, texture_util::GetTiledOffset2D),
// bytes from the start of the texture.
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
  // 4 blocks from a multiple of 4 along X are 16 contiguous bytes.
  uint guest_offset =
      xe_texture_load_guest_offset +
      XeTextureTiledOffset2D(block_index, xe_texture_load_guest_pitch_aligned, 2u);
  uint4 blocks = XeEndianSwap32(xe_texture_load_source[guest_offset >> 4u],
                                (xe_texture_load_is_tiled_3d_endian_scale >> 2u) & 3u);
  [unroll] for (uint i = 0u; i < 4u; ++i) {
    if (block_index.x + i < xe_texture_load_size_blocks.x) {
      uint block = blocks[i];
      xe_texture_load_dest[uint2(block_index.x + i, block_index.y)] =
          uint4(block & 0xFFu, (block >> 8u) & 0xFFu, (block >> 16u) & 0xFFu, block >> 24u);
    }
  }
}
