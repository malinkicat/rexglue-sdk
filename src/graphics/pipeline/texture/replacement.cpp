/**
 * @file        graphics/pipeline/texture/replacement.cpp
 *
 * @brief       Texture dump and replacement pipeline implementation.
 *
 */
#include <rex/graphics/pipeline/texture/replacement.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <system_error>
#include <thread>

#include <rex/graphics/pipeline/texture/conversion.h>
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/perf/frame_rate.h>
#include <rex/thread.h>

#include "replacement_alpha.h"

#ifndef XXH_INLINE_ALL
#define XXH_INLINE_ALL
#endif
#include <xxhash.h>

// stb_image — PNG/JPEG/etc. loader (implementation compiled exactly once here)
// STB_IMAGE_STATIC gives all symbols internal linkage, preventing duplicate
// symbol conflicts if the consuming application also vendors stb_image.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO  // we pass memory buffers directly
#include <stb_image.h>

namespace rex::graphics {

// ---------------------------------------------------------------------------
// DDS constants and structures
// ---------------------------------------------------------------------------

static constexpr uint32_t kDdsMagic = 0x20534444u;  // "DDS "
static constexpr uint32_t kDdsdCaps = 0x00000001u;
static constexpr uint32_t kDdsdHeight = 0x00000002u;
static constexpr uint32_t kDdsdWidth = 0x00000004u;
static constexpr uint32_t kDdsdPitch = 0x00000008u;
static constexpr uint32_t kDdsdLinearSize = 0x00080000u;
static constexpr uint32_t kDdsdPixelFormat = 0x00001000u;
static constexpr uint32_t kDdsPfRgb = 0x00000040u;
static constexpr uint32_t kDdsPfAlphaPixels = 0x00000001u;
static constexpr uint32_t kDdsPfFourCC = 0x00000004u;
static constexpr uint32_t kDdsCapsTexture = 0x00001000u;

static constexpr uint32_t kFourCC_DXT1 = 0x31545844u;  // "DXT1"
static constexpr uint32_t kFourCC_DXT3 = 0x33545844u;  // "DXT3"
static constexpr uint32_t kFourCC_DXT5 = 0x35545844u;  // "DXT5"
static constexpr uint32_t kFourCC_DX10 = 0x30315844u;  // "DX10"
static constexpr uint32_t kFourCC_ATI1 = 0x31495441u;  // "ATI1" (BC4 / DXN red)
static constexpr uint32_t kFourCC_ATI2 = 0x32495441u;  // "ATI2" (BC5 / DXN rg)

static constexpr uint32_t kDxgiFormatR8G8B8A8UNorm = 28;
static constexpr uint32_t kDxgiFormatR8G8B8A8UNormSRGB = 29;
static constexpr uint32_t kDxgiFormatBC1UNorm = 71;
static constexpr uint32_t kDxgiFormatBC1UNormSRGB = 72;
static constexpr uint32_t kDxgiFormatBC2UNorm = 74;
static constexpr uint32_t kDxgiFormatBC2UNormSRGB = 75;
static constexpr uint32_t kDxgiFormatBC3UNorm = 77;
static constexpr uint32_t kDxgiFormatBC3UNormSRGB = 78;
static constexpr uint32_t kDxgiFormatB8G8R8A8UNorm = 87;
static constexpr uint32_t kDxgiFormatB8G8R8X8UNorm = 88;
static constexpr uint32_t kDxgiFormatB8G8R8A8UNormSRGB = 91;
static constexpr uint32_t kDxgiFormatB8G8R8X8UNormSRGB = 93;
static constexpr uint32_t kDxgiFormatBC7UNorm = 98;
static constexpr uint32_t kDxgiFormatBC7UNormSRGB = 99;

#pragma pack(push, 1)
struct DdsPixelFormat {
  uint32_t size = 32;
  uint32_t flags = 0;
  uint32_t four_cc = 0;
  uint32_t rgb_bit_count = 0;
  uint32_t r_bit_mask = 0;
  uint32_t g_bit_mask = 0;
  uint32_t b_bit_mask = 0;
  uint32_t a_bit_mask = 0;
};
struct DdsHeader {
  uint32_t magic = kDdsMagic;
  uint32_t size = 124;
  uint32_t flags = 0;
  uint32_t height = 0;
  uint32_t width = 0;
  uint32_t pitch_or_linear = 0;
  uint32_t depth = 0;
  uint32_t mip_map_count = 1;
  uint32_t reserved1[11] = {};
  DdsPixelFormat ddspf;
  uint32_t caps = kDdsCapsTexture;
  uint32_t caps2 = 0;
  uint32_t caps3 = 0;
  uint32_t caps4 = 0;
  uint32_t reserved2 = 0;
};
struct DdsHeaderDX10 {
  uint32_t dxgi_format = 0;
  uint32_t resource_dimension = 0;
  uint32_t misc_flag = 0;
  uint32_t array_size = 0;
  uint32_t misc_flags2 = 0;
};
#pragma pack(pop)
static_assert(sizeof(DdsHeader) == 128);
static_assert(sizeof(DdsHeaderDX10) == 20);

// ---------------------------------------------------------------------------
// Internal helpers: tiled address decode (mirrors conversion.cpp)
// ---------------------------------------------------------------------------
static uint32_t TiledOffset2DRow(uint32_t y, uint32_t width, uint32_t log2_bpp) {
  uint32_t macro = ((y / 32) * (width / 32)) << (log2_bpp + 7);
  uint32_t micro = ((y & 6) << 2) << log2_bpp;
  return macro + ((micro & ~0xFu) << 1) + (micro & 0xFu) + ((y & 8) << (3 + log2_bpp)) +
         ((y & 1) << 4);
}

static uint32_t TiledOffset2DColumn(uint32_t x, uint32_t y, uint32_t log2_bpp,
                                    uint32_t base_offset) {
  uint32_t macro = (x / 32) << (log2_bpp + 7);
  uint32_t micro = (x & 7) << log2_bpp;
  uint32_t offset = base_offset + (macro + ((micro & ~0xFu) << 1) + (micro & 0xFu));
  return ((offset & ~0x1FFu) << 3) + ((offset & 0x1C0u) << 2) + (offset & 0x3Fu) + ((y & 16) << 7) +
         (((((y & 8) >> 2) + (x >> 3)) & 3) << 6);
}

// Untile a 2-D block-based texture into a linear output buffer.
//   src              : tiled guest bytes
//   dst              : output buffer (row-major, no padding)
//   width_blocks     : visible width in blocks
//   height_blocks    : visible height in blocks
//   pitch_blocks     : row pitch in blocks (aligned to 32 for tiled)
//   bytes_per_block  : bytes per compressed block or texel
static void UntileBlocks(const uint8_t* src, uint8_t* dst, uint32_t width_blocks,
                         uint32_t height_blocks, uint32_t pitch_blocks, uint32_t bytes_per_block) {
  // log2(bytes_per_block / 4) + extra bias matching Xenia's formula
  const uint32_t log2_bpp =
      (bytes_per_block / 4) + ((bytes_per_block / 2) >> (bytes_per_block / 4));

  const uint32_t out_row_bytes = width_blocks * bytes_per_block;

  for (uint32_t y = 0; y < height_blocks; ++y) {
    const uint32_t row_offset = TiledOffset2DRow(y, pitch_blocks, log2_bpp);
    for (uint32_t x = 0; x < width_blocks; ++x) {
      uint32_t src_offset = TiledOffset2DColumn(x, y, log2_bpp, row_offset);
      src_offset >>= log2_bpp;
      std::memcpy(dst + y * out_row_bytes + x * bytes_per_block, src + src_offset * bytes_per_block,
                  bytes_per_block);
    }
  }
}

// ---------------------------------------------------------------------------
// RGBA8 expansion helpers
// ---------------------------------------------------------------------------
// Each returns an RGBA8 pixel from a pointer into the (already endian-swapped)
// source data.

static uint32_t Expand5To8(uint32_t v) {
  return (v << 3) | (v >> 2);
}
static uint32_t Expand6To8(uint32_t v) {
  return (v << 2) | (v >> 4);
}

// Convert one texel from the given format (already endian-corrected) to RGBA8.
// Returns false for formats that need the BC path (compressed blocks).
static bool TexelToRGBA8(const uint8_t* src, xenos::TextureFormat fmt, uint8_t out[4]) {
  using F = xenos::TextureFormat;
  switch (fmt) {
    case F::k_8_8_8_8:
    case F::k_8_8_8_8_A:
    case F::k_8_8_8_8_GAMMA_EDRAM:
      out[0] = src[0];
      out[1] = src[1];
      out[2] = src[2];
      out[3] = src[3];
      return true;
    case F::k_8:
    case F::k_8_A:
    case F::k_8_B:
      out[0] = out[1] = out[2] = src[0];
      out[3] = 255;
      return true;
    case F::k_8_8:
      out[0] = src[0];
      out[1] = src[1];
      out[2] = 0;
      out[3] = 255;
      return true;
    case F::k_5_6_5: {
      uint16_t v;
      std::memcpy(&v, src, 2);
      out[0] = static_cast<uint8_t>(Expand5To8((v >> 11) & 0x1F));
      out[1] = static_cast<uint8_t>(Expand6To8((v >> 5) & 0x3F));
      out[2] = static_cast<uint8_t>(Expand5To8(v & 0x1F));
      out[3] = 255;
      return true;
    }
    case F::k_1_5_5_5: {
      uint16_t v;
      std::memcpy(&v, src, 2);
      out[0] = static_cast<uint8_t>(Expand5To8((v >> 10) & 0x1F));
      out[1] = static_cast<uint8_t>(Expand5To8((v >> 5) & 0x1F));
      out[2] = static_cast<uint8_t>(Expand5To8(v & 0x1F));
      out[3] = static_cast<uint8_t>(((v >> 15) & 1) ? 255 : 0);
      return true;
    }
    case F::k_4_4_4_4: {
      uint16_t v;
      std::memcpy(&v, src, 2);
      out[0] = static_cast<uint8_t>(((v >> 12) & 0xF) * 17);
      out[1] = static_cast<uint8_t>(((v >> 8) & 0xF) * 17);
      out[2] = static_cast<uint8_t>(((v >> 4) & 0xF) * 17);
      out[3] = static_cast<uint8_t>((v & 0xF) * 17);
      return true;
    }
    case F::k_2_10_10_10: {
      uint32_t v;
      std::memcpy(&v, src, 4);
      out[0] = static_cast<uint8_t>((v >> 22) & 0xFF);
      out[1] = static_cast<uint8_t>((v >> 12) & 0xFF);
      out[2] = static_cast<uint8_t>((v >> 2) & 0xFF);
      out[3] = static_cast<uint8_t>(((v & 3) * 85));
      return true;
    }
    default:
      // Unsupported / compressed — caller should use BC path or skip
      out[0] = out[1] = out[2] = out[3] = 0;
      return false;
  }
}

static void DecodeRGB565(uint16_t value, uint8_t out[4]) {
  out[0] = static_cast<uint8_t>(Expand5To8((value >> 11) & 0x1F));
  out[1] = static_cast<uint8_t>(Expand6To8((value >> 5) & 0x3F));
  out[2] = static_cast<uint8_t>(Expand5To8(value & 0x1F));
  out[3] = 255;
}

static void DecodeBCColorBlock(const uint8_t* block, bool allow_1bit_alpha, uint8_t colors[16][4]) {
  uint16_t color0 = 0;
  uint16_t color1 = 0;
  std::memcpy(&color0, block, sizeof(color0));
  std::memcpy(&color1, block + 2, sizeof(color1));

  uint8_t palette[4][4]{};
  DecodeRGB565(color0, palette[0]);
  DecodeRGB565(color1, palette[1]);
  if (color0 > color1 || !allow_1bit_alpha) {
    for (uint32_t c = 0; c < 3; ++c) {
      palette[2][c] = static_cast<uint8_t>((2 * palette[0][c] + palette[1][c]) / 3);
      palette[3][c] = static_cast<uint8_t>((palette[0][c] + 2 * palette[1][c]) / 3);
    }
    palette[2][3] = 255;
    palette[3][3] = 255;
  } else {
    for (uint32_t c = 0; c < 3; ++c) {
      palette[2][c] = static_cast<uint8_t>((palette[0][c] + palette[1][c]) / 2);
      palette[3][c] = 0;
    }
    palette[2][3] = 255;
    palette[3][3] = 0;
  }

  uint32_t indices = 0;
  std::memcpy(&indices, block + 4, sizeof(indices));
  for (uint32_t i = 0; i < 16; ++i) {
    std::memcpy(colors[i], palette[indices & 0x3], 4);
    indices >>= 2;
  }
}

static void DecodeBC2AlphaBlock(const uint8_t* block, uint8_t alpha[16]) {
  uint64_t alpha_bits = 0;
  std::memcpy(&alpha_bits, block, sizeof(alpha_bits));
  for (uint32_t i = 0; i < 16; ++i) {
    alpha[i] = static_cast<uint8_t>((alpha_bits & 0xFu) * 17);
    alpha_bits >>= 4;
  }
}

static void DecodeBC3AlphaBlock(const uint8_t* block, uint8_t alpha[16]) {
  uint8_t palette[8]{};
  palette[0] = block[0];
  palette[1] = block[1];
  if (palette[0] > palette[1]) {
    palette[2] = static_cast<uint8_t>((6 * palette[0] + palette[1]) / 7);
    palette[3] = static_cast<uint8_t>((5 * palette[0] + 2 * palette[1]) / 7);
    palette[4] = static_cast<uint8_t>((4 * palette[0] + 3 * palette[1]) / 7);
    palette[5] = static_cast<uint8_t>((3 * palette[0] + 4 * palette[1]) / 7);
    palette[6] = static_cast<uint8_t>((2 * palette[0] + 5 * palette[1]) / 7);
    palette[7] = static_cast<uint8_t>((palette[0] + 6 * palette[1]) / 7);
  } else {
    palette[2] = static_cast<uint8_t>((4 * palette[0] + palette[1]) / 5);
    palette[3] = static_cast<uint8_t>((3 * palette[0] + 2 * palette[1]) / 5);
    palette[4] = static_cast<uint8_t>((2 * palette[0] + 3 * palette[1]) / 5);
    palette[5] = static_cast<uint8_t>((palette[0] + 4 * palette[1]) / 5);
    palette[6] = 0;
    palette[7] = 255;
  }

  uint64_t indices = 0;
  for (uint32_t i = 0; i < 6; ++i) {
    indices |= uint64_t(block[2 + i]) << (i * 8);
  }
  for (uint32_t i = 0; i < 16; ++i) {
    alpha[i] = palette[indices & 0x7];
    indices >>= 3;
  }
}

// (replacement_alpha.h)
using replacement_image::ParallelRows;

enum class DdsDecodeFormat {
  kRGBA8,
  kBGRA8,
  kBGRX8,
  kBC1,
  kBC2,
  kBC3,
  kBC7,
};

// BC7 block decoder (D3D11 functional spec, 3.1 "BC7 Format").

static constexpr uint8_t kBc7Partition2[64][16] = {
    {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1}, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},
    {0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1}, {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1}, {0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1},
    {0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1}, {0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1},
    {0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
    {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1},
    {0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1}, {0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0}, {0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0},
    {0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0}, {0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1},
    {0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0},
    {0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0}, {0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0},
    {0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
    {0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0}, {0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0},
    {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}, {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1},
    {0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0}, {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0},
    {0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0}, {0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0},
    {0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1}, {0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1},
    {0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0}, {0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0},
    {0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0}, {0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0},
    {0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0}, {0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1},
    {0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1}, {0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0},
    {0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1}, {0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1},
    {0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0}, {0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0},
    {0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1}, {0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1},
    {0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1},
    {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1}, {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0}, {0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1},
};

static constexpr uint8_t kBc7Partition3[64][16] = {
    {0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 1, 2, 2, 2, 2}, {0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1},
    {0, 0, 0, 0, 2, 0, 0, 1, 2, 2, 1, 1, 2, 2, 1, 1}, {0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 0, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2}, {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 2, 2},
    {0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}, {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2},
    {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2}, {0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2},
    {0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2}, {0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2},
    {0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2}, {0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0, 2, 2, 2, 0},
    {0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2}, {0, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0},
    {0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2}, {0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1},
    {0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2}, {0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1},
    {0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2}, {0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 1, 0, 2, 2, 1, 0},
    {0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0}, {0, 0, 1, 2, 0, 0, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2},
    {0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0}, {0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1},
    {0, 0, 2, 2, 1, 1, 0, 2, 1, 1, 0, 2, 0, 0, 2, 2}, {0, 1, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 2, 2, 2, 2},
    {0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1}, {0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1},
    {0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 2, 2, 2}, {0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1, 1},
    {0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 2, 2, 2}, {0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0},
    {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0}, {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0},
    {0, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0}, {0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1},
    {0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1}, {0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2},
    {0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1}, {0, 0, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2},
    {0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 1}, {0, 2, 2, 0, 1, 2, 2, 1, 0, 2, 2, 0, 1, 2, 2, 1},
    {0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0, 1}, {0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1},
    {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2}, {0, 2, 2, 2, 0, 1, 1, 1, 0, 2, 2, 2, 0, 1, 1, 1},
    {0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 2, 1, 1, 1, 2}, {0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2},
    {0, 2, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2}, {0, 0, 0, 2, 1, 1, 1, 2, 1, 1, 1, 2, 0, 0, 0, 2},
    {0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2}, {0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2},
    {0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2}, {0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2},
    {0, 0, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2},
    {0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1}, {0, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 1, 2, 2, 2},
    {0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, {0, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0},
};

// Pixel whose index has one bit less: subset 1 of 2, subsets 1 and 2 of 3
// (subset 0 always uses pixel 0).
static constexpr uint8_t kBc7Anchor2[64] = {
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2,  8,  2,  2, 8,
    8,  15, 2,  8,  2,  2,  8,  8,  2,  2,  15, 15, 6,  8,  2,  8,  15, 15, 2,  8,  2, 2,
    2,  15, 15, 6,  6,  2,  6,  8,  15, 15, 2,  2,  15, 15, 15, 15, 15, 2,  2,  15};
static constexpr uint8_t kBc7Anchor3a[64] = {
    3,  3,  15, 15, 8, 3,  15, 15, 8,  8,  6,  6,  6,  5,  3,  3,  3,  3,  8,  15, 3, 3,
    6,  10, 5,  8,  8, 6,  8,  5,  15, 15, 8,  15, 3,  5,  6,  10, 8,  15, 15, 3,  15, 5,
    15, 15, 15, 15, 3, 15, 5,  5,  5,  8,  5,  10, 5,  10, 8,  13, 15, 12, 3,  3};
static constexpr uint8_t kBc7Anchor3b[64] = {
    15, 8,  8,  3,  15, 15, 3,  8,  15, 15, 15, 15, 15, 15, 15, 8,  15, 8,  15, 3,  15, 8,
    15, 8,  3,  15, 6,  10, 15, 15, 10, 8,  15, 3,  15, 10, 10, 8,  9,  10, 6,  15, 8,  15,
    3,  6,  6,  8,  15, 3,  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3,  15, 15, 8};

static constexpr uint8_t kBc7Weights2[4] = {0, 21, 43, 64};
static constexpr uint8_t kBc7Weights3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
static constexpr uint8_t kBc7Weights4[16] = {0,  4,  9,  13, 17, 21, 26, 30,
                                            34, 38, 43, 47, 51, 55, 60, 64};

struct Bc7Bits {
  uint64_t lo, hi;
  uint32_t pos = 0;
  uint32_t Read(uint32_t count) {
    uint32_t value;
    if (pos >= 64) {
      value = uint32_t(hi >> (pos - 64));
    } else if (pos + count <= 64) {
      value = uint32_t(lo >> pos);
    } else {
      value = uint32_t((lo >> pos) | (hi << (64 - pos)));
    }
    pos += count;
    return count >= 32 ? value : value & ((1u << count) - 1);
  }
};

static uint8_t Bc7Interpolate(uint32_t e0, uint32_t e1, uint32_t index, uint32_t bits) {
  const uint32_t w = bits == 2 ? kBc7Weights2[index] : bits == 3 ? kBc7Weights3[index]
                                                                  : kBc7Weights4[index];
  return uint8_t(((64 - w) * e0 + w * e1 + 32) >> 6);
}

// One 16-byte BC7 block -> 16 RGBA8 pixels (row-major in the 4x4 block).
static void DecodeBC7Block(const uint8_t* block, uint8_t out[16][4]) {
  Bc7Bits bits;
  std::memcpy(&bits.lo, block, 8);
  std::memcpy(&bits.hi, block + 8, 8);

  uint32_t mode = 0;
  while (mode < 8 && !bits.Read(1)) {
    ++mode;
  }
  if (mode >= 8) {
    // Reserved: transparent black.
    std::memset(out, 0, 16 * 4);
    return;
  }

  struct ModeInfo {
    uint8_t subsets, partition_bits, rotation_bits, index_mode_bits, color_bits, alpha_bits,
        endpoint_pbits, shared_pbits, index_bits, index2_bits;
  };
  static constexpr ModeInfo kModes[8] = {
      {3, 4, 0, 0, 4, 0, 1, 0, 3, 0}, {2, 6, 0, 0, 6, 0, 0, 1, 3, 0},
      {3, 6, 0, 0, 5, 0, 0, 0, 2, 0}, {2, 6, 0, 0, 7, 0, 1, 0, 2, 0},
      {1, 0, 2, 1, 5, 6, 0, 0, 2, 3}, {1, 0, 2, 0, 7, 8, 0, 0, 2, 2},
      {1, 0, 0, 0, 7, 7, 1, 0, 4, 0}, {2, 6, 0, 0, 5, 5, 1, 0, 2, 0},
  };
  const ModeInfo& m = kModes[mode];
  const uint32_t partition = bits.Read(m.partition_bits);
  const uint32_t rotation = bits.Read(m.rotation_bits);
  const uint32_t index_mode = bits.Read(m.index_mode_bits);
  const uint32_t endpoint_count = m.subsets * 2u;

  uint32_t endpoints[6][4];
  for (uint32_t c = 0; c < 3; ++c) {
    for (uint32_t e = 0; e < endpoint_count; ++e) {
      endpoints[e][c] = bits.Read(m.color_bits);
    }
  }
  for (uint32_t e = 0; e < endpoint_count; ++e) {
    endpoints[e][3] = m.alpha_bits ? bits.Read(m.alpha_bits) : 255;
  }
  uint32_t color_bits = m.color_bits, alpha_bits = m.alpha_bits;
  if (m.endpoint_pbits || m.shared_pbits) {
    uint32_t pbits[6];
    if (m.endpoint_pbits) {
      for (uint32_t e = 0; e < endpoint_count; ++e) {
        pbits[e] = bits.Read(1);
      }
    } else {
      for (uint32_t s = 0; s < m.subsets; ++s) {
        pbits[s * 2] = pbits[s * 2 + 1] = bits.Read(1);
      }
    }
    for (uint32_t e = 0; e < endpoint_count; ++e) {
      for (uint32_t c = 0; c < 3; ++c) {
        endpoints[e][c] = (endpoints[e][c] << 1) | pbits[e];
      }
      if (m.alpha_bits) {
        endpoints[e][3] = (endpoints[e][3] << 1) | pbits[e];
      }
    }
    ++color_bits;
    if (m.alpha_bits) {
      ++alpha_bits;
    }
  }
  // Expand to 8 bits by replicating the top bits.
  for (uint32_t e = 0; e < endpoint_count; ++e) {
    for (uint32_t c = 0; c < 3; ++c) {
      const uint32_t v = endpoints[e][c] << (8 - color_bits);
      endpoints[e][c] = v | (v >> color_bits);
    }
    if (m.alpha_bits) {
      const uint32_t v = endpoints[e][3] << (8 - alpha_bits);
      endpoints[e][3] = v | (v >> alpha_bits);
    }
  }

  auto subset_of = [&](uint32_t i) -> uint32_t {
    return m.subsets == 2 ? kBc7Partition2[partition][i]
           : m.subsets == 3 ? kBc7Partition3[partition][i]
                            : 0;
  };
  auto is_anchor = [&](uint32_t i) -> bool {
    if (i == 0) {
      return true;
    }
    if (m.subsets == 2) {
      return i == kBc7Anchor2[partition];
    }
    if (m.subsets == 3) {
      return i == kBc7Anchor3a[partition] || i == kBc7Anchor3b[partition];
    }
    return false;
  };

  uint32_t index1[16], index2[16];
  for (uint32_t i = 0; i < 16; ++i) {
    index1[i] = bits.Read(is_anchor(i) ? m.index_bits - 1 : m.index_bits);
  }
  if (m.index2_bits) {
    for (uint32_t i = 0; i < 16; ++i) {
      index2[i] = bits.Read(i == 0 ? m.index2_bits - 1 : m.index2_bits);
    }
  }

  for (uint32_t i = 0; i < 16; ++i) {
    const uint32_t s = subset_of(i);
    const uint32_t* e0 = endpoints[s * 2];
    const uint32_t* e1 = endpoints[s * 2 + 1];
    uint8_t* p = out[i];
    if (m.index2_bits) {
      // Modes 4/5: colour and alpha have their own indices (swapped by the
      // index mode in mode 4).
      const bool swap = index_mode != 0;
      const uint32_t ci = swap ? index2[i] : index1[i];
      const uint32_t cb = swap ? m.index2_bits : m.index_bits;
      const uint32_t ai = swap ? index1[i] : index2[i];
      const uint32_t ab = swap ? m.index_bits : m.index2_bits;
      for (uint32_t c = 0; c < 3; ++c) {
        p[c] = Bc7Interpolate(e0[c], e1[c], ci, cb);
      }
      p[3] = Bc7Interpolate(e0[3], e1[3], ai, ab);
    } else {
      for (uint32_t c = 0; c < 4; ++c) {
        p[c] = Bc7Interpolate(e0[c], e1[c], index1[i], m.index_bits);
      }
    }
    switch (rotation) {
      case 1: std::swap(p[0], p[3]); break;
      case 2: std::swap(p[1], p[3]); break;
      case 3: std::swap(p[2], p[3]); break;
      default: break;
    }
  }
}

static bool DecodeBCToRGBA8(const uint8_t* blocks, size_t block_bytes, uint32_t width,
                            uint32_t height, DdsDecodeFormat format, std::vector<uint8_t>& rgba) {
  const uint32_t block_count_x = (width + 3) / 4;
  const uint32_t block_count_y = (height + 3) / 4;
  const uint32_t bytes_per_block = format == DdsDecodeFormat::kBC1 ? 8 : 16;
  const size_t expected_bytes =
      static_cast<size_t>(block_count_x) * block_count_y * bytes_per_block;
  if (block_bytes < expected_bytes) {
    return false;
  }

  rgba.assign(static_cast<size_t>(width) * height * 4, 0);

  ParallelRows(block_count_y, uint64_t(width) * height, [&](uint32_t by_begin, uint32_t by_end) {
  for (uint32_t by = by_begin; by < by_end; ++by) {
    for (uint32_t bx = 0; bx < block_count_x; ++bx) {
      const uint8_t* block =
          blocks + (static_cast<size_t>(by) * block_count_x + bx) * bytes_per_block;

      if (format == DdsDecodeFormat::kBC7) {
        uint8_t pixels[16][4];
        DecodeBC7Block(block, pixels);
        for (uint32_t py = 0; py < 4 && by * 4 + py < height; ++py) {
          const uint32_t columns = std::min(4u, width - bx * 4);
          std::memcpy(rgba.data() + ((static_cast<size_t>(by) * 4 + py) * width + bx * 4) * 4,
                      pixels[py * 4], columns * 4);
        }
        continue;
      }

      uint8_t colors[16][4]{};
      uint8_t alpha[16];
      std::fill(alpha, alpha + 16, uint8_t(255));
      const uint8_t* color_block = block;
      if (format == DdsDecodeFormat::kBC2) {
        DecodeBC2AlphaBlock(block, alpha);
        color_block = block + 8;
      } else if (format == DdsDecodeFormat::kBC3) {
        DecodeBC3AlphaBlock(block, alpha);
        color_block = block + 8;
      }
      DecodeBCColorBlock(color_block, format == DdsDecodeFormat::kBC1, colors);

      for (uint32_t py = 0; py < 4; ++py) {
        const uint32_t y = by * 4 + py;
        if (y >= height) {
          continue;
        }
        for (uint32_t px = 0; px < 4; ++px) {
          const uint32_t x = bx * 4 + px;
          if (x >= width) {
            continue;
          }
          const uint32_t source_index = py * 4 + px;
          uint8_t* dst = rgba.data() + (static_cast<size_t>(y) * width + x) * 4;
          std::memcpy(dst, colors[source_index], 4);
          if (format == DdsDecodeFormat::kBC2 || format == DdsDecodeFormat::kBC3) {
            dst[3] = alpha[source_index];
          }
        }
      }
    }
  }
  });

  return true;
}

static bool ReadExact(std::ifstream& f, void* data, size_t size) {
  f.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size));
  return f.good();
}

static bool ReadRemainingBytes(std::ifstream& f, std::vector<uint8_t>& bytes) {
  const std::streampos start = f.tellg();
  if (start < std::streampos(0)) {
    return false;
  }
  f.seekg(0, std::ios::end);
  const std::streampos end = f.tellg();
  if (end < start) {
    return false;
  }
  f.seekg(start);
  bytes.resize(static_cast<size_t>(end - start));
  return bytes.empty() || ReadExact(f, bytes.data(), bytes.size());
}

static bool DecodeLinear32ToRGBA8(const uint8_t* data, size_t data_size, uint32_t width,
                                  uint32_t height, uint32_t row_pitch, DdsDecodeFormat format,
                                  std::vector<uint8_t>& rgba) {
  const uint32_t min_row_pitch = width * 4;
  if (row_pitch < min_row_pitch) {
    return false;
  }
  const size_t required_size =
      height == 0 ? 0 : static_cast<size_t>(row_pitch) * (height - 1) + min_row_pitch;
  if (data_size < required_size) {
    return false;
  }

  rgba.resize(static_cast<size_t>(width) * height * 4);
  for (uint32_t y = 0; y < height; ++y) {
    const uint8_t* src = data + static_cast<size_t>(y) * row_pitch;
    uint8_t* dst = rgba.data() + static_cast<size_t>(y) * width * 4;
    if (format == DdsDecodeFormat::kRGBA8) {
      std::memcpy(dst, src, min_row_pitch);
    } else {
      for (uint32_t x = 0; x < width; ++x) {
        dst[x * 4 + 0] = src[x * 4 + 2];
        dst[x * 4 + 1] = src[x * 4 + 1];
        dst[x * 4 + 2] = src[x * 4 + 0];
        dst[x * 4 + 3] = format == DdsDecodeFormat::kBGRX8 ? 255 : src[x * 4 + 3];
      }
    }
  }
  return true;
}

static uint32_t GetDDSRowPitch(const DdsHeader& hdr) {
  const uint32_t tight_pitch = hdr.width * 4;
  if ((hdr.flags & kDdsdPitch) && hdr.pitch_or_linear >= tight_pitch) {
    return hdr.pitch_or_linear;
  }
  return tight_pitch;
}

static bool IsSupportedDDSDimension(uint32_t width, uint32_t height) {
  constexpr uint32_t kMaxReplacementExtent = 32768;
  return width != 0 && height != 0 && width <= kMaxReplacementExtent &&
         height <= kMaxReplacementExtent;
}

static std::string LowerASCII(std::string value) {
  for (char& c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return value;
}

// ---------------------------------------------------------------------------
// DDS file writers
// ---------------------------------------------------------------------------

bool TextureReplacement::WriteDDS_RGBA8(const std::filesystem::path& path, uint32_t width,
                                        uint32_t height, const uint8_t* rgba8_rows,
                                        uint32_t row_pitch_bytes) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);

  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f.is_open())
    return false;

  DdsHeader hdr;
  hdr.flags = kDdsdCaps | kDdsdHeight | kDdsdWidth | kDdsdPixelFormat | kDdsdPitch;
  hdr.height = height;
  hdr.width = width;
  hdr.pitch_or_linear = width * 4;
  hdr.ddspf.flags = kDdsPfRgb | kDdsPfAlphaPixels;
  hdr.ddspf.rgb_bit_count = 32;
  hdr.ddspf.r_bit_mask = 0x000000FFu;
  hdr.ddspf.g_bit_mask = 0x0000FF00u;
  hdr.ddspf.b_bit_mask = 0x00FF0000u;
  hdr.ddspf.a_bit_mask = 0xFF000000u;

  f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

  const uint32_t row_bytes = width * 4;
  for (uint32_t y = 0; y < height; ++y) {
    f.write(reinterpret_cast<const char*>(rgba8_rows + y * row_pitch_bytes), row_bytes);
  }
  return f.good();
}

bool TextureReplacement::WriteDDS_BC(const std::filesystem::path& path, uint32_t width,
                                     uint32_t height, const uint8_t* bc_blocks,
                                     uint32_t bytes_per_block, uint32_t fourcc) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);

  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f.is_open())
    return false;

  const uint32_t w_blocks = (width + 3) / 4;
  const uint32_t h_blocks = (height + 3) / 4;
  const uint32_t linear_size = w_blocks * h_blocks * bytes_per_block;

  DdsHeader hdr;
  hdr.flags = kDdsdCaps | kDdsdHeight | kDdsdWidth | kDdsdPixelFormat | kDdsdLinearSize;
  hdr.height = height;
  hdr.width = width;
  hdr.pitch_or_linear = linear_size;
  hdr.ddspf.flags = kDdsPfFourCC;
  hdr.ddspf.four_cc = fourcc;

  f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
  f.write(reinterpret_cast<const char*>(bc_blocks), linear_size);
  return f.good();
}

// ---------------------------------------------------------------------------
// DDS reader: decodes supported top-level DDS formats to RGBA8 replacement pixels.
// ---------------------------------------------------------------------------
bool TextureReplacement::ReadDDS(const std::filesystem::path& path, TextureReplacementData& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
    return false;

  DdsHeader hdr{};
  f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
  if (!f || hdr.magic != kDdsMagic || hdr.size != 124)
    return false;

  out.width = hdr.width;
  out.height = hdr.height;
  out.mip_levels = std::max(1u, hdr.mip_map_count);

  if (!IsSupportedDDSDimension(out.width, out.height)) {
    REXLOG_WARN("TextureReplacement: {} has unsupported DDS dimensions {}x{}",
                path.filename().string(), out.width, out.height);
    return false;
  }

  DdsDecodeFormat decode_format = DdsDecodeFormat::kRGBA8;
  bool compressed = false;

  if ((hdr.ddspf.flags & kDdsPfFourCC) != 0) {
    switch (hdr.ddspf.four_cc) {
      case kFourCC_DXT1:
        decode_format = DdsDecodeFormat::kBC1;
        compressed = true;
        break;
      case kFourCC_DXT3:
        decode_format = DdsDecodeFormat::kBC2;
        compressed = true;
        break;
      case kFourCC_DXT5:
        decode_format = DdsDecodeFormat::kBC3;
        compressed = true;
        break;
      case kFourCC_DX10: {
        DdsHeaderDX10 hdr_dx10{};
        if (!ReadExact(f, &hdr_dx10, sizeof(hdr_dx10))) {
          return false;
        }
        switch (hdr_dx10.dxgi_format) {
          case kDxgiFormatR8G8B8A8UNorm:
          case kDxgiFormatR8G8B8A8UNormSRGB:
            decode_format = DdsDecodeFormat::kRGBA8;
            break;
          case kDxgiFormatB8G8R8A8UNorm:
          case kDxgiFormatB8G8R8A8UNormSRGB:
            decode_format = DdsDecodeFormat::kBGRA8;
            break;
          case kDxgiFormatB8G8R8X8UNorm:
          case kDxgiFormatB8G8R8X8UNormSRGB:
            decode_format = DdsDecodeFormat::kBGRX8;
            break;
          case kDxgiFormatBC1UNorm:
          case kDxgiFormatBC1UNormSRGB:
            decode_format = DdsDecodeFormat::kBC1;
            compressed = true;
            break;
          case kDxgiFormatBC2UNorm:
          case kDxgiFormatBC2UNormSRGB:
            decode_format = DdsDecodeFormat::kBC2;
            compressed = true;
            break;
          case kDxgiFormatBC3UNorm:
          case kDxgiFormatBC3UNormSRGB:
            decode_format = DdsDecodeFormat::kBC3;
            compressed = true;
            break;
          case kDxgiFormatBC7UNorm:
          case kDxgiFormatBC7UNormSRGB:
            decode_format = DdsDecodeFormat::kBC7;
            compressed = true;
            break;
          default:
            REXLOG_WARN("TextureReplacement: {} has unsupported DX10 DDS format {}",
                        path.filename().string(), hdr_dx10.dxgi_format);
            return false;
        }
        break;
      }
      default:
        REXLOG_WARN("TextureReplacement: {} has unsupported DDS FOURCC {:08X}",
                    path.filename().string(), hdr.ddspf.four_cc);
        return false;
    }
  } else {
    if (hdr.ddspf.rgb_bit_count != 32 || hdr.ddspf.g_bit_mask != 0x0000FF00u) {
      REXLOG_WARN("TextureReplacement: {} has unsupported DDS pixel format",
                  path.filename().string());
      return false;
    }
    if (hdr.ddspf.r_bit_mask == 0x000000FFu && hdr.ddspf.b_bit_mask == 0x00FF0000u) {
      decode_format = DdsDecodeFormat::kRGBA8;
    } else if (hdr.ddspf.r_bit_mask == 0x00FF0000u && hdr.ddspf.b_bit_mask == 0x000000FFu) {
      decode_format =
          (hdr.ddspf.a_bit_mask == 0) ? DdsDecodeFormat::kBGRX8 : DdsDecodeFormat::kBGRA8;
    } else {
      REXLOG_WARN("TextureReplacement: {} has unsupported DDS channel masks",
                  path.filename().string());
      return false;
    }
  }

  std::vector<uint8_t> bytes;
  if (!ReadRemainingBytes(f, bytes)) {
    return false;
  }

  if (compressed) {
    if (!DecodeBCToRGBA8(bytes.data(), bytes.size(), out.width, out.height, decode_format,
                         out.pixels)) {
      REXLOG_WARN("TextureReplacement: {} has truncated DDS block data", path.filename().string());
      return false;
    }
    return true;
  }

  if (!DecodeLinear32ToRGBA8(bytes.data(), bytes.size(), out.width, out.height, GetDDSRowPitch(hdr),
                             decode_format, out.pixels)) {
    REXLOG_WARN("TextureReplacement: {} has truncated DDS pixel data", path.filename().string());
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// TextureReplacement — construction / rescan
// ---------------------------------------------------------------------------

TextureReplacement::TextureReplacement(std::filesystem::path textures_dir)
    : textures_dir_(std::move(textures_dir)) {
  Rescan();
}

TextureReplacement::~TextureReplacement() {
  // The survey first: loads waiting for its answer get it (and once more at
  // the end, for one a stopping load may have started).
  StopAlphaSurvey();
  StopAsync();
  StopPreload();
  StopAlphaSurvey();
}

void TextureReplacement::Rescan() {
  StopAlphaSurvey();
  StopAsync();
  StopPreload();
  StopAlphaSurvey();
  std::lock_guard<std::mutex> lock(cache_mutex_);
  replacements_.clear();
  pixel_cache_.clear();
  pixel_cache_use_.clear();
  pixel_cache_bytes_ = 0;
  failed_cache_.clear();
  size_cache_.clear();
  guest_refs_.clear();
  premultiplied_votes_ = 0;
  straight_votes_ = 0;
  pack_verdict_logged_ = false;
  // Files for the premultiplied-alpha survey: up to 8 MB (quick to decode),
  // in the order of their keys (as good as random).
  std::vector<std::pair<uint64_t, std::filesystem::path>> survey_files;
  {
    std::lock_guard<std::mutex> alpha_lock(alpha_mutex_);
    pack_alpha_ = PackAlpha::kUnknown;
    alpha_survey_started_ = false;
    alpha_survey_files_.clear();
  }

  std::error_code ec;
  if (!std::filesystem::exists(replace_dir(), ec))
    return;

  for (auto& entry : std::filesystem::directory_iterator(replace_dir(), ec)) {
    if (ec)
      break;
    if (!entry.is_regular_file())
      continue;
    auto& p = entry.path();
    const auto ext = LowerASCII(p.extension().string());
    if (ext != ".dds" && ext != ".png" && ext != ".jpg")
      continue;

    const std::string stem = p.stem().string();
    if (stem.size() < 16)
      continue;

    uint64_t hash = 0;
    bool ok = true;
    for (int i = 0; i < 16; ++i) {
      char c = stem[i];
      uint64_t nibble = 0;
      if (c >= '0' && c <= '9')
        nibble = static_cast<uint64_t>(c - '0');
      else if (c >= 'a' && c <= 'f')
        nibble = static_cast<uint64_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F')
        nibble = static_cast<uint64_t>(c - 'A' + 10);
      else {
        ok = false;
        break;
      }
      hash = (hash << 4) | nibble;
    }
    if (!ok)
      continue;

    replacements_[hash] = p;
    std::error_code size_ec;
    const uintmax_t file_size = entry.file_size(size_ec);
    if (!size_ec && file_size <= (uintmax_t(8) << 20)) {
      survey_files.emplace_back(hash, p);
    }
  }

  std::sort(survey_files.begin(), survey_files.end());
  {
    std::lock_guard<std::mutex> alpha_lock(alpha_mutex_);
    for (auto& [hash, path] : survey_files) {
      // (A later file with the same key replaced this one in the index.)
      auto it = replacements_.find(hash);
      if (it != replacements_.end() && it->second == path) {
        alpha_survey_files_.push_back(std::move(path));
      }
    }
  }

  REXLOG_INFO("TextureReplacement: {} replacement(s) indexed from {}", replacements_.size(),
              replace_dir().string());
}

size_t TextureReplacement::GetReplacementCount() const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return replacements_.size();
}

// ---------------------------------------------------------------------------
// Hash
// ---------------------------------------------------------------------------
uint64_t TextureReplacement::HashGuestData(const uint8_t* data, size_t size) {
  return XXH3_64bits(data, size);
}

uint64_t TextureReplacement::FingerprintGuestData(const uint8_t* data, size_t size) {
  // Small textures whole, bigger ones by 16 samples of 512 bytes spread over
  // them.
  constexpr size_t kSampleCount = 16;
  constexpr size_t kSampleSize = 512;
  if (size <= kSampleCount * kSampleSize * 2) {
    return XXH3_64bits(data, size);
  }
  uint64_t fingerprint = size;
  for (size_t i = 0; i < kSampleCount; ++i) {
    const size_t offset = (size - kSampleSize) * i / (kSampleCount - 1);
    fingerprint = XXH3_64bits_withSeed(data + offset, kSampleSize, fingerprint);
  }
  return fingerprint;
}

// ---------------------------------------------------------------------------
// DumpTexture — untile + endian-swap + write DDS
// ---------------------------------------------------------------------------
bool TextureReplacement::GetSwizzleColorOrder(uint32_t guest_swizzle, uint32_t channel_of[4]) {
  uint32_t seen = 0;
  for (uint32_t i = 0; i < 3; ++i) {
    const uint32_t c = (guest_swizzle >> (3 * i)) & 0b111;
    if (c > 2 || (seen & (1u << c))) {
      return false;
    }
    seen |= 1u << c;
    channel_of[c] = i;
  }
  channel_of[3] = 3;
  return true;
}

void TextureReplacement::DumpTexture(uint64_t content_hash, uint32_t width, uint32_t height,
                                     uint32_t pitch_blocks, bool tiled, xenos::TextureFormat format,
                                     xenos::Endian endianness, const uint8_t* guest_bytes,
                                     uint32_t guest_size, uint32_t guest_swizzle) const {
  using F = xenos::TextureFormat;

  const FormatInfo* fi = FormatInfo::Get(format);
  if (!fi)
    return;

  // Build filename: <hash16>_<w>x<h>_<format_name>.dds
  char name[128];
  std::snprintf(name, sizeof(name), "%016llx_%ux%u_%s.dds",
                static_cast<unsigned long long>(content_hash), width, height, fi->name);
  const auto dest = dump_dir() / name;

  // Only write once per unique texture to avoid hammering the disk.
  if (std::filesystem::exists(dest))
    return;

  const uint32_t bpb = fi->bytes_per_block();
  const uint32_t w_blocks = (width + fi->block_width - 1) / fi->block_width;
  const uint32_t h_blocks = (height + fi->block_height - 1) / fi->block_height;
  // pitch_blocks is in units of 32 texels, convert to block units
  const uint32_t pitch_b32 = pitch_blocks * 32;  // pitch in texels
  const uint32_t pitch_blk = (pitch_b32 + fi->block_width - 1) / fi->block_width;

  // Step 1 — allocate a linear staging buffer and untile (or copy linear)
  const uint32_t linear_bytes = w_blocks * h_blocks * bpb;
  std::vector<uint8_t> linear(linear_bytes);

  if (tiled) {
    UntileBlocks(guest_bytes, linear.data(), w_blocks, h_blocks, pitch_blk, bpb);
  } else {
    // Linear: rows are already in order but may have pitch padding — copy
    // only the visible region.
    const uint32_t src_row_bytes = pitch_blk * bpb;
    const uint32_t dst_row_bytes = w_blocks * bpb;
    for (uint32_t y = 0; y < h_blocks; ++y) {
      const uint32_t src_off = y * src_row_bytes;
      if (src_off + dst_row_bytes > guest_size)
        break;
      std::memcpy(linear.data() + y * dst_row_bytes, guest_bytes + src_off, dst_row_bytes);
    }
  }

  // Step 2 — endian-swap the staging buffer in-place using CopySwapBlock
  if (endianness != xenos::Endian::kNone) {
    texture_conversion::CopySwapBlock(endianness, linear.data(), linear.data(), linear_bytes);
  }

  // Step 3 — write to DDS
  if (fi->type == FormatType::kCompressed) {
    // Determine DDS FOURCC
    uint32_t fourcc = 0;
    switch (format) {
      case F::k_DXT1:
      case F::k_DXT1_AS_16_16_16_16:
        fourcc = kFourCC_DXT1;
        break;
      case F::k_DXT2_3:
      case F::k_DXT2_3_AS_16_16_16_16:
        fourcc = kFourCC_DXT3;
        break;
      case F::k_DXT4_5:
      case F::k_DXT4_5_AS_16_16_16_16:
        fourcc = kFourCC_DXT5;
        break;
      case F::k_DXN:
        fourcc = kFourCC_ATI2;
        break;
      case F::k_DXT5A:
        fourcc = kFourCC_ATI1;
        break;
      // DXT3A variants: treat as DXT3 (same block layout)
      case F::k_DXT3A:
      case F::k_DXT3A_AS_1_1_1_1:
        fourcc = kFourCC_DXT3;
        break;
      default:
        fourcc = kFourCC_DXT5;
        break;
    }

    if (!WriteDDS_BC(dest, width, height, linear.data(), bpb, fourcc)) {
      REXLOG_WARN("TextureReplacement: failed to write BC dump {}", dest.string());
    } else {
      REXLOG_DEBUG("TextureReplacement: dumped BC  {}", dest.filename().string());
    }
  } else {
    // Uncompressed — expand each texel to RGBA8
    const uint32_t out_row_bytes = w_blocks * 4;  // w_blocks == width for uncompressed
    std::vector<uint8_t> rgba(static_cast<size_t>(w_blocks) * h_blocks * 4);

    bool ok = true;
    for (uint32_t y = 0; y < h_blocks && ok; ++y) {
      for (uint32_t x = 0; x < w_blocks; ++x) {
        const uint8_t* src = linear.data() + (y * w_blocks + x) * bpb;
        uint8_t* dst = rgba.data() + y * out_row_bytes + x * 4;
        if (!TexelToRGBA8(src, format, dst)) {
          // Unsupported format — write raw bytes zero-padded to RGBA8 as
          // a best-effort so at least something useful shows up.
          dst[0] = bpb > 0 ? src[0] : 0;
          dst[1] = bpb > 1 ? src[1] : 0;
          dst[2] = bpb > 2 ? src[2] : 0;
          dst[3] = bpb > 3 ? src[3] : 255;
        }
      }
    }

    // Write the colours the game shows (replacement files are read that
    // way): e.g. ARGB textures read with a ZYXW swizzle come out blue in the
    // guest component order. Only for formats with red, green and blue.
    uint32_t channel_of[4];
    const bool rgb_format = format == F::k_8_8_8_8 || format == F::k_8_8_8_8_A ||
                            format == F::k_8_8_8_8_GAMMA_EDRAM || format == F::k_2_10_10_10 ||
                            format == F::k_1_5_5_5 || format == F::k_5_6_5 ||
                            format == F::k_4_4_4_4;
    if (rgb_format && GetSwizzleColorOrder(guest_swizzle, channel_of) &&
        (channel_of[0] != 0 || channel_of[1] != 1 || channel_of[2] != 2)) {
      for (size_t i = 0; i + 3 < rgba.size(); i += 4) {
        uint8_t texel[4] = {rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]};
        for (uint32_t c = 0; c < 4; ++c) {
          rgba[i + channel_of[c]] = texel[c];
        }
      }
    }

    if (!WriteDDS_RGBA8(dest, width, height, rgba.data(), out_row_bytes)) {
      REXLOG_WARN("TextureReplacement: failed to write RGBA8 dump {}", dest.string());
    } else {
      REXLOG_DEBUG("TextureReplacement: dumped RGBA8 {}", dest.filename().string());
    }
  }
}

// ---------------------------------------------------------------------------
// PNG reader (RGBA8 — via stb_image)
// ---------------------------------------------------------------------------
bool TextureReplacement::ReadPNG(const std::filesystem::path& path, TextureReplacementData& out) {
  // Read the whole file into memory first so we can use stbi_load_from_memory
  // (avoids any stdio FILE* locale issues on Windows).
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open())
    return false;

  const auto file_size = static_cast<size_t>(f.tellg());
  f.seekg(0);
  std::vector<uint8_t> buf(file_size);
  f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(file_size));
  if (!f)
    return false;

  int w = 0, h = 0, channels = 0;
  uint8_t* data =
      stbi_load_from_memory(buf.data(), static_cast<int>(file_size), &w, &h, &channels, 4 /*RGBA*/);
  if (!data) {
    REXLOG_WARN("TextureReplacement: stb_image failed to load {}: {}", path.filename().string(),
                stbi_failure_reason());
    return false;
  }

  out.width = static_cast<uint32_t>(w);
  out.height = static_cast<uint32_t>(h);
  out.mip_levels = 1;
  out.pixels.assign(data, data + static_cast<size_t>(w) * h * 4);
  stbi_image_free(data);
  return true;
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------
void TextureReplacement::Downsample2x(const uint8_t* source, uint32_t source_width,
                                      uint32_t source_height, uint8_t* dest, uint32_t dest_width,
                                      uint32_t dest_height) {
  ParallelRows(dest_height, uint64_t(source_width) * source_height,
               [&](uint32_t y_begin, uint32_t y_end) {
    for (uint32_t y = y_begin; y < y_end; ++y) {
      const uint32_t y0 = std::min(y * 2, source_height - 1);
      const uint32_t y1 = std::min(y * 2 + 1, source_height - 1);
      for (uint32_t x = 0; x < dest_width; ++x) {
        const uint32_t x0 = std::min(x * 2, source_width - 1);
        const uint32_t x1 = std::min(x * 2 + 1, source_width - 1);
        const uint8_t* texels[4] = {
            source + (size_t(y0) * source_width + x0) * 4,
            source + (size_t(y0) * source_width + x1) * 4,
            source + (size_t(y1) * source_width + x0) * 4,
            source + (size_t(y1) * source_width + x1) * 4,
        };
        uint32_t alpha_sum = 0, color_sum[3] = {}, weighted_sum[3] = {};
        for (const uint8_t* texel : texels) {
          alpha_sum += texel[3];
          for (uint32_t c = 0; c < 3; ++c) {
            color_sum[c] += texel[c];
            weighted_sum[c] += uint32_t(texel[c]) * texel[3];
          }
        }
        uint8_t* out = dest + (size_t(y) * dest_width + x) * 4;
        for (uint32_t c = 0; c < 3; ++c) {
          out[c] = uint8_t(alpha_sum ? (weighted_sum[c] + alpha_sum / 2) / alpha_sum
                                     : (color_sum[c] + 2) / 4);
        }
        out[3] = uint8_t((alpha_sum + 2) / 4);
      }
    }
  });
}

std::vector<std::vector<uint8_t>> TextureReplacement::BuildMips(const uint8_t* pixels,
                                                                uint32_t width, uint32_t height) {
  std::vector<std::vector<uint8_t>> mips;
  const uint8_t* source = pixels;
  uint32_t source_width = width, source_height = height;
  while (source_width > 1 || source_height > 1) {
    const uint32_t mip_width = std::max(source_width >> 1, uint32_t(1));
    const uint32_t mip_height = std::max(source_height >> 1, uint32_t(1));
    std::vector<uint8_t>& mip = mips.emplace_back(size_t(mip_width) * mip_height * 4);
    Downsample2x(source, source_width, source_height, mip.data(), mip_width, mip_height);
    source = mip.data();
    source_width = mip_width;
    source_height = mip_height;
  }
  return mips;
}

void TextureReplacement::FitReplacementSize(uint32_t& width, uint32_t& height) {
  replacement_image::FitExtent(width, height, kMaxReplacementSize);
}

// ---------------------------------------------------------------------------
// Premultiplied alpha
// ---------------------------------------------------------------------------
namespace {
// Half-transparent texels the alpha checks look at.
constexpr uint32_t kSoftAlphaMin = 8;
constexpr uint32_t kSoftAlphaMax = 247;
// How much brighter than alpha a premultiplied texel may come out (rounding,
// and upscalers that work on the colour and alpha separately).
constexpr uint32_t kPremultipliedTolerance = 8;
// An image tells something only with this many half-transparent texels
// sampled (and at least 1 in 500 of them).
constexpr uint64_t kAlphaMinSoft = 256;

bool AlphaStatsInformative(const TextureReplacement::AlphaStats& stats) {
  return stats.soft >= kAlphaMinSoft && stats.soft * 500 >= stats.sampled;
}

// Logs a line once per file name (re-decoding after the RAM budget dropped a
// texture doesn't log it again).
bool FirstTimeFor(const std::filesystem::path& path) {
  static std::mutex mutex;
  static std::unordered_set<std::string> seen;
  std::lock_guard<std::mutex> lock(mutex);
  return seen.size() < 65536 && seen.insert(path.filename().string()).second;
}
}  // namespace

TextureReplacement::AlphaStats TextureReplacement::GetAlphaStats(const uint8_t* pixels,
                                                                 uint32_t width,
                                                                 uint32_t height) {
  // Big images every second row and column for the counts (all texels for
  // whether there's any transparency).
  const uint32_t step = uint64_t(width) * height > 1024 * 1024 ? 2 : 1;
  std::atomic<uint64_t> soft{0}, brighter{0}, sampled{0};
  std::atomic<bool> transparent{false};
  ParallelRows(height, uint64_t(width) * height, [&](uint32_t y_begin, uint32_t y_end) {
    uint64_t local_soft = 0, local_brighter = 0, local_sampled = 0;
    bool local_transparent = false;
    for (uint32_t y = y_begin; y < y_end; ++y) {
      const uint8_t* row = pixels + size_t(y) * width * 4;
      const bool sampled_row = y % step == 0;
      for (uint32_t x = 0; x < width; ++x) {
        const uint8_t* texel = row + size_t(x) * 4;
        const uint32_t alpha = texel[3];
        if (alpha != 255) {
          local_transparent = true;
        }
        if (!sampled_row || x % step) {
          continue;
        }
        ++local_sampled;
        if (alpha < kSoftAlphaMin || alpha > kSoftAlphaMax) {
          continue;
        }
        ++local_soft;
        if (std::max({texel[0], texel[1], texel[2]}) > alpha + kPremultipliedTolerance) {
          ++local_brighter;
        }
      }
    }
    soft.fetch_add(local_soft, std::memory_order_relaxed);
    brighter.fetch_add(local_brighter, std::memory_order_relaxed);
    sampled.fetch_add(local_sampled, std::memory_order_relaxed);
    if (local_transparent) {
      transparent.store(true, std::memory_order_relaxed);
    }
  });
  AlphaStats stats;
  stats.soft = soft.load();
  stats.brighter = brighter.load();
  stats.sampled = sampled.load();
  stats.transparent = transparent.load();
  return stats;
}

bool TextureReplacement::DecodeFile(const std::filesystem::path& path,
                                    TextureReplacementData& out) {
  const auto ext = LowerASCII(path.extension().string());
  // JPEG (opaque art such as full-screen pictures: a tenth of a PNG's size)
  // goes through stb_image like PNG.
  return ext == ".png" || ext == ".jpg" ? ReadPNG(path, out) : ReadDDS(path, out);
}

// ---------------------------------------------------------------------------
// The original texture (for the alpha check)
// ---------------------------------------------------------------------------
bool TextureReplacement::CanDecodeGuestFormat(xenos::TextureFormat format) {
  using F = xenos::TextureFormat;
  switch (format) {
    // Formats with alpha worth comparing (DXT1's is 1-bit and black under
    // it, telling nothing).
    case F::k_DXT2_3:
    case F::k_DXT2_3_AS_16_16_16_16:
    case F::k_DXT4_5:
    case F::k_DXT4_5_AS_16_16_16_16:
    case F::k_8_8_8_8:
    case F::k_8_8_8_8_A:
    case F::k_4_4_4_4:
      return true;
    default:
      return false;
  }
}

bool TextureReplacement::DecodeGuest(const TextureReplacementGuest& guest,
                                     std::vector<uint8_t>& rgba) {
  using F = xenos::TextureFormat;
  const FormatInfo* format_info = FormatInfo::Get(guest.format);
  if (!format_info || !guest.width || !guest.height || !CanDecodeGuestFormat(guest.format)) {
    return false;
  }
  const uint32_t bytes_per_block = format_info->bytes_per_block();
  if (!bytes_per_block || (bytes_per_block & (bytes_per_block - 1)) || !guest.row_pitch_bytes) {
    return false;
  }
  const uint32_t x_blocks = (guest.width + format_info->block_width - 1) / format_info->block_width;
  const uint32_t y_blocks =
      (guest.height + format_info->block_height - 1) / format_info->block_height;
  // The blocks in rows, as the guest addresses them (like HashReplacementBase).
  std::vector<uint8_t> linear(size_t(x_blocks) * y_blocks * bytes_per_block);
  const uint32_t bytes_per_block_log2 = rex::log2_floor(bytes_per_block);
  const uint32_t pitch_blocks = guest.row_pitch_bytes / bytes_per_block;
  for (uint32_t y = 0; y < y_blocks; ++y) {
    for (uint32_t x = 0; x < x_blocks; ++x) {
      const uint64_t offset =
          guest.tiled ? uint64_t(uint32_t(texture_util::GetTiledOffset2D(
                            int32_t(x), int32_t(y), pitch_blocks, bytes_per_block_log2)))
                      : uint64_t(y) * guest.row_pitch_bytes + uint64_t(x) * bytes_per_block;
      if (offset + bytes_per_block > guest.bytes.size()) {
        return false;
      }
      std::memcpy(linear.data() + (size_t(y) * x_blocks + x) * bytes_per_block,
                  guest.bytes.data() + offset, bytes_per_block);
    }
  }
  if (guest.endianness != xenos::Endian::kNone) {
    texture_conversion::CopySwapBlock(guest.endianness, linear.data(), linear.data(),
                                      linear.size());
  }
  if (format_info->type == FormatType::kCompressed) {
    const bool dxt3 = guest.format == F::k_DXT2_3 || guest.format == F::k_DXT2_3_AS_16_16_16_16;
    if (!DecodeBCToRGBA8(linear.data(), linear.size(), guest.width, guest.height,
                         dxt3 ? DdsDecodeFormat::kBC2 : DdsDecodeFormat::kBC3, rgba)) {
      return false;
    }
  } else {
    rgba.resize(size_t(guest.width) * guest.height * 4);
    for (size_t i = 0; i < size_t(guest.width) * guest.height; ++i) {
      if (!TexelToRGBA8(linear.data() + i * bytes_per_block, guest.format, &rgba[i * 4])) {
        return false;
      }
    }
  }
  // In the colours the game shows (replacement images hold those).
  uint32_t channel_of[4];
  if (GetSwizzleColorOrder(guest.swizzle, channel_of) &&
      (channel_of[0] != 0 || channel_of[1] != 1 || channel_of[2] != 2)) {
    for (size_t i = 0; i + 3 < rgba.size(); i += 4) {
      const uint8_t texel[4] = {rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]};
      for (uint32_t c = 0; c < 4; ++c) {
        rgba[i + channel_of[c]] = texel[c];
      }
    }
  }
  return true;
}

bool TextureReplacement::WantsGuestReference(uint64_t content_hash) const {
  if (REXCVAR_GET(texture_replace_alpha) == "straight") {
    return false;
  }
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return replacements_.count(content_hash) && !pixel_cache_.count(content_hash) &&
         !failed_cache_.count(content_hash) && !guest_refs_.count(content_hash);
}

void TextureReplacement::SetGuestReference(
    uint64_t content_hash, std::shared_ptr<const TextureReplacementGuest> guest) const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  // Only those waiting to be decoded - a few at a time, normally.
  if (guest_refs_.size() >= 1024) {
    guest_refs_.clear();
  }
  guest_refs_[content_hash] = std::move(guest);
}

// ---------------------------------------------------------------------------
// Loading a replacement
// ---------------------------------------------------------------------------
bool TextureReplacement::FixAlpha(uint64_t hash, const std::filesystem::path& path,
                                  TextureReplacementData& data, bool allow_defer) const {
  namespace image = replacement_image;
  const std::string mode = REXCVAR_GET(texture_replace_alpha);
  if (mode == "straight" ||
      !image::HasTransparency(data.pixels.data(), uint64_t(data.width) * data.height)) {
    return true;
  }
  const bool forced = mode == "premultiplied";

  // The game's own texture, when it was handed over with the lookup.
  std::shared_ptr<const TextureReplacementGuest> guest;
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = guest_refs_.find(hash);
    if (it != guest_refs_.end()) {
      guest = it->second;
    }
  }
  std::vector<uint8_t> original;
  if (guest && DecodeGuest(*guest, original)) {
    // Compared with the original: decided for this texture alone.
    const image::AlphaCheck check = image::ClassifyAgainstOriginal(
        data.pixels.data(), data.width, data.height, original.data(), guest->width,
        guest->height);
    if (!check.cutout) {
      if (check.kind == image::AlphaKind::kPremultiplied) {
        ++premultiplied_votes_;
      } else if (check.kind == image::AlphaKind::kStraight) {
        ++straight_votes_;
      }
    }
    image::AlphaKind kind = check.kind;
    if (forced && kind != image::AlphaKind::kLostColour) {
      kind = image::AlphaKind::kPremultiplied;
    }
    if (kind != image::AlphaKind::kPremultiplied && kind != image::AlphaKind::kLostColour) {
      return true;
    }
    image::RestoreStraightAlpha(data.pixels.data(), data.width, data.height, kind,
                                original.data(), guest->width, guest->height);
    const uint32_t fixed = ++alpha_fixed_count_;
    if (FirstTimeFor(path) && (fixed <= 48 || fixed % 100 == 0)) {
      if (kind == image::AlphaKind::kLostColour) {
        REXLOG_INFO(
            "TextureReplacement: {} is black under its transparent texels where the original "
            "has colour (original {:.0f}, image {:.0f}) - the original's colour put back ({} "
            "fixed so far)",
            path.filename().string(), check.error_a, check.error_b, fixed);
      } else if (check.cutout || forced) {
        REXLOG_INFO(
            "TextureReplacement: {} has premultiplied alpha{} - converted to straight alpha, the "
            "original's colour under its transparent texels ({} fixed so far)",
            path.filename().string(),
            forced ? " (texture_replace_alpha = premultiplied)" : " (black under transparency)",
            fixed);
      } else {
        REXLOG_INFO(
            "TextureReplacement: {} has premultiplied alpha (colour difference from the "
            "original: {:.1f} as straight, {:.1f} as premultiplied) - converted to straight "
            "alpha, the original's colour under its transparent texels ({} fixed so far)",
            path.filename().string(), check.error_a, check.error_b, fixed);
      }
    }
    if (!pack_verdict_logged_ && premultiplied_votes_ >= 3 &&
        premultiplied_votes_ >= 3 * straight_votes_ && !pack_verdict_logged_.exchange(true)) {
      REXLOG_INFO(
          "TextureReplacement: this pack has premultiplied alpha ({} of {} textures compared "
          "with the game's originals so far) - such images are converted as they load",
          uint32_t(premultiplied_votes_), uint32_t(premultiplied_votes_ + straight_votes_));
    }
    return true;
  }

  // No original to compare with (decoded ahead by the preload, or a guest
  // format the check can't read): the image's own texels and the pack as a
  // whole decide.
  const AlphaStats alpha = GetAlphaStats(data.pixels.data(), data.width, data.height);
  // Most of its colour brighter than its alpha: straight (whatever the pack).
  const bool straight = AlphaStatsInformative(alpha) && alpha.brighter * 2 > alpha.soft;
  if (straight) {
    return true;
  }
  if (allow_defer) {
    // Loaded again when the game asks for it, with its original.
    return false;
  }
  if (!forced && !IsPackPremultiplied()) {
    return true;
  }
  image::RestoreStraightAlpha(data.pixels.data(), data.width, data.height,
                              image::AlphaKind::kPremultiplied, nullptr, 0, 0);
  const uint32_t fixed = ++alpha_fixed_count_;
  if (FirstTimeFor(path) && (fixed <= 48 || fixed % 100 == 0)) {
    REXLOG_INFO(
        "TextureReplacement: {} has premultiplied alpha (the pack's; no original to compare "
        "with) - converted to straight alpha ({} fixed so far)",
        path.filename().string(), fixed);
  }
  return true;
}

TextureReplacement::LoadResult TextureReplacement::LoadReplacement(
    uint64_t hash, const std::filesystem::path& path, TextureReplacementData& out,
    bool allow_defer) const {
  if (!DecodeFile(path, out)) {
    REXLOG_WARN("TextureReplacement: failed to load {}", path.filename().string());
    return LoadResult::kFailed;
  }

  // Premultiplied alpha to straight, before anything averages texels.
  if (!FixAlpha(hash, path, out, allow_defer)) {
    return LoadResult::kDeferred;
  }

  // Bigger than a texture can be: downscaled to fit, area-averaged (the size
  // lookups say the same, see ReadImageSize).
  uint32_t fit_width = out.width, fit_height = out.height;
  FitReplacementSize(fit_width, fit_height);
  if (fit_width != out.width || fit_height != out.height) {
    std::vector<uint8_t> fitted = replacement_image::DownscaleArea(
        out.pixels.data(), out.width, out.height, fit_width, fit_height);
    if (FirstTimeFor(path.filename().string() + "#size")) {
      REXLOG_INFO(
          "TextureReplacement: {} is {}x{}, more than a texture can be ({} per side) - used at "
          "{}x{}",
          path.filename().string(), out.width, out.height, kMaxReplacementSize, fit_width,
          fit_height);
    }
    out.pixels.swap(fitted);
    out.width = fit_width;
    out.height = fit_height;
  }

  out.mip_levels = 1;
  out.mips = BuildMips(out.pixels.data(), out.width, out.height);
  // Its original isn't needed any more (given again if it has to be decoded
  // again).
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    guest_refs_.erase(hash);
  }
  return LoadResult::kLoaded;
}

bool TextureReplacement::ReadImageSize(const std::filesystem::path& path, uint32_t& width,
                                       uint32_t& height) {
  uint8_t header[24] = {};
  std::ifstream file(path, std::ios::binary);
  if (!file.read(reinterpret_cast<char*>(header), sizeof(header))) {
    return false;
  }
  auto big_endian = [&](size_t offset) {
    return (uint32_t(header[offset]) << 24) | (uint32_t(header[offset + 1]) << 16) |
           (uint32_t(header[offset + 2]) << 8) | uint32_t(header[offset + 3]);
  };
  auto little_endian = [&](size_t offset) {
    return uint32_t(header[offset]) | (uint32_t(header[offset + 1]) << 8) |
           (uint32_t(header[offset + 2]) << 16) | (uint32_t(header[offset + 3]) << 24);
  };
  width = 0;
  height = 0;
  if (!std::memcmp(header, "\x89PNG", 4)) {
    width = big_endian(16);  // IHDR
    height = big_endian(20);
  } else if (!std::memcmp(header, "DDS ", 4)) {
    height = little_endian(12);
    width = little_endian(16);
  }
  if (!width || !height || width > 16384 || height > 16384) {
    return false;
  }
  // The size it's used at (LoadFile halves bigger images).
  FitReplacementSize(width, height);
  return true;
}

uint64_t TextureReplacement::DecodedSize(const std::filesystem::path& path) {
  uint32_t width, height;
  if (!ReadImageSize(path, width, height)) {
    return 0;
  }
  // RGBA8, and a third more for the mips.
  return uint64_t(width) * height * 4 * 4 / 3;
}

void TextureReplacement::MarkFailed(uint64_t content_hash) const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  failed_cache_.insert(content_hash);
  size_cache_.erase(content_hash);
}

bool TextureReplacement::IsJpeg(uint64_t content_hash) const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  const auto it = replacements_.find(content_hash);
  return it != replacements_.end() && LowerASCII(it->second.extension().string()) == ".jpg";
}

bool TextureReplacement::FindReplacementSize(uint64_t content_hash, uint32_t& width,
                                             uint32_t& height) const {
  // The program's own replacements first, also without a texture pack.
  if (auto image = FindTextureMemoryReplacement(content_hash)) {
    width = image->width;
    height = image->height;
    return true;
  }
  if (!REXCVAR_GET(texture_replace_enabled)) {
    return false;
  }
  std::filesystem::path path;
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    // The background decoding starts with the first texture looked up.
    if (!preload_started_) {
      StartPreload();
    }
    auto cached = pixel_cache_.find(content_hash);
    if (cached != pixel_cache_.end()) {
      width = cached->second.width;
      height = cached->second.height;
      return width && height;
    }
    if (failed_cache_.count(content_hash)) {
      return false;
    }
    auto size = size_cache_.find(content_hash);
    if (size != size_cache_.end()) {
      width = size->second.first;
      height = size->second.second;
      return true;
    }
    auto it = replacements_.find(content_hash);
    if (it == replacements_.end()) {
      return false;
    }
    path = it->second;
  }
  const bool ok = ReadImageSize(path, width, height);
  if (!ok) {
    REXLOG_WARN(
        "TextureReplacement: {} can't be used (not a PNG or DDS image, or more than 16384 per "
        "side) - the original texture is shown",
        path.filename().string());
  }
  std::lock_guard<std::mutex> lock(cache_mutex_);
  if (ok) {
    size_cache_[content_hash] = {width, height};
  } else {
    failed_cache_.insert(content_hash);
  }
  return ok;
}

uint64_t TextureReplacement::CachedSize(const TextureReplacementData& data) {
  uint64_t bytes = data.pixels.size() + data.prepared_bytes;
  for (const auto& mip : data.mips) {
    bytes += mip.size();
  }
  return bytes;
}

TextureReplacementData* TextureReplacement::AddToPixelCache(uint64_t hash,
                                                            TextureReplacementData&& data) const {
  auto [it, added] = pixel_cache_.emplace(hash, std::move(data));
  if (added) {
    PixelCacheUse& use = pixel_cache_use_[hash];
    use.bytes = CachedSize(it->second);
    use.last_use = ++pixel_cache_clock_;
    pixel_cache_bytes_ += use.bytes;
  }
  return &it->second;
}

void TextureReplacement::TrimPixelCache(uint64_t keep) const {
  if (pixel_cache_bytes_ <= pixel_cache_budget_) {
    return;
  }
  const uint64_t before = pixel_cache_bytes_;
  size_t dropped = 0;
  // The ones already handed out (their textures are made) first, oldest
  // first; those decoded in the background and not used yet last - the game
  // is about to ask for them. A few thousand entries at most, and only when
  // the cache is over its budget.
  std::vector<std::tuple<bool, uint64_t, uint64_t>> by_age;  // not used yet, last use, hash
  by_age.reserve(pixel_cache_use_.size());
  for (const auto& [hash, use] : pixel_cache_use_) {
    if (hash != keep) {
      by_age.emplace_back(!use.used, use.last_use, hash);
    }
  }
  std::sort(by_age.begin(), by_age.end());
  for (const auto& [not_used, last_use, hash] : by_age) {
    if (pixel_cache_bytes_ <= pixel_cache_budget_) {
      break;
    }
    auto use = pixel_cache_use_.find(hash);
    pixel_cache_bytes_ -= use->second.bytes;
    pixel_cache_use_.erase(use);
    pixel_cache_.erase(hash);
    ++dropped;
  }
  // Logged at most every 10 s (with the background loading this runs often
  // once the budget is full).
  trim_log_dropped_ += dropped;
  const auto now = std::chrono::steady_clock::now();
  if (trim_log_dropped_ && now - trim_log_time_ >= std::chrono::seconds(10)) {
    REXLOG_INFO(
        "TextureReplacement: dropped {} decoded texture(s) from RAM (now {} MB, "
        "texture_replace_ram_mb {} MB; was {} MB)",
        trim_log_dropped_, pixel_cache_bytes_ >> 20, pixel_cache_budget_ >> 20, before >> 20);
    trim_log_dropped_ = 0;
    trim_log_time_ = now;
  }
}

void TextureReplacement::StartPreload() const {
  preload_started_ = true;
  pixel_cache_budget_ = uint64_t(std::max(REXCVAR_GET(texture_replace_ram_mb), 1)) << 20;
  // Whether the pack has premultiplied alpha is needed by every load: the
  // survey starts with the first lookup, so it's usually done before the first
  // replacement is decoded.
  if (!replacements_.empty() && REXCVAR_GET(texture_replace_alpha) != "straight" &&
      REXCVAR_GET(texture_replace_alpha) != "premultiplied") {
    std::lock_guard<std::mutex> alpha_lock(alpha_mutex_);
    StartAlphaSurvey();
  }
  if (!REXCVAR_GET(texture_replace_preload) || replacements_.empty()) {
    return;
  }
  // A pack too big to keep decoded is loaded as it's used instead.
  uint64_t pack_bytes = 0;
  for (const auto& [hash, path] : replacements_) {
    pack_bytes += DecodedSize(path);
  }
  if (pack_bytes > pixel_cache_budget_) {
    REXLOG_INFO(
        "TextureReplacement: not preloading - the {} replacement(s) take {} MB decoded, more "
        "than texture_replace_ram_mb ({} MB); they load as they're used",
        replacements_.size(), pack_bytes >> 20, pixel_cache_budget_ >> 20);
    return;
  }
  preload_queue_.assign(replacements_.begin(), replacements_.end());
  preload_next_ = 0;
  preload_remaining_ = preload_queue_.size();
  preload_start_time_ = std::chrono::steady_clock::now();
  // Below normal priority, a few threads - the game's own threads come first.
  const uint32_t thread_count = std::clamp(std::thread::hardware_concurrency() / 4, 1u, 4u);
  for (uint32_t i = 0; i < thread_count; ++i) {
    rex::thread::Thread::CreationParameters params;
    params.stack_size = 1024 * 1024;
    auto thread = rex::thread::Thread::Create(params, [this]() { PreloadWorker(); });
    if (!thread) {
      break;
    }
    thread->set_name("Texture Preload");
    thread->set_priority(rex::thread::ThreadPriority::kBelowNormal);
    preload_threads_.push_back(std::move(thread));
  }
  REXLOG_INFO("TextureReplacement: decoding {} replacement(s) in the background ({} thread(s))",
              preload_queue_.size(), preload_threads_.size());
}

void TextureReplacement::StopPreload() {
  preload_stop_ = true;
  for (auto& thread : preload_threads_) {
    rex::thread::Wait(thread.get(), false);
  }
  preload_threads_.clear();
  preload_queue_.clear();
  preload_in_progress_.clear();
  preload_started_ = false;
  preload_stop_ = false;
}

void TextureReplacement::PreloadWorker() const {
  while (!preload_stop_) {
    const size_t index = preload_next_.fetch_add(1);
    if (index >= preload_queue_.size()) {
      return;
    }
    const auto& [hash, path] = preload_queue_[index];
    bool needed;
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      needed = !pixel_cache_.count(hash) && !failed_cache_.count(hash) &&
               preload_in_progress_.insert(hash).second;
    }
    if (needed) {
      TextureReplacementData loaded;
      // Deferred: its alpha needs its original, given when the game asks for
      // it - left for that load.
      const LoadResult result = LoadReplacement(hash, path, loaded, true);
      const bool ok = result == LoadResult::kLoaded;
      if (ok) {
        Prepare(loaded);
      }
      {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (ok) {
          AddToPixelCache(hash, std::move(loaded));
        } else if (result == LoadResult::kFailed) {
          failed_cache_.insert(hash);
        }
        preload_in_progress_.erase(hash);
      }
      preload_done_.notify_all();
    }
    if (preload_remaining_.fetch_sub(1) == 1) {
      REXLOG_INFO("TextureReplacement: background decoding done in {} ms",
                  std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - preload_start_time_)
                      .count());
    }
  }
}

// ---------------------------------------------------------------------------
// Premultiplied-alpha survey (texture_replace_alpha = auto)
// ---------------------------------------------------------------------------
// A pack is either: whoever made it exported all of it the same way. One image
// can't always tell - dark straight images (shadows, shading masks) also keep
// their colour below their alpha - so a sample of the pack's images votes:
// premultiplied ones have almost no texel brighter than its alpha, straight
// ones mostly have.
namespace {
// Whether an image file can have half-transparent texels, from its header.
bool MayHaveSoftAlpha(const std::filesystem::path& path) {
  uint8_t header[88] = {};
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char*>(header), sizeof(header));
  if (!std::memcmp(header, "\x89PNG", 4)) {
    const uint8_t color_type = header[25];  // IHDR: palette, gray + alpha, RGBA
    return color_type == 3 || color_type == 4 || color_type == 6;
  }
  if (!std::memcmp(header, "DDS ", 4)) {
    uint32_t four_cc;
    std::memcpy(&four_cc, header + 84, sizeof(four_cc));
    return four_cc != kFourCC_DXT1;  // 1-bit alpha
  }
  return false;
}
// Images decoded at most, and how many that tell something are enough.
constexpr uint32_t kAlphaSurveyMaxImages = 160;
constexpr uint32_t kAlphaSurveyVotes = 24;
}  // namespace

bool TextureReplacement::IsPackPremultiplied() const {
  std::unique_lock<std::mutex> lock(alpha_mutex_);
  if (pack_alpha_ == PackAlpha::kUnknown) {
    const std::string mode = REXCVAR_GET(texture_replace_alpha);
    if (mode == "straight") {
      pack_alpha_ = PackAlpha::kStraight;
    } else if (mode == "premultiplied") {
      pack_alpha_ = PackAlpha::kPremultiplied;
      REXLOG_INFO(
          "TextureReplacement: texture_replace_alpha = premultiplied - the pack's images are "
          "converted to straight alpha as they load");
    } else {
      StartAlphaSurvey();
      alpha_decided_.wait(lock, [this]() { return pack_alpha_ != PackAlpha::kUnknown; });
    }
  }
  return pack_alpha_ == PackAlpha::kPremultiplied;
}

void TextureReplacement::StartAlphaSurvey() const {
  if (alpha_survey_started_ || pack_alpha_ != PackAlpha::kUnknown) {
    return;
  }
  alpha_survey_started_ = true;
  alpha_survey_next_ = 0;
  alpha_votes_premultiplied_ = 0;
  alpha_votes_straight_ = 0;
  alpha_votes_looked_ = 0;
  alpha_survey_start_time_ = std::chrono::steady_clock::now();
  const uint32_t thread_count = std::clamp(std::thread::hardware_concurrency() / 4, 1u, 4u);
  alpha_survey_active_ = thread_count;
  for (uint32_t i = 0; i < thread_count; ++i) {
    rex::thread::Thread::CreationParameters params;
    params.stack_size = 1024 * 1024;
    auto thread = rex::thread::Thread::Create(params, [this]() { AlphaSurveyWorker(); });
    if (!thread) {
      if (alpha_survey_active_.fetch_sub(1) == 1) {
        FinishAlphaSurvey();
      }
      continue;
    }
    thread->set_name("Texture Alpha Survey");
    alpha_survey_threads_.push_back(std::move(thread));
  }
}

void TextureReplacement::StopAlphaSurvey() {
  std::vector<std::unique_ptr<rex::thread::Thread>> threads;
  {
    std::lock_guard<std::mutex> lock(alpha_mutex_);
    threads.swap(alpha_survey_threads_);
    alpha_survey_stop_ = true;
  }
  for (auto& thread : threads) {
    rex::thread::Wait(thread.get(), false);
  }
  std::lock_guard<std::mutex> lock(alpha_mutex_);
  alpha_survey_stop_ = false;
}

void TextureReplacement::AlphaSurveyWorker() const {
  while (!alpha_survey_stop_ &&
         alpha_votes_premultiplied_ + alpha_votes_straight_ < kAlphaSurveyVotes &&
         alpha_votes_looked_ < kAlphaSurveyMaxImages) {
    const size_t index = alpha_survey_next_.fetch_add(1);
    if (index >= alpha_survey_files_.size()) {
      break;
    }
    const std::filesystem::path& path = alpha_survey_files_[index];
    if (!MayHaveSoftAlpha(path)) {
      continue;
    }
    TextureReplacementData image;
    if (!DecodeFile(path, image)) {
      continue;
    }
    alpha_votes_looked_.fetch_add(1);
    const AlphaStats stats = GetAlphaStats(image.pixels.data(), image.width, image.height);
    if (!AlphaStatsInformative(stats)) {
      continue;
    }
    if (stats.brighter * 20 <= stats.soft) {
      // At most 5% brighter than alpha.
      alpha_votes_premultiplied_.fetch_add(1);
    } else if (stats.brighter * 4 >= stats.soft) {
      // A quarter or more.
      alpha_votes_straight_.fetch_add(1);
    }
  }
  if (alpha_survey_active_.fetch_sub(1) == 1) {
    // The last one decides.
    std::lock_guard<std::mutex> lock(alpha_mutex_);
    FinishAlphaSurvey();
  }
}

void TextureReplacement::FinishAlphaSurvey() const {
  const uint32_t premultiplied = alpha_votes_premultiplied_;
  const uint32_t straight = alpha_votes_straight_;
  const bool is_premultiplied = premultiplied >= 3 && premultiplied >= straight * 4;
  pack_alpha_ = is_premultiplied ? PackAlpha::kPremultiplied : PackAlpha::kStraight;
  REXLOG_INFO(
      "TextureReplacement: {} ({} of {} images looked at premultiplied, {} straight, {} ms; "
      "texture_replace_alpha = auto)",
      is_premultiplied ? "the pack's images have premultiplied alpha - they're converted to "
                         "straight alpha (the game's) as they load"
                       : "the pack's images have straight alpha",
      premultiplied, uint32_t(alpha_votes_looked_), straight,
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                            alpha_survey_start_time_)
          .count());
  alpha_decided_.notify_all();
}

// ---------------------------------------------------------------------------
// Asynchronous loading
// ---------------------------------------------------------------------------
bool TextureReplacement::IsResident(uint64_t content_hash) const {
  if (FindTextureMemoryReplacement(content_hash)) {
    return true;
  }
  std::lock_guard<std::mutex> lock(cache_mutex_);
  auto use = pixel_cache_use_.find(content_hash);
  if (use != pixel_cache_use_.end()) {
    // About to be made into a texture: the newest, so trimming the RAM
    // budget before FindReplacement gets to it doesn't drop it.
    use->second.last_use = ++pixel_cache_clock_;
    return true;
  }
  return pixel_cache_.count(content_hash) || failed_cache_.count(content_hash) ||
         !replacements_.count(content_hash);
}

void TextureReplacement::RequestAsync(uint64_t content_hash) const {
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    // The loading threads add without trimming (FindReplacement's pointers
    // must stay valid until its next call): texture_replace_ram_mb is kept
    // here and in FindReplacement, on the thread asking for textures.
    TrimPixelCache(0);
    if (pixel_cache_.count(content_hash) || failed_cache_.count(content_hash) ||
        preload_in_progress_.count(content_hash) || !replacements_.count(content_hash) ||
        std::find(async_queue_.begin(), async_queue_.end(), content_hash) !=
            async_queue_.end()) {
      return;
    }
    async_queue_.push_back(content_hash);
    if (async_threads_.empty() && !async_stop_) {
      // Normal priority: what's on screen now waits for these. Each big image
      // is also split across threads (ParallelRows).
      for (uint32_t i = 0; i < 2; ++i) {
        rex::thread::Thread::CreationParameters params;
        params.stack_size = 1024 * 1024;
        auto thread = rex::thread::Thread::Create(params, [this]() { AsyncWorker(); });
        if (!thread) {
          break;
        }
        thread->set_name("Texture Async Load");
        async_threads_.push_back(std::move(thread));
      }
    }
  }
  async_wake_.notify_one();
}

void TextureReplacement::AsyncWorker() const {
  for (;;) {
    uint64_t hash;
    std::filesystem::path path;
    {
      std::unique_lock<std::mutex> lock(cache_mutex_);
      async_wake_.wait(lock, [this]() { return async_stop_ || !async_queue_.empty(); });
      if (async_stop_) {
        return;
      }
      hash = async_queue_.back();
      async_queue_.pop_back();
      auto it = replacements_.find(hash);
      if (it == replacements_.end() || pixel_cache_.count(hash) || failed_cache_.count(hash) ||
          !preload_in_progress_.insert(hash).second) {
        continue;
      }
      path = it->second;
    }
    const auto load_start = std::chrono::steady_clock::now();
    TextureReplacementData loaded;
    const bool ok = LoadReplacement(hash, path, loaded, false) == LoadResult::kLoaded;
    if (ok) {
      Prepare(loaded);
    }
    {
      rex::perf::TextureCacheStats& stats = rex::perf::GetTextureCacheStats();
      stats.replacement_async_loads.fetch_add(1, std::memory_order_relaxed);
      stats.replacement_async_ns.fetch_add(
          uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now() - load_start)
                       .count()),
          std::memory_order_relaxed);
    }
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      if (ok) {
        // Not trimmed here: FindReplacement's pointers must stay valid until
        // its next call - it and RequestAsync trim.
        AddToPixelCache(hash, std::move(loaded));
      } else {
        failed_cache_.insert(hash);
      }
      preload_in_progress_.erase(hash);
    }
    preload_done_.notify_all();
    if (ok && ready_callback_) {
      ready_callback_();
    }
  }
}

void TextureReplacement::Prepare(TextureReplacementData& data) const {
  if (!prepare_callback_) {
    return;
  }
  std::shared_ptr<void> prepared = prepare_callback_(data);
  if (!prepared) {
    return;
  }
  data.prepared_bytes = CachedSize(data);
  data.prepared_upload = std::move(prepared);
  std::vector<uint8_t>().swap(data.pixels);
  std::vector<std::vector<uint8_t>>().swap(data.mips);
}

void TextureReplacement::StopAsync() {
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    async_stop_ = true;
    async_queue_.clear();
  }
  async_wake_.notify_all();
  for (auto& thread : async_threads_) {
    rex::thread::Wait(thread.get(), false);
  }
  async_threads_.clear();
  std::lock_guard<std::mutex> lock(cache_mutex_);
  async_stop_ = false;
}

// ---------------------------------------------------------------------------
// FindReplacement
// ---------------------------------------------------------------------------
const TextureReplacementData* TextureReplacement::FindReplacement(uint64_t content_hash) const {
  if (auto image = FindTextureMemoryReplacement(content_hash)) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = memory_cache_.find(content_hash);
    if (it == memory_cache_.end()) {
      TextureReplacementData data;
      data.width = image->width;
      data.height = image->height;
      data.pixels = image->rgba;
      data.mips = BuildMips(data.pixels.data(), data.width, data.height);
      it = memory_cache_.emplace(content_hash, std::move(data)).first;
      REXLOG_INFO("TextureReplacement: in-memory replacement {}x{} for {:016x}", image->width,
                  image->height, content_hash);
    }
    return &it->second;
  }
  if (!REXCVAR_GET(texture_replace_enabled)) {
    return nullptr;
  }
  std::filesystem::path path;
  {
    std::unique_lock<std::mutex> lock(cache_mutex_);
    if (!preload_started_) {
      StartPreload();
    }
    // Being decoded by the preload - waiting for it is quicker than starting
    // over.
    if (preload_in_progress_.count(content_hash)) {
      const auto wait_start = std::chrono::steady_clock::now();
      preload_done_.wait(lock, [&]() { return !preload_in_progress_.count(content_hash); });
      REXLOG_INFO("TextureReplacement: waited {} ms for the preload of {:016x}",
                  std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - wait_start)
                      .count(),
                  content_hash);
    }

    auto cached = pixel_cache_.find(content_hash);
    if (cached != pixel_cache_.end()) {
      auto use = pixel_cache_use_.find(content_hash);
      if (use != pixel_cache_use_.end()) {
        use->second.last_use = ++pixel_cache_clock_;
        use->second.used = true;
      }
      // texture_replace_ram_mb also with the background loading (which only
      // adds): the pointers handed out before are used by now.
      TrimPixelCache(content_hash);
      return &cached->second;
    }
    // Known failure — don't retry.
    if (failed_cache_.count(content_hash)) {
      return nullptr;
    }
    auto it = replacements_.find(content_hash);
    if (it == replacements_.end()) {
      failed_cache_.insert(content_hash);
      return nullptr;
    }
    path = it->second;
    preload_in_progress_.insert(content_hash);
  }

  // Not decoded yet (needed before the preload got to it, or it's off).
  const auto load_start = std::chrono::steady_clock::now();
  TextureReplacementData loaded;
  const bool ok = LoadReplacement(content_hash, path, loaded, false) == LoadResult::kLoaded;
  {
    rex::perf::TextureCacheStats& stats = rex::perf::GetTextureCacheStats();
    stats.replacement_decodes.fetch_add(1, std::memory_order_relaxed);
    stats.replacement_decode_ns.fetch_add(
        uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - load_start)
                     .count()),
        std::memory_order_relaxed);
  }
  if (!preload_queue_.empty()) {
    REXLOG_INFO("TextureReplacement: {} was needed before the preload got to it, loaded in {} ms",
                path.filename().string(),
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - load_start)
                    .count());
  }
  const TextureReplacementData* result = nullptr;
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (ok) {
      result = AddToPixelCache(content_hash, std::move(loaded));
      pixel_cache_use_[content_hash].used = true;
      TrimPixelCache(content_hash);
    } else {
      failed_cache_.insert(content_hash);
    }
    preload_in_progress_.erase(content_hash);
  }
  preload_done_.notify_all();
  return result;
}

// ---------------------------------------------------------------------------
// In-memory replacements (from burstlimit's draw_overrides)
// ---------------------------------------------------------------------------
namespace {
std::mutex g_memory_replacements_mutex;
std::unordered_map<uint64_t, std::shared_ptr<const TextureMemoryReplacement>>
    g_memory_replacements;
std::atomic<bool> g_has_memory_replacements{false};

// Xenos 2D tiling (XGAddress2DTiledOffset), as texture_util::GetTiledOffset2D.
uint32_t MemoryReplacementTiledOffset2D(uint32_t x, uint32_t y, uint32_t pitch,
                                        uint32_t bytes_per_block_log2) {
  pitch = (pitch + 31) & ~31u;
  const uint32_t macro = ((x >> 5) + (y >> 5) * (pitch >> 5)) << (bytes_per_block_log2 + 7);
  const uint32_t micro = ((x & 7) + ((y & 0xE) << 2)) << bytes_per_block_log2;
  const uint32_t offset = macro + ((micro & ~0xFu) << 1) + (micro & 0xF) + ((y & 1) << 4);
  return ((offset & ~0x1FFu) << 3) + ((y & 16) << 7) + ((offset & 0x1C0) << 2) +
         (((((y & 8) >> 2) + (x >> 3)) & 3) << 6) + (offset & 0x3F);
}
}  // namespace

void SetTextureMemoryReplacement(uint64_t key, uint32_t width, uint32_t height,
                                 std::vector<uint8_t> rgba) {
  if (!width || !height || rgba.size() != size_t(width) * height * 4) {
    return;
  }
  auto image = std::make_shared<TextureMemoryReplacement>();
  image->width = width;
  image->height = height;
  image->rgba = std::move(rgba);
  std::lock_guard lock(g_memory_replacements_mutex);
  g_memory_replacements.emplace(key, std::move(image));
  g_has_memory_replacements.store(true, std::memory_order_release);
}

std::shared_ptr<const TextureMemoryReplacement> FindTextureMemoryReplacement(uint64_t key) {
  if (!g_has_memory_replacements.load(std::memory_order_acquire)) {
    return nullptr;
  }
  std::lock_guard lock(g_memory_replacements_mutex);
  auto it = g_memory_replacements.find(key);
  return it != g_memory_replacements.end() ? it->second : nullptr;
}

bool HasTextureMemoryReplacements() {
  return g_has_memory_replacements.load(std::memory_order_acquire);
}

uint64_t TiledTexture2DReplacementKey(const uint8_t* blocks, uint32_t width, uint32_t height,
                                      uint32_t bytes_per_block, uint32_t block_texels) {
  if (!blocks || !width || !height || !bytes_per_block || !block_texels ||
      (bytes_per_block & (bytes_per_block - 1))) {
    return 0;
  }
  uint32_t log2 = 0;
  while ((1u << log2) < bytes_per_block) {
    ++log2;
  }
  const uint32_t x_blocks = (width + block_texels - 1) / block_texels;
  const uint32_t y_blocks = (height + block_texels - 1) / block_texels;
  // The tiled base level takes whole 32x32-block tiles.
  const size_t size = size_t((x_blocks + 31) & ~31u) * ((y_blocks + 31) & ~31u) * bytes_per_block;
  std::vector<uint8_t> tiled(size, 0);
  for (uint32_t y = 0; y < y_blocks; ++y) {
    for (uint32_t x = 0; x < x_blocks; ++x) {
      const size_t offset = MemoryReplacementTiledOffset2D(x, y, x_blocks, log2);
      if (offset + bytes_per_block <= size) {
        std::memcpy(tiled.data() + offset,
                    blocks + (size_t(y) * x_blocks + x) * bytes_per_block, bytes_per_block);
      }
    }
  }
  return XXH3_64bits(tiled.data(), tiled.size());
}

}  // namespace rex::graphics
