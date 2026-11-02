/**
 * @file        graphics/pipeline/texture/replacement_alpha.h
 *
 * @brief       Image work of the texture replacement path, on RGBA8 images:
 *              telling a replacement with premultiplied alpha from a
 *              straight one by comparing it with the game's own texture,
 *              turning it back into straight alpha, and downscaling images
 *              too big for the GPU.
 *
 *              The game blends its textures as straight alpha, and so do the
 *              originals a pack is made from. Some upscalers and exporters
 *              write colour premultiplied by alpha instead (and black under
 *              the transparent texels): drawn as they are, every soft edge,
 *              glow, cloud and see-through layer comes out darker
 *              (colour x alpha x alpha), often nearly black.
 *
 *              Header-only and independent of the rest of the SDK, so an
 *              offline test can build it on its own.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace rex::graphics::replacement_image {

// Runs body(begin, end) over the rows [0, count), split across a few threads
// when the image is big (a 4096x4096 replacement decodes and mips about 8x
// faster on 8 threads, which is what the frame waits for).
template <typename Body>
inline void ParallelRows(uint32_t count, uint64_t pixels, Body&& body) {
  const uint32_t threads = std::clamp(std::thread::hardware_concurrency() / 2, 1u, 8u);
  if (threads <= 1 || pixels < uint64_t(512) * 512 || count < threads * 4) {
    body(0u, count);
    return;
  }
  const uint32_t step = (count + threads - 1) / threads;
  std::vector<std::thread> workers;
  for (uint32_t begin = step; begin < count; begin += step) {
    const uint32_t end = std::min(count, begin + step);
    workers.emplace_back([&body, begin, end] { body(begin, end); });
  }
  body(0u, std::min(count, step));
  for (std::thread& worker : workers) {
    worker.join();
  }
}

// Alpha (0-255) of the partly transparent texels the colours are compared on.
constexpr uint32_t kSoftAlphaMin = 16;
constexpr uint32_t kSoftAlphaMax = 239;
// Original texels at or below this alpha count as fully transparent.
constexpr uint32_t kTransparentAlphaMax = 4;
// Converting: colour / alpha from kBlendAlphaHigh up, the original's colour
// below kBlendAlphaLow (colour x alpha keeps too few bits there - and none
// at all under alpha 0), a smooth blend in between.
constexpr float kBlendAlphaLow = 16.0f;
constexpr float kBlendAlphaHigh = 64.0f;
// Alpha this low where the original is fully transparent is noise of the
// upscaler (a faint black veil over what's behind): made 0.
constexpr uint32_t kVeilAlphaMax = 6;
// Straight images with the colour under the transparent texels lost: the
// texels up to this alpha get the original's colour.
constexpr uint32_t kLostColourAlphaMax = 4;

enum class AlphaKind : uint8_t {
  kUnknown,
  kStraight,
  // Colour = straight colour x alpha.
  kPremultiplied,
  // Straight, but black under the transparent texels where the original has
  // colour (bilinear filtering pulls that black into the edges).
  kLostColour,
};

struct AlphaCheck {
  AlphaKind kind = AlphaKind::kUnknown;
  // Decided by the texels the original has fully transparent (the original
  // has too few partly transparent ones).
  bool cutout = false;
  // Partly transparent texels: mean colour difference from the original as
  // straight alpha (a) and as premultiplied (b). Cut-outs: the mean colour
  // under the transparent texels of the original (a) and of the replacement
  // (b).
  float error_a = 0.0f;
  float error_b = 0.0f;
};

inline bool HasTransparency(const uint8_t* rgba, uint64_t texel_count) {
  for (uint64_t i = 0; i < texel_count; ++i) {
    if (rgba[i * 4 + 3] != 255) {
      return true;
    }
  }
  return false;
}

// The share of the partly transparent texels whose colour is brighter than
// their alpha (max(r, g, b) > a + 2): none in a premultiplied image, most in a
// straight one (dark straight images pass too, so this alone doesn't decide).
// -1 without such texels. Big images are sampled every few rows.
inline float PremultipliedViolation(const uint8_t* rgba, uint32_t width, uint32_t height,
                                    uint64_t* soft_count_out = nullptr) {
  const uint32_t row_step =
      std::max(uint32_t(1), uint32_t(uint64_t(width) * height / (uint64_t(4) << 20)));
  uint64_t soft = 0, bright = 0;
  for (uint32_t y = 0; y < height; y += row_step) {
    const uint8_t* texel = rgba + size_t(y) * width * 4;
    for (uint32_t x = 0; x < width; ++x, texel += 4) {
      if (texel[3] < kSoftAlphaMin || texel[3] > kSoftAlphaMax) {
        continue;
      }
      ++soft;
      if (uint32_t(std::max(std::max(texel[0], texel[1]), texel[2])) > uint32_t(texel[3]) + 2) {
        ++bright;
      }
    }
  }
  if (soft_count_out) {
    *soft_count_out = soft;
  }
  return soft ? float(double(bright) / double(soft)) : -1.0f;
}

// Compares a replacement with the original texture it replaces (both RGBA8,
// the original at its own size): the replacement is averaged over the area
// of each original texel. Premultiplied when its colour matches the
// original's colour x alpha clearly better than the original's colour on the
// partly transparent texels; for cut-outs, when it's black where the
// original has colour under its transparent texels (kLostColour instead when
// its own partly transparent texels are straight).
inline AlphaCheck ClassifyAgainstOriginal(const uint8_t* replacement, uint32_t width,
                                          uint32_t height, const uint8_t* original,
                                          uint32_t original_width, uint32_t original_height) {
  AlphaCheck check;
  if (!width || !height || !original_width || !original_height) {
    return check;
  }
  // Up to about 256K original texels.
  uint32_t step = 1;
  while (step < std::max(original_width, original_height) &&
         uint64_t(original_width / step) * (original_height / step) > (uint64_t(256) << 10)) {
    ++step;
  }
  double soft_count = 0, straight_error = 0, premultiplied_error = 0, separation = 0;
  double zero_count = 0, zero_original = 0, zero_replacement = 0;
  for (uint32_t oy = 0; oy < original_height; oy += step) {
    const uint32_t y0 = uint32_t(uint64_t(oy) * height / original_height);
    const uint32_t y1 =
        std::min(height, std::max(uint32_t(uint64_t(oy + 1) * height / original_height), y0 + 1));
    for (uint32_t ox = 0; ox < original_width; ox += step) {
      const uint8_t* o = original + (size_t(oy) * original_width + ox) * 4;
      const uint32_t oa = o[3];
      const bool soft = oa >= kSoftAlphaMin && oa <= kSoftAlphaMax;
      const bool zero = oa <= kTransparentAlphaMax;
      if (!soft && !zero) {
        continue;
      }
      const uint32_t x0 = uint32_t(uint64_t(ox) * width / original_width);
      const uint32_t x1 =
          std::min(width, std::max(uint32_t(uint64_t(ox + 1) * width / original_width), x0 + 1));
      uint64_t sum[3] = {};
      for (uint32_t y = y0; y < y1; ++y) {
        const uint8_t* r = replacement + (size_t(y) * width + x0) * 4;
        for (uint32_t x = x0; x < x1; ++x, r += 4) {
          sum[0] += r[0];
          sum[1] += r[1];
          sum[2] += r[2];
        }
      }
      const double area = double(x1 - x0) * double(y1 - y0);
      if (soft) {
        const double a = oa / 255.0;
        for (uint32_t c = 0; c < 3; ++c) {
          const double r = sum[c] / area;
          straight_error += std::abs(r - o[c]);
          premultiplied_error += std::abs(r - o[c] * a);
          separation += o[c] * (1.0 - a);
        }
        soft_count += 1;
      } else {
        for (uint32_t c = 0; c < 3; ++c) {
          zero_original += o[c];
          zero_replacement += sum[c] / area;
        }
        zero_count += 1;
      }
    }
  }
  if (soft_count >= 64) {
    check.error_a = float(straight_error / (soft_count * 3));
    check.error_b = float(premultiplied_error / (soft_count * 3));
    // Only when the two readings differ enough to tell (alpha near 1 or dark
    // colours make them the same).
    if (separation / (soft_count * 3) >= 6.0) {
      if (premultiplied_error < 0.6 * straight_error) {
        check.kind = AlphaKind::kPremultiplied;
      } else if (straight_error < 0.6 * premultiplied_error) {
        check.kind = AlphaKind::kStraight;
      }
      return check;
    }
  }
  if (zero_count >= 64) {
    const double original_mean = zero_original / (zero_count * 3);
    const double replacement_mean = zero_replacement / (zero_count * 3);
    check.cutout = true;
    check.error_a = float(original_mean);
    check.error_b = float(replacement_mean);
    if (original_mean >= 12.0) {
      if (replacement_mean < std::min(12.0, 0.15 * original_mean)) {
        uint64_t soft_texels = 0;
        const float bright = PremultipliedViolation(replacement, width, height, &soft_texels);
        check.kind = soft_texels >= 64 && bright >= 0.02f ? AlphaKind::kLostColour
                                                           : AlphaKind::kPremultiplied;
      } else {
        check.kind = AlphaKind::kStraight;
      }
    }
  }
  return check;
}

// Bilinear sampling positions of an axis of `in_size` texels stretched over
// `out_size` (texel centres aligned, edges clamped).
struct BilinearTap {
  uint32_t i0, i1;
  float f;
};
inline std::vector<BilinearTap> MakeBilinearTaps(uint32_t out_size, uint32_t in_size) {
  std::vector<BilinearTap> taps(out_size);
  const float scale = float(in_size) / float(out_size);
  for (uint32_t o = 0; o < out_size; ++o) {
    float s = (float(o) + 0.5f) * scale - 0.5f;
    s = std::clamp(s, 0.0f, float(in_size - 1));
    const uint32_t i0 = std::min(uint32_t(s), in_size - 1);
    taps[o] = {i0, std::min(i0 + 1, in_size - 1), s - float(i0)};
  }
  return taps;
}

// Turns a replacement into straight alpha in place:
//   kPremultiplied: colour / alpha, the original's colour (bilinear) under low
//     alpha, blended in between;
//   kLostColour: the original's colour only under (nearly) fully transparent
//     texels.
// Alpha up to kVeilAlphaMax where the original is fully transparent becomes
// 0. Without the original (null), a premultiplied image's low-alpha texels
// are filled from the alpha-weighted colour around them instead.
inline void RestoreStraightAlpha(uint8_t* rgba, uint32_t width, uint32_t height, AlphaKind kind,
                                 const uint8_t* original, uint32_t original_width,
                                 uint32_t original_height) {
  if (kind != AlphaKind::kPremultiplied && kind != AlphaKind::kLostColour) {
    return;
  }
  if (!original && kind == AlphaKind::kLostColour) {
    return;
  }
  std::vector<BilinearTap> taps_x, taps_y;
  if (original) {
    taps_x = MakeBilinearTaps(width, original_width);
    taps_y = MakeBilinearTaps(height, original_height);
  }

  // Without the original: the colour around low-alpha texels - levels of the
  // average premultiplied colour and alpha, each half the size of the one
  // before (level 1 first).
  struct Level {
    uint32_t width, height;
    std::vector<float> texels;  // r, g, b (premultiplied), a - 0-255
  };
  std::vector<Level> levels;
  if (!original) {
    uint32_t source_width = width, source_height = height;
    while (source_width > 1 || source_height > 1) {
      Level level;
      level.width = std::max(source_width >> 1, uint32_t(1));
      level.height = std::max(source_height >> 1, uint32_t(1));
      level.texels.resize(size_t(level.width) * level.height * 4);
      const Level* previous = levels.empty() ? nullptr : &levels.back();
      for (uint32_t y = 0; y < level.height; ++y) {
        for (uint32_t x = 0; x < level.width; ++x) {
          float sum[4] = {};
          for (uint32_t dy = 0; dy < 2; ++dy) {
            for (uint32_t dx = 0; dx < 2; ++dx) {
              const uint32_t sx = std::min(x * 2 + dx, source_width - 1);
              const uint32_t sy = std::min(y * 2 + dy, source_height - 1);
              if (previous) {
                const float* s = &previous->texels[(size_t(sy) * source_width + sx) * 4];
                for (uint32_t c = 0; c < 4; ++c) {
                  sum[c] += s[c];
                }
              } else {
                const uint8_t* s = rgba + (size_t(sy) * width + sx) * 4;
                for (uint32_t c = 0; c < 4; ++c) {
                  sum[c] += s[c];
                }
              }
            }
          }
          float* d = &level.texels[(size_t(y) * level.width + x) * 4];
          for (uint32_t c = 0; c < 4; ++c) {
            d[c] = sum[c] * 0.25f;
          }
        }
      }
      source_width = level.width;
      source_height = level.height;
      levels.push_back(std::move(level));
    }
  }

  ParallelRows(height, uint64_t(width) * height, [&](uint32_t y_begin, uint32_t y_end) {
    for (uint32_t y = y_begin; y < y_end; ++y) {
      uint8_t* texel = rgba + size_t(y) * width * 4;
      for (uint32_t x = 0; x < width; ++x, texel += 4) {
        const uint32_t a = texel[3];
        float straight[3];
        float weight;
        if (kind == AlphaKind::kLostColour) {
          if (a > kLostColourAlphaMax && a > kVeilAlphaMax) {
            continue;
          }
          for (uint32_t c = 0; c < 3; ++c) {
            straight[c] = float(texel[c]);
          }
          weight = a > kLostColourAlphaMax ? 1.0f : 0.0f;
        } else {
          for (uint32_t c = 0; c < 3; ++c) {
            straight[c] = a ? std::min(255.0f, float(texel[c]) * 255.0f / float(a)) : 0.0f;
          }
          const float t =
              std::clamp((float(a) - kBlendAlphaLow) / (kBlendAlphaHigh - kBlendAlphaLow), 0.0f,
                         1.0f);
          weight = t * t * (3.0f - 2.0f * t);
        }
        float fill[3] = {0.0f, 0.0f, 0.0f};
        bool transparent_in_original = false;
        if (original) {
          if (weight < 1.0f || a <= kVeilAlphaMax) {
            const BilinearTap& tx = taps_x[x];
            const BilinearTap& ty = taps_y[y];
            const uint8_t* o00 = original + (size_t(ty.i0) * original_width + tx.i0) * 4;
            const uint8_t* o01 = original + (size_t(ty.i0) * original_width + tx.i1) * 4;
            const uint8_t* o10 = original + (size_t(ty.i1) * original_width + tx.i0) * 4;
            const uint8_t* o11 = original + (size_t(ty.i1) * original_width + tx.i1) * 4;
            float sampled[4];
            for (uint32_t c = 0; c < 4; ++c) {
              const float top = float(o00[c]) + (float(o01[c]) - float(o00[c])) * tx.f;
              const float bottom = float(o10[c]) + (float(o11[c]) - float(o10[c])) * tx.f;
              sampled[c] = top + (bottom - top) * ty.f;
            }
            for (uint32_t c = 0; c < 3; ++c) {
              fill[c] = sampled[c];
            }
            transparent_in_original = sampled[3] < 0.5f;
          }
        } else if (weight < 1.0f) {
          for (const Level& level : levels) {
            const uint32_t shift = uint32_t(&level - levels.data()) + 1;
            const uint32_t lx = std::min(x >> shift, level.width - 1);
            const uint32_t ly = std::min(y >> shift, level.height - 1);
            const float* l = &level.texels[(size_t(ly) * level.width + lx) * 4];
            if (l[3] >= 8.0f || &level == &levels.back()) {
              if (l[3] > 0.0f) {
                for (uint32_t c = 0; c < 3; ++c) {
                  fill[c] = std::min(255.0f, l[c] * 255.0f / l[3]);
                }
              }
              break;
            }
          }
        }
        for (uint32_t c = 0; c < 3; ++c) {
          const float value = straight[c] * weight + fill[c] * (1.0f - weight);
          texel[c] = uint8_t(std::clamp(value + 0.5f, 0.0f, 255.0f));
        }
        if (transparent_in_original && a <= kVeilAlphaMax) {
          texel[3] = 0;
        }
      }
    }
  });
}

// The size an image gets when its longer side is fitted to max_extent
// (aspect ratio kept); unchanged when it fits.
inline void FitExtent(uint32_t& width, uint32_t& height, uint32_t max_extent) {
  if (!max_extent || (width <= max_extent && height <= max_extent)) {
    return;
  }
  if (width >= height) {
    height = std::max(uint32_t(1),
                      uint32_t((uint64_t(height) * max_extent + width / 2) / width));
    width = max_extent;
  } else {
    width = std::max(uint32_t(1),
                     uint32_t((uint64_t(width) * max_extent + height / 2) / height));
    height = max_extent;
  }
}

// Area-average downscale of an RGBA8 image (every source texel weighted by
// how much of it each destination texel covers), colour weighted by alpha so
// transparent texels don't darken the edges - like the mips. Exact for whole
// ratios (a 2x reduction is the 2x2 box of the mips).
inline std::vector<uint8_t> DownscaleArea(const uint8_t* rgba, uint32_t width, uint32_t height,
                                          uint32_t new_width, uint32_t new_height) {
  struct Tap {
    uint32_t index;
    float weight;
  };
  auto make_taps = [](uint32_t out_size, uint32_t in_size, std::vector<uint32_t>& first,
                      std::vector<Tap>& taps) {
    first.assign(out_size + 1, 0);
    taps.clear();
    const double scale = double(in_size) / double(out_size);
    for (uint32_t o = 0; o < out_size; ++o) {
      first[o] = uint32_t(taps.size());
      const double start = o * scale;
      const double end = std::min(double(in_size), (o + 1) * scale);
      for (uint32_t i = uint32_t(start); i < in_size && double(i) < end; ++i) {
        const double overlap = std::min(end, double(i) + 1.0) - std::max(start, double(i));
        if (overlap > 0.0) {
          taps.push_back({i, float(overlap / (end - start))});
        }
      }
    }
    first[out_size] = uint32_t(taps.size());
  };
  std::vector<uint32_t> first_x, first_y;
  std::vector<Tap> taps_x, taps_y;
  make_taps(new_width, width, first_x, taps_x);
  make_taps(new_height, height, first_y, taps_y);

  std::vector<uint8_t> result(size_t(new_width) * new_height * 4);
  ParallelRows(new_height, uint64_t(width) * height, [&](uint32_t y_begin, uint32_t y_end) {
    // Per destination texel: colour x alpha, alpha, plain colour.
    std::vector<float> sums(size_t(new_width) * 7);
    for (uint32_t oy = y_begin; oy < y_end; ++oy) {
      std::fill(sums.begin(), sums.end(), 0.0f);
      for (uint32_t ty = first_y[oy]; ty < first_y[oy + 1]; ++ty) {
        const uint8_t* row = rgba + size_t(taps_y[ty].index) * width * 4;
        const float wy = taps_y[ty].weight;
        for (uint32_t ox = 0; ox < new_width; ++ox) {
          float* s = &sums[size_t(ox) * 7];
          for (uint32_t tx = first_x[ox]; tx < first_x[ox + 1]; ++tx) {
            const uint8_t* texel = row + size_t(taps_x[tx].index) * 4;
            const float w = wy * taps_x[tx].weight;
            const float wa = w * texel[3];
            s[0] += wa * texel[0];
            s[1] += wa * texel[1];
            s[2] += wa * texel[2];
            s[3] += wa;
            s[4] += w * texel[0];
            s[5] += w * texel[1];
            s[6] += w * texel[2];
          }
        }
      }
      uint8_t* out = result.data() + size_t(oy) * new_width * 4;
      for (uint32_t ox = 0; ox < new_width; ++ox, out += 4) {
        const float* s = &sums[size_t(ox) * 7];
        for (uint32_t c = 0; c < 3; ++c) {
          const float value = s[3] > 0.001f ? s[c] / s[3] : s[4 + c];
          out[c] = uint8_t(std::clamp(value + 0.5f, 0.0f, 255.0f));
        }
        out[3] = uint8_t(std::clamp(s[3] + 0.5f, 0.0f, 255.0f));
      }
    }
  });
  return result;
}

}  // namespace rex::graphics::replacement_image
