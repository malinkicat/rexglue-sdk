// SMAA 1x (Jorge Jimenez et al., https://github.com/iryoku/smaa, MIT) as the
// swap post effect (swap_post_effect=smaa): three compute passes over the
// gamma-applied guest output, calling SMAA.hlsl's own vertex and pixel shader
// functions per pixel.
//   XE_SMAA_EDGES:   t0 color -> u0 edges (luma edge detection)
//   XE_SMAA_WEIGHTS: t0 edges, t1 AreaTex, t2 SearchTex -> u0 blend weights
//   XE_SMAA_BLEND:   t0 color, t1 blend weights -> u0 output
// s0 linear clamp, s1 point clamp (static samplers).
//
// Build (from this folder, see build_direct_resolve_shaders.cmd):
//   fxc /nologo /T cs_5_1 /E main /O3 /D XE_SMAA_EDGES /Vn smaa_edges_cs
//       /Fh bytecode/d3d12_5_1/smaa_edges_cs.h smaa.cs.hlsl
//   (and XE_SMAA_WEIGHTS -> smaa_weights_cs, XE_SMAA_BLEND -> smaa_blend_cs)

cbuffer xe_smaa_constants : register(b0) {
  uint2 xe_smaa_size;
};

SamplerState xe_smaa_linear : register(s0);
SamplerState xe_smaa_point : register(s1);

#define SMAA_RT_METRICS float4(1.0 / float2(xe_smaa_size), float2(xe_smaa_size))
#define SMAA_PRESET_HIGH 1
#define SMAA_CUSTOM_SL 1
#define SMAATexture2D(tex) Texture2D tex
#define SMAATexturePass2D(tex) tex
#define SMAASampleLevelZero(tex, coord) tex.SampleLevel(xe_smaa_linear, coord, 0)
#define SMAASampleLevelZeroPoint(tex, coord) tex.SampleLevel(xe_smaa_point, coord, 0)
#define SMAASampleLevelZeroOffset(tex, coord, offset) \
  tex.SampleLevel(xe_smaa_linear, coord, 0, offset)
#define SMAASample(tex, coord) tex.SampleLevel(xe_smaa_linear, coord, 0)
#define SMAASamplePoint(tex, coord) tex.SampleLevel(xe_smaa_point, coord, 0)
#define SMAASampleOffset(tex, coord, offset) tex.SampleLevel(xe_smaa_linear, coord, 0, offset)
#define SMAA_FLATTEN [flatten]
#define SMAA_BRANCH [branch]
#define SMAAGather(tex, coord) tex.Gather(xe_smaa_linear, coord, 0)
#ifdef XE_SMAA_EDGES
// The edge detection pixel shader discards pixels without edges; here they
// get no edges instead.
#define discard return float2(0.0, 0.0)
#endif

#include "../../../thirdparty/smaa/SMAA.hlsl"

#ifdef XE_SMAA_EDGES
#undef discard
#endif

#if defined(XE_SMAA_EDGES)
Texture2D xe_smaa_color : register(t0);
RWTexture2D<unorm float4> xe_smaa_dest : register(u0);
#elif defined(XE_SMAA_WEIGHTS)
Texture2D xe_smaa_edges : register(t0);
Texture2D xe_smaa_area : register(t1);
Texture2D xe_smaa_search : register(t2);
RWTexture2D<unorm float4> xe_smaa_dest : register(u0);
#elif defined(XE_SMAA_BLEND)
Texture2D xe_smaa_color : register(t0);
Texture2D xe_smaa_blend : register(t1);
RWTexture2D<unorm float4> xe_smaa_dest : register(u0);
#endif

[numthreads(8, 8, 1)]
void main(uint3 xe_thread_id : SV_DispatchThreadID) {
  if (any(xe_thread_id.xy >= xe_smaa_size)) {
    return;
  }
  float2 texcoord = (float2(xe_thread_id.xy) + 0.5) * SMAA_RT_METRICS.xy;
#if defined(XE_SMAA_EDGES)
  float4 offset[3];
  SMAAEdgeDetectionVS(texcoord, offset);
  float2 edges = SMAALumaEdgeDetectionPS(texcoord, offset, xe_smaa_color);
  xe_smaa_dest[xe_thread_id.xy] = float4(edges, 0.0, 0.0);
#elif defined(XE_SMAA_WEIGHTS)
  float2 pixcoord;
  float4 offset[3];
  SMAABlendingWeightCalculationVS(texcoord, pixcoord, offset);
  xe_smaa_dest[xe_thread_id.xy] =
      SMAABlendingWeightCalculationPS(texcoord, pixcoord, offset, xe_smaa_edges, xe_smaa_area,
                                      xe_smaa_search, float4(0.0, 0.0, 0.0, 0.0));
#elif defined(XE_SMAA_BLEND)
  float4 offset;
  SMAANeighborhoodBlendingVS(texcoord, offset);
  xe_smaa_dest[xe_thread_id.xy] =
      SMAANeighborhoodBlendingPS(texcoord, offset, xe_smaa_color, xe_smaa_blend);
#endif
}
