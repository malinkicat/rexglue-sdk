// Shows DLSS's inputs in place of its output, to check them (cvar
// dlss_debug_view): 1 - motion vectors, 2 - depth.

cbuffer XeDlssDebugConstants : register(b0) {
  uint2 xe_dlss_debug_size;
  uint xe_dlss_debug_view;
  uint xe_dlss_debug_padding;
};

Texture2D<float2> xe_dlss_debug_motion : register(t0);
Texture2D<float> xe_dlss_debug_depth : register(t1);
RWTexture2D<float4> xe_dlss_debug_dest : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 xe_thread_id : SV_DispatchThreadID) {
  [branch] if (any(xe_thread_id.xy >= xe_dlss_debug_size)) {
    return;
  }
  int3 pixel = int3(xe_thread_id.xy, 0);
  float4 result;
  [branch] if (xe_dlss_debug_view == 1u) {
    // Red - motion to the right, green - down, blue - up or left; brighter =
    // faster (full at 8 pixels per frame).
    float2 motion = xe_dlss_debug_motion.Load(pixel);
    result = float4(saturate(motion.x / 8.0), saturate(motion.y / 8.0),
                    saturate(-(motion.x + motion.y) / 8.0), 1.0);
  } else {
    // Most of the scene is close to the far plane with perspective depth.
    float depth = xe_dlss_debug_depth.Load(pixel);
    result = float4(saturate((1.0 - depth) * 30.0).xxx, 1.0);
  }
  xe_dlss_debug_dest[xe_thread_id.xy] = result;
}
