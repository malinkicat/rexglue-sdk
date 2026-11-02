// DLSS inputs from the scene's depth render target, taken before the HUD is
// drawn: the depth itself as R32_FLOAT, and the camera motion of every pixel -
// where it was in the previous frame, in render target pixels (DLSS's
// convention: previous position minus current, X right, Y down, no jitter).
//
// Only the camera is known, so objects moving by themselves (the fighters,
// their effects) get the motion of what's behind them.

cbuffer XeDlssInputsConstants : register(b0) {
  uint2 xe_dlss_size;
  // 0 - no previous camera (first frame, a cut): no motion.
  uint xe_dlss_has_motion;
  uint xe_dlss_padding;
  // This frame's (unjittered) clip space to the previous frame's, as row
  // vectors: previous = current.x * r[0] + current.y * r[1] + ...
  float4 xe_dlss_reprojection[4];
};

Texture2D<float> xe_dlss_depth_source : register(t0);
RWTexture2D<float> xe_dlss_depth_dest : register(u0);
RWTexture2D<float2> xe_dlss_motion_dest : register(u1);

[numthreads(8, 8, 1)]
void main(uint3 xe_thread_id : SV_DispatchThreadID) {
  [branch] if (any(xe_thread_id.xy >= xe_dlss_size)) {
    return;
  }
  float depth = xe_dlss_depth_source.Load(int3(xe_thread_id.xy, 0));
  xe_dlss_depth_dest[xe_thread_id.xy] = depth;

  float2 motion = float2(0.0, 0.0);
  [branch] if (xe_dlss_has_motion != 0u) {
    float2 size = float2(xe_dlss_size);
    float2 ndc = float2((float(xe_thread_id.x) + 0.5) / size.x * 2.0 - 1.0,
                        1.0 - (float(xe_thread_id.y) + 0.5) / size.y * 2.0);
    float4 previous = ndc.x * xe_dlss_reprojection[0] + ndc.y * xe_dlss_reprojection[1] +
                      depth * xe_dlss_reprojection[2] + xe_dlss_reprojection[3];
    [branch] if (previous.w > 1.0e-6) {
      motion = (previous.xy / previous.w - ndc) * float2(0.5 * size.x, -0.5 * size.y);
    }
  }
  xe_dlss_motion_dest[xe_thread_id.xy] = motion;
}
