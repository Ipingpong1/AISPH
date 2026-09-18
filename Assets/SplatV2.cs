// SplatV2.cs — in-engine port of splatRender.splat_render_v2 (ML repo, renderer v2, 2026-09-17).
//
// Subpixel-placed, physically projected SPHERES instead of the legacy 5 px square Gaussian stamp:
//   * continuous centre (no rounding); drawn radius r_px = worldRadius * focal * (H/2) / depth, clamped to
//     [minRadiusPx, maxRadiusPx] for the FOOTPRINT only (depth and volume use the true radius);
//   * a pixel belongs to the footprint iff the distance from its CENTRE (col+.5, row+.5) to the true centre < r_px;
//   * thickness kernel k(u) = sqrt(1-u^2), normalised PER PARTICLE so its weights sum to
//     T = thicknessScale * (4/3) pi r_w r_px_true^2 (VOLUME mode: sum of chord lengths [m] x scale; the k-sum
//     runs over the whole footprint including off-image pixels, so a clipped particle loses that mass, as Python);
//   * per-pixel depth = d - sqrt(r_w^2 - s^2) with s = rho_px * d / (focal*H/2) (paraxial), min-reduce;
//   * velocity / density are weight sums here; the caller divides by thickness (weight-normalised means) and
//     builds mask = thickness > 0 with background depth 0 — exactly the Python finalize.
// Float32 op order follows the torch reference so the parity gate (Helpers/compare_unity_splat_v2.py) holds
// to its tolerances (mask <= 5 px, depth 1e-4, other channels 1e-3 on >= 99.99 %).
// Buffers must be pre-filled by the caller: depthBuf = DepthSentinel, the others 0.
using System;
using UnityEngine;

public static class SplatV2
{
    // splatRender.THICKNESS_SCALE_LEGACY: legacy LR per-particle weight 24.86 / volume weight of one trunk coarse
    // particle (r 0.04144 m) at the training-camera midpoint (4.25 m, fov 45, H 512). Metas ship the exact value.
    public const float ThicknessScaleLegacy = 3.9448388f;
    public const float DepthSentinel = 1e6f;

    public static int Splat(float[] data, int off, int count,
                            Vector3 eye, Vector3 right, Vector3 up, Vector3 fwd, float focal,
                            int H, int W, float worldRadius, float thicknessScale,
                            float minRadiusPx, float maxRadiusPx,
                            float[] depthBuf, float[] thickBuf,
                            float[] velX, float[] velY, float[] velZ, float[] den)
    {
        if (!(worldRadius > 0f)) throw new ArgumentException($"SplatV2: worldRadius must be > 0 (got {worldRadius})");
        float aspect = (float)W / H;
        float halfH = H * 0.5f;
        float pxPerUnit = focal * halfH;                       // pixels per world unit of lateral offset at depth 1
        float rw2 = worldRadius * worldRadius;
        float volK = thicknessScale * (4f / 3f) * (float)Math.PI * worldRadius;
        int drawn = 0;

        for (int pi = 0; pi < count; pi++)
        {
            int b = off + pi * 7;
            float rx = data[b] - eye.x, ry = data[b + 1] - eye.y, rz = data[b + 2] - eye.z;
            float cx = rx * right.x + ry * right.y + rz * right.z;
            float cy = rx * up.x + ry * up.y + rz * up.z;
            float cz = -(rx * fwd.x + ry * fwd.y + rz * fwd.z);   // camera looks -z
            float depth = -cz;
            if (!(depth > 1e-3f)) continue;                        // near-plane cull

            float pxf = (focal * cx / depth / aspect + 1f) * 0.5f * W;
            float pyf = (1f - focal * cy / depth) * 0.5f * H;     // row 0 = top
            float rTrue = worldRadius * focal * halfH / depth;
            float r = rTrue < minRadiusPx ? minRadiusPx : (rTrue > maxRadiusPx ? maxRadiusPx : rTrue);
            if (!(pxf + r > 0f && pxf - r < W && pyf + r > 0f && pyf - r < H)) continue;

            float T = volK * rTrue * rTrue;
            int S = (int)Math.Ceiling(r);
            int x0 = (int)Math.Floor(pxf), y0 = (int)Math.Floor(pyf);
            float r2 = r * r;

            // pass 1: per-particle kernel sum over the WHOLE footprint (off-image pixels included)
            float kSum = 0f;
            for (int oy = -S; oy <= S; oy++)
            {
                float dy = (y0 + oy) + 0.5f - pyf;
                for (int ox = -S; ox <= S; ox++)
                {
                    float dx = (x0 + ox) + 0.5f - pxf;
                    float rho2 = dx * dx + dy * dy;
                    if (rho2 < r2)
                    {
                        float u = 1f - rho2 / r2;
                        kSum += (float)Math.Sqrt(u > 0f ? u : 0f);
                    }
                }
            }
            if (!(kSum > 1e-12f)) continue;

            float wx = data[b + 3], wy = data[b + 4], wz = data[b + 5], dn = data[b + 6];
            float vcx = wx * right.x + wy * right.y + wz * right.z;   // v_world @ R.T (camera space, as legacy)
            float vcy = wx * up.x + wy * up.y + wz * up.z;
            float vcz = -(wx * fwd.x + wy * fwd.y + wz * fwd.z);
            float dScale = depth / pxPerUnit;
            float dScale2 = dScale * dScale;

            // pass 2: deposit
            for (int oy = -S; oy <= S; oy++)
            {
                int ty = y0 + oy;
                float dy = ty + 0.5f - pyf;
                for (int ox = -S; ox <= S; ox++)
                {
                    int tx = x0 + ox;
                    float dx = tx + 0.5f - pxf;
                    float rho2 = dx * dx + dy * dy;
                    if (!(rho2 < r2)) continue;
                    if (tx < 0 || tx >= W || ty < 0 || ty >= H) continue;
                    float u = 1f - rho2 / r2;
                    float k = (float)Math.Sqrt(u > 0f ? u : 0f);
                    float w = T * k / kSum;
                    int i = ty * W + tx;
                    thickBuf[i] += w;
                    float s2 = rho2 * dScale2;
                    float h2 = rw2 - s2;
                    float dPix = depth - (float)Math.Sqrt(h2 > 0f ? h2 : 0f);   // sphere front surface
                    if (dPix < depthBuf[i]) depthBuf[i] = dPix;
                    velX[i] += vcx * w;
                    velY[i] += vcy * w;
                    velZ[i] += vcz * w;
                    den[i] += dn * w;
                }
            }
            drawn++;
        }
        return drawn;
    }
}
