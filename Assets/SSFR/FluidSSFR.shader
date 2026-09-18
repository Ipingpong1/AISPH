// FluidSSFR.shader — screen-space fluid shading, a faithful HLSL port of the ML repo's
// fluid_shader.py (van der Laan et al. 2009 / Green 2010 style; see Claude/renderer_spec.md there).
//
// Input texture layout (RGBAFloat): R = depth (raw view-space D=-z_cam, world units, background 0),
// G = thickness (raw accumulated splat weight, >=0), B = alpha (binary occupancy).
// Pass 0: foreground-masked Gaussian depth smoothing (fluid_shader._masked_blur).
// Pass 1: shade — back-project, silhouette-aware normals, Beer-Lambert, screen-space refraction,
//         Schlick Fresnel + sky reflection, Blinn-Phong specular, composite over procedural bg.
//
// Coordinate notes: the viewer uploads tensors flipped to Unity's bottom-up textures, so uv.y=1 is
// the image TOP; ndc = uv*2-1 reproduces splatRender's projection (focal=1/tan(fov/2), aspect=1).
Shader "Hidden/FluidSSFR"
{
    Properties
    {
        _MainTex ("Fields (R=depth, G=thickness, B=alpha)", 2D) = "black" {}
        _FoamTex ("Foam density (R, screen space, FoamLayer)", 2D) = "black" {}
        _FoamOn ("Foam on", Float) = 0
        _FoamK ("Foam coverage k (1-exp(-k D))", Float) = 0.8
        _FoamColor ("Foam color", Color) = (0.96, 0.98, 1.0, 1)
    }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        // ---- Pass 0: masked depth smooth (normalize by blurred mask so bg-0 doesn't bleed).
        //      _BilateralRangeSigma > 0 switches to one iteration of render_compare.py's
        //      masked_bilateral_depth: weight = spatial-Gaussian * range-Gaussian(depth similarity)
        //      * neighbor-is-foreground, radius int(2*sigma_s), bg pixels pass through untouched.
        //      Iterate by blitting this pass repeatedly (the reference runs 4 iterations).
        //      Border taps clamp instead of the reference's torch.roll wraparound. ----
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_TexelSize;
            float _BlurSigma;
            float _BilateralRangeSigma;   // world-space depth units; 0 = plain Gaussian

            float4 frag(v2f_img i) : SV_Target
            {
                float4 c = tex2D(_MainTex, i.uv);
                if (_BlurSigma <= 0.0) return c;
                bool bilateral = _BilateralRangeSigma > 0.0;
                // Reference: d = where(a > 0.5, num/den, d) — bg depth is never rewritten.
                if (bilateral && c.b < 0.5) return c;
                int r = bilateral ? clamp((int)(2.0 * _BlurSigma), 1, 8)
                                  : clamp((int)(3.0 * _BlurSigma), 1, 8);
                float inv2s2 = 1.0 / (2.0 * _BlurSigma * _BlurSigma);
                float invR2 = bilateral ? 1.0 / (2.0 * _BilateralRangeSigma * _BilateralRangeSigma) : 0.0;
                float num = 0.0, den = 0.0;
                for (int dy = -r; dy <= r; dy++)
                {
                    for (int dx = -r; dx <= r; dx++)
                    {
                        float w = exp(-(dx * dx + dy * dy) * inv2s2);
                        float4 s = tex2D(_MainTex, i.uv + float2(dx, dy) * _MainTex_TexelSize.xy);
                        if (bilateral)
                        {
                            float dd = s.r - c.r;
                            w *= exp(-dd * dd * invR2);
                        }
                        num += w * s.r * s.b;
                        den += w * s.b;
                    }
                }
                return float4(num / max(den, 1e-6), c.g, c.b, 1.0);
            }
            ENDCG
        }

        // ---- Pass 1: shade ----
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_TexelSize;
            sampler2D _FoamTex;
            float _FoamOn, _FoamK;
            float3 _FoamColor;
            float _Focal, _KThick, _RefrStrength, _F0, _Shininess, _Ks, _FrMax;

            // 059 whitewater layer: density -> coverage, lerp towards the foam colour (before gamma).
            float3 foamOver(float3 c, float2 uv)
            {
                if (_FoamOn < 0.5) return c;
                float fa = 1.0 - exp(-_FoamK * tex2D(_FoamTex, uv).r);
                return lerp(c, _FoamColor, saturate(fa));
            }
            float3 _AbsorbSigma, _LightDir, _SkyTop, _SkyHor, _FloorA, _FloorB, _BodyTint, _SpecTint;

            // Sky gradient above a horizon at 52% height, perspective checker floor below
            // (fluid_shader._background; ybg = 0 at image top).
            float3 background(float2 uv)
            {
                float ybg = 1.0 - uv.y;
                const float horizon = 0.52;
                float t = saturate(ybg / horizon);
                float3 sky = lerp(_SkyTop, _SkyHor, t);
                float fy = saturate((ybg - horizon) / (1.0 - horizon));
                // Perspective row density: fastest change at the horizon (thin distant rows),
                // slowest at the bottom (thick near rows).
                float persp = 1.0 - (1.0 - fy) * (1.0 - fy);
                float chk = fmod(floor(uv.x * 18.0) + floor(persp * 26.0), 2.0);
                float3 flo = lerp(_FloorB, _FloorA, chk);
                return ybg >= horizon ? flo : sky;
            }

            // Environment approximation: ray elevation -> sky gradient, below horizon -> dark floor
            // (fluid_shader._sky_lookup).
            float3 skyLookup(float3 dir)
            {
                float ey = clamp(dir.y, -1.0, 1.0);
                float3 sky = lerp(_SkyHor, _SkyTop, saturate(ey));
                return ey >= 0.0 ? sky : _FloorB;
            }

            // Back-project a pixel to view space: X = ndc_x*D/focal, Y = ndc_y*D/focal, Z = -D (aspect=1).
            float3 viewPos(float2 uv)
            {
                float D = max(tex2D(_MainTex, uv).r, 1e-4);
                float2 ndc = uv * 2.0 - 1.0;
                return float3(ndc.x * D / _Focal, ndc.y * D / _Focal, -D);
            }

            float4 frag(v2f_img i) : SV_Target
            {
                float4 f = tex2D(_MainTex, i.uv);
                float3 bg = background(i.uv);
                if (f.b < 0.5)
                {
                    float3 bgc = foamOver(bg, i.uv);
                    #if !defined(UNITY_COLORSPACE_GAMMA)
                    bgc = GammaToLinearSpace(bgc);
                    #endif
                    return float4(bgc, 1.0);
                }

                float2 tx = _MainTex_TexelSize.xy;
                float3 P = viewPos(i.uv);
                float D = -P.z;

                // Silhouette-aware normals: per axis take the one-sided difference with the
                // smaller depth step so normals never straddle the silhouette.
                float Dxf = abs(max(tex2D(_MainTex, i.uv + float2(tx.x, 0)).r, 1e-4) - D);
                float Dxb = abs(D - max(tex2D(_MainTex, i.uv - float2(tx.x, 0)).r, 1e-4));
                float Dyf = abs(max(tex2D(_MainTex, i.uv + float2(0, tx.y)).r, 1e-4) - D);
                float Dyb = abs(D - max(tex2D(_MainTex, i.uv - float2(0, tx.y)).r, 1e-4));
                float3 dPx = (Dxf <= Dxb) ? viewPos(i.uv + float2(tx.x, 0)) - P : P - viewPos(i.uv - float2(tx.x, 0));
                float3 dPy = (Dyf <= Dyb) ? viewPos(i.uv + float2(0, tx.y)) - P : P - viewPos(i.uv - float2(0, tx.y));
                float3 Nc = cross(dPx, dPy);
                float3 N = Nc / (length(Nc) + 1e-8);
                float3 V = normalize(-P);
                if (dot(N, V) < 0.0) N = -N;
                float NdotV = saturate(dot(N, V));

                // Beer-Lambert absorption through the fluid thickness.
                float thick = max(f.g, 0.0);
                float3 T = exp(-_AbsorbSigma * _KThick * thick);

                // Screen-space refraction: offset the background lookup along the normal.
                float thn = saturate(thick * _KThick);
                float2 duv = N.xy * _RefrStrength * thn * tx;
                float3 refr = background(clamp(i.uv + duv, 0.0, 1.0));
                refr = refr * T + _BodyTint * (1.0 - T);

                // Schlick Fresnel + environment reflection. _FrMax caps how much a grazing angle
                // can flip the surface over to pure environment reflection — at 1.0 (water) this
                // is a no-op; lower values (blood) stop thin/foamy silhouettes from flashing white.
                float Fr = _F0 + (1.0 - _F0) * pow(1.0 - NdotV, 5.0);
                Fr = min(Fr, _FrMax);
                float3 refl = skyLookup(reflect(-V, N));

                // Blinn-Phong specular (light defined in view space). _SpecTint colors the
                // highlight (white for water; a warm dim tint for blood so glints don't read white).
                float3 L = normalize(_LightDir);
                float3 Hh = normalize(L + V);
                float3 spec = _Ks * pow(saturate(dot(N, Hh)), _Shininess) * _SpecTint;

                float3 surface = refr * (1.0 - Fr) + refl * Fr + spec;
                float3 outc = foamOver(saturate(surface), i.uv);
                // The reference pipeline's colors are display-ready (sRGB); in a Linear-space
                // project convert so the displayed/encoded value matches the reference exactly.
                #if !defined(UNITY_COLORSPACE_GAMMA)
                outc = GammaToLinearSpace(outc);
                #endif
                return float4(outc, 1.0);
            }
            ENDCG
        }
    }
}
