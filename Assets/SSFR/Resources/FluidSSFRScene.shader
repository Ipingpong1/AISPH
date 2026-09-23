// FluidSSFRScene.shader — scene-space variant of FluidSSFR: shades the neural fluid fields
// against the REAL Unity scene instead of the procedural floor/sky box.
//
// URP 17 (Render Graph) constraint: _CameraOpaqueTexture/_CameraDepthTexture are transient —
// they are only bound while the camera is rendering, so anything that samples them must run
// in-render, on the composite quad. Everything scene-INDEPENDENT is shaded at the model's
// native 512² during LateUpdate (Graphics.Blit, passes 0-2), packed premultiplied-by-alpha
// so the quad can sample it bilinearly, and the final combine happens per screen pixel:
//
//   out.rgb = sceneColor(screenUV + refractionOffset) * M + C          (premultiplied)
//   passes 0-2 (Blit only; the LightMode tag keeps URP from drawing them on the quad):
//     0 "SHADE_C": C = bodyTint*(1-T)*(1-Fr) + envReflection*Fr + specular   → (C, alpha)
//     1 "SHADE_M": M = T*(1-Fr)  (Beer-Lambert transmission × non-reflected) → (M, depth)
//     2 "SHADE_N": refraction offset = N.xy * strength * saturate(thick*k)   → (duv, 0, 0)
//   pass 3 "COMPOSITE" (camera-child quad, transparent queue): reconstructs the per-pixel
//     view ray in world space, maps it into the model sub-window (the splat used a square
//     frustum covering the whole camera frustum), samples the packs, occlusion-tests against
//     _CameraDepthTexture, refracts _CameraOpaqueTexture, and alpha-composites.
//
// Normal reconstruction runs in the splat camera's view space (sim units); the components
// transfer to world space through the camera basis (_CamRightWS/_CamUpWS/_CamFwdWS).
Shader "Hidden/FluidSSFRScene"
{
    Properties
    {
        _MainTex ("Fields (R=depth sim units, G=thickness, B=alpha)", 2D) = "black" {}
        _EnvCube ("Reflection cubemap", Cube) = "" {}
    }

    CGINCLUDE
    #include "UnityCG.cginc"

    sampler2D _MainTex;
    float4 _MainTex_TexelSize;
    samplerCUBE _EnvCube;
    float _UseEnvCube;

    float _FocalM;
    float _WinAspect;   // model window W/H (067; unset/0 = the square window of before)
    float3 _CamRightWS, _CamUpWS, _CamFwdWS;
    float _KThick, _RefrStrength, _F0, _Shininess, _Ks, _FrMax;
    float3 _AbsorbSigma, _BodyTint, _SpecTint;
    float3 _LightDirWS, _LightColor;
    float3 _SkyTopFallback, _SkyHorFallback, _GroundFallback;

    // Back-project a field texel to the splat camera's view space (sim units, aspect 1).
    float3 viewPos(float2 uv)
    {
        float D = max(tex2D(_MainTex, uv).r, 1e-4);
        float2 ndc = uv * 2.0 - 1.0;
        float wa = _WinAspect > 0.0 ? _WinAspect : 1.0;
        return float3(ndc.x * wa * D / _FocalM, ndc.y * D / _FocalM, -D);
    }

    // Environment for reflections: real probe/skybox cubemap, else a procedural sky.
    float3 envLookup(float3 dir)
    {
        if (_UseEnvCube > 0.5)
            return texCUBElod(_EnvCube, float4(dir, 1.0)).rgb;
        float ey = clamp(dir.y, -1.0, 1.0);
        float3 sky = lerp(_SkyHorFallback, _SkyTopFallback, saturate(ey));
        return ey >= 0.0 ? sky : _GroundFallback;
    }

    struct ShadeOut
    {
        float3 C;      // scene-independent surface color (premultiplied by alpha)
        float3 M;      // per-channel multiplier on the refracted scene color
        float2 duv;    // refraction offset in screen texels
        float  D;      // fluid front-surface depth, sim units (eye-z along camera fwd)
        float  a;      // coverage
    };

    ShadeOut Shade(float2 uv)
    {
        ShadeOut o = (ShadeOut)0;
        float4 f = tex2D(_MainTex, uv);
        if (f.b < 0.5) return o;

        float2 tx = _MainTex_TexelSize.xy;
        float3 P = viewPos(uv);
        float D = -P.z;

        // Silhouette-aware normals (identical scheme to FluidSSFR).
        float Dxf = abs(max(tex2D(_MainTex, uv + float2(tx.x, 0)).r, 1e-4) - D);
        float Dxb = abs(D - max(tex2D(_MainTex, uv - float2(tx.x, 0)).r, 1e-4));
        float Dyf = abs(max(tex2D(_MainTex, uv + float2(0, tx.y)).r, 1e-4) - D);
        float Dyb = abs(D - max(tex2D(_MainTex, uv - float2(0, tx.y)).r, 1e-4));
        float3 dPx = (Dxf <= Dxb) ? viewPos(uv + float2(tx.x, 0)) - P : P - viewPos(uv - float2(tx.x, 0));
        float3 dPy = (Dyf <= Dyb) ? viewPos(uv + float2(0, tx.y)) - P : P - viewPos(uv - float2(0, tx.y));
        float3 Nc = cross(dPx, dPy);
        float3 N = Nc / (length(Nc) + 1e-8);
        float3 V = normalize(-P);
        if (dot(N, V) < 0.0) N = -N;
        float NdotV = saturate(dot(N, V));

        // Beer-Lambert transmission through the fluid thickness.
        float thick = max(f.g, 0.0);
        float3 T = exp(-_AbsorbSigma * _KThick * thick);

        // World-space normal / view ray for environment reflection + specular.
        float2 slope = (uv * 2.0 - 1.0) / _FocalM;
        slope.x *= _WinAspect > 0.0 ? _WinAspect : 1.0;
        float3 Nw = _CamRightWS * N.x + _CamUpWS * N.y - _CamFwdWS * N.z;
        float3 rd = normalize(_CamRightWS * slope.x + _CamUpWS * slope.y + _CamFwdWS);

        float Fr = _F0 + (1.0 - _F0) * pow(1.0 - NdotV, 5.0);
        Fr = min(Fr, _FrMax);
        float3 refl = envLookup(reflect(rd, Nw));

        float3 L = normalize(_LightDirWS);
        float3 Hh = normalize(L - rd);
        float3 spec = _Ks * pow(saturate(dot(Nw, Hh)), _Shininess) * _SpecTint * _LightColor;

        float thn = saturate(thick * _KThick);
        o.C = max(_BodyTint * (1.0 - T) * (1.0 - Fr) + refl * Fr + spec, 0.0);
        o.M = T * (1.0 - Fr);
        o.duv = N.xy * _RefrStrength * thn;
        o.D = D;
        o.a = 1.0;
        return o;
    }
    ENDCG

    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" "IgnoreProjector"="True" }

        // ---- Pass 0: scene-independent surface color + coverage (Blit only) ----
        Pass
        {
            Name "SHADE_C"
            Tags { "LightMode" = "FluidSceneShade" }
            Cull Off ZWrite Off ZTest Always
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            float4 frag(v2f_img i) : SV_Target
            {
                ShadeOut s = Shade(i.uv);
                return float4(s.C * s.a, s.a);
            }
            ENDCG
        }

        // ---- Pass 1: scene-color multiplier + fluid depth (Blit only) ----
        Pass
        {
            Name "SHADE_M"
            Tags { "LightMode" = "FluidSceneShade" }
            Cull Off ZWrite Off ZTest Always
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            float4 frag(v2f_img i) : SV_Target
            {
                ShadeOut s = Shade(i.uv);
                return float4(s.M * s.a, s.D * s.a);
            }
            ENDCG
        }

        // ---- Pass 2: refraction offset (Blit only) ----
        Pass
        {
            Name "SHADE_N"
            Tags { "LightMode" = "FluidSceneShade" }
            Cull Off ZWrite Off ZTest Always
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            float4 frag(v2f_img i) : SV_Target
            {
                ShadeOut s = Shade(i.uv);
                return float4(s.duv * s.a, 0.0, s.a);
            }
            ENDCG
        }

        // ---- Pass 3: composite into the camera frame (drawn on the quad, in-render) ----
        Pass
        {
            Name "COMPOSITE"
            Cull Off ZWrite Off ZTest Always
            Blend One OneMinusSrcAlpha

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            sampler2D _CTex, _MTex, _NTex;
            sampler2D _CameraOpaqueTexture;
            float4 _CameraOpaqueTexture_TexelSize;
            sampler2D _CameraDepthTexture;
            float _SimScale;
            float _DepthBias;   // world-units slack on the occlusion test (0 = unbiased)

            struct appdata { float4 vertex : POSITION; };
            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 ray : TEXCOORD0;   // world-space ray from camera through this vertex
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.ray = mul(unity_ObjectToWorld, v.vertex).xyz - _WorldSpaceCameraPos;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                // model sub-window UV from the world-space view ray (platform-flip proof)
                float df = dot(i.ray, _CamFwdWS);
                clip(df - 1e-6);
                float2 slope = float2(dot(i.ray, _CamRightWS), dot(i.ray, _CamUpWS)) / df;
                float2 uvm = slope * _FocalM * 0.5;
                uvm.x /= _WinAspect > 0.0 ? _WinAspect : 1.0;
                uvm += 0.5;
                clip(uvm);
                clip(1.0 - uvm);

                float4 c0 = tex2D(_CTex, uvm);
                float a = c0.a;
                clip(a - 0.004);
                float inva = 1.0 / a;

                // occlusion by scene geometry, per screen pixel (both eye-z along camera fwd);
                // URP convention: screen UV = pixel coords / screen size (no flip)
                float2 suv = i.pos.xy / _ScreenParams.xy;
                float4 c1 = tex2D(_MTex, uvm);
                float worldD = c1.a * inva * _SimScale;
                float sceneEye = LinearEyeDepth(tex2D(_CameraDepthTexture, suv).r);
                clip(sceneEye - worldD + _DepthBias);

                // refracted scene color; offset stored in screen texels, alpha-normalized
                float2 duv = tex2D(_NTex, uvm).rg * inva * _CameraOpaqueTexture_TexelSize.xy;
                float3 scene = tex2D(_CameraOpaqueTexture, saturate(suv + duv)).rgb;

                return float4(scene * c1.rgb + c0.rgb, a);   // premultiplied
            }
            ENDCG
        }
    }
}
