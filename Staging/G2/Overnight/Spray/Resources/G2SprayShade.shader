// G2SprayShade.shader — G2 (overnight 2026-09-30): shades the learned-spray droplets with the SAME lighting model as
// FluidSSFRScene's Shade() (Beer-Lambert through the thickness, Schlick Fresnel capped at _FrMax, environment
// reflection from the probe / fallback sky, Blinn-Phong specular, body tint), with ONE difference: the normal is the
// droplet sphere's analytic (paraxial) normal rasterised on the CPU (_NrmTex, view space), not the depth-derivative
// normal. At 1-2 model-window pixels a droplet has no usable depth derivatives: FluidSSFRScene's scheme then returns
// N = 0 (Fresnel 1, a sky-coloured dot). The uniforms are FluidSceneMVP.SetShadeParams' names, so the look knobs
// (kThick, ks, shininess, env, light) are shared with the bulk. Keep the lighting in sync with FluidSSFRScene.shader.
//   pass 0: C (premultiplied by coverage) -> (C, a);   pass 1: M (premultiplied) + depth -> (M a, D a)
Shader "Hidden/G2SprayShade"
{
    Properties
    {
        _MainTex ("Droplet field (R depth, G thickness, B alpha)", 2D) = "black" {}
        _NrmTex ("Droplet normal (view space xyz)", 2D) = "black" {}
        _EnvCube ("Reflection cubemap", Cube) = "" {}
    }

    CGINCLUDE
    #include "UnityCG.cginc"

    sampler2D _MainTex, _NrmTex;
    samplerCUBE _EnvCube;
    float _UseEnvCube;
    float _FocalM, _WinAspect;
    float3 _CamRightWS, _CamUpWS, _CamFwdWS;
    float _KThick, _F0, _Shininess, _Ks, _FrMax;
    float3 _AbsorbSigma, _BodyTint, _SpecTint;
    float3 _LightDirWS, _LightColor;
    float3 _SkyTopFallback, _SkyHorFallback, _GroundFallback;

    float3 envLookup(float3 dir)
    {
        if (_UseEnvCube > 0.5)
            return texCUBElod(_EnvCube, float4(dir, 1.0)).rgb;
        float ey = clamp(dir.y, -1.0, 1.0);
        float3 sky = lerp(_SkyHorFallback, _SkyTopFallback, saturate(ey));
        return ey >= 0.0 ? sky : _GroundFallback;
    }

    struct ShadeOut { float3 C; float3 M; float D; float a; };

    ShadeOut Shade(float2 uv)
    {
        ShadeOut o = (ShadeOut)0;
        float4 f = tex2D(_MainTex, uv);
        if (f.b < 0.5) return o;
        float D = max(f.r, 1e-4);
        float2 ndc = uv * 2.0 - 1.0;
        float wa = _WinAspect > 0.0 ? _WinAspect : 1.0;
        float3 P = float3(ndc.x * wa * D / _FocalM, ndc.y * D / _FocalM, -D);

        float3 N = tex2D(_NrmTex, uv).xyz;
        N = N / (length(N) + 1e-8);
        float3 V = normalize(-P);
        if (dot(N, V) < 0.0) N = -N;
        float NdotV = saturate(dot(N, V));

        float thick = max(f.g, 0.0);
        float3 T = exp(-_AbsorbSigma * _KThick * thick);

        float2 slope = ndc / _FocalM;
        slope.x *= wa;
        float3 Nw = _CamRightWS * N.x + _CamUpWS * N.y - _CamFwdWS * N.z;
        float3 rd = normalize(_CamRightWS * slope.x + _CamUpWS * slope.y + _CamFwdWS);

        float Fr = _F0 + (1.0 - _F0) * pow(1.0 - NdotV, 5.0);
        Fr = min(Fr, _FrMax);
        float3 refl = envLookup(reflect(rd, Nw));

        float3 L = normalize(_LightDirWS);
        float3 Hh = normalize(L - rd);
        float3 spec = _Ks * pow(saturate(dot(Nw, Hh)), _Shininess) * _SpecTint * _LightColor;

        o.C = max(_BodyTint * (1.0 - T) * (1.0 - Fr) + refl * Fr + spec, 0.0);
        o.M = T * (1.0 - Fr);
        o.D = D;
        o.a = 1.0;
        return o;
    }
    ENDCG

    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" "IgnoreProjector"="True" }
        Pass
        {
            Name "SPRAY_SHADE_C"
            Tags { "LightMode" = "G2SprayShade" }
            Cull Off ZWrite Off ZTest Always
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            float4 frag(v2f_img i) : SV_Target { ShadeOut s = Shade(i.uv); return float4(s.C * s.a, s.a); }
            ENDCG
        }
        Pass
        {
            Name "SPRAY_SHADE_M"
            Tags { "LightMode" = "G2SprayShade" }
            Cull Off ZWrite Off ZTest Always
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            float4 frag(v2f_img i) : SV_Target { ShadeOut s = Shade(i.uv); return float4(s.M * s.a, s.D * s.a); }
            ENDCG
        }
    }
}
