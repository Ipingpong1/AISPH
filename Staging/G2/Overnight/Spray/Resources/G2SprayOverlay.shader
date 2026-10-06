// G2SprayOverlay.shader — G2 (overnight 2026-09-30): composites the learned-spray droplets as a thin transparent
// layer OVER whatever is already in the frame (scene + the fluid composite quad + foam), after the fluid quad.
//
// The droplets are first shaded at model-window resolution by FluidSSFRScene passes 0-1 (the bulk's own Shade():
// normals from their sphere depth, Beer-Lambert through their chord, Fresnel env reflection, Blinn-Phong specular),
// from a droplet-only field (FluidSceneMVP.Spray.cs). Here, per screen pixel, the same model-window mapping as
// FluidSSFRScene pass 3 is used, occlusion by scene geometry is tested against _CameraDepthTexture, and the layer is
// applied as   dst = dst * ((1 - a) + a M) + a C   in two draws (this shader with _Mode 0 = MUL, then _Mode 1 = ADD):
// the droplet transmits what is behind it (the bulk or the scene) instead of replacing it, so a droplet in front of an
// object that hides the bulk does not inherit the bulk's thickness. No refraction offset (droplets are 1-10 px).
Shader "Hidden/G2SprayOverlay"
{
    Properties
    {
        _CTex ("Droplet C (premultiplied, a = coverage)", 2D) = "black" {}
        _MTex ("Droplet M (premultiplied), a = D * coverage", 2D) = "black" {}
        _Mode ("0 = multiply pass, 1 = add pass", Float) = 0
        _SrcBlend ("Src blend", Float) = 0
        _DstBlend ("Dst blend", Float) = 3
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" "IgnoreProjector"="True" }
        Pass
        {
            Name "SPRAY_OVERLAY"
            Cull Off ZWrite Off ZTest Always
            Blend [_SrcBlend] [_DstBlend]

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _CTex, _MTex;
            sampler2D _CameraDepthTexture;
            float4 _ScaledScreenParams;
            float _FocalM, _WinAspect, _SimScale, _DepthBias, _Mode;
            float3 _CamRightWS, _CamUpWS, _CamFwdWS;

            struct appdata { float4 vertex : POSITION; };
            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 ray : TEXCOORD0;
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
                float4 c1 = tex2D(_MTex, uvm);
                float worldD = c1.a / max(a, 1e-4) * _SimScale;
                float2 suv = i.pos.xy / _ScaledScreenParams.xy;
                float sceneEye = LinearEyeDepth(tex2D(_CameraDepthTexture, suv).r);
                float vis = (sceneEye - worldD + _DepthBias >= 0.0) ? 1.0 : 0.0;
                if (_Mode < 0.5)
                    return float4((1.0 - vis * a) + vis * c1.rgb, 1.0);   // MUL: dst *= (1 - a) + a M
                return float4(vis * c0.rgb, 0.0);                            // ADD: dst += a C
            }
            ENDCG
        }
    }
}
