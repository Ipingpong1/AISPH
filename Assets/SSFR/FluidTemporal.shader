// FluidTemporal.shader — 067-EMA3: the temporal stage between the presmooth and the shading passes of FluidSceneMVP.
// Exact port of SSU_restart/Helpers/live_gap_temporal.py (temporal_step + reproject); edit the two together.
//   1. reproject the history through the camera motion (unproject with the CURRENT depth, project into the PREVIOUS camera,
//      mask-weighted bilinear tap, convert the tapped depth back to the current axis); reject on disagreement > _RejectM
//   2. adaptive EMA on depth + thickness: alpha = aRest + (1 - aRest) * smoothstep(v0, v1, |v_cam| from the INPUT tensor)
//   3. hysteresis on a smoothed occupancy mask: ON above 0.7, OFF below 0.3 (on / off after 2 consecutive frames)
// In : _MainTex  current field after presmooth (r depth, g thickness, b mask)      _FieldTex raw staged field (a = input speed, m/s)
//      _HistTex  previous output of this pass                                        Out: r D, g T, b on (the mask the shading passes read), a M
// All camera vectors are in SIM space (FluidSceneMVP.eyeSim ...); texture y is UP (row 0 = bottom), as in FluidSSFRScene.viewPos.
Shader "Hidden/FluidTemporal"
{
    Properties { _MainTex ("Current field", 2D) = "black" {} }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex, _FieldTex, _HistTex;
            float4 _HistTex_TexelSize;          // xy = 1/size, zw = size
            float _HasHistory, _Reproject, _Hyst, _ARest, _V0, _V1, _RejectM, _WinAspect, _FocalC, _FocalP, _DGate0, _DGate1;
            float3 _EyeC, _RightC, _UpC, _FwdC, _EyeP, _RightP, _UpP, _FwdP;

            float4 tapHist(float2 ip) { return tex2Dlod(_HistTex, float4((ip + 0.5) * _HistTex_TexelSize.xy, 0, 0)); }

            float4 frag(v2f_img i) : SV_Target
            {
                float4 c = tex2D(_MainTex, i.uv);
                bool m = c.b > 0.5;
                if (_HasHistory < 0.5) return float4(c.r, c.g, m ? 1.0 : 0.0, m ? 1.0 : 0.0);

                float wa = _WinAspect > 0.0 ? _WinAspect : 1.0;
                float2 size = _HistTex_TexelSize.zw;
                float2 ip = i.uv * size - 0.5;          // same-uv fallback (no current depth to unproject with)
                float2 uvp = i.uv;
                bool inside = false;
                if (m && _Reproject > 0.5)
                {
                    float2 n = float2((i.uv.x * 2.0 - 1.0) * wa, i.uv.y * 2.0 - 1.0) / _FocalC;
                    float3 P = _EyeC + c.r * (_FwdC + n.x * _RightC + n.y * _UpC);
                    float3 rel = P - _EyeP;
                    float z = dot(rel, _FwdP);
                    if (z > 1e-3)
                    {
                        uvp = float2(dot(rel, _RightP) / z * _FocalP / wa, dot(rel, _UpP) / z * _FocalP) * 0.5 + 0.5;
                        inside = uvp.x >= 0.0 && uvp.x <= 1.0 && uvp.y >= 0.0 && uvp.y <= 1.0;
                    }
                    if (inside) ip = uvp * size - 0.5; else uvp = i.uv;
                }

                float2 i0 = clamp(floor(ip), 0.0, size - 1.0);
                float2 i1 = clamp(i0 + 1.0, 0.0, size - 1.0);
                float2 f = saturate(ip - i0);
                float4 h00 = tapHist(i0), h10 = tapHist(float2(i1.x, i0.y)), h01 = tapHist(float2(i0.x, i1.y)), h11 = tapHist(i1);
                float w00 = (1.0 - f.x) * (1.0 - f.y), w10 = f.x * (1.0 - f.y), w01 = (1.0 - f.x) * f.y, w11 = f.x * f.y;
                float onw = w00 * h00.b + w10 * h10.b + w01 * h01.b + w11 * h11.b;      // mask-weighted bilinear
                float inv = 1.0 / max(onw, 1e-6);
                float Dh = (w00 * h00.b * h00.r + w10 * h10.b * h10.r + w01 * h01.b * h01.r + w11 * h11.b * h11.r) * inv;
                float Th = (w00 * h00.b * h00.g + w10 * h10.b * h10.g + w01 * h01.b * h01.g + w11 * h11.b * h11.g) * inv;
                float Mh = w00 * h00.a + w10 * h10.a + w01 * h01.a + w11 * h11.a;
                if (inside)
                {   // the tapped depth is along the PREVIOUS axis: rebuild that point, measure it along the CURRENT axis
                    float2 n0 = float2((uvp.x * 2.0 - 1.0) * wa, uvp.y * 2.0 - 1.0) / _FocalP;
                    float3 Q = _EyeP + Dh * (_FwdP + n0.x * _RightP + n0.y * _UpP);
                    Dh = dot(Q - _EyeC, _FwdC);
                }

                bool have = onw > 0.5;
                bool agree = have && m && abs(Dh - c.r) <= _RejectM;
                float speed = tex2D(_FieldTex, i.uv).a;
                float t = saturate((speed - _V0) / max(_V1 - _V0, 1e-6));
                float a = _ARest + (1.0 - _ARest) * t * t * (3.0 - 2.0 * t);
                if (_DGate1 > 0.0)
                {   // cm-scale disagreement with the history = real motion, not flicker (mm-scale): trust the current frame
                    float g = saturate((abs(Dh - c.r) - _DGate0) / max(_DGate1 - _DGate0, 1e-6));
                    a = max(a, g * g * (3.0 - 2.0 * g));
                }
                if (!agree) a = 1.0;
                float D = m ? a * c.r + (1.0 - a) * Dh : Dh;
                float T = m ? a * c.g + (1.0 - a) * Th : Th;

                float M = m ? 1.0 : 0.0;
                bool on = m;
                if (_Hyst > 0.5)
                {
                    M = 0.5 * M + 0.5 * Mh;
                    on = (have ? M > 0.3 : M > 0.7) && (m || have);   // 0.3 / 0.7, not 0.25 / 0.75: M is dyadic under a static camera, so those would be exact ties
                }
                return float4(D, T, on ? 1.0 : 0.0, M);
            }
            ENDCG
        }
    }
}
