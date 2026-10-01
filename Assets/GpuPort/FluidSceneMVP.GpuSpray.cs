// FluidSceneMVP.GpuSpray.cs — GPU1001: the learned spray inside the GPU chain (GpuSprayLayer + GpuSpray.compute).
// Stepping: once per SOLVER frame from GpuSphProvider.OnSolverStepGpu (no readback; the layer reads the solver's
// buffers), or once per new played frame for a baked provider. Drawing: every rendered frame the droplets are rastered
// against the decoded bulk field inside the frame's command buffer; Overlay mode then shades the two RTs with the G2
// materials (G2SprayShade + G2SprayOverlay quads), Field mode merges into the field before FieldToTex.
// The CPU layer (LearnedSprayLayer) is shut down while the GPU layer runs, so its per-solver-frame readback stops too.
using System;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.InferenceEngine;

public partial class FluidSceneMVP
{
    const string GpuSprayLayerSample = "GPU1001.Spray";

    GpuSprayLayer gpuSprayLayer;
    CommandBuffer gpuSprayCb;
    GpuSphProvider gpuSprayProvider;
    float gpuSprayTimeLast = -1f;
    int gpuSprayPlayFrameLast = -1;
    string gpuSprayError;
    bool gpuSprayDrawn;

    public GpuSprayLayer GpuSpray => gpuSprayLayer;

    bool EnsureGpuSpray()
    {
        if (gpuSprayLayer != null) return true;
        if (gpuSprayError != null) return false;
        try
        {
            SprayShutdown();   // the CPU layer (and its per-solver-frame readback) is replaced
            var shader = Resources.Load<ComputeShader>("GpuSpray");
            if (shader == null) throw new Exception("GpuSpray.compute not found (Assets/GpuPort/Resources)");
            var asset = sprayModel != null ? sprayModel : Resources.Load<ModelAsset>(sprayModelResource);
            if (asset == null) throw new Exception($"emitter model '{sprayModelResource}' not found under any Resources folder");
            gpuSprayLayer = new GpuSprayLayer(shader) { seed = (uint)spraySeed };
            gpuSprayLayer.SetModel(asset);
            var metaTA = Resources.Load<TextAsset>(asset.name + "_meta");
            if (metaTA != null)
            {
                var m = JsonUtility.FromJson<SprayMeta>(metaTA.text);
                if (m != null && m.sig_in > 0f) gpuSprayLayer.sigIn = m.sig_in;
            }
            gpuSprayCb = new CommandBuffer { name = "GPU1001.SprayStep" };
            gpuSprayProvider = provider as GpuSphProvider;
            if (gpuSprayProvider != null) gpuSprayProvider.OnSolverStepGpu += OnGpuSpraySolverStep;
            gpuSprayTimeLast = -1f; gpuSprayPlayFrameLast = -1;
            Debug.Log($"FluidSceneMVP: GPU1001 learned spray ON — model {asset.name}, sig_in {gpuSprayLayer.sigIn:F4}, " +
                      $"stepping {(gpuSprayProvider != null ? "per GPU solver frame (no readback)" : "per played frame")}");
            return true;
        }
        catch (Exception e)
        {
            gpuSprayError = e.Message;
            Debug.LogError($"FluidSceneMVP: GPU1001 spray disabled — {e.Message}");
            DisposeGpuSpray();
            gpuSprayError = e.Message;
            return false;
        }
    }

    void DisposeGpuSpray()
    {
        if (gpuSprayProvider != null) gpuSprayProvider.OnSolverStepGpu -= OnGpuSpraySolverStep;
        gpuSprayProvider = null;
        gpuSprayLayer?.Dispose(); gpuSprayLayer = null;
        gpuSprayCb?.Release(); gpuSprayCb = null;
        gpuSprayError = null;
        gpuSprayDrawn = false;
    }

    void ApplyGpuSprayKnobs()
    {
        var L = gpuSprayLayer;
        L.rateScale = sprayRateScale;
        L.launchSpeed = sprayLaunchSpeed;
        L.dropletRadius = sprayDropletRadius;
        L.maxDroplets = Mathf.Max(sprayMaxDroplets, 0);
        var gpu = provider as GpuSphProvider;
        L.frameDt = 1.0 / (gpu != null ? gpu.simHz : (provider.NativeFps > 0f ? provider.NativeFps : 25f));
        L.obstacles.Clear();
        if (gpu == null) return;
        L.gravity = gpu.gravity;
        L.domainMin = gpu.domainMin; L.domainMax = gpu.domainMax;
        if (gpu.obstacles == null) return;
        // as ApplySprayKnobs (G2): GpuSphProvider.AdvanceObstacles' sim -> world mapping, obstacle pose of this solver frame
        Vector3 off = new Vector3(gpu.domainCenterXZ.x, 0f, gpu.domainCenterXZ.y);
        Matrix4x4 simToWorld = gpu.transform.localToWorldMatrix * Matrix4x4.Scale(new Vector3(1f, 1f, -1f)) * Matrix4x4.Translate(-off);
        float rdm = L.RD > 0f ? L.RD : SprayCoarseRadius / 2.924f;
        foreach (var o in gpu.obstacles)
        {
            if (o == null || o.transform == null || !o.transform.gameObject.activeInHierarchy) continue;
            Matrix4x4 toLocal = o.transform.worldToLocalMatrix * simToWorld;
            float s = Mathf.Max(new Vector3(toLocal.m00, toLocal.m01, toLocal.m02).magnitude,
                                new Vector3(toLocal.m10, toLocal.m11, toLocal.m12).magnitude,
                                new Vector3(toLocal.m20, toLocal.m21, toLocal.m22).magnitude);
            L.obstacles.Add(new LearnedSprayLayer.Obstacle { toLocal = toLocal, shape = (int)o.shape, padLocal = rdm * s });
        }
    }

    void OnGpuSpraySolverStep(int idx, float solverTime)
    {
        if (!sprayEnabled || gpuSprayLayer == null) return;
        if (solverTime < gpuSprayTimeLast) gpuSprayLayer.Reset();   // ResetSim
        gpuSprayTimeLast = solverTime;
        try
        {
            ApplyGpuSprayKnobs();
            var s = gpuSprayProvider.Solver;
            gpuSprayCb.Clear();
            gpuSprayCb.BeginSample(SamplerSpray);
            gpuSprayLayer.RecordStep(gpuSprayCb, s.PositionBuffer, s.VelocityBuffer, s.Count, SprayCoarseRadius);
            gpuSprayCb.EndSample(SamplerSpray);
            Graphics.ExecuteCommandBuffer(gpuSprayCb);
        }
        catch (Exception e)
        {   // never take the solver loop down: log once, switch the layer off (J re-tries)
            Debug.LogException(e);
            sprayEnabled = false;
            DisposeGpuSpray();
        }
    }

    // Called inside GpuLateUpdate after Decode, before FieldToTex (records into the frame's command buffer).
    void RecordGpuSprayComposite(CommandBuffer cb)
    {
        gpuSprayDrawn = false;
        if (!sprayEnabled)
        {
            if (gpuSprayLayer != null) DisposeGpuSpray();
            SetSprayOverlayVisible(false);
            return;
        }
        if (!EnsureGpuSpray()) { SetSprayOverlayVisible(false); return; }
        if (gpuSprayProvider == null && frameIdx != gpuSprayPlayFrameLast)
        {   // baked provider: one emitter step per new played frame, on the uploaded frame
            if (frameIdx < gpuSprayPlayFrameLast) gpuSprayLayer.Reset();
            gpuSprayPlayFrameLast = frameIdx;
            ApplyGpuSprayKnobs();
            GpuParticles(out var pos, out var vel, out var dens, out int n);
            cb.BeginSample(SamplerSpray);
            gpuSprayLayer.RecordStep(cb, pos, vel, n, SprayCoarseRadius);
            cb.EndSample(SamplerSpray);
        }
        float ts = useV2 ? v2TS : SplatV2.ThicknessScaleLegacy;
        bool field = sprayRenderMode == SprayRenderMode.Field;
        gpuSprayLayer.RecordRaster(cb, gpuSplat.Field, W, H, eyeSim, rightSim, upSim, fwdSim, focalM, winAspect, ts,
                                   sprayMinRadiusPx, sprayRadiusScale, field);
        gpuSprayDrawn = !field;
        if (field) SetSprayOverlayVisible(false);
    }

    // After the frame's command buffer ran: Overlay mode shades the droplet RTs with the G2 materials.
    void AfterGpuExecute()
    {
        if (!gpuSprayDrawn || gpuSprayLayer == null) return;
        EnsureSprayOverlay();
        Material shade = sprayShadeMat;
        if (sprayAnalyticNormals)
        {
            shade = sprayShadeNMat;
            shade.SetTexture("_NrmTex", gpuSprayLayer.SprayNrmRT);
        }
        SetShadeParams(shade);
        Graphics.Blit(gpuSprayLayer.SprayFieldRT, sprayCRT, shade, 0);
        Graphics.Blit(gpuSprayLayer.SprayFieldRT, sprayMRT, shade, 1);
        foreach (var m in new[] { sprayMulMat, sprayAddMat })
        {
            m.SetTexture("_CTex", sprayCRT);
            m.SetTexture("_MTex", sprayMRT);
            m.SetFloat("_FocalM", focalM);
            m.SetFloat("_WinAspect", winAspect);
            m.SetFloat("_SimScale", transform.lossyScale.x);
            m.SetFloat("_DepthBias", depthBias);
            m.SetVector("_CamRightWS", shadeRightWS);
            m.SetVector("_CamUpWS", shadeUpWS);
            m.SetVector("_CamFwdWS", shadeFwdWS);
        }
        SetSprayOverlayVisible(true);
    }

    string GpuSprayStatus()
    {
        var L = gpuSprayLayer;
        return $"   GPU SPRAY {L.Alive} drops +{L.LastBorn} ({L.LastEvents} ev, Σλ {L.LastRateSum:F1}" +
               $"{(L.LastDropped > 0 ? $", capped {L.LastDropped}" : "")}{(L.LastEventOverflow > 0 ? $", ev overflow {L.LastEventOverflow}" : "")}) " +
               $"gpu {GpuMsSpray:F2} ms ({sprayRenderMode})";
    }
}
