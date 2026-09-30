// FluidSceneMVP.Spray.cs — G2 (overnight 2026-09-30): the LEARNED spray emitter (LearnedSprayLayer: 4e emitter L +
// 4f LOR knobs) on the live coarse particles, composited into the fluid field before shading so droplets are shaded
// as water (the bulk's normals / refraction / specular / scene occlusion). OFF by default; key J toggles it.
//
// Stepping: once per SOLVER frame via GpuSphProvider.OnSolverFrame (the emitter is a per-1/25-s model), or, for a
// playback provider, whenever the played frame index changes. A solver reset (solver time going backwards) clears the
// droplets. Zero cost while off: nothing is allocated and the provider event is not subscribed.
// Notes / parity gate / timings: SSU_restart/Experiments/Overnight0930/G2_spray/NOTES.md.
using System;
using UnityEngine;
using Unity.InferenceEngine;

public partial class FluidSceneMVP
{
    [Header("Learned spray (G2 overnight 2026-09-30, key J) — off by default")]
    [Tooltip("Emit spray droplets from the live coarse particles with the learned 3D emitter (4e emitter L, trained on DFSPH true-coarse states, with the 4f LOR launch/rate knobs), fly them ballistically, cull them at walls / obstacles / re-entry into the bulk / 2 s, and draw them as water spheres in the fluid field. Toggle at runtime with J.")]
    public bool sprayEnabled = false;
    [Tooltip("Emitter ONNX (Inference Engine ModelAsset) under a Resources folder, loaded at runtime. spray_emitter_L_s0 = 4e emitter L, seed 0. Re-export any same-architecture checkpoint with SSU_restart/Experiments/Overnight0930/G2_spray/export_emitter.py.")]
    public string sprayModelResource = "spray_emitter_L_s0";
    [Tooltip("Optional explicit emitter asset; overrides Spray Model Resource.")]
    public ModelAsset sprayModel;
    [Tooltip("Inference Engine backend for the emitter MLP (124 -> 256x3 -> heads, one row per coarse particle per solver frame).")]
    public BackendType sprayBackend = BackendType.GPUCompute;
    [Tooltip("Multiplier s_R on the emitter's Poisson rate. 2.9 = the 4f LOR arm (fitted on DFSPH TRAIN scenes so the outside-silhouette spray area matched GT). The live GPU-PBF pool is a transfer from that training distribution, so this is a user knob, not a calibrated value.")]
    public float sprayRateScale = 2.9f;
    [Tooltip("Outward launch speed u_o [m/s] added along the parent's colour-field normal n^ (4f LOR: 3.0). 0 = the 4e emitter as trained.")]
    public float sprayLaunchSpeed = 3f;
    [Tooltip("Droplet radius r_d [sim m]. 0 = r_c / 25^(1/3) (one dense particle of the 4x training data; 0.0142 at r_c 0.0414).")]
    public float sprayDropletRadius = 0f;
    [Tooltip("Global cap on live droplets; events that would exceed it are dropped (counted in the status line).")]
    public int sprayMaxDroplets = 30000;
    [Tooltip("Display only: smallest drawn droplet footprint radius in model-window pixels (a far droplet smaller than this is drawn at this size so it covers a pixel centre).")]
    [Range(0f, 3f)] public float sprayMinRadiusPx = 0.75f;
    [Tooltip("Display only: multiplier on r_d for the drawn sphere (1 = physical).")]
    [Range(0.25f, 4f)] public float sprayRadiusScale = 1f;
    [Tooltip("Seed of the birth sampler (Poisson counts, mixture, sizes, blob jitter).")]
    public int spraySeed = 0;

    [Serializable] class SprayMeta { public float sig_in; }

    LearnedSprayLayer spray;
    GpuSphProvider sprayGpu;                 // provider whose OnSolverFrame we are subscribed to
    float spraySolverTimeLast = -1f;
    int sprayPlayFrameLast = -1;
    string sprayError;

    public bool SprayEnabled => sprayEnabled;
    public LearnedSprayLayer Spray => spray;
    public void ToggleSpray() { sprayEnabled = !sprayEnabled; if (!sprayEnabled) SprayShutdown(); }

    float SprayCoarseRadius => provider != null && provider.ParticleRadius > 0f ? provider.ParticleRadius
        : meta != null && meta.coarseRadius > 0f ? meta.coarseRadius : 0.0414f;

    bool EnsureSpray()
    {
        if (spray != null) return true;
        if (sprayError != null) return false;
        try
        {
            var asset = sprayModel != null ? sprayModel : Resources.Load<ModelAsset>(sprayModelResource);
            if (asset == null) throw new Exception($"emitter model '{sprayModelResource}' not found under any Resources folder");
            spray = new LearnedSprayLayer((ulong)(uint)spraySeed);
            spray.SetModel(asset, sprayBackend);
            // sigma_in belongs to the emitter's training data: export_emitter.py writes it into <name>_meta.json
            var metaTA = Resources.Load<TextAsset>(asset.name + "_meta");
            if (metaTA != null)
            {
                var m = JsonUtility.FromJson<SprayMeta>(metaTA.text);
                if (m != null && m.sig_in > 0f) spray.sigIn = m.sig_in;
            }
            sprayGpu = provider as GpuSphProvider;
            if (sprayGpu != null) sprayGpu.OnSolverFrame += OnSpraySolverFrame;
            spraySolverTimeLast = -1f; sprayPlayFrameLast = -1;
            Debug.Log($"FluidSceneMVP: learned spray ON — model {asset.name}, backend {sprayBackend}, r_c {SprayCoarseRadius:F4}, sig_in {spray.sigIn:F4}, " +
                      $"stepping {(sprayGpu != null ? "per GPU solver frame" : "per played frame")}");
            return true;
        }
        catch (Exception e)
        {
            sprayError = e.Message;
            Debug.LogError($"FluidSceneMVP: learned spray disabled — {e.Message}");
            spray?.Dispose(); spray = null;
            return false;
        }
    }

    void SprayShutdown()
    {
        if (sprayGpu != null) sprayGpu.OnSolverFrame -= OnSpraySolverFrame;
        sprayGpu = null;
        spray?.Dispose(); spray = null;
        sprayError = null;
    }

    void ApplySprayKnobs()
    {
        spray.rateScale = sprayRateScale;
        spray.launchSpeed = sprayLaunchSpeed;
        spray.dropletRadius = sprayDropletRadius;
        spray.maxDroplets = Mathf.Max(sprayMaxDroplets, 0);
        var gpu = provider as GpuSphProvider;
        spray.frameDt = 1.0 / (gpu != null ? gpu.simHz : (provider.NativeFps > 0f ? provider.NativeFps : 25f));
        spray.obstacles.Clear();
        if (gpu != null)
        {
            spray.gravity = gpu.gravity;
            spray.domainMin = gpu.domainMin; spray.domainMax = gpu.domainMax;
            if (gpu.obstacles != null)
            {
                // GpuSphProvider.AdvanceObstacles' sim -> world mapping; the obstacle transforms are already at this
                // solver frame's pose when OnSolverFrame fires
                Vector3 off = new Vector3(gpu.domainCenterXZ.x, 0f, gpu.domainCenterXZ.y);
                Matrix4x4 simToWorld = gpu.transform.localToWorldMatrix * Matrix4x4.Scale(new Vector3(1f, 1f, -1f)) * Matrix4x4.Translate(-off);
                float rdm = spray.RD > 0f ? spray.RD : SprayCoarseRadius / 2.924f;
                foreach (var o in gpu.obstacles)
                {
                    if (o == null || o.transform == null || !o.transform.gameObject.activeInHierarchy) continue;
                    Matrix4x4 toLocal = o.transform.worldToLocalMatrix * simToWorld;
                    float s = Mathf.Max(new Vector3(toLocal.m00, toLocal.m01, toLocal.m02).magnitude,
                                        new Vector3(toLocal.m10, toLocal.m11, toLocal.m12).magnitude,
                                        new Vector3(toLocal.m20, toLocal.m21, toLocal.m22).magnitude);
                    spray.obstacles.Add(new LearnedSprayLayer.Obstacle { toLocal = toLocal, shape = (int)o.shape, padLocal = rdm * s });
                }
            }
        }
    }

    void OnSpraySolverFrame(int idx, float solverTime, float[] records, int count)
    {
        if (!sprayEnabled || spray == null) return;
        if (solverTime < spraySolverTimeLast) spray.Reset();   // ResetSim: the solver clock restarted
        spraySolverTimeLast = solverTime;
        try { ApplySprayKnobs(); spray.Step(records, 0, count, SprayCoarseRadius); }
        catch (Exception e)
        {   // never let the spray take the solver loop down: log once, switch the layer off (J re-tries)
            Debug.LogException(e);
            sprayEnabled = false;
            SprayShutdown();
        }
    }

    // Called in ProcessPrediction right before the field texture upload: droplets into px (texture rows, row 0 = bottom).
    void SprayComposite()
    {
        if (!sprayEnabled || !EnsureSpray()) return;
        if (sprayGpu == null)
        {   // playback provider: one emitter step per new played frame
            if (frameIdx != sprayPlayFrameLast)
            {
                if (frameIdx < sprayPlayFrameLast) spray.Reset();
                sprayPlayFrameLast = frameIdx;
                provider.GetFrame(frameIdx, out var data, out int off, out int count);
                ApplySprayKnobs();
                spray.Step(data, off, count, SprayCoarseRadius);
            }
        }
        float ts = useV2 ? v2TS : SplatV2.ThicknessScaleLegacy;
        spray.Composite(px, W, H, eyeSim, rightSim, upSim, fwdSim, focalM, winAspect, ts, sprayMinRadiusPx, sprayRadiusScale);
    }

    string SprayStatus()
    {
        if (!sprayEnabled) return "";
        if (spray == null) return sprayError != null ? $"   SPRAY ERROR: {sprayError}" : "   SPRAY (starting)";
        return $"   SPRAY {spray.Visible} drops +{spray.LastBorn} ({spray.LastEvents} ev, Σλ {spray.LastRateSum:F1}" +
               $"{(spray.LastDropped > 0 ? $", capped {spray.LastDropped}" : "")})  " +
               $"feat {spray.MsFeatures:F1} mlp {spray.MsMlp:F1} birth {spray.MsBirths:F1} fly {spray.MsDroplets:F1} draw {spray.MsComposite:F1} ms";
    }
}
