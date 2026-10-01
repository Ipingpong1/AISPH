// FluidSceneMVP.Spray.cs — G2 (overnight 2026-09-30): the LEARNED spray emitter (LearnedSprayLayer: 4e emitter L +
// 4f LOR knobs) on the live coarse particles, shaded as water by the bulk's own SSFR shading. OFF by default; key J.
// Rendering (sprayRenderMode): Overlay (default) = a droplet-only (depth, chord thickness, alpha) field, depth-tested
// against the predicted bulk, shaded by FluidSSFRScene passes 0-1 and laid over the frame after the fluid quad by
// Resources/G2SprayOverlay.shader (dst = dst ((1-a) + a M) + a C, scene-depth occluded); Field = droplets merged into
// the bulk's own field before shading (the brief's suggestion; see the caveat on LearnedSprayLayer.Composite).
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
    [Tooltip("Emit spray droplets from the live coarse particles with the learned 3D emitter (4e emitter L, trained on DFSPH true-coarse states, with the 4f LOR launch/rate knobs), fly them ballistically, cull them at walls / obstacles / re-entry into the bulk / 2 s, and draw them as water spheres shaded like the bulk (see Spray Render Mode). Toggle at runtime with J.")]
    public bool sprayEnabled = false;
    [Tooltip("Emitter ONNX (Inference Engine ModelAsset) under a Resources folder, loaded at runtime. spray_emitter_L_s0 = 4e emitter L, seed 0. Re-export any same-architecture checkpoint with SSU_restart/Experiments/Overnight0930/G2_spray/export_emitter.py.")]
    public string sprayModelResource = "spray_emitter_L_s0";
    [Tooltip("Optional explicit emitter asset; overrides Spray Model Resource.")]
    public ModelAsset sprayModel;
    [Tooltip("Inference Engine backend for the emitter MLP (124 -> 256x3 -> heads, one row per coarse particle per solver frame).")]
    public BackendType sprayBackend = BackendType.GPUCompute;
    [Tooltip("Multiplier s_R on the emitter's Poisson rate. Default 2.9 = the 4f LOR arm, a DFSPH SCREEN-AREA calibration (fitted on DFSPH train scenes so the outside-silhouette spray area matched GT; emitter L paints small blobs). Here it is a look choice, not a calibration: on Unity-PBF pools emitter L already over-produces spray EVENTS 2.1-2.7x at s_R = 1 (SSU_restart Experiments/Overnight0930/G1_pbf_rs/emitter_transfer.json), so the event-count-matched value is ~0.4 and 2.9 gives ~6-8x the dense PBF event count.")]
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

    public enum SprayRenderMode { Overlay, Field }
    [Tooltip("Overlay (default): the droplets are shaded by the bulk's own SSFR shading (FluidSSFRScene passes 0-1: normals from their sphere depth, Fresnel reflection, specular, Beer-Lambert through their chord) from a droplet-only field, then laid over the frame as a thin transparent layer that transmits whatever is behind it (the bulk or the scene), occluded by scene depth. Field: the droplets are merged into the bulk's own depth/thickness field before shading (the bulk bulges where a droplet is in front; a droplet in front of an object that hides the bulk inherits the hidden bulk's thickness).")]
    public SprayRenderMode sprayRenderMode = SprayRenderMode.Overlay;
    [Tooltip("Overlay only: shade each droplet with its analytic sphere normal (Resources/G2SprayShade.shader, the same lighting model as FluidSSFRScene) instead of FluidSSFRScene's depth-derivative normal, which degenerates for droplets of 1-2 model-window pixels (N = 0 -> full Fresnel -> a sky-coloured dot).")]
    public bool sprayAnalyticNormals = true;

    [Serializable] class SprayMeta { public float sig_in; }

    LearnedSprayLayer spray;
    // overlay mode resources (created on first use, released by SprayShutdown)
    Color[] sprayPx, sprayNrmPx; Texture2D sprayFieldTex, sprayNrmTex; RenderTexture sprayCRT, sprayMRT;
    Material sprayShadeMat, sprayShadeNMat, sprayMulMat, sprayAddMat; GameObject sprayMulGO, sprayAddGO;
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
        ReleaseSprayOverlay();
    }

    void ReleaseSprayOverlay()
    {
        if (sprayMulGO != null) Destroy(sprayMulGO);
        if (sprayAddGO != null) Destroy(sprayAddGO);
        if (sprayMulMat != null) Destroy(sprayMulMat);
        if (sprayAddMat != null) Destroy(sprayAddMat);
        if (sprayShadeMat != null) Destroy(sprayShadeMat);
        if (sprayShadeNMat != null) Destroy(sprayShadeNMat);
        if (sprayFieldTex != null) Destroy(sprayFieldTex);
        if (sprayNrmTex != null) Destroy(sprayNrmTex);
        if (sprayCRT != null) sprayCRT.Release();
        if (sprayMRT != null) sprayMRT.Release();
        sprayMulGO = sprayAddGO = null; sprayMulMat = sprayAddMat = sprayShadeMat = sprayShadeNMat = null;
        sprayFieldTex = sprayNrmTex = null; sprayCRT = sprayMRT = null; sprayPx = sprayNrmPx = null;
    }

    void EnsureSprayOverlay()
    {
        if (sprayMulGO != null && sprayPx != null && sprayPx.Length == W * H) return;
        ReleaseSprayOverlay();
        var sh = Shader.Find("Hidden/G2SprayOverlay");
        if (sh == null) throw new Exception("Hidden/G2SprayOverlay shader not found (Assets/Overnight/Spray/Resources)");
        var shN = Shader.Find("Hidden/G2SprayShade");
        if (shN == null) throw new Exception("Hidden/G2SprayShade shader not found (Assets/Overnight/Spray/Resources)");
        sprayPx = new Color[W * H]; sprayNrmPx = new Color[W * H];
        sprayFieldTex = new Texture2D(W, H, TextureFormat.RGBAFloat, false) { filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp };
        sprayNrmTex = new Texture2D(W, H, TextureFormat.RGBAFloat, false) { filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp };
        sprayShadeNMat = new Material(shN);
        sprayCRT = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBHalf) { filterMode = FilterMode.Bilinear, wrapMode = TextureWrapMode.Clamp };
        sprayMRT = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBHalf) { filterMode = FilterMode.Bilinear, wrapMode = TextureWrapMode.Clamp };
        sprayShadeMat = new Material(sceneShader);   // FluidSSFRScene: passes 0-1 shade the droplet field exactly like the bulk
        sprayMulMat = new Material(sh) { renderQueue = 3101 };   // after the fluid composite quad (3100)
        sprayMulMat.SetFloat("_Mode", 0f);
        sprayMulMat.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.Zero);
        sprayMulMat.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.SrcColor);
        sprayAddMat = new Material(sh) { renderQueue = 3102 };
        sprayAddMat.SetFloat("_Mode", 1f);
        sprayAddMat.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.One);
        sprayAddMat.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.One);
        sprayMulGO = MakeSprayQuad("G2 Spray Overlay (mul)", sprayMulMat);
        sprayAddGO = MakeSprayQuad("G2 Spray Overlay (add)", sprayAddMat);
    }

    GameObject MakeSprayQuad(string name, Material m)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Quad);
        go.name = name;
        Destroy(go.GetComponent<Collider>());
        float qz = Mathf.Max(1f, targetCamera.nearClipPlane * 2f);   // same placement as the fluid composite quad
        go.transform.SetParent(targetCamera.transform, false);
        go.transform.localPosition = new Vector3(0, 0, qz);
        go.transform.localScale = new Vector3(5f * qz, 5f * qz, 1f);
        var mr = go.GetComponent<MeshRenderer>();
        mr.sharedMaterial = m;
        mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        mr.receiveShadows = false;
        mr.lightProbeUsage = UnityEngine.Rendering.LightProbeUsage.Off;
        mr.enabled = false;
        return go;
    }

    void SetSprayOverlayVisible(bool on)
    {
        if (sprayMulGO != null) sprayMulGO.GetComponent<MeshRenderer>().enabled = on;
        if (sprayAddGO != null) sprayAddGO.GetComponent<MeshRenderer>().enabled = on;
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

    // Called in ProcessPrediction (and by G3's classical path) right before the field texture upload, with px = the
    // bulk field (texture rows, row 0 = bottom). Field mode writes the droplets into px; Overlay mode only reads px
    // (depth test against the bulk) and shades / draws the droplets as their own layer.
    float sprayDrawMs;

    void SprayComposite()
    {
        if (!sprayEnabled || !EnsureSpray()) { SetSprayOverlayVisible(false); return; }
        var sw = System.Diagnostics.Stopwatch.StartNew();
        SprayCompositeInner();
        sprayDrawMs = (float)sw.Elapsed.TotalMilliseconds;
    }

    void SprayCompositeInner()
    {
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
        if (sprayRenderMode == SprayRenderMode.Field)
        {
            SetSprayOverlayVisible(false);
            spray.Composite(px, W, H, eyeSim, rightSim, upSim, fwdSim, focalM, winAspect, ts, sprayMinRadiusPx, sprayRadiusScale);
            return;
        }
        EnsureSprayOverlay();
        spray.CompositeOverlay(px, sprayPx, W, H, eyeSim, rightSim, upSim, fwdSim, focalM, winAspect, ts, sprayMinRadiusPx, sprayRadiusScale,
                               sprayAnalyticNormals ? sprayNrmPx : null);
        if (spray.DrawnPixels == 0) { SetSprayOverlayVisible(false); return; }
        sprayFieldTex.SetPixelData(sprayPx, 0);       // RGBAFloat == Color layout
        sprayFieldTex.Apply(false, false);
        Material shade = sprayShadeMat;
        if (sprayAnalyticNormals)
        {
            sprayNrmTex.SetPixelData(sprayNrmPx, 0);
            sprayNrmTex.Apply(false, false);
            shade = sprayShadeNMat;
            shade.SetTexture("_NrmTex", sprayNrmTex);
        }
        SetShadeParams(shade);                         // the bulk's look (kThick, Fresnel, specular, env) and splat camera
        Graphics.Blit(sprayFieldTex, sprayCRT, shade, 0);
        Graphics.Blit(sprayFieldTex, sprayMRT, shade, 1);
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

    string SprayStatus()
    {
        if (!sprayEnabled) return "";
        if (gpuSprayLayer != null) return GpuSprayStatus();   // GPU1001: the GPU layer is running instead
        if (spray == null) return sprayError != null ? $"   SPRAY ERROR: {sprayError}" : "   SPRAY (starting)";
        return $"   SPRAY {spray.Visible} drops +{spray.LastBorn} ({spray.LastEvents} ev, Σλ {spray.LastRateSum:F1}" +
               $"{(spray.LastDropped > 0 ? $", capped {spray.LastDropped}" : "")})  " +
               $"feat {spray.MsFeatures:F1} mlp {spray.MsMlp:F1} birth {spray.MsBirths:F1} fly {spray.MsDroplets:F1} draw {sprayDrawMs:F1} ms ({sprayRenderMode})";
    }
}
