// FluidSceneMVP.Classical.cs — G3 overnight 2026-09-30: a runtime switch (key C) between the network surface and the
// classical C* surface (the `t3_aniso+nr` row of the SSU_restart PBF tables: Yu & Turk anisotropic ellipsoids +
// Truong & Yuksel narrow-range filter; ClassicalSurface.cs). OFF by default (surfaceSource = Network): nothing in the
// network path changes unless C is pressed or the inspector field is set.
//
// Classical mode, per rendered frame (LateUpdate calls ClassicalLateUpdate instead of the splat + inference):
//   same particles (provider.GetFrame(frameIdx)), same sim-space camera (ComputeSimCamera: eye/right/up/fwd, focalM),
//   same model window (H x W, CoverFrustum / FitVertical / windowWidth) -> ClassicalSurface.Run on the GPU ->
//   the (depth, thickness, alpha) field in the network path's texture layout -> smoothedRT -> the SAME FluidSSFRScene
//   shading packs (passes 0-2) and composite (pass 3) with the same SetShadeParams, so lighting / refraction /
//   occlusion are identical and only the surface differs.
// Double-filtering policy (classicalApplyPostFilters = false, the default): presmooth, bilateral and the temporal
//   EMA are BYPASSED in classical mode — the research row has none of them and the NR filter is its smoothing. The
//   network's own "+s2" row corresponds to presmoothSigma 2, which classical mode deliberately does not get.
//   With classicalApplyPostFilters on, presmooth / bilateral are applied like the network path; temporal is controlled separately.
// kThick (Beer-Lambert scale): kThickOverride if set (SampleScene: 0.5); else the network's locked value (one shared
//   material); else derived from the classical thickness with the network path's own rule (1.2 / median, same lock).
//   Classical thickness (shading only; the research row scores depth + mask) = path length through the ellipsoids x
//   15.851, the pysurf GT scale 067a / 060a were trained on, i.e. the network's own thickness unit (see ClassicalSurface).
// Foam (G / FoamLayer): works in both modes; in classical mode its fluid depth is an async readback (one frame late).
// Learned spray (G2, key J): if present and on, the classical field is read back into the CPU field so the spray
//   layer composites its droplets exactly as it does for the network (found by reflection: no compile-time coupling).
// Not in classical mode: V (raw input), OnInferred / LastPred taps (LiveClipRecorder).
using System;
using System.Collections.Generic;
using UnityEngine;

public partial class FluidSceneMVP
{
    public enum SurfaceSource { Network, Classical }

    [Header("Surface source (G3: classical C* comparison, key C) — Network by default")]
    [Tooltip("Network = the U-Net path (unchanged). Classical = the classical C* surface (t3_aniso+nr: Yu & Turk anisotropic ellipsoids at 3 r + Truong & Yuksel narrow-range filter, SSU_restart PBF-suite parameters) computed on the GPU from the same particles and camera and shaded by the SAME FluidSSFRScene passes. Toggle at runtime with C.")]
    public SurfaceSource surfaceSource = SurfaceSource.Network;
    [Tooltip("Classical C* parameters: the frozen PBF-suite values (sources: SSU_restart Experiments/Overnight0930/G3_classical/NOTES.md).")]
    public ClassicalSurfaceSettings classicalSettings = new ClassicalSurfaceSettings();
    [Tooltip("Off (default): classical depth reaches the shading as the research row defines it (no presmooth / bilateral / temporal on top of the NR filter). On: the network path's presmooth / bilateral settings are applied on top (temporal is controlled separately).")]
    public bool classicalApplyPostFilters = false;
    [Tooltip("Opt-in shared adaptive temporal stage. Set Temporal Mode to Adaptive before starting. Uses the SAME input velocity splat as the network; adds CPU splat/upload cost. Off preserves the classical baseline.")]
    public bool classicalApplyTemporal = false;
    Texture2D classicalSpeedTex;
    Color[] classicalSpeedPixels;
    int classicalTemporalFrame = -1;

    ClassicalSurface classical;
    float[] classicalDepthCpu, classicalAuxCpu, classicalAux2Cpu;
    float classicalKThick;
    bool classicalKThickLocked;
    float classicalMs;                 // smoothed CPU ms of the classical frame (Run + any readbacks), GPU work is async

    public bool ClassicalActive => surfaceSource == SurfaceSource.Classical;

    public void ToggleSurfaceSource()
    {
        surfaceSource = ClassicalActive ? SurfaceSource.Network : SurfaceSource.Classical;
        hasHistory = false;            // the temporal history holds the other source's surface
        classicalFoamHave = false;     // re-prime the foam depth synchronously on the next classical frame
    }

    /// <summary>The classical surface of the last classical frame (null before the first one).</summary>
    public ClassicalSurface Classical => classical;

    float ClassicalRadius => classicalSettings.particleRadius > 0f ? classicalSettings.particleRadius
        : useV2 && v2R > 0f ? v2R
        : provider.ParticleRadius > 0f ? provider.ParticleRadius
        : meta.coarseRadius > 0f ? meta.coarseRadius : 0.0414f;

    // Replaces the splat + inference + ProcessPrediction of one rendered frame (LateUpdate hook).
    void ClassicalLateUpdate()
    {
        DrainPending();                // a spread inference in flight when C was pressed is finished and dropped
        try { ClassicalFrame(); }
        catch (Exception e)
        {
            Debug.LogException(e);
            surfaceSource = SurfaceSource.Network;   // never leave the scene without a surface
            status = $"classical surface failed ({e.Message}) — back to Network";
            return;
        }
        CaptureAndStatus();
        // the network is not run in classical mode: drop its (stale) inference timing from the status line
        status = System.Text.RegularExpressions.Regex.Replace(status, @"infer (spread/\d+ )?[0-9.]+ ms( latency, fluid [0-9.]+ Hz)?", "network idle");
        var cs = classicalSettings;
        status += $"   CLASSICAL C* (k {cs.radiusMult:0.##}, NR {(cs.narrowRange ? $"{cs.nrFilterSize}/{cs.nrIters}/{cs.nrThresholdRatio:0.#}" : "off")}" +
                  $"{(classicalApplyPostFilters ? ", +post" : "")}) {classical.Count}p cpu {classicalMs:F2} ms";
    }

    void ClassicalFrame()
    {
        ComputeSimCamera();
        StepFoam();
        var sw = System.Diagnostics.Stopwatch.StartNew();   // classical work only (the FOAM status block times the foam)
        Transform c = targetCamera.transform;
        shadeRightWS = c.right; shadeUpWS = c.up; shadeFwdWS = c.forward;
        if (classical == null) classical = new ClassicalSurface(classicalSettings);
        provider.GetFrame(frameIdx, out var data, out int off, out int count);
        float ts = useV2 && v2TS > 0f ? v2TS : SplatV2.ThicknessScaleLegacy;
        classical.Run(data, off, count, 7, eyeSim, rightSim, upSim, fwdSim, focalM, H, W, ClassicalRadius, ts);

        float now = Time.realtimeSinceStartup;
        if (lastPredTime >= 0f && now > lastPredTime)
            fluidHz = fluidHz <= 0f ? 1f / (now - lastPredTime) : Mathf.Lerp(fluidHz, 1f / (now - lastPredTime), 0.2f);
        lastPredTime = now;

        // G2's learned spray (key J) composites droplets into the CPU field `px`: bring the classical field over first
        Texture field = classical.Output;
        if (ClassicalSprayOn())
        {
            if (classicalReadTex == null || classicalReadTex.width != W || classicalReadTex.height != H)
                classicalReadTex = new Texture2D(W, H, TextureFormat.RGBAFloat, false);
            var prev = RenderTexture.active;
            RenderTexture.active = classical.Output;
            classicalReadTex.ReadPixels(new Rect(0, 0, W, H), 0, 0, false);
            RenderTexture.active = prev;
            classicalReadTex.GetPixelData<Color>(0).CopyTo(px);
            sprayCompositeMI.Invoke(this, null);
            fieldTex.SetPixels(px);
            fieldTex.Apply(false, false);
            field = fieldTex;
        }

        // the field: bypass (default) or the network path's presmooth / bilateral on top
        if (classicalApplyPostFilters)
        {
            if (bilateralSmoothing) bilateralSigmaR = ClassicalDepthStd() * bilateralRangeScale;
            smoothMat.SetFloat("_BlurSigma", bilateralSmoothing ? bilateralSigmaS : presmoothSigma);
            smoothMat.SetFloat("_BilateralRangeSigma", bilateralSmoothing ? Mathf.Max(bilateralSigmaR, 1e-3f) : 0f);
            Graphics.Blit(field, smoothedRT, smoothMat, 0);
            if (bilateralSmoothing)
                for (int it = 1; it < bilateralIters; it++)
                {
                    Graphics.Blit(smoothedRT, smoothedRT2, smoothMat, 0);
                    (smoothedRT, smoothedRT2) = (smoothedRT2, smoothedRT);
                }
        }
        else
        {
            if (!smoothedRT.IsCreated()) smoothedRT.Create();
            Graphics.CopyTexture(field, smoothedRT);   // bit-exact: no presmooth / bilateral
        }
        bool applyTemporal = classicalApplyTemporal && temporalMat != null;
        if (provider.FrameCount > 1 && frameIdx < classicalTemporalFrame) hasHistory = false;
        classicalTemporalFrame = frameIdx;
        RenderTexture shadeSource = ApplyTemporalField(applyTemporal ? ClassicalInputSpeed() : field, applyTemporal);

        // same material for both sources (kThick policy in the header)
        float kNet = kThick;
        kThick = ClassicalKThick();
        SetShadeParams(shadeMat);
        Graphics.Blit(shadeSource, cRT, shadeMat, 0);
        Graphics.Blit(shadeSource, mRT, shadeMat, 1);
        Graphics.Blit(shadeSource, nRT, shadeMat, 2);
        OnShaded?.Invoke();

        compositeMat.SetTexture("_CTex", cRT);
        compositeMat.SetTexture("_MTex", mRT);
        compositeMat.SetTexture("_NTex", nRT);
        compositeMat.SetFloat("_FocalM", focalM);
        compositeMat.SetFloat("_WinAspect", winAspect);
        compositeMat.SetFloat("_SimScale", transform.lossyScale.x);
        compositeMat.SetFloat("_DepthBias", depthBias);
        compositeMat.SetVector("_CamRightWS", shadeRightWS);
        compositeMat.SetVector("_CamUpWS", shadeUpWS);
        compositeMat.SetVector("_CamFwdWS", shadeFwdWS);
        if (foamEnabled) ClassicalFoamDepth();
        kThick = kNet;                 // the network path's kThick state is untouched
        float ms = (float)sw.Elapsed.TotalMilliseconds;
        classicalMs = classicalMs <= 0f ? ms : Mathf.Lerp(classicalMs, ms, 0.1f);
        DrawFoam();
        quadMR.enabled = true;
    }

    // Reuse the network's complete input splat, including its footprint and velocity
    // normalization. The classical surface can extend beyond it; speed is then zero,
    // exactly as for the network. This is an opt-in reference path, not a GPU optimization.
    Texture ClassicalInputSpeed()
    {
        SplatToInput(frameIdx);
        if (classicalSpeedTex == null || classicalSpeedTex.width != W || classicalSpeedTex.height != H)
        {
            if (classicalSpeedTex != null) Destroy(classicalSpeedTex);
            classicalSpeedTex = new Texture2D(W, H, TextureFormat.RGBAFloat, false)
                { filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp };
            classicalSpeedPixels = new Color[H * W];
        }
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
        {
            int i = y * W + x;
            float speed = 0f;
            if (in7[6 * HW + i] > 0f)
            {
                float vx = in7[2 * HW + i] * meta.std[2] + meta.mean[2];
                float vy = in7[3 * HW + i] * meta.std[3] + meta.mean[3];
                float vz = in7[4 * HW + i] * meta.std[4] + meta.mean[4];
                speed = Mathf.Sqrt(vx * vx + vy * vy + vz * vz);
            }
            classicalSpeedPixels[(H - 1 - y) * W + x] = new Color(0f, 0f, 0f, speed);
        }
        classicalSpeedTex.SetPixels(classicalSpeedPixels);
        classicalSpeedTex.Apply(false, false);
        return classicalSpeedTex;
    }

    // G2's learned spray is looked up by reflection so this file never depends on G2's (separately committed) code
    System.Reflection.MethodInfo sprayCompositeMI;
    System.Reflection.FieldInfo sprayEnabledFI;
    bool sprayProbed;
    Texture2D classicalReadTex;

    bool ClassicalSprayOn()
    {
        if (!sprayProbed)
        {
            sprayProbed = true;
            const System.Reflection.BindingFlags bf = System.Reflection.BindingFlags.Instance |
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic;
            sprayCompositeMI = GetType().GetMethod("SprayComposite", bf, null, Type.EmptyTypes, null);
            sprayEnabledFI = GetType().GetField("sprayEnabled", bf);
        }
        return sprayCompositeMI != null && sprayEnabledFI != null && sprayEnabledFI.GetValue(this) is bool on && on;
    }

    float ClassicalKThick()
    {
        if (kThickOverride > 0f) return kThickOverride;
        if (kThickLocked && kThick > 0f) return kThick;          // the network's locked value: one shared material
        if (!classicalKThickLocked)
        {
            int n = H * W;
            if (classicalAuxCpu == null || classicalAuxCpu.Length != n) { classicalAuxCpu = new float[n]; classicalAux2Cpu = new float[n]; }
            classical.ReadSplat(null, classicalAux2Cpu, classicalAuxCpu);   // mask, thickness
            var vals = new List<float>(n / 4);
            for (int i = 0; i < n; i++) if (classicalAux2Cpu[i] > 0.5f) vals.Add(Mathf.Max(classicalAuxCpu[i], 0f));
            float med = 1f;
            if (vals.Count > 0) { vals.Sort(); med = vals[vals.Count / 2]; }
            classicalKThick = 1.2f / Mathf.Max(med, 1e-3f);
            if (provider.FrameCount <= 1 ? simTime >= 2.5f : frameIdx >= provider.FrameCount / 2)
                classicalKThickLocked = true;
        }
        return classicalKThick;
    }

    // FoamLayer depth-tests diffuse particles against the fluid depth (tensor layout, row 0 = top, 0 = no fluid).
    // Asynchronous: the foam of this frame is tested against the classical depth of the previous classical frame
    // (one frame late, invisible at 25 Hz solver motion); only the very first classical frame reads synchronously.
    bool classicalFoamPending, classicalFoamHave;
    void ClassicalFoamDepth()
    {
        int n = H * W;
        if (!classicalFoamHave)
        {
            classical.ReadShadedDepth(foamFluidDepth);
            classicalFoamHave = true;
        }
        if (classicalFoamPending) return;
        classicalFoamPending = true;
        classical.RequestShadedDepth(req =>
        {
            classicalFoamPending = false;
            if (req.hasError || foamFluidDepth == null || !ClassicalActive) return;
            var data = req.GetData<float>();
            if (data.Length == foamFluidDepth.Length) data.CopyTo(foamFluidDepth);
        });
    }

    float ClassicalDepthStd()
    {
        int n = H * W;
        if (classicalDepthCpu == null || classicalDepthCpu.Length != n) classicalDepthCpu = new float[n];
        if (classicalAux2Cpu == null || classicalAux2Cpu.Length != n) { classicalAuxCpu = new float[n]; classicalAux2Cpu = new float[n]; }
        classical.ReadFinalDepth(classicalDepthCpu);
        classical.ReadSplat(null, classicalAux2Cpu, null);
        double s = 0, s2 = 0; int k = 0;
        for (int i = 0; i < n; i++)
            if (classicalAux2Cpu[i] > 0.5f) { double d = classicalDepthCpu[i]; s += d; s2 += d * d; k++; }
        if (k < 2) return 0.05f;
        return (float)Math.Sqrt(Math.Max((s2 - s * s / k) / (k - 1), 0.0));
    }

    void DisposeClassical()
    {
        classical?.Dispose();
        classical = null;
        if (classicalReadTex != null) Destroy(classicalReadTex);
        classicalReadTex = null;
        if (classicalSpeedTex != null) Destroy(classicalSpeedTex);
        classicalSpeedTex = null;
        classicalSpeedPixels = null;
        classicalTemporalFrame = -1;
    }
}
