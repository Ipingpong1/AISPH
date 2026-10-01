// FluidSceneMVP.GpuPort.cs — GPU1001 (2026-10-01): the splat -> U-Net -> decode chain with no CPU round trip.
// Key U toggles it (off by default; the CPU path is the reference). One CommandBuffer per rendered frame:
//   GpuSplat.compute Clear/Splat/Pack  (GpuSphSolver's position/velocity/density buffers -> the U-Net input tensor)
//   -> ScheduleWorker (the input tensor is pinned to GPU memory; the output stays there)
//   -> Decode (output tensor -> field buffer, texture layout) -> [spray] -> FieldToTex -> ShadeField (unchanged passes).
// What still reads back, and only when needed: FoamLayer (CPU, out of scope) needs the particles (provider.GetFrame)
// and the fluid depth (async, one frame late) while foam is on; kThick auto (kThickOverride <= 0) reads the prediction
// asynchronously until it locks; the bilateral range sigma comes from an async per-block reduction (one frame late).
// Not covered (falls back to the CPU path): spread inference, LiveClipRecorder taps (OnInferred subscribers), the
// legacy splat, a non-GPUCompute backend. Gates + timings: SSU_restart/Experiments/GPU1001/PLAN.md.
using System;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;
using Unity.InferenceEngine;

public partial class FluidSceneMVP
{
    [Header("GPU1001: splat / decode / spray on the GPU (key U) — off by default")]
    [Tooltip("Run the V2 splat, the network input, the decode and (once ported) the learned spray as compute shaders: the particles stay in the GPU solver's buffers and the prediction never comes back to the CPU. Toggle at runtime with U.")]
    public bool gpuPath = false;
    [Tooltip("GpuSplat.compute; empty = Resources/GpuSplat.")]
    public ComputeShader gpuSplatShader;

    GpuSplatPass gpuSplat;
    // The GPU chain needs its OWN worker: Inference Engine 2.6.1's ScheduleWorker(cb, ...) calls SetCommandBuffer, and the
    // GPUCompute backend then records every later Schedule() into that external buffer for good (m_InternalCommandBuffer
    // stays false), so the CPU path's RunModel would silently read back the previous prediction. Found by gate S3.
    Worker gpuWorker, gpuWorkerTwin;   // twin = the CPU worker it mirrors (rebuilt when the model bank swaps it)
    Tensor<float> gpuInputT;
    ComputeBuffer gpuIn7;
    RenderTexture gpuFieldRT;
    CommandBuffer gpuCb;
    string gpuError;
    bool gpuWasActive;
    // async readbacks (at most one in flight each)
    bool gpuDepthPending, gpuFgPending, gpuPredPending;
    // GPU timings (ms, from the profiler Recorders of the named command-buffer samples; a frame or two late)
    Recorder recSplat, recNet, recDecode, recSpray;
    public float GpuMsSplat { get; private set; }
    public float GpuMsNet { get; private set; }
    public float GpuMsDecode { get; private set; }
    public float GpuMsSpray { get; private set; }
    const string SampleNet = "GPU1001.UNet";
    CustomSampler SamplerNet, SamplerSpray;   // created in EnsureGpuPath (main thread)

    public bool GpuPathEnabled => gpuPath;
    public void ToggleGpuPath() { gpuPath = !gpuPath; gpuError = null; }

    /// <summary>True when this frame runs the GPU chain.</summary>
    bool GpuPathActive => gpuPath && useV2 && ActiveSpread == 0 && OnInferred == null && EnsureGpuPath();

    bool EnsureGpuPath()
    {
        if (gpuSplat != null) return true;
        if (gpuError != null) return false;   // failed once: stay on the CPU path until toggled again (no per-frame retry)
        try
        {
            if (backend != BackendType.GPUCompute) throw new Exception($"backend {backend} — the GPU path needs GPUCompute");
            var shader = gpuSplatShader != null ? gpuSplatShader : Resources.Load<ComputeShader>("GpuSplat");
            if (shader == null) throw new Exception("GpuSplat.compute not found (Assets/GpuPort/Resources)");
            gpuSplat = new GpuSplatPass(shader, W, H);
            gpuInputT = new Tensor<float>(new TensorShape(1, 7, H, W));
            var td = ComputeTensorData.Pin(gpuInputT, true);
            gpuIn7 = td.buffer;
            if (gpuIn7.stride != 4 || gpuIn7.count < 7 * HW) throw new Exception($"input tensor buffer stride {gpuIn7.stride} count {gpuIn7.count}");
            gpuFieldRT = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBFloat)
            { enableRandomWrite = true, filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp, name = "GPU1001.Field" };
            gpuFieldRT.Create();
            gpuCb = new CommandBuffer { name = "GPU1001" };
            SamplerNet = CustomSampler.Create(SampleNet, true);
            SamplerSpray = CustomSampler.Create(GpuSprayLayerSample, true);
            recSplat = gpuSplat.SamplerSplat.GetRecorder(); recSplat.enabled = true;
            recNet = SamplerNet.GetRecorder(); recNet.enabled = true;
            recDecode = gpuSplat.SamplerDecode.GetRecorder(); recDecode.enabled = true;
            recSpray = SamplerSpray.GetRecorder(); recSpray.enabled = true;
            Debug.Log($"FluidSceneMVP: GPU1001 path ready ({W}x{H}, input tensor buffer {gpuIn7.count} floats, GPU recorder {SystemInfo.supportsGpuRecorder})");
            return true;
        }
        catch (Exception e)
        {
            gpuError = e.Message;
            Debug.LogError($"FluidSceneMVP: GPU1001 path disabled — {e.Message}");
            DisposeGpuPath();
            gpuError = e.Message;
            return false;
        }
    }

    void DisposeGpuPath()
    {
        if (gpuDepthPending || gpuFgPending || gpuPredPending) AsyncGPUReadback.WaitAllRequests();
        gpuDepthPending = gpuFgPending = gpuPredPending = false;
        DisposeGpuSpray();
        gpuSplat?.Dispose(); gpuSplat = null;
        gpuWorker?.Dispose(); gpuWorker = null; gpuWorkerTwin = null;
        gpuInputT?.Dispose(); gpuInputT = null; gpuIn7 = null;   // the buffer belongs to the tensor
        if (gpuFieldRT != null) { gpuFieldRT.Release(); Destroy(gpuFieldRT); gpuFieldRT = null; }
        gpuCb?.Release(); gpuCb = null;
    }

    GpuSplatPass.Camera GpuCam() => new GpuSplatPass.Camera { eye = eyeSim, right = rightSim, up = upSim, fwd = fwdSim, focal = focalM };
    GpuSplatPass.SplatParams GpuSplatParams() => new GpuSplatPass.SplatParams { worldR = v2R, thickScaleV2 = v2TS, minR = v2MinR, maxR = v2MaxR };

    /// <summary>The current particle frame as GPU buffers: the solver's own (no copy) or an upload of the CPU frame.</summary>
    void GpuParticles(out GraphicsBuffer pos, out GraphicsBuffer vel, out GraphicsBuffer dens, out int n)
    {
        if (provider is GpuSphProvider g)
        {
            var s = g.Solver;
            pos = s.PositionBuffer; vel = s.VelocityBuffer; dens = s.DensityBuffer; n = s.Count;
            return;
        }
        provider.GetFrame(frameIdx, out var data, out int off, out int count);
        gpuSplat.UploadRecords(data, off, count, out pos, out vel, out dens);
        n = count;
    }

    /// <summary>Record splat + U-Net + decode for the current camera into gpuCb (not executed). Returns the output buffer.</summary>
    ComputeBuffer RecordGpuInference()
    {
        EnsureGpuWorker();
        GpuParticles(out var pos, out var vel, out var dens, out int n);
        gpuSplat.RecordSplat(gpuCb, pos, vel, dens, n, GpuCam(), GpuSplatParams(), gpuIn7, meta.mean, meta.std, thickScale);
        gpuCb.BeginSample(SamplerNet);
        gpuCb.ScheduleWorker(gpuWorker, gpuInputT);
        gpuCb.EndSample(SamplerNet);
        var outT = gpuWorker.PeekOutput() as Tensor<float>;
        var predBuf = ComputeTensorData.Pin(outT).buffer;
        gpuSplat.RecordDecode(gpuCb, predBuf, gpuIn7, meta.mean, meta.std, thickScale, showRawInput, temporalMat != null);
        return predBuf;
    }

    void EnsureGpuWorker()
    {
        if (gpuWorker != null && gpuWorkerTwin == worker) return;
        gpuWorker?.Dispose();
        var asset = bankModels != null && bankIdx > 0 && bankIdx < bankModels.Length ? bankModels[bankIdx] : modelAsset;
        gpuWorker = new Worker(ModelLoader.Load(asset), BackendType.GPUCompute);   // fp32 (fp16 fails at schedule time on IE 2.6.1)
        gpuWorkerTwin = worker;
    }

    void GpuLateUpdate()
    {
        if (!gpuWasActive) { DrainPending(); gpuWasActive = true; }
        ComputeSimCamera();
        gpuCb.Clear();
        var predBuf = RecordGpuInference();
        StepFoam();   // CPU foam (out of scope): reads the particles back only while foam is on
        Transform c = targetCamera.transform;
        shadeRightWS = c.right; shadeUpWS = c.up; shadeFwdWS = c.forward;

        float now = Time.realtimeSinceStartup;
        if (lastPredTime >= 0f && now > lastPredTime)
            fluidHz = fluidHz <= 0f ? 1f / (now - lastPredTime) : Mathf.Lerp(fluidHz, 1f / (now - lastPredTime), 0.2f);
        lastPredTime = now;

        // kThick: override, or the median predicted thickness from an async readback until it locks
        if (kThickOverride > 0f) kThick = kThickOverride;
        else if (!kThickLocked && !gpuPredPending)
        {
            gpuPredPending = true;
            int hw = HW;
            gpuCb.RequestAsyncReadback(predBuf, 7 * hw * 4, 0, req =>
            {
                gpuPredPending = false;
                if (req.hasError || kThickLocked || kThickOverride > 0f) return;
                var a = req.GetData<float>();
                var vals = new System.Collections.Generic.List<float>(hw / 4);
                float tm = meta.mean[1], ts = meta.std[1];
                for (int i = 0; i < hw; i++) if (a[6 * hw + i] > 0f) vals.Add(Mathf.Max(a[hw + i] * ts + tm, 0f));
                float med = 1f;
                if (vals.Count > 0) { vals.Sort(); med = vals[vals.Count / 2]; }
                kThick = 1.2f / Mathf.Max(med, 1e-3f);
                if (provider.FrameCount <= 1 ? simTime >= 2.5f : frameIdx >= provider.FrameCount / 2) kThickLocked = true;
            });
        }

        // bilateral range sigma: async per-block foreground depth moments (one frame late)
        if (bilateralSmoothing && !gpuFgPending)
        {
            gpuFgPending = true;
            gpuSplat.RecordFgStats(gpuCb);
            gpuCb.RequestAsyncReadback(gpuSplat.FgPartial, req =>
            {
                gpuFgPending = false;
                if (req.hasError) return;
                var a = req.GetData<Vector4>();
                double s = 0, s2 = 0, cnt = 0;
                for (int i = 0; i < a.Length; i++) { s += a[i].x; s2 += a[i].y; cnt += a[i].z; }
                bilateralSigmaR = 0.05f;
                if (cnt > 1)
                {
                    double var = (s2 - s * s / cnt) / (cnt - 1);
                    bilateralSigmaR = Mathf.Max(bilateralRangeScale * (float)Math.Sqrt(Math.Max(var, 0.0)), 1e-3f);
                }
            });
        }

        RecordGpuSprayComposite(gpuCb);   // learned spray (GPU layer when available, else the CPU layer via one field readback)

        gpuSplat.RecordFieldToTex(gpuCb, gpuFieldRT);

        // CPU foam's depth test reads the fluid depth one frame late
        if (foamEnabled && !gpuDepthPending)
        {
            gpuDepthPending = true;
            gpuCb.RequestAsyncReadback(gpuSplat.FluidDepth, req =>
            {
                gpuDepthPending = false;
                if (!req.hasError && foamFluidDepth != null) req.GetData<float>().CopyTo(foamFluidDepth);
            });
        }

        Graphics.ExecuteCommandBuffer(gpuCb);
        AfterGpuExecute();
        ShadeField(gpuFieldRT);
        ReadGpuTimings();
    }

    void ReadGpuTimings()
    {
        if (recSplat != null) GpuMsSplat = recSplat.gpuElapsedNanoseconds * 1e-6f;
        if (recNet != null) GpuMsNet = recNet.gpuElapsedNanoseconds * 1e-6f;
        if (recDecode != null) GpuMsDecode = recDecode.gpuElapsedNanoseconds * 1e-6f;
        if (recSpray != null) GpuMsSpray = recSpray.gpuElapsedNanoseconds * 1e-6f;
    }

    string GpuStatus()
    {
        if (!gpuPath) return "";
        if (gpuError != null) return $"   GPU1001 ERROR: {gpuError}";
        if (!GpuPathActive) return "   GPU1001 (inactive: spread / recorder / legacy splat)";
        return $"   GPU1001 gpu splat {GpuMsSplat:F2} net {GpuMsNet:F2} decode {GpuMsDecode:F2} spray {GpuMsSpray:F2} ms, main {LateUpdateMs:F1} ms";
    }
}
