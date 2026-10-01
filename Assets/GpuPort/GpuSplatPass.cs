// GpuSplatPass.cs — GPU1001 (2026-10-01): buffer owner + command recorder for Resources/GpuSplat.compute.
// SplatV2 + the network-input normalisation (SplatToInput) and the decode loop of ProcessPrediction, recorded into a
// CommandBuffer so the whole chain (splat -> pack -> U-Net -> decode -> field texture) runs without a CPU round trip.
// Every stage is wrapped in a named sample: Recorder.Get(name).gpuElapsedNanoseconds gives its GPU time.
// Gates: SSU_restart/Experiments/GPU1001/PLAN.md (S1 splat vs SplatV2.cs, S2 decode vs ProcessPrediction).
using System;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;

public sealed class GpuSplatPass : IDisposable
{
    public const string SampleSplat = "GPU1001.Splat", SampleDecode = "GPU1001.Decode";
    // GPU timing needs samplers created with collectGpuData (string samples report 0 GPU ns)
    public readonly CustomSampler SamplerSplat, SamplerDecode;   // created in the constructor (main thread)
    const int GroupsX = 1024;   // GpuSplat.compute GROUPS_X

    readonly ComputeShader cs;
    readonly int kClear, kSplat, kPack, kDecode, kFieldToTex, kFgStats;
    public readonly int W, H, HW;

    // accumulators (tensor layout) + outputs
    public GraphicsBuffer DepthKey { get; private set; }
    public GraphicsBuffer Thick { get; private set; }
    public GraphicsBuffer VelX { get; private set; }
    public GraphicsBuffer VelY { get; private set; }
    public GraphicsBuffer VelZ { get; private set; }
    public GraphicsBuffer Den { get; private set; }
    public GraphicsBuffer Field { get; private set; }        // float4, TEXTURE layout (row 0 = bottom)
    public GraphicsBuffer FluidDepth { get; private set; }   // float, tensor layout
    public GraphicsBuffer FgPartial { get; private set; }    // float4 per 256-px block
    public int FgBlocks => (HW + 255) / 256;

    // CPU-side particle upload (non-GPU providers only)
    GraphicsBuffer upPos, upVel, upDens;
    Vector3[] upPosStage, upVelStage; float[] upDensStage;

    public GpuSplatPass(ComputeShader shader, int w, int h)
    {
        cs = shader ?? throw new ArgumentNullException(nameof(shader));
        W = w; H = h; HW = w * h;
        kClear = cs.FindKernel("Clear"); kSplat = cs.FindKernel("Splat"); kPack = cs.FindKernel("Pack");
        SamplerSplat = CustomSampler.Create(SampleSplat, true);
        SamplerDecode = CustomSampler.Create(SampleDecode, true);
        kDecode = cs.FindKernel("Decode"); kFieldToTex = cs.FindKernel("FieldToTex"); kFgStats = cs.FindKernel("FgStats");
        DepthKey = Buf(HW, 4); Thick = Buf(HW, 4); VelX = Buf(HW, 4); VelY = Buf(HW, 4); VelZ = Buf(HW, 4); Den = Buf(HW, 4);
        Field = Buf(HW, 16); FluidDepth = Buf(HW, 4); FgPartial = Buf(FgBlocks, 16);
    }

    static GraphicsBuffer Buf(int count, int stride) => new GraphicsBuffer(GraphicsBuffer.Target.Structured, Math.Max(count, 1), stride);

    public struct Camera
    {
        public Vector3 eye, right, up, fwd; public float focal;
    }

    public struct SplatParams
    {
        public float worldR, thickScaleV2, minR, maxR;
    }

    /// <summary>Upload a 7-float record frame (ParticleFrameProvider contract) for providers without GPU buffers.</summary>
    public void UploadRecords(float[] data, int off, int count, out GraphicsBuffer pos, out GraphicsBuffer vel, out GraphicsBuffer dens)
    {
        if (upPos == null || upPos.count < count)
        {
            upPos?.Release(); upVel?.Release(); upDens?.Release();
            int cap = Math.Max(count, 1024);
            upPos = Buf(cap, 12); upVel = Buf(cap, 12); upDens = Buf(cap, 4);
            upPosStage = new Vector3[cap]; upVelStage = new Vector3[cap]; upDensStage = new float[cap];
        }
        for (int i = 0; i < count; i++)
        {
            int b = off + i * 7;
            upPosStage[i] = new Vector3(data[b], data[b + 1], data[b + 2]);
            upVelStage[i] = new Vector3(data[b + 3], data[b + 4], data[b + 5]);
            upDensStage[i] = data[b + 6];
        }
        if (count > 0) { upPos.SetData(upPosStage, 0, 0, count); upVel.SetData(upVelStage, 0, 0, count); upDens.SetData(upDensStage, 0, 0, count); }
        pos = upPos; vel = upVel; dens = upDens;
    }

    /// <summary>Clear + splat + pack into `in7` (the U-Net input tensor's buffer, [7, H, W]).</summary>
    public void RecordSplat(CommandBuffer cb, GraphicsBuffer pos, GraphicsBuffer vel, GraphicsBuffer dens, int n, Camera c, SplatParams sp,
                            ComputeBuffer in7, float[] mean, float[] std, float thickScale)
    {
        cb.BeginSample(SamplerSplat);
        cb.SetComputeIntParam(cs, "_N", n);
        cb.SetComputeIntParam(cs, "_H", H);
        cb.SetComputeIntParam(cs, "_W", W);
        cb.SetComputeVectorParam(cs, "_Eye", c.eye);
        cb.SetComputeVectorParam(cs, "_Right", c.right);
        cb.SetComputeVectorParam(cs, "_Up", c.up);
        cb.SetComputeVectorParam(cs, "_Fwd", c.fwd);
        cb.SetComputeFloatParam(cs, "_Focal", c.focal);
        cb.SetComputeFloatParam(cs, "_WorldR", sp.worldR);
        cb.SetComputeFloatParam(cs, "_ThickScaleV2", sp.thickScaleV2);
        cb.SetComputeFloatParam(cs, "_MinR", sp.minR);
        cb.SetComputeFloatParam(cs, "_MaxR", sp.maxR);
        BindAccum(cb, kClear);
        cb.DispatchCompute(cs, kClear, (HW + 63) / 64, 1, 1);
        if (n > 0)
        {
            BindAccum(cb, kSplat);
            cb.SetComputeBufferParam(cs, kSplat, "_PosIn", pos);
            cb.SetComputeBufferParam(cs, kSplat, "_VelIn", vel);
            cb.SetComputeBufferParam(cs, kSplat, "_DensIn", dens);
            cb.DispatchCompute(cs, kSplat, Math.Min(n, GroupsX), (n + GroupsX - 1) / GroupsX, 1);
        }
        SetNorm(cb, mean, std, thickScale);
        BindAccum(cb, kPack);
        cb.SetComputeBufferParam(cs, kPack, "_In7", in7);
        cb.DispatchCompute(cs, kPack, (HW + 63) / 64, 1, 1);
        cb.EndSample(SamplerSplat);
    }

    /// <summary>Prediction ([7, H, W]) -> Field (texture layout) + FluidDepth; optionally the raw splat instead.</summary>
    public void RecordDecode(CommandBuffer cb, ComputeBuffer pred, ComputeBuffer in7, float[] mean, float[] std, float thickScale,
                             bool showRaw, bool temporal)
    {
        cb.BeginSample(SamplerDecode);
        SetNorm(cb, mean, std, thickScale);
        cb.SetComputeIntParam(cs, "_ShowRaw", showRaw ? 1 : 0);
        cb.SetComputeIntParam(cs, "_Temporal", temporal ? 1 : 0);
        BindAccum(cb, kDecode);
        cb.SetComputeBufferParam(cs, kDecode, "_Pred", pred);
        cb.SetComputeBufferParam(cs, kDecode, "_In7R", in7);
        cb.SetComputeBufferParam(cs, kDecode, "_Field", Field);
        cb.SetComputeBufferParam(cs, kDecode, "_FluidDepth", FluidDepth);
        cb.DispatchCompute(cs, kDecode, (HW + 63) / 64, 1, 1);
        cb.EndSample(SamplerDecode);
    }

    public void RecordFgStats(CommandBuffer cb)
    {
        cb.SetComputeBufferParam(cs, kFgStats, "_FieldR", Field);
        cb.SetComputeBufferParam(cs, kFgStats, "_FgPartial", FgPartial);
        cb.DispatchCompute(cs, kFgStats, FgBlocks, 1, 1);
    }

    public void RecordFieldToTex(CommandBuffer cb, RenderTexture rt)
    {
        cb.SetComputeIntParam(cs, "_H", H);
        cb.SetComputeIntParam(cs, "_W", W);
        cb.SetComputeBufferParam(cs, kFieldToTex, "_FieldR", Field);
        cb.SetComputeTextureParam(cs, kFieldToTex, "_FieldTex", rt);
        cb.DispatchCompute(cs, kFieldToTex, (W + 7) / 8, (H + 7) / 8, 1);
    }

    void BindAccum(CommandBuffer cb, int k)
    {
        cb.SetComputeBufferParam(cs, k, "_DepthKey", DepthKey);
        cb.SetComputeBufferParam(cs, k, "_Thick", Thick);
        cb.SetComputeBufferParam(cs, k, "_VelX", VelX);
        cb.SetComputeBufferParam(cs, k, "_VelY", VelY);
        cb.SetComputeBufferParam(cs, k, "_VelZ", VelZ);
        cb.SetComputeBufferParam(cs, k, "_Den", Den);
    }

    void SetNorm(CommandBuffer cb, float[] mean, float[] std, float thickScale)
    {
        cb.SetComputeIntParam(cs, "_H", H);
        cb.SetComputeIntParam(cs, "_W", W);
        cb.SetComputeVectorParam(cs, "_Mean0123", new Vector4(mean[0], mean[1], mean[2], mean[3]));
        cb.SetComputeVectorParam(cs, "_Mean45", new Vector4(mean[4], mean[5], 0f, 0f));
        cb.SetComputeVectorParam(cs, "_InvStd0123", new Vector4(1f / std[0], 1f / std[1], 1f / std[2], 1f / std[3]));
        cb.SetComputeVectorParam(cs, "_InvStd45", new Vector4(1f / std[4], 1f / std[5], 0f, 0f));
        cb.SetComputeVectorParam(cs, "_Std0123", new Vector4(std[0], std[1], std[2], std[3]));
        cb.SetComputeVectorParam(cs, "_Std45", new Vector4(std[4], std[5], 0f, 0f));
        cb.SetComputeFloatParam(cs, "_ThickScale", thickScale);
    }

    public void Dispose()
    {
        DepthKey?.Release(); Thick?.Release(); VelX?.Release(); VelY?.Release(); VelZ?.Release(); Den?.Release();
        Field?.Release(); FluidDepth?.Release(); FgPartial?.Release();
        upPos?.Release(); upVel?.Release(); upDens?.Release();
        DepthKey = Thick = VelX = VelY = VelZ = Den = Field = FluidDepth = FgPartial = null;
        upPos = upVel = upDens = null;
    }
}
