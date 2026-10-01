// GpuSprayLayer.cs — GPU1001 (2026-10-01): buffer owner + command recorder for Resources/GpuSpray.compute, the GPU
// port of LearnedSprayLayer (G2). Same knobs and defaults; per solver frame RecordStep() records grid -> flight/culls ->
// features -> MLP (its own Inference Engine worker, scheduled into the same command buffer) -> births; per rendered
// frame RecordRaster() draws the droplets against the bulk field. Status counters come back by async readback (one
// step late). Parity hooks (gates P1-P5, SSU_restart/Experiments/GPU1001/PLAN.md) read individual stages back.
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.InferenceEngine;

public sealed class GpuSprayLayer : IDisposable
{
    public const int NFeat = LearnedSprayLayer.NFeat, KNN = LearnedSprayLayer.KNN, NPack = LearnedSprayLayer.NPack;
    const int Hist = 1024, MaxObst = 8;

    // ---- knobs (LearnedSprayLayer defaults = 4f LOR on emitter L)
    public float rateScale = 2.9f, launchSpeed = 3f, sigIn = 0.08748386701656483f, dropletRadius = 0f, volumeRatio = 25f;
    public int maxDroplets = 30000, maxEvents = 65536, mergeGrace = 2, substeps = 4, batchQuantum = 4096;
    public float mergeR = 1f, maxAge = 2f, rateCap = 50f;
    public double frameDt = 1.0 / 25.0;
    public Vector3 gravity = new Vector3(0f, -9.81f, 0f), domainMin = Vector3.zero, domainMax = new Vector3(3f, 3f, 3f);
    public bool cullObstacles = true, cullMerge = true;
    public uint seed;
    public readonly List<LearnedSprayLayer.Obstacle> obstacles = new List<LearnedSprayLayer.Obstacle>();

    // ---- readouts (async, one step late)
    public int LastEvents, LastBorn, LastDropped, LastEventOverflow, Alive, KillWall, KillObst, KillMerge, KillAge;
    public float LastRateSum;
    public long StepCount;
    public float RD { get; private set; }

    readonly ComputeShader cs;
    readonly int kReset, kGClear, kGCount, kGScan, kGScatter, kAdvance, kFeatA, kFeatHist, kFeatPct, kFeatB, kBirths,
                 kEventArgs, kSpawn, kDropArgs, kRClear, kRDepth, kRNormal, kRResolve;

    Worker worker; ModelAsset modelAsset;
    Tensor<float> xT; ComputeBuffer xBuf; int nPadCur = -1;

    int capN = -1, cells = -1;
    GraphicsBuffer cellCount, cellStart, cellFill, cellOf, sorted, cnt, ita, nhat, e1, e2, knn, hist, scalarU, scalarF;
    GraphicsBuffer events, eventCount, argsEvent, argsDrop, argsRaster, poolA, poolB, countA, countB, stats, obst, lat, latCum, sizeEdges;
    int rasterHW = -1;
    GraphicsBuffer zKey, zChord, zSpeed, zNrm;
    public RenderTexture SprayFieldRT { get; private set; }
    public RenderTexture SprayNrmRT { get; private set; }
    bool statsPending;
    readonly ObstGpu[] obstStage = new ObstGpu[MaxObst];

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    struct ObstGpu { public Vector4 r0, r1, r2; public int shape; public float pad; public Vector2 pad2; }   // 64 bytes
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct EventGpu { public Vector4 c, u; public uint parent, k, compBin, round; }                  // 48 bytes
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct DropGpu { public Vector4 xa; public Vector3 v; public uint id; }                            // 32 bytes; xa.w = age

    public GpuSprayLayer(ComputeShader shader)
    {
        cs = shader ?? throw new ArgumentNullException(nameof(shader));
        kReset = cs.FindKernel("ResetStep"); kGClear = cs.FindKernel("GridClear"); kGCount = cs.FindKernel("GridCount");
        kGScan = cs.FindKernel("GridScan"); kGScatter = cs.FindKernel("GridScatter"); kAdvance = cs.FindKernel("Advance");
        kFeatA = cs.FindKernel("FeatA"); kFeatHist = cs.FindKernel("FeatHist"); kFeatPct = cs.FindKernel("FeatPercentile");
        kFeatB = cs.FindKernel("FeatB"); kBirths = cs.FindKernel("Births"); kEventArgs = cs.FindKernel("EventArgs");
        kSpawn = cs.FindKernel("Spawn"); kDropArgs = cs.FindKernel("DropArgs"); kRClear = cs.FindKernel("RClear");
        kRDepth = cs.FindKernel("RDepth"); kRNormal = cs.FindKernel("RNormal"); kRResolve = cs.FindKernel("RResolve");

        hist = Buf(Hist, 4); scalarU = Buf(4, 4); scalarF = Buf(4, 4);
        eventCount = Buf(4, 4); stats = Buf(16, 4); obst = Buf(MaxObst, 64);
        argsEvent = Args(); argsDrop = Args(); argsRaster = Args();
        countA = Buf(4, 4); countB = Buf(4, 4);
        countA.SetData(new uint[4]); countB.SetData(new uint[4]); eventCount.SetData(new uint[4]);
        sizeEdges = Buf(LearnedSprayLayer.SizeEdges.Length, 4); sizeEdges.SetData(LearnedSprayLayer.SizeEdges);
        BuildLatticeBuffers();
        EnsureEvents(maxEvents);
    }

    static GraphicsBuffer Buf(int count, int stride) => new GraphicsBuffer(GraphicsBuffer.Target.Structured, Math.Max(count, 1), stride);
    static GraphicsBuffer Args() => new GraphicsBuffer(GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments, 3, 4);

    void BuildLatticeBuffers()
    {
        // the C# layer's lattice (15^3 offsets sorted by |o| + 1e-6 index tie-break), prefix sums in double, cast once
        int L = 15, N = L * L * L;
        var key = new double[N]; var idx = new int[N]; var o = new int[3 * N];
        int q = 0;
        for (int a = -7; a <= 7; a++) for (int b = -7; b <= 7; b++) for (int c = -7; c <= 7; c++)
        { o[3 * q] = a; o[3 * q + 1] = b; o[3 * q + 2] = c; key[q] = Math.Sqrt(a * a + b * b + c * c) + 1e-6 * q; idx[q] = q; q++; }
        Array.Sort(key, idx);
        var latV = new Vector4[N]; var cumV = new Vector4[N + 1];
        double sx = 0, sy = 0, sz = 0;
        for (int m = 0; m < N; m++)
        {
            int s = idx[m];
            latV[m] = new Vector4(o[3 * s], o[3 * s + 1], o[3 * s + 2], 0f);
            sx += o[3 * s]; sy += o[3 * s + 1]; sz += o[3 * s + 2];
            cumV[m + 1] = new Vector4((float)sx, (float)sy, (float)sz, 0f);
        }
        lat = Buf(N, 16); lat.SetData(latV);
        latCum = Buf(N + 1, 16); latCum.SetData(cumV);
    }

    void EnsureEvents(int cap)
    {
        if (events != null && events.count >= cap) return;
        events?.Release();
        events = Buf(cap, 48);
    }

    void EnsureCapacity(int n)
    {
        if (n <= capN && poolA != null) return;
        int c = Math.Max(n, Math.Max(capN * 2, 4096));
        foreach (var b in new[] { cellOf, sorted, cnt, ita, nhat, e1, e2, knn }) b?.Release();
        cellOf = Buf(c, 4); sorted = Buf(c, 4); cnt = Buf(c, 4); ita = Buf(c, 4);
        nhat = Buf(c, 16); e1 = Buf(c, 16); e2 = Buf(c, 16); knn = Buf(c * KNN, 4);
        capN = c;
        if (poolA == null || poolA.count < maxDroplets)
        {
            poolA?.Release(); poolB?.Release();
            poolA = Buf(Math.Max(maxDroplets, 1024), 32); poolB = Buf(Math.Max(maxDroplets, 1024), 32);
            countA.SetData(new uint[4]); countB.SetData(new uint[4]);
        }
    }

    public void SetModel(ModelAsset asset)
    {
        if (asset == null) throw new ArgumentNullException(nameof(asset));
        if (worker != null && asset == modelAsset) return;
        worker?.Dispose();
        worker = new Worker(ModelLoader.Load(asset), BackendType.GPUCompute);   // own worker: ScheduleWorker(cb) binds it to cb for good
        modelAsset = asset;
    }

    public void Reset()
    {
        if (countA != null) countA.SetData(new uint[4]);
        if (countB != null) countB.SetData(new uint[4]);
        LastEvents = LastBorn = LastDropped = Alive = 0; LastRateSum = 0f;
    }

    int PaddedRows(int rows) { int q = Math.Max(batchQuantum, 1); return Math.Max(q, (rows + q - 1) / q * q); }

    void EnsureInput(int n)
    {
        int nPad = PaddedRows(n);
        if (xT != null && nPadCur == nPad) return;
        xT?.Dispose();
        xT = new Tensor<float>(new TensorShape(nPad, NFeat));
        xBuf = ComputeTensorData.Pin(xT, true).buffer;
        nPadCur = nPad;
    }

    // ------------------------------------------------------------------ parameters
    void SetParams(CommandBuffer cb, int n, float rc)
    {
        double rdD = dropletRadius > 0f ? dropletRadius : rc / Math.Pow(Math.Max(volumeRatio, 1e-6), 1.0 / 3.0);
        RD = (float)rdD;
        float h = 4f * rc;
        float cell = h > 0f ? h : 1f;
        Vector3 ext = domainMax - domainMin;
        int gx = (int)Math.Floor(ext.x / cell) + 3, gy = (int)Math.Floor(ext.y / cell) + 3, gz = (int)Math.Floor(ext.z / cell) + 3;
        int nc = gx * gy * gz;
        if (nc != cells)
        {
            cellCount?.Release(); cellStart?.Release(); cellFill?.Release();
            cellCount = Buf(nc, 4); cellStart = Buf(nc + 1, 4); cellFill = Buf(nc, 4);
            cells = nc;
        }
        Vector3 g = gravity;
        float gn = g.magnitude;
        Vector3 up = gn > 0f ? -g / gn : Vector3.up;
        // the C# layer's fixed horizontal axis (in double there)
        double ux = up.x, uy = up.y, uz = up.z;
        double rx = Math.Abs(ux) < 0.9 ? 1.0 : 0.0, ry = 0.0, rz = Math.Abs(ux) < 0.9 ? 0.0 : 1.0;
        double ru = rx * ux + ry * uy + rz * uz; rx -= ru * ux; ry -= ru * uy; rz -= ru * uz;
        double rn = Math.Sqrt(rx * rx + ry * ry + rz * rz);

        cb.SetComputeIntParam(cs, "_N", n);
        cb.SetComputeFloatParam(cs, "_Rc", rc);
        cb.SetComputeFloatParam(cs, "_H", h);
        cb.SetComputeFloatParam(cs, "_Rd", RD);
        cb.SetComputeVectorParam(cs, "_Upv", up);
        cb.SetComputeVectorParam(cs, "_Ref", new Vector3((float)(rx / rn), (float)(ry / rn), (float)(rz / rn)));
        cb.SetComputeVectorParam(cs, "_G", gravity);
        cb.SetComputeVectorParam(cs, "_DomMin", domainMin);
        cb.SetComputeVectorParam(cs, "_DomMax", domainMax);
        cb.SetComputeVectorParam(cs, "_GridMin", domainMin - Vector3.one * cell);
        cb.SetComputeFloatParam(cs, "_Cell", cell);
        cb.SetComputeIntParams(cs, "_Gn", gx, gy, gz);
        cb.SetComputeIntParam(cs, "_Cells", nc);
        float sR = rateScale;
        cb.SetComputeFloatParam(cs, "_SR", sR);
        cb.SetComputeFloatParam(cs, "_Cap", rateCap * Mathf.Max(sR, 1f));
        cb.SetComputeFloatParam(cs, "_Launch", launchSpeed);
        cb.SetComputeFloatParam(cs, "_SigIn", sigIn);
        cb.SetComputeIntParam(cs, "_MaxDrops", maxDroplets);
        cb.SetComputeIntParam(cs, "_MaxEvents", Math.Min(maxEvents, events.count));
        cb.SetComputeIntParam(cs, "_MergeGrace", mergeGrace);
        cb.SetComputeIntParam(cs, "_Substeps", substeps);
        cb.SetComputeFloatParam(cs, "_MergeR2", (float)Math.Pow(Math.Min(mergeR, 4f) * rc, 2));
        cb.SetComputeFloatParam(cs, "_MaxAge", maxAge);
        cb.SetComputeFloatParam(cs, "_FrameDt", (float)frameDt);
        cb.SetComputeIntParam(cs, "_CullObst", cullObstacles ? 1 : 0);
        cb.SetComputeIntParam(cs, "_CullMerge", cullMerge ? 1 : 0);
        int no = Math.Min(obstacles.Count, MaxObst);
        for (int o = 0; o < no; o++)
        {
            var ob = obstacles[o];
            obstStage[o] = new ObstGpu { r0 = ob.toLocal.GetRow(0), r1 = ob.toLocal.GetRow(1), r2 = ob.toLocal.GetRow(2), shape = ob.shape, pad = ob.padLocal };
        }
        cb.SetBufferData(obst, obstStage, 0, 0, MaxObst);
        cb.SetComputeIntParam(cs, "_NumObst", no);
        cb.SetComputeIntParam(cs, "_Seed", (int)seed);
        cb.SetComputeIntParam(cs, "_Step", (int)(uint)StepCount);
        cb.SetComputeIntParam(cs, "_Round", 0);
        cb.SetComputeIntParam(cs, "_KeepEvents", 0);
    }

    void B(CommandBuffer cb, int k, string name, GraphicsBuffer b) => cb.SetComputeBufferParam(cs, k, name, b);
    void B(CommandBuffer cb, int k, string name, ComputeBuffer b) => cb.SetComputeBufferParam(cs, k, name, b);
    static int G64(int n) => Math.Max((n + 63) / 64, 1);

    // ------------------------------------------------------------------ one solver frame
    /// <summary>Grid, flight + culls, features, MLP, births for the coarse state in (pos, vel) — recorded, not executed.</summary>
    public void RecordStep(CommandBuffer cb, GraphicsBuffer pos, GraphicsBuffer vel, int n, float rc, bool withBirths = true)
    {
        if (worker == null) throw new InvalidOperationException("GpuSprayLayer: SetModel first");
        StepCount++;
        EnsureCapacity(n);
        EnsureInput(n);
        SetParams(cb, n, rc);
        RecordReset(cb, countB);
        RecordGrid(cb, pos, n);
        RecordAdvance(cb, pos, poolA, countA, poolB, countB);
        if (n > 0)
        {
            RecordFeatures(cb, pos, vel, n);
            var packed = RecordMlp(cb);
            if (withBirths) RecordBirths(cb, pos, vel, n, packed, poolB, countB);
        }
        (poolA, poolB) = (poolB, poolA);
        (countA, countB) = (countB, countA);
        if (!statsPending)
        {
            statsPending = true;
            cb.RequestAsyncReadback(stats, req =>
            {
                statsPending = false;
                if (req.hasError) return;
                var s = req.GetData<uint>();
                LastEvents = (int)s[0]; LastEventOverflow = (int)s[1]; LastBorn = (int)s[2]; LastDropped = (int)s[3];
                KillWall = (int)s[4]; KillObst = (int)s[5]; KillMerge = (int)s[6]; KillAge = (int)s[7];
                LastRateSum = BitConverter.ToSingle(BitConverter.GetBytes(s[8]), 0);
                Alive = (int)s[9] + (int)s[2];
            });
        }
    }

    void RecordReset(CommandBuffer cb, GraphicsBuffer countOut)
    {
        B(cb, kReset, "_Stats", stats); B(cb, kReset, "_CountOut", countOut); B(cb, kReset, "_EventCount", eventCount); B(cb, kReset, "_ScalarU", scalarU);
        cb.DispatchCompute(cs, kReset, 1, 1, 1);
    }

    void RecordGrid(CommandBuffer cb, GraphicsBuffer pos, int n)
    {
        B(cb, kGClear, "_CellCount", cellCount); B(cb, kGClear, "_Hist", hist);
        cb.DispatchCompute(cs, kGClear, G64(Math.Max(cells, Hist)), 1, 1);
        if (n > 0)
        {
            B(cb, kGCount, "_PosIn", pos); B(cb, kGCount, "_CellOf", cellOf); B(cb, kGCount, "_CellCount", cellCount);
            cb.DispatchCompute(cs, kGCount, G64(n), 1, 1);
        }
        B(cb, kGScan, "_CellCount", cellCount); B(cb, kGScan, "_CellStart", cellStart); B(cb, kGScan, "_CellFill", cellFill);
        cb.DispatchCompute(cs, kGScan, 1, 1, 1);
        if (n > 0)
        {
            B(cb, kGScatter, "_CellFill", cellFill); B(cb, kGScatter, "_CellOf", cellOf); B(cb, kGScatter, "_Sorted", sorted);
            cb.DispatchCompute(cs, kGScatter, G64(n), 1, 1);
        }
    }

    void RecordAdvance(CommandBuffer cb, GraphicsBuffer pos, GraphicsBuffer pIn, GraphicsBuffer cIn, GraphicsBuffer pOut, GraphicsBuffer cOut)
    {
        B(cb, kDropArgs, "_CountIn", cIn); B(cb, kDropArgs, "_Args", argsDrop);
        cb.DispatchCompute(cs, kDropArgs, 1, 1, 1);
        B(cb, kAdvance, "_CountIn", cIn); B(cb, kAdvance, "_PoolIn", pIn); B(cb, kAdvance, "_Obst", obst);
        B(cb, kAdvance, "_CellStart", cellStart); B(cb, kAdvance, "_Sorted", sorted); B(cb, kAdvance, "_PosIn", pos);
        B(cb, kAdvance, "_Stats", stats); B(cb, kAdvance, "_CountOut", cOut); B(cb, kAdvance, "_PoolOut", pOut);
        cb.DispatchCompute(cs, kAdvance, argsDrop, 0);
    }

    void RecordFeatures(CommandBuffer cb, GraphicsBuffer pos, GraphicsBuffer vel, int n)
    {
        B(cb, kFeatA, "_PosIn", pos); B(cb, kFeatA, "_VelIn", vel); B(cb, kFeatA, "_CellStart", cellStart); B(cb, kFeatA, "_Sorted", sorted);
        B(cb, kFeatA, "_Cnt", cnt); B(cb, kFeatA, "_Ita", ita); B(cb, kFeatA, "_NHat", nhat); B(cb, kFeatA, "_Knn", knn);
        cb.DispatchCompute(cs, kFeatA, G64(n), 1, 1);
        B(cb, kFeatHist, "_Hist", hist); B(cb, kFeatHist, "_Cnt", cnt); B(cb, kFeatHist, "_ScalarU", scalarU); B(cb, kFeatHist, "_PosIn", pos);
        cb.DispatchCompute(cs, kFeatHist, G64(n), 1, 1);
        B(cb, kFeatPct, "_Hist", hist); B(cb, kFeatPct, "_ScalarF", scalarF); B(cb, kFeatPct, "_ScalarU", scalarU);
        cb.DispatchCompute(cs, kFeatPct, 1, 1, 1);
        B(cb, kFeatB, "_ScalarF", scalarF); B(cb, kFeatB, "_PosIn", pos); B(cb, kFeatB, "_VelIn", vel); B(cb, kFeatB, "_NHat", nhat);
        B(cb, kFeatB, "_Cnt", cnt); B(cb, kFeatB, "_CellStart", cellStart); B(cb, kFeatB, "_Sorted", sorted); B(cb, kFeatB, "_E1", e1);
        B(cb, kFeatB, "_E2", e2); B(cb, kFeatB, "_X", xBuf); B(cb, kFeatB, "_Knn", knn); B(cb, kFeatB, "_Ita", ita);
        cb.DispatchCompute(cs, kFeatB, G64(n), 1, 1);
    }

    ComputeBuffer RecordMlp(CommandBuffer cb)
    {
        cb.ScheduleWorker(worker, xT);
        var o = worker.PeekOutput("packed") as Tensor<float>;
        return ComputeTensorData.Pin(o).buffer;
    }

    void RecordBirths(CommandBuffer cb, GraphicsBuffer pos, GraphicsBuffer vel, int n, ComputeBuffer packed, GraphicsBuffer pOut, GraphicsBuffer cOut)
    {
        B(cb, kBirths, "_Packed", packed); B(cb, kBirths, "_Stats", stats); B(cb, kBirths, "_PosIn", pos); B(cb, kBirths, "_VelIn", vel);
        B(cb, kBirths, "_E1", e1); B(cb, kBirths, "_E2", e2); B(cb, kBirths, "_NHat", nhat); B(cb, kBirths, "_SizeEdges", sizeEdges);
        B(cb, kBirths, "_EventCount", eventCount); B(cb, kBirths, "_Events", events);
        cb.DispatchCompute(cs, kBirths, G64(n), 1, 1);
        if (pOut == null) return;
        B(cb, kEventArgs, "_EventCount", eventCount); B(cb, kEventArgs, "_Args", argsEvent);
        cb.DispatchCompute(cs, kEventArgs, 1, 1, 1);
        B(cb, kSpawn, "_Events", events); B(cb, kSpawn, "_CountOut", cOut); B(cb, kSpawn, "_Stats", stats);
        B(cb, kSpawn, "_LatCum", latCum); B(cb, kSpawn, "_Lat", lat); B(cb, kSpawn, "_PoolOut", pOut);
        cb.DispatchCompute(cs, kSpawn, argsEvent, 0);
    }

    // ------------------------------------------------------------------ raster (every rendered frame)
    void EnsureRaster(int W, int H)
    {
        if (rasterHW == W * H && SprayFieldRT != null && SprayFieldRT.width == W) return;
        zKey?.Release(); zChord?.Release(); zSpeed?.Release(); zNrm?.Release();
        if (SprayFieldRT != null) { SprayFieldRT.Release(); UnityEngine.Object.Destroy(SprayFieldRT); }
        if (SprayNrmRT != null) { SprayNrmRT.Release(); UnityEngine.Object.Destroy(SprayNrmRT); }
        zKey = Buf(W * H, 4); zChord = Buf(W * H, 4); zSpeed = Buf(W * H, 4); zNrm = Buf(W * H, 16);
        SprayFieldRT = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBFloat) { enableRandomWrite = true, filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp, name = "GPU1001.SprayField" };
        SprayNrmRT = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBFloat) { enableRandomWrite = true, filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp, name = "GPU1001.SprayNrm" };
        SprayFieldRT.Create(); SprayNrmRT.Create();
        rasterHW = W * H;
    }

    /// <summary>Draw the current droplets (age >= 1) against the bulk field (texture layout). Field mode merges into it.</summary>
    public void RecordRaster(CommandBuffer cb, GraphicsBuffer field, int W, int H, Vector3 eye, Vector3 right, Vector3 up, Vector3 fwd,
                             float focal, float aspect, float thicknessScale, float minRadiusPx, float radiusScale, bool fieldMode)
    {
        EnsureRaster(W, H);
        if (poolA == null) EnsureCapacity(1);
        cb.SetComputeIntParam(cs, "_RW", W); cb.SetComputeIntParam(cs, "_RH", H);
        cb.SetComputeVectorParam(cs, "_Eye", eye); cb.SetComputeVectorParam(cs, "_Right", right);
        cb.SetComputeVectorParam(cs, "_Up", up); cb.SetComputeVectorParam(cs, "_Fwd", fwd);
        cb.SetComputeFloatParam(cs, "_Focal", focal); cb.SetComputeFloatParam(cs, "_Aspect", aspect);
        cb.SetComputeFloatParam(cs, "_MinRpx", minRadiusPx); cb.SetComputeFloatParam(cs, "_RScale", radiusScale);
        cb.SetComputeFloatParam(cs, "_TScale", thicknessScale); cb.SetComputeIntParam(cs, "_FieldMode", fieldMode ? 1 : 0);
        cb.SetComputeFloatParam(cs, "_Rd", RD);
        B(cb, kRClear, "_ZKey", zKey); B(cb, kRClear, "_ZChord", zChord); B(cb, kRClear, "_ZSpeed", zSpeed);
        cb.DispatchCompute(cs, kRClear, G64(W * H), 1, 1);
        B(cb, kDropArgs, "_CountIn", countA); B(cb, kDropArgs, "_Args", argsRaster);
        cb.DispatchCompute(cs, kDropArgs, 1, 1, 1);
        foreach (int k in new[] { kRDepth, kRNormal })
        {
            B(cb, k, "_CountIn", countA); B(cb, k, "_PoolIn", poolA); B(cb, k, "_Field", field);
            B(cb, k, "_ZKey", zKey); B(cb, k, "_ZChord", zChord); B(cb, k, "_ZSpeed", zSpeed); B(cb, k, "_ZNrm", zNrm);
            cb.DispatchCompute(cs, k, argsRaster, 0);
        }
        B(cb, kRResolve, "_ZKey", zKey); B(cb, kRResolve, "_Field", field); B(cb, kRResolve, "_ZChord", zChord);
        B(cb, kRResolve, "_ZSpeed", zSpeed); B(cb, kRResolve, "_ZNrm", zNrm);
        cb.SetComputeTextureParam(cs, kRResolve, "_SprayField", SprayFieldRT);
        cb.SetComputeTextureParam(cs, kRResolve, "_SprayNrm", SprayNrmRT);
        cb.DispatchCompute(cs, kRResolve, (W + 7) / 8, (H + 7) / 8, 1);
    }

    // ------------------------------------------------------------------ parity hooks (P1-P5)
    public struct FeatureDump { public float[] X; public int[] Knn; public float NFull, MinH; public Vector4[] NHat; }

    /// <summary>P1: grid + features only on (pos, vel); executes and reads back.</summary>
    public FeatureDump ParityFeatures(GraphicsBuffer pos, GraphicsBuffer vel, int n, float rc)
    {
        var cb = new CommandBuffer { name = "GPU1001.spray.parity" };
        EnsureCapacity(n); EnsureInput(n);
        SetParams(cb, n, rc);
        RecordReset(cb, countB);
        RecordGrid(cb, pos, n);
        RecordFeatures(cb, pos, vel, n);
        Graphics.ExecuteCommandBuffer(cb); cb.Release();
        var d = new FeatureDump { X = new float[n * NFeat], Knn = new int[n * KNN], NHat = new Vector4[n] };
        xBuf.GetData(d.X, 0, 0, n * NFeat);
        knn.GetData(d.Knn, 0, 0, n * KNN);
        nhat.GetData(d.NHat, 0, 0, n);
        var sf = new float[4]; scalarF.GetData(sf);
        d.NFull = sf[0]; d.MinH = sf[1];
        return d;
    }

    /// <summary>P2: MLP on given feature rows (uploaded into the input tensor); returns packed [n, 63].</summary>
    public float[] ParityMlp(float[] X, int n)
    {
        EnsureInput(n);
        var pad = new float[nPadCur * NFeat];
        Array.Copy(X, pad, Math.Min(X.Length, pad.Length));
        xBuf.SetData(pad);
        var cb = new CommandBuffer { name = "GPU1001.spray.mlp" };
        var packed = RecordMlp(cb);
        Graphics.ExecuteCommandBuffer(cb); cb.Release();
        var o = new float[n * NPack];
        packed.GetData(o, 0, 0, n * NPack);
        return o;
    }

    /// <summary>P3: `rounds` independent birth rounds on the features of (pos, vel) (features + MLP once). Returns the events
    /// as 12 floats each: c xyz, parent, u xyz, k, comp, bin, round, 0 (parent / k / comp / bin / round as integers).</summary>
    public float[] ParityBirthRounds(GraphicsBuffer pos, GraphicsBuffer vel, int n, float rc, int rounds, int eventCap, out int total, out float rateSum)
    {
        EnsureEvents(eventCap);
        var cb = new CommandBuffer { name = "GPU1001.spray.births" };
        EnsureCapacity(n); EnsureInput(n);
        int keep = maxEvents; maxEvents = eventCap;
        SetParams(cb, n, rc);
        maxEvents = keep;
        RecordReset(cb, countB);
        RecordGrid(cb, pos, n);
        RecordFeatures(cb, pos, vel, n);
        var packed = RecordMlp(cb);
        cb.SetComputeIntParam(cs, "_KeepEvents", 1);
        for (int r = 0; r < rounds; r++)
        {
            cb.SetComputeIntParam(cs, "_Round", r);
            RecordBirths(cb, pos, vel, n, packed, null, null);
        }
        Graphics.ExecuteCommandBuffer(cb); cb.Release();
        var ec = new uint[4]; eventCount.GetData(ec);
        total = (int)Math.Min(ec[0], (uint)eventCap);
        // sum of lambda in double from the GPU's own MLP output (a float atomic sum over many rounds loses precision)
        var pk = new float[n * NPack];
        if (n > 0) packed.GetData(pk, 0, 0, n * NPack);
        double sR = rateScale, cap = rateCap * Math.Max(sR, 1.0), sum = 0;
        for (int i = 0; i < n; i++) { double lam = Math.Exp((double)pk[i * NPack]) * sR; if (lam > cap) lam = cap; if (lam > 0) sum += lam; }
        rateSum = (float)sum;
        var raw = new EventGpu[total];
        if (total > 0) events.GetData(raw, 0, 0, total);
        var outv = new float[total * 12];
        for (int e = 0; e < total; e++)
        {
            int b = e * 12; var ev = raw[e];
            outv[b] = ev.c.x; outv[b + 1] = ev.c.y; outv[b + 2] = ev.c.z; outv[b + 3] = ev.parent;
            outv[b + 4] = ev.u.x; outv[b + 5] = ev.u.y; outv[b + 6] = ev.u.z; outv[b + 7] = ev.k;
            outv[b + 8] = ev.compBin & 0xFF; outv[b + 9] = ev.compBin >> 8; outv[b + 10] = ev.round;
        }
        return outv;
    }

    /// <summary>P4/P5 setup: replace the live pool with the given droplets (ids = index).</summary>
    public void ParitySetPool(Vector3[] p, Vector3[] v, int[] age, int count)
    {
        EnsureCapacity(1);
        var d = new DropGpu[count];
        for (int i = 0; i < count; i++)
            d[i] = new DropGpu { xa = new Vector4(p[i].x, p[i].y, p[i].z, age[i]), v = v[i], id = (uint)i };
        if (count > 0) poolA.SetData(d, 0, 0, count);
        countA.SetData(new uint[] { (uint)count, 0, 0, 0 });
    }

    /// <summary>P4: grid + flight/culls only (no births) against (pos, n); returns the survivors (xa = pos, age; vi = vel, id bits).</summary>
    public DropGpu[] ParityAdvance(GraphicsBuffer pos, int n, float rc, out int survivors, out uint[] statsOut)
    {
        var cb = new CommandBuffer { name = "GPU1001.spray.advance" };
        EnsureCapacity(Math.Max(n, 1));
        SetParams(cb, n, rc);
        RecordReset(cb, countB);
        RecordGrid(cb, pos, n);
        RecordAdvance(cb, pos, poolA, countA, poolB, countB);
        Graphics.ExecuteCommandBuffer(cb); cb.Release();
        var c = new uint[4]; countB.GetData(c);
        survivors = (int)c[0];
        var o = new DropGpu[survivors];
        if (survivors > 0) poolB.GetData(o, 0, 0, survivors);
        statsOut = new uint[16]; stats.GetData(statsOut);
        return o;
    }

    public void ReadRaster(int W, int H, Vector4[] field, Vector4[] nrm)
    {
        var tf = RenderTexture.active;
        var t = new Texture2D(W, H, TextureFormat.RGBAFloat, false);
        RenderTexture.active = SprayFieldRT; t.ReadPixels(new Rect(0, 0, W, H), 0, 0); t.Apply();
        var a = t.GetPixels(); for (int i = 0; i < a.Length; i++) field[i] = a[i];
        RenderTexture.active = SprayNrmRT; t.ReadPixels(new Rect(0, 0, W, H), 0, 0); t.Apply();
        a = t.GetPixels(); for (int i = 0; i < a.Length; i++) nrm[i] = a[i];
        RenderTexture.active = tf;
        UnityEngine.Object.Destroy(t);
    }

    public void Dispose()
    {
        if (statsPending) AsyncGPUReadback.WaitAllRequests();
        xT?.Dispose(); xT = null; xBuf = null;
        worker?.Dispose(); worker = null;
        foreach (var b in new[] { cellCount, cellStart, cellFill, cellOf, sorted, cnt, ita, nhat, e1, e2, knn, hist, scalarU, scalarF,
                                  events, eventCount, argsEvent, argsDrop, argsRaster, poolA, poolB, countA, countB, stats, obst, lat, latCum,
                                  sizeEdges, zKey, zChord, zSpeed, zNrm })
            b?.Release();
        if (SprayFieldRT != null) { SprayFieldRT.Release(); UnityEngine.Object.Destroy(SprayFieldRT); SprayFieldRT = null; }
        if (SprayNrmRT != null) { SprayNrmRT.Release(); UnityEngine.Object.Destroy(SprayNrmRT); SprayNrmRT = null; }
    }
}
