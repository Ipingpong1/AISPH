// DfsphSolver.cs — DFSPH1001 (2026-10-01): buffer owner + dispatch driver for Dfsph.compute, the GPU port of SPlisHSPlasH
// c5a063d's DFSPH with Bender 2019 volume maps (analytic primitives). Plain class like GpuSphSolver: DfsphProvider wraps
// it for the live scene, DfsphParity drives it headlessly for the known-answer / parity gates
// (SSU_restart/Experiments/DFSPH1001/PLAN.md).
//
// All solver state lives on the GPU, including dt (CFL) and the iteration counters, so one step is a FIXED command buffer
// (recorded once per particle count / config) that the caller executes as often as it likes; nothing is read back
// mid-step. Per-step statistics land in a 64-entry ring (_Stats) for async readback.
//
// Conventions copied from the reference build (Real = float, AVX + USE_PERFORMANCE_OPTIMIZATION):
//   h = 4 r; V = 0.8 (2r)^3; mass = V rho0; fluid-fluid kernel = CubicKernel_AVX; boundary kernel = PrecomputedKernel
//   <CubicKernel, 10000>; first dt = Configuration.timeStepSize; then the CFL (method 1). Fluid blocks are sampled by
//   SimulatorBase::createFluidBlocks (modes 0/1/2). Volume maps: SimulatorBase::initVolumeMap on the analytic SDF.
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;

public class DfsphSolver : IDisposable
{
    public const int T = 128, MaxNbr = 128, MaxBodies = 8, Ring = 64, StatWords = 16, StateWords = 32;
    const int PreRes = 10000;

    // ---------- configuration (SPlisHSPlasH names; set before Init) ----------
    public float particleRadius = 0.0424032509f;
    public float density0 = 1000f;
    public float viscosity = 0.00999999978f;            // "Standard viscosity" / viscosity
    public Vector3 gravity = new Vector3(0f, -9.81000042f, 0f);
    public float maxError = 0.0500000007f, maxErrorV = 0.100000001f;
    public int minIterations = 2, maxIterations = 100, maxIterationsV = 100;
    public bool enableDivergenceSolver = true;
    public float cflFactor = 1f, cflMinTimeStepSize = 9.99999975e-05f, cflMaxTimeStepSize = 0.00499999989f;
    public float initialTimeStep = 0.00153995235f;      // Configuration.timeStepSize (TimeManager h before the first CFL)
    public int maxParticles = 65536;
    [Tooltip("Iteration slots recorded per solve. >= maxIterations reproduces the reference loop exactly; fewer = live mode (DfsphProvider adapts it; a step that needed more is counted in Stats.truncated).")]
    public int kCap = 100, kCapV = 100;
    public Vector3 gridMin = new Vector3(-0.5f, -0.5f, -0.5f), gridMax = new Vector3(3.5f, 3.5f, 3.5f);

    public enum Shape { Box = 0, Sphere = 1, Torus = 2 }

    /// <summary>One boundary body = one SPlisHSPlasH RigidBodies entry: stock mesh (UnitBox.obj / sphere.obj r 1 /
    /// torus.obj R 1, r 0.5, axis y) scaled by `scale`, then rotated + translated. The volume map lives in the scaled,
    /// unrotated local frame (as in the reference).</summary>
    [Serializable]
    public class Body
    {
        public Shape shape = Shape.Box;
        public Vector3 scale = Vector3.one;
        public Vector3 translation;
        public Vector3 rotationAxis = Vector3.right;
        public float rotationAngle;                      // radians
        public bool mapInvert;
        public float mapThickness;
        public Vector3Int mapResolution = new Vector3Int(20, 20, 20);
        [Tooltip("Field 0 from the corpus' tessellated stock mesh (sphere.obj / torus.obj, Discregrid TriangleMeshDistance) instead of the analytic primitive: exact corpus geometry.")]
        public bool tessellated;
        [NonSerialized] public DfsphStockMesh mesh;
        // a rotation given directly (Unity props) instead of rotationAxis / rotationAngle
        [NonSerialized] public Matrix4x4 rotationMatrix = Matrix4x4.identity;
        [NonSerialized] public bool useRotationMatrix;
        // kinematic motion (sim space), set per frame by the provider; zero = static
        [NonSerialized] public Vector3 linearVelocity, angularVelocity;
        [NonSerialized] public bool moving;
        // filled by BuildVolumeMaps
        [NonSerialized] public double[] dmin, dmax, cell;
        [NonSerialized] public int nodeOffset, nodeCount;
        public Matrix4x4 Rotation()
        {
            if (useRotationMatrix) return rotationMatrix;
            Vector3 a = rotationAxis.sqrMagnitude > 0f ? rotationAxis.normalized : Vector3.right;
            float c = Mathf.Cos(rotationAngle), s = Mathf.Sin(rotationAngle), t = 1f - c;
            var m = Matrix4x4.identity;   // Eigen AngleAxis (right-handed), rows
            m.m00 = t * a.x * a.x + c;       m.m01 = t * a.x * a.y - s * a.z; m.m02 = t * a.x * a.z + s * a.y;
            m.m10 = t * a.x * a.y + s * a.z; m.m11 = t * a.y * a.y + c;       m.m12 = t * a.y * a.z - s * a.x;
            m.m20 = t * a.x * a.z - s * a.y; m.m21 = t * a.y * a.z + s * a.x; m.m22 = t * a.z * a.z + c;
            return m;
        }
        /// <summary>Local AABB of the scaled stock mesh (what initVolumeMap extends by 8h + thickness).</summary>
        public void MeshAabb(out Vector3 lo, out Vector3 hi)
        {
            if (tessellated)
            {
                if (mesh == null) mesh = new DfsphStockMesh(shape, scale);
                lo = new Vector3((float)mesh.aabbMin[0], (float)mesh.aabbMin[1], (float)mesh.aabbMin[2]);
                hi = new Vector3((float)mesh.aabbMax[0], (float)mesh.aabbMax[1], (float)mesh.aabbMax[2]);
                return;
            }
            Vector3 s = scale;
            switch (shape)
            {
                case Shape.Box: lo = -0.5f * s; hi = 0.5f * s; break;
                case Shape.Sphere: lo = -s; hi = s; break;
                default: lo = Vector3.Scale(new Vector3(-1.5f, -0.5f, -1.5f), s); hi = Vector3.Scale(new Vector3(1.5f, 0.5f, 1.5f), s); break;
            }
        }
        /// <summary>Max distance of a mesh vertex to the centre (BoundaryModel maxDist, for the CFL of moving bodies).</summary>
        public float MaxDist()
        {
            switch (shape)
            {
                case Shape.Box: return (0.5f * scale).magnitude;
                case Shape.Sphere: return scale.x;
                default: return 1.5f * scale.x;
            }
        }
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    struct BodyGpu
    {
        public Vector4 r0, r1, r2, t, dmin, dmax, cell, invCell;
        public uint resX, resY, resZ, nodeOff;
        public Vector4 vlin, vang;
    }

    /// <summary>One step's statistics (from the _Stats ring).</summary>
    public struct StepStats
    {
        public uint step; public float t, dtOld, dtNew; public uint itV, it; public float errV, err;
        public uint nbrMax, pushed, truncV, trunc; public float vmax; public uint recV, rec; public bool valid;
        public override string ToString() => $"step {step} t {t:F4} dt {dtOld * 1e3f:F3}->{dtNew * 1e3f:F3} ms itV {itV} it {it} errV {errV:G3} err {err:G3} nbrMax {nbrMax} push {pushed} vmax {vmax:F2}";
    }

    // ---------- GPU state ----------
    ComputeShader cs;
    int kGClear, kGCount, kGScan, kGScatter, kGSort, kNbr, kBnd, kDens, kFactor, kDivInit, kPAccelV, kDivUpd, kDivChk,
        kDivFin, kNonP, kCflRed, kCflFin, kIntV, kPresInit, kPAccel, kPresUpd, kPresChk, kPresFin, kIntP, kEnd,
        kVmSdf, kVmVol, kProbeK, kProbeB;
    GraphicsBuffer bPos, bVel, bDens, bAcc, bFactor, bDensAdv, bP, bPV, bPAcc, bNbrCount, bNbrs, bBVol, bBXj,
        bCellOf, bCellCount, bCellStart, bCellFill, bSorted, bState, bPartial, bArgs, bStats, bWT, bGT, bBodies, bNodes;
    GraphicsBuffer bProbeIn, bProbeOut;
    readonly List<Body> bodies = new List<Body>();
    BodyGpu[] bodyStage = new BodyGpu[MaxBodies];
    int n, cells, groups;
    Vector3Int gridDim;
    float h;
    CommandBuffer stepCb, bodyCb;
    bool stepCbDirty = true;
    CustomSampler sampler;

    public int Count => n;
    public float ParticleRadius => particleRadius;
    public float SupportRadius => h;
    public GraphicsBuffer PositionBuffer => bPos;
    public GraphicsBuffer VelocityBuffer => bVel;
    public GraphicsBuffer DensityBuffer => bDens;
    public GraphicsBuffer StateBuffer => bState;
    public GraphicsBuffer StatsBuffer => bStats;
    public IReadOnlyList<Body> Bodies => bodies;
    public CustomSampler Sampler => sampler;

    public DfsphSolver(ComputeShader shader)
    {
        cs = UnityEngine.Object.Instantiate(shader);   // own copy: two solvers never share bindings
    }

    // ---------- lifecycle ----------
    public void Init()
    {
        h = 4f * particleRadius;
        string[] names = { "GridClear", "GridCount", "GridScan", "GridScatter", "GridSortCells", "Neighbours", "Boundary",
            "Density", "Factor", "DivInit", "PAccelV", "DivUpdate", "DivCheck", "DivFinal", "NonPressure", "CflReduce",
            "CflFinal", "IntegrateVel", "PresInit", "PAccel", "PresUpdate", "PresCheck", "PresFinal", "IntegratePos",
            "EndStep", "VmSdf", "VmVolume", "ProbeKernels", "ProbeBoundary" };
        var k = new int[names.Length];
        for (int i = 0; i < names.Length; i++) k[i] = cs.FindKernel(names[i]);
        (kGClear, kGCount, kGScan, kGScatter, kGSort, kNbr, kBnd, kDens, kFactor, kDivInit) = (k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8], k[9]);
        (kPAccelV, kDivUpd, kDivChk, kDivFin, kNonP, kCflRed, kCflFin, kIntV, kPresInit, kPAccel) = (k[10], k[11], k[12], k[13], k[14], k[15], k[16], k[17], k[18], k[19]);
        (kPresUpd, kPresChk, kPresFin, kIntP, kEnd, kVmSdf, kVmVol, kProbeK, kProbeB) = (k[20], k[21], k[22], k[23], k[24], k[25], k[26], k[27], k[28]);

        int N = maxParticles;
        bPos = Buf(N, 12); bVel = Buf(N, 12); bDens = Buf(N, 4); bAcc = Buf(N, 12); bFactor = Buf(N, 4);
        bDensAdv = Buf(N, 4); bP = Buf(N, 4); bPV = Buf(N, 4); bPAcc = Buf(N, 12); bNbrCount = Buf(N, 4);
        bNbrs = Buf(N * MaxNbr, 16); bBVol = Buf(N * MaxBodies, 4); bBXj = Buf(N * MaxBodies, 12); bCellOf = Buf(N, 4);
        bSorted = Buf(N, 4);
        bState = Buf(StateWords, 4); bPartial = Buf((N + T - 1) / T + 1, 4);
        bArgs = new GraphicsBuffer(GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments, 6, 4);
        bStats = Buf(Ring * StatWords, 4);
        bBodies = Buf(MaxBodies, System.Runtime.InteropServices.Marshal.SizeOf<BodyGpu>());
        BuildKernelTables();
        SetupGrid();
        Zero(bP, N); Zero(bPV, N); Zero(bDens, N); Zero(bVel, N);
        bStats.SetData(new uint[Ring * StatWords]);
        ResetState();
        sampler = CustomSampler.Create("DFSPH1001.Step", true);
        n = 0;
        stepCbDirty = true;
    }

    static GraphicsBuffer Buf(int count, int stride) => new GraphicsBuffer(GraphicsBuffer.Target.Structured, Math.Max(count, 1), stride);
    static void Zero(GraphicsBuffer b, int count)
    {
        var z = new float[count * (b.stride / 4)];
        b.SetData(z);
    }

    public void Dispose()
    {
        foreach (var b in new[] { bPos, bVel, bDens, bAcc, bFactor, bDensAdv, bP, bPV, bPAcc, bNbrCount, bNbrs, bBVol, bBXj,
                                  bCellOf, bCellCount, bCellStart, bCellFill, bSorted, bState, bPartial, bArgs, bStats, bWT,
                                  bGT, bBodies, bNodes, bProbeIn, bProbeOut })
            b?.Release();
        bPos = bVel = bDens = bAcc = bFactor = bDensAdv = bP = bPV = bPAcc = bNbrCount = bNbrs = bBVol = bBXj = null;
        bCellOf = bCellCount = bCellStart = bCellFill = bSorted = bState = bPartial = bArgs = bStats = bWT = bGT = null;
        bBodies = bNodes = bProbeIn = bProbeOut = null;
        stepCb?.Release(); stepCb = null;
        bodyCb?.Release(); bodyCb = null;
        if (cs != null) { if (Application.isPlaying) UnityEngine.Object.Destroy(cs); else UnityEngine.Object.DestroyImmediate(cs); }
        cs = null;
    }

    /// <summary>Back to t = 0 with the initial dt and zero warm-start pressures (particles are removed; spawn again).</summary>
    public void Reset()
    {
        n = 0;
        Zero(bP, maxParticles); Zero(bPV, maxParticles);
        bStats.SetData(new uint[Ring * StatWords]);   // no stale step records after a reset
        ResetState();
        stepCbDirty = true;
    }

    void ResetState()
    {
        var s = new uint[StateWords];
        s[0] = FloatBits(initialTimeStep); s[1] = FloatBits(initialTimeStep);
        bState.SetData(s);
    }

    static uint FloatBits(float f) => BitConverter.ToUInt32(BitConverter.GetBytes(f), 0);
    static float BitsFloat(uint u) => BitConverter.ToSingle(BitConverter.GetBytes(u), 0);

    // ---------- kernels: SPH::CubicKernel (float) and PrecomputedKernel<CubicKernel, 10000> ----------
    float cubicK, cubicL;
    float CubicW(float r)
    {
        float q = r / h;
        if (q <= 1f)
        {
            if (q <= 0.5f) { float q2 = q * q; float q3 = q2 * q; return cubicK * (6f * q3 - 6f * q2 + 1f); }
            return cubicK * (2f * MathF.Pow(1f - q, 3f));
        }
        return 0f;
    }
    float CubicGradX(float x)   // CubicKernel::gradW(Vector3r(x, 0, 0))[0]
    {
        float rl = MathF.Sqrt(x * x);
        float q = rl / h;
        if (rl > 1.0e-9f && q <= 1f)
        {
            float gradq = x / rl;
            gradq /= h;
            if (q <= 0.5f) return cubicL * q * (3f * q - 2f) * gradq;
            float factor = 1f - q;
            return cubicL * (-factor * factor) * gradq;
        }
        return 0f;
    }

    void BuildKernelTables()
    {
        float pi = (float)Math.PI;
        float h3 = h * h * h;
        cubicK = 8f / (pi * h3);
        cubicL = 48f / (pi * h3);
        float stepSize = h / (float)(PreRes - 1);
        float invStep = 1f / stepSize;
        var W = new float[PreRes];
        var G = new float[PreRes + 1];
        for (int i = 0; i < PreRes; i++)
        {
            float posX = stepSize * (float)i;
            W[i] = CubicW(posX);
            G[i] = posX > 1.0e-9f ? CubicGradX(posX) / posX : 0f;
        }
        G[PreRes] = 0f;
        bWT?.Release(); bGT?.Release();
        bWT = Buf(PreRes, 4); bWT.SetData(W);
        bGT = Buf(PreRes + 1, 4); bGT.SetData(G);

        float diam = 2f * particleRadius;
        Volume = 0.8f * diam * diam * diam;
        preInvStep = invStep;
    }

    public float Volume { get; private set; }
    float preInvStep;

    /// <summary>Every uniform of Dfsph.compute, recorded into c (and set on the shader object for immediate dispatches).</summary>
    void Uniforms(CommandBuffer c)
    {
        groups = Math.Max(1, (n + T - 1) / T);
        void F(string name, float v) { c?.SetComputeFloatParam(cs, name, v); cs.SetFloat(name, v); }
        void I(string name, int v) { c?.SetComputeIntParam(cs, name, v); cs.SetInt(name, v); }
        void V4(string name, Vector4 v) { c?.SetComputeVectorParam(cs, name, v); cs.SetVector(name, v); }
        float pi = (float)Math.PI, h3 = h * h * h, invR = 1f / h;
        I("_N", n); I("_Groups", groups);
        F("_R", particleRadius); F("_H", h); F("_H2", h * h); F("_V", Volume); F("_Rho0", density0);
        F("_InvH", invR); F("_InvH2", invR * invR);
        F("_KAvx", 8f / (pi * h3)); F("_LAvx", 48f / (pi * h3)); F("_W0Avx", 8f / (pi * h3));   // W_avx(0) = k
        F("_PreInvStep", preInvStep); F("_PreRadius2", h * h);
        F("_CubicK", cubicK); F("_CubicW0", CubicW(0f));
        V4("_Gravity", gravity);
        F("_Visc", 10f * viscosity * density0); F("_ViscEps", 0.01f * (h * h));
        F("_MaxError", maxError); F("_MaxErrorV", maxErrorV);
        I("_MinIter", minIterations); I("_MaxIter", maxIterations); I("_MaxIterV", maxIterationsV);
        I("_KCap", Math.Min(kCap, maxIterations)); I("_KCapV", Math.Min(kCapV, maxIterationsV));
        F("_CflFactor", cflFactor); F("_CflMin", cflMinTimeStepSize); F("_CflMax", cflMaxTimeStepSize);
        I("_NumBodies", bodies.Count); I("_DivEnabled", enableDivergenceSolver ? 1 : 0);
        V4("_GridMin", gridMin); F("_CellInv", 1f / h);
        if (c != null) c.SetComputeIntParams(cs, "_GridDim", gridDim.x, gridDim.y, gridDim.z, cells);
        cs.SetInts("_GridDim", gridDim.x, gridDim.y, gridDim.z, cells);
        I("_Cells", cells);
    }

    void SetupGrid()
    {
        gridDim = new Vector3Int(Mathf.Max(1, Mathf.CeilToInt((gridMax.x - gridMin.x) / h)),
                                 Mathf.Max(1, Mathf.CeilToInt((gridMax.y - gridMin.y) / h)),
                                 Mathf.Max(1, Mathf.CeilToInt((gridMax.z - gridMin.z) / h)));
        cells = gridDim.x * gridDim.y * gridDim.z;
        bCellCount?.Release(); bCellStart?.Release(); bCellFill?.Release();
        bCellCount = Buf(cells, 4); bCellStart = Buf(cells + 1, 4); bCellFill = Buf(cells, 4);
        stepCbDirty = true;
    }

    /// <summary>Grid bounds = the union of the inverted (container) bodies' world AABBs (+ margin), else the given box.</summary>
    public void FitGridToBodies(float margin)
    {
        bool any = false;
        Vector3 lo = Vector3.zero, hi = Vector3.zero;
        foreach (var b in bodies)
        {
            if (!b.mapInvert) continue;
            b.MeshAabb(out var l, out var u);
            var R = b.Rotation();
            for (int c = 0; c < 8; c++)
            {
                var p = new Vector3((c & 1) != 0 ? u.x : l.x, (c & 2) != 0 ? u.y : l.y, (c & 4) != 0 ? u.z : l.z);
                var w = (Vector3)(R.MultiplyVector(p)) + b.translation;
                if (!any) { lo = hi = w; any = true; } else { lo = Vector3.Min(lo, w); hi = Vector3.Max(hi, w); }
            }
        }
        if (!any) return;
        gridMin = lo - Vector3.one * margin;
        gridMax = hi + Vector3.one * margin;
        SetupGrid();
    }

    // ---------- bodies + volume maps (SimulatorBase::initVolumeMap, analytic SDF) ----------
    public void SetBodies(IList<Body> list)
    {
        if (list.Count > MaxBodies) throw new ArgumentException($"at most {MaxBodies} bodies");
        bodies.Clear();
        bodies.AddRange(list);
        BuildVolumeMaps();
        UploadBodies();
        stepCbDirty = true;
    }

    public static void NodeCounts(Vector3Int res, out int nv, out int nex, out int ney, out int nez)
    {
        nv = (res.x + 1) * (res.y + 1) * (res.z + 1);
        nex = res.x * (res.y + 1) * (res.z + 1);
        ney = (res.x + 1) * res.y * (res.z + 1);
        nez = (res.x + 1) * (res.y + 1) * res.z;
    }

    /// <summary>indexToNodePosition in double + Discregrid's AlignedBox::contains (inclusive) of that double point.</summary>
    static void NodePositions(double[] dmin, double[] dmax, double[] cell, Vector3Int res, Vector4[] outPos, double[] outD = null)
    {
        NodeCounts(res, out int nv, out int nex, out int ney, out int nez);
        int n0 = res.x, n1 = res.y, n2 = res.z;
        int total = nv + 2 * (nex + ney + nez);
        for (int L = 0; L < total; L++)
        {
            long l = L; long i, j, k; double ox = 0, oy = 0, oz = 0;
            if (l < nv) { k = l / ((n1 + 1) * (n0 + 1)); long t = l % ((n1 + 1) * (n0 + 1)); j = t / (n0 + 1); i = t % (n0 + 1); }
            else if (l < nv + 2 * nex)
            {
                l -= nv; long e = l / 2; k = e / ((n1 + 1) * n0); long t = e % ((n1 + 1) * n0); j = t / n0; i = t % n0;
                ox = (1.0 + (double)(l % 2)) / 3.0 * cell[0];
            }
            else if (l < nv + 2 * (nex + ney))
            {
                l -= nv + 2 * nex; long e = l / 2; i = e / ((n2 + 1) * n1); long t = e % ((n2 + 1) * n1); k = t / n1; j = t % n1;
                oy = (1.0 + (double)(l % 2)) / 3.0 * cell[1];
            }
            else
            {
                l -= nv + 2 * (nex + ney); long e = l / 2; j = e / ((n0 + 1) * n2); long t = e % ((n0 + 1) * n2); i = t / n2; k = t % n2;
                oz = (1.0 + (double)(l % 2)) / 3.0 * cell[2];
            }
            double x = dmin[0] + cell[0] * i + ox, y = dmin[1] + cell[1] * j + oy, z = dmin[2] + cell[2] * k + oz;
            bool inside = x >= dmin[0] && x <= dmax[0] && y >= dmin[1] && y <= dmax[1] && z >= dmin[2] && z <= dmax[2];
            outPos[L] = new Vector4((float)x, (float)y, (float)z, inside ? 1f : 0f);
            if (outD != null) { outD[3 * L] = x; outD[3 * L + 1] = y; outD[3 * L + 2] = z; }
        }
    }

    void BuildVolumeMaps()
    {
        int total = 0;
        foreach (var b in bodies)
        {
            NodeCounts(b.mapResolution, out int nv, out int nex, out int ney, out int nez);
            b.nodeOffset = total;
            b.nodeCount = nv + 2 * (nex + ney + nez);
            total += b.nodeCount;
            b.MeshAabb(out var lo, out var hi);
            double m = 8.0 * (double)h + (double)b.mapThickness;      // (8 supportRadius + tolerance) in double
            if (b.tessellated)
            {
                b.dmin = new[] { b.mesh.aabbMin[0] - m, b.mesh.aabbMin[1] - m, b.mesh.aabbMin[2] - m };
                b.dmax = new[] { b.mesh.aabbMax[0] + m, b.mesh.aabbMax[1] + m, b.mesh.aabbMax[2] + m };
            }
            else
            {
                b.dmin = new[] { lo.x - m, lo.y - m, lo.z - m };
                b.dmax = new[] { hi.x + m, hi.y + m, hi.z + m };
            }
            b.cell = new[] { (b.dmax[0] - b.dmin[0]) / b.mapResolution.x, (b.dmax[1] - b.dmin[1]) / b.mapResolution.y,
                             (b.dmax[2] - b.dmin[2]) / b.mapResolution.z };
        }
        bNodes?.Release();
        bNodes = Buf(Math.Max(total, 1), 8);
        UploadBodies();
        var cmd = new CommandBuffer { name = "DFSPH1001.VolumeMaps" };
        Uniforms(cmd);
        var tmp = new List<GraphicsBuffer>();
        for (int bi = 0; bi < bodies.Count; bi++)
        {
            var b = bodies[bi];
            var pos = new Vector4[b.nodeCount];
            var posD = b.tessellated ? new double[3 * b.nodeCount] : null;
            NodePositions(b.dmin, b.dmax, b.cell, b.mapResolution, pos, posD);
            if (b.tessellated)
            {   // field 0 = sign (TriangleMeshDistance(x) - thickness) on the stock mesh, in double, uploaded before the GPU passes
                var f0 = new Vector2[b.nodeCount];
                double sign = b.mapInvert ? -1.0 : 1.0, tol = b.mapThickness;
                var mesh = b.mesh;
                System.Threading.Tasks.Parallel.For(0, b.nodeCount, l =>
                {
                    double d = mesh.SignedDistance(posD[3 * l], posD[3 * l + 1], posD[3 * l + 2]);
                    f0[l] = new Vector2((float)(sign * (d - tol)), 0f);
                });
                bNodes.SetData(f0, 0, b.nodeOffset, b.nodeCount);
            }
            var bpos = Buf(b.nodeCount, 16); bpos.SetData(pos); tmp.Add(bpos);
            Vector4 param; int shape = (int)b.shape;
            switch (b.shape)
            {
                case Shape.Box: param = 0.5f * b.scale; break;
                case Shape.Sphere: param = new Vector4(b.scale.x, 0, 0, 0); break;
                default: param = new Vector4(1f * b.scale.x, 0.5f * b.scale.x, 0, 0); break;
            }
            cmd.SetComputeIntParam(cs, "_VmBody", bi);
            cmd.SetComputeIntParam(cs, "_VmShape", shape);
            cmd.SetComputeVectorParam(cs, "_VmShapeParam", param);
            cmd.SetComputeFloatParam(cs, "_VmSign", b.mapInvert ? -1f : 1f);
            cmd.SetComputeFloatParam(cs, "_VmThickness", b.mapThickness);
            cmd.SetComputeIntParam(cs, "_VmNodesTotal", b.nodeCount);
            foreach (int k in new[] { kVmSdf, kVmVol })
            {
                cmd.SetComputeBufferParam(cs, k, "_Bodies", bBodies);
                cmd.SetComputeBufferParam(cs, k, "_VmNodes", bNodes);
                cmd.SetComputeBufferParam(cs, k, "_VmPos", bpos);
            }
            cmd.SetComputeIntParam(cs, "_VmNodeStart", 0);
            cmd.SetComputeIntParam(cs, "_VmNodeCount", b.nodeCount);
            if (!b.tessellated) cmd.DispatchCompute(cs, kVmSdf, (b.nodeCount + T - 1) / T, 1, 1);
            const int chunk = 8192;   // keep each dispatch short (16^3 quadrature x 32-node interpolation per node)
            for (int s = 0; s < b.nodeCount; s += chunk)
            {
                cmd.SetComputeIntParam(cs, "_VmNodeStart", s);
                cmd.SetComputeIntParam(cs, "_VmNodeCount", Math.Min(chunk, b.nodeCount - s));
                cmd.DispatchCompute(cs, kVmVol, (Math.Min(chunk, b.nodeCount - s) + 63) / 64, 1, 1);
            }
        }
        Graphics.ExecuteCommandBuffer(cmd);
        cmd.Release();
        // make sure the build finished before the staging buffers go away
        var probe = new float[2]; bNodes.GetData(probe, 0, 0, 2);
        foreach (var t in tmp) t.Release();
    }

    /// <summary>Upload body poses / kinematic velocities (call after changing translation / rotation / velocities).</summary>
    public void UploadBodies()
    {
        StageBodies();
        bBodies.SetData(bodyStage);
    }

    /// <summary>UploadBodies through a command buffer, so the upload sits between ExecuteSteps calls on the GPU timeline
    /// (per-step prop poses: upload, ExecuteSteps(1), upload, ...).</summary>
    public void UploadBodiesOrdered()
    {
        StageBodies();
        if (bodyCb == null) bodyCb = new CommandBuffer { name = "DFSPH1001.Bodies" };
        bodyCb.Clear();
        bodyCb.SetBufferData(bBodies, bodyStage);
        Graphics.ExecuteCommandBuffer(bodyCb);
    }

    void StageBodies()
    {
        for (int i = 0; i < MaxBodies; i++) bodyStage[i] = default;
        for (int i = 0; i < bodies.Count; i++)
        {
            var b = bodies[i];
            var R = b.Rotation();
            var g = new BodyGpu
            {
                r0 = new Vector4(R.m00, R.m01, R.m02, 0), r1 = new Vector4(R.m10, R.m11, R.m12, 0), r2 = new Vector4(R.m20, R.m21, R.m22, 0),
                t = new Vector4(b.translation.x, b.translation.y, b.translation.z, 0f),
                resX = (uint)b.mapResolution.x, resY = (uint)b.mapResolution.y, resZ = (uint)b.mapResolution.z, nodeOff = (uint)b.nodeOffset,
            };
            if (b.dmin != null)
            {
                g.dmin = new Vector4((float)b.dmin[0], (float)b.dmin[1], (float)b.dmin[2], 0);
                g.dmax = new Vector4((float)b.dmax[0], (float)b.dmax[1], (float)b.dmax[2], 0);
                g.cell = new Vector4((float)b.cell[0], (float)b.cell[1], (float)b.cell[2], 0);
                g.invCell = new Vector4((float)(1.0 / b.cell[0]), (float)(1.0 / b.cell[1]), (float)(1.0 / b.cell[2]), 0);
            }
            if (b.moving)
            {
                g.vlin = new Vector4(b.linearVelocity.x, b.linearVelocity.y, b.linearVelocity.z, 1f);
                g.vang = new Vector4(b.angularVelocity.x, b.angularVelocity.y, b.angularVelocity.z, 0f);
                g.t.w = (Vector3.Cross(b.angularVelocity, new Vector3(b.MaxDist(), 0f, 0f)) + b.linearVelocity).magnitude;
            }
            bodyStage[i] = g;
        }
    }

    // ---------- particles ----------
    /// <summary>SimulatorBase::createFluidBlocks for one block (float math as Real = float). Returns positions.</summary>
    public static List<Vector3> SampleFluidBlock(Vector3 boxMin, Vector3 boxMax, Vector3 scale, Vector3 translation, int mode, float radius)
    {
        var res = new List<Vector3>();
        float diam = 2f * radius;
        float xshift = diam, yshift = diam;
        const float eps = 1.0e-9f;
        if (mode == 1) yshift = MathF.Sqrt(3f) * radius + eps;
        else if (mode == 2) { xshift = MathF.Sqrt(6f) * diam / 3f + eps; yshift = MathF.Sqrt(3f) * radius + eps; }
        Vector3 minX = Vector3.Scale(scale, boxMin) + translation;
        Vector3 maxX = Vector3.Scale(scale, boxMax) + translation;
        Vector3 diff = maxX - minX;
        if (mode == 1) { diff.x -= diam; diff.z -= diam; }
        else if (mode == 2) { diff.x -= xshift; diff.z -= diam; }
        int stepsX = (int)MathF.Round(diff.x / xshift, MidpointRounding.AwayFromZero) - 1;
        int stepsY = (int)MathF.Round(diff.y / yshift, MidpointRounding.AwayFromZero) - 1;
        int stepsZ = (int)MathF.Round(diff.z / diam, MidpointRounding.AwayFromZero) - 1;
        Vector3 start = minX + 2f * radius * Vector3.one;
        if (stepsX <= 1 || stepsY <= 1 || stepsZ <= 1) return res;
        for (int j = 0; j < stepsX; j++)
            for (int k = 0; k < stepsY; k++)
                for (int l = 0; l < stepsZ; l++)
                {
                    Vector3 p = new Vector3(j * xshift, k * yshift, l * diam) + start;
                    if (mode == 1)
                    {
                        if (k % 2 == 0) p += new Vector3(0, 0, radius); else p += new Vector3(radius, 0, 0);
                    }
                    else if (mode == 2)
                    {
                        p += new Vector3(0, 0, radius);
                        Vector3 sv = Vector3.zero;
                        if ((j % 2) != 0) sv.z += diam / (2f * (k % 2 != 0 ? -1 : 1));
                        if (k % 2 == 0) sv.x += xshift / 2f;
                        p += sv;
                    }
                    res.Add(p);
                }
        return res;
    }

    /// <summary>Append particles (positions / velocities in sim space). Warm-start pressures of new particles are 0.</summary>
    public int AddParticles(IList<Vector3> pos, IList<Vector3> vel)
    {
        int add = Math.Min(pos.Count, maxParticles - n);
        if (add <= 0) return 0;
        var p = new Vector3[add]; var v = new Vector3[add];
        for (int i = 0; i < add; i++) { p[i] = pos[i]; v[i] = vel != null ? vel[i] : Vector3.zero; }
        bPos.SetData(p, 0, n, add);
        bVel.SetData(v, 0, n, add);
        var z = new float[add];
        bP.SetData(z, 0, n, add); bPV.SetData(z, 0, n, add);
        var d = new float[add]; for (int i = 0; i < add; i++) d[i] = density0;
        bDens.SetData(d, 0, n, add);
        n += add;
        stepCbDirty = true;
        return add;
    }

    // ---------- stepping ----------
    void BindAll(CommandBuffer cmd)
    {
        void B(int k, string name, GraphicsBuffer b) => cmd.SetComputeBufferParam(cs, k, name, b);
        int[] all = { kGClear, kGCount, kGScan, kGScatter, kGSort, kNbr, kBnd, kDens, kFactor, kDivInit, kPAccelV, kDivUpd,
                      kDivChk, kDivFin, kNonP, kCflRed, kCflFin, kIntV, kPresInit, kPAccel, kPresUpd, kPresChk, kPresFin, kIntP, kEnd };
        foreach (int k in all)
        {
            B(k, "_Pos", bPos); B(k, "_PosR", bPos); B(k, "_Vel", bVel); B(k, "_VelR", bVel);
            B(k, "_Dens", bDens); B(k, "_DensR", bDens); B(k, "_Acc", bAcc); B(k, "_AccR", bAcc);
            B(k, "_Factor", bFactor); B(k, "_FactorR", bFactor); B(k, "_DensAdv", bDensAdv); B(k, "_DensAdvR", bDensAdv);
            B(k, "_PRho2", bP); B(k, "_PRho2R", bP); B(k, "_PRho2V", bPV); B(k, "_PRho2VR", bPV);
            B(k, "_PAcc", bPAcc); B(k, "_PAccR", bPAcc); B(k, "_NbrCount", bNbrCount); B(k, "_NbrCountR", bNbrCount);
            B(k, "_Nbrs", bNbrs); B(k, "_NbrsR", bNbrs); B(k, "_BVol", bBVol); B(k, "_BVolR", bBVol);
            B(k, "_BXj", bBXj); B(k, "_BXjR", bBXj); B(k, "_CellOf", bCellOf); B(k, "_CellOfR", bCellOf);
            B(k, "_CellCount", bCellCount); B(k, "_CellStart", bCellStart); B(k, "_CellStartR", bCellStart);
            B(k, "_CellFill", bCellFill); B(k, "_Sorted", bSorted); B(k, "_SortedR", bSorted);
            B(k, "_State", bState); B(k, "_StateR", bState); B(k, "_Partial", bPartial); B(k, "_PartialR", bPartial);
            B(k, "_Args", bArgs); B(k, "_Stats", bStats); B(k, "_WTable", bWT); B(k, "_GTable", bGT);
            B(k, "_Bodies", bBodies); B(k, "_VmNodesR", bNodes);
        }
    }

    /// <summary>The fixed command buffer of ONE reference step (rebuilt when N, bodies or the config change).</summary>
    CommandBuffer StepCommandBuffer()
    {
        if (!stepCbDirty && stepCb != null) return stepCb;
        stepCb?.Release();
        var c = new CommandBuffer { name = "DFSPH1001.Step" };
        c.BeginSample(sampler);
        Uniforms(c);
        if (n > 0)
        {
            BindAll(c);
            int g = groups;
            int gc = Math.Max(1, (cells + T - 1) / T);
            c.DispatchCompute(cs, kGClear, gc, 1, 1);
            c.DispatchCompute(cs, kGCount, g, 1, 1);
            c.DispatchCompute(cs, kGScan, 1, 1, 1);
            c.DispatchCompute(cs, kGScatter, g, 1, 1);
            c.DispatchCompute(cs, kGSort, gc, 1, 1);
            c.DispatchCompute(cs, kNbr, g, 1, 1);
            if (bodies.Count > 0) c.DispatchCompute(cs, kBnd, g, 1, 1);
            c.DispatchCompute(cs, kDens, g, 1, 1);
            c.DispatchCompute(cs, kFactor, g, 1, 1);
            if (enableDivergenceSolver)
            {
                c.DispatchCompute(cs, kDivInit, g, 1, 1);
                int kv = Math.Min(kCapV, maxIterationsV);
                for (int it = 0; it < kv; it++)
                {
                    c.DispatchCompute(cs, kPAccelV, bArgs, 0);
                    c.DispatchCompute(cs, kDivUpd, bArgs, 0);
                    c.DispatchCompute(cs, kDivChk, 1, 1, 1);
                }
                c.DispatchCompute(cs, kPAccelV, g, 1, 1);
                c.DispatchCompute(cs, kDivFin, g, 1, 1);
            }
            c.DispatchCompute(cs, kNonP, g, 1, 1);
            c.DispatchCompute(cs, kCflRed, g, 1, 1);
            c.DispatchCompute(cs, kCflFin, 1, 1, 1);
            c.DispatchCompute(cs, kIntV, g, 1, 1);
            c.DispatchCompute(cs, kPresInit, g, 1, 1);
            int kp = Math.Min(kCap, maxIterations);
            for (int it = 0; it < kp; it++)
            {
                c.DispatchCompute(cs, kPAccel, bArgs, 12);
                c.DispatchCompute(cs, kPresUpd, bArgs, 12);
                c.DispatchCompute(cs, kPresChk, 1, 1, 1);
            }
            c.DispatchCompute(cs, kPAccel, g, 1, 1);
            c.DispatchCompute(cs, kPresFin, g, 1, 1);
            c.DispatchCompute(cs, kIntP, g, 1, 1);
            c.DispatchCompute(cs, kEnd, 1, 1, 1);
        }
        c.EndSample(sampler);
        stepCb = c;
        stepCbDirty = false;
        return c;
    }

    /// <summary>Record one reference step into cmd (the fixed step command buffer is re-used: execute it instead when
    /// possible — see ExecuteSteps).</summary>
    public void ExecuteSteps(int count)
    {
        if (n == 0 || count <= 0) return;
        var c = StepCommandBuffer();
        for (int i = 0; i < count; i++) Graphics.ExecuteCommandBuffer(c);
    }

    /// <summary>Changing kCap / kCapV / config after Init: mark the step buffer for re-recording.</summary>
    public void MarkDirty() => stepCbDirty = true;

    // ---------- readback (parity / stats) ----------
    public void ReadState(uint[] dst) => bState.GetData(dst);
    public float ReadTime() { var s = new uint[StateWords]; bState.GetData(s); return BitsFloat(s[2]); }

    public static StepStats DecodeStats(uint[] ring, int slot)
    {
        int o = slot * StatWords;
        return new StepStats
        {
            step = ring[o], t = BitsFloat(ring[o + 1]), dtOld = BitsFloat(ring[o + 2]), dtNew = BitsFloat(ring[o + 3]),
            itV = ring[o + 4], it = ring[o + 5], errV = BitsFloat(ring[o + 6]), err = BitsFloat(ring[o + 7]),
            nbrMax = ring[o + 8], pushed = ring[o + 9], truncV = ring[o + 10], trunc = ring[o + 11], vmax = BitsFloat(ring[o + 12]),
            recV = ring[o + 13], rec = ring[o + 14], valid = ring[o + 15] == 0xD5F5u,
        };
    }

    public void ReadParticles(Vector3[] pos, Vector3[] vel, float[] dens)
    {
        if (n == 0) return;
        if (pos != null) bPos.GetData(pos, 0, 0, n);
        if (vel != null) bVel.GetData(vel, 0, 0, n);
        if (dens != null) bDens.GetData(dens, 0, 0, n);
    }

    public void ReadExtra(float[] factor, float[] densAdv, float[] p, float[] pv, uint[] nbrCount)
    {
        if (factor != null) bFactor.GetData(factor, 0, 0, n);
        if (densAdv != null) bDensAdv.GetData(densAdv, 0, 0, n);
        if (p != null) bP.GetData(p, 0, 0, n);
        if (pv != null) bPV.GetData(pv, 0, 0, n);
        if (nbrCount != null) bNbrCount.GetData(nbrCount, 0, 0, n);
    }

    public void ReadBoundary(float[] vol, Vector3[] xj)
    {
        if (vol != null) bBVol.GetData(vol, 0, 0, n * bodies.Count);
        if (xj != null) bBXj.GetData(xj, 0, 0, n * bodies.Count);
    }

    /// <summary>All volume-map nodes (field 0, field 1), bodies concatenated in order.</summary>
    public Vector2[] ReadNodes()
    {
        int total = 0; foreach (var b in bodies) total += b.nodeCount;
        var a = new Vector2[total];
        if (total > 0) bNodes.GetData(a);
        return a;
    }

    /// <summary>Gate K0 / M1 probes: kernels at r vectors, or boundary interpolation at (x, body).</summary>
    public Vector4[] Probe(bool boundary, Vector4[] input)
    {
        bProbeIn?.Release(); bProbeOut?.Release();
        bProbeIn = Buf(input.Length, 16); bProbeIn.SetData(input);
        bProbeOut = Buf(input.Length * 2, 16);
        int k = boundary ? kProbeB : kProbeK;
        Uniforms(null);
        cs.SetInt("_ProbeCount", input.Length);
        cs.SetBuffer(k, "_ProbeIn", bProbeIn);
        cs.SetBuffer(k, "_ProbeOut", bProbeOut);
        cs.SetBuffer(k, "_WTable", bWT);
        cs.SetBuffer(k, "_GTable", bGT);
        if (boundary) { cs.SetBuffer(k, "_Bodies", bBodies); cs.SetBuffer(k, "_VmNodesR", bNodes); }
        cs.Dispatch(k, (input.Length + T - 1) / T, 1, 1);
        var o = new Vector4[input.Length * 2];
        bProbeOut.GetData(o);
        return o;
    }
}
