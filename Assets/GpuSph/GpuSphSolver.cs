// GpuSphSolver.cs — dispatch driver and buffer owner for GpuSph.compute (the GPU port of
// LiveSphProvider's PBF solver; that C# solver is the parity oracle — see GpuSph.compute
// header for the ported conventions and the recorded deliberate differences).
//
// Plain class, not a MonoBehaviour: GpuSphProvider wraps it for the live scene seam and
// SimExportRunner drives it headlessly for training-data generation. All simulation state
// lives in GraphicsBuffers; the CPU touches it only at spawn (upload) and export (readback).
//
// The position/velocity/density buffers + Count + radius are exactly the contract
// FUTURE_PLAN.md specifies for the eventual compute-shader splat — expose, don't read back,
// when that lands.

using System;
using UnityEngine;

public class GpuSphSolver : IDisposable
{
    // ---------- config (set before Init; mirrors LiveSphProvider fields) ----------
    public float particleRadius = 0.0414f;
    public float restDensity = 1000f;
    public int maxParticles = 4096;
    public int solverIters = 3;
    public float cfl = 0.4f;
    public int maxSubsteps = 8;
    public float lambdaEps = 100f;
    public float sCorrK = 0.001f;
    public float xsph = 0.05f;
    [Tooltip("Per-iteration position-correction cap, as a fraction of h (0 = off). PBD stabilizer: prevents impact-frame velocity spikes (v = dx/dt). Dense-class solves spiked to 40-4000 m/s without it (2026-08-19); coarse-class corrections rarely reach it.")]
    public float dpClampFrac = 0.2f;
    [Tooltip("Hard velocity cap in m/s after the position-based update (0 = off). Reference DFSPH data never exceeds ~9 m/s in the 3m domain; PBF correction spikes reached 42 m/s even with dpClamp. 12 = above the physical envelope, cleans tails only.")]
    public float maxVelocity = 12f;
    [Tooltip("067, data-gen only: 1/s decay of the tangential velocity of particles resting on a domain wall (0 = off, the live behaviour).")]
    public float wallDamp = 0f;
    public Vector3 gravity = new Vector3(0f, -9.81f, 0f);
    public Vector3 domainMin = Vector3.zero;
    public Vector3 domainMax = new Vector3(3f, 3f, 3f);

    public const int MaxObstacles = 8;

    // ---------- state ----------
    ComputeShader cs;
    int kIntegrate, kClearGrid, kBuildGrid, kLambda, kDelta, kApply, kFinalize,
        kXsphAccum, kXsphApply, kVmax;

    GraphicsBuffer bPos, bVel, bPred, bDp, bDens, bLambda, bHead, bNext, bVmax, bObstacles;
    int tableSize;
    int n;
    float h, mass, kernelSigma, invSCorrDenom;

    readonly ObstacleGpu[] obstacleStage = new ObstacleGpu[MaxObstacles];
    int numObstacles;
    readonly uint[] vmaxStage = new uint[1];

    Vector3[] spawnStage;   // CPU staging for block uploads only
    float[] spawnDensStage;

    public int Count => n;
    public float ParticleRadius => particleRadius;
    public GraphicsBuffer PositionBuffer => bPos;
    public GraphicsBuffer VelocityBuffer => bVel;
    public GraphicsBuffer DensityBuffer => bDens;

    // Matches ObstacleGpu in GpuSph.compute: 8 float4 + int + int + float2 pad = 144 bytes.
    // Transforms travel as EXPLICIT ROWS. They used to be Matrix4x4 fields uploaded `.transpose`
    // on the belief that HLSL reads a structured-buffer float4x4 as consecutive rows; it does not
    // (matrix packing there defaults to column-major, which is also Unity's Matrix4x4 layout), so
    // every obstacle arrived transposed and collided with nothing. Silent: no error, no NaN, the
    // fluid simply ignored props. Found 2026-08-20 by GpuSphObstacleGate (0 of 267 oracle-moved
    // particles moved on the GPU). Rows + dot products have one meaning on every backend.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    struct ObstacleGpu
    {
        public Vector4 t0, t1, t2, t3;   // toLocal rows   (sim -> obstacle local)
        public Vector4 f0, f1, f2, f3;   // fromLocal rows (obstacle local -> sim)
        public int shape;                // 0 sphere, 1 box, 2 torus
        public int active;
        public Vector2 _pad;
    }

    // ---------- lifecycle ----------

    public GpuSphSolver(ComputeShader shader)
    {
        cs = shader;
    }

    public void Init()
    {
        h = 4f * particleRadius;
        float d = 2f * particleRadius;
        mass = 0.8f * d * d * d * restDensity;               // SPlisHSPlasH convention
        kernelSigma = 8f / (Mathf.PI * h * h * h);
        float sCorrDenom = W(0.2f * h);
        invSCorrDenom = sCorrDenom > 0f ? 1f / sCorrDenom : 0f;

        kIntegrate = cs.FindKernel("Integrate");
        kClearGrid = cs.FindKernel("ClearGrid");
        kBuildGrid = cs.FindKernel("BuildGrid");
        kLambda = cs.FindKernel("ComputeLambda");
        kDelta = cs.FindKernel("ComputeDelta");
        kApply = cs.FindKernel("ApplyDelta");
        kFinalize = cs.FindKernel("FinalizeVelocity");
        kXsphAccum = cs.FindKernel("XsphAccum");
        kXsphApply = cs.FindKernel("XsphApply");
        kVmax = cs.FindKernel("VmaxReduce");

        tableSize = Mathf.NextPowerOfTwo(maxParticles * 2);
        bPos = NewBuffer(maxParticles, 12);
        bVel = NewBuffer(maxParticles, 12);
        bPred = NewBuffer(maxParticles, 12);
        bDp = NewBuffer(maxParticles, 12);
        bDens = NewBuffer(maxParticles, 4);
        bLambda = NewBuffer(maxParticles, 4);
        bHead = NewBuffer(tableSize, 4);
        bNext = NewBuffer(maxParticles, 4);
        bVmax = NewBuffer(1, 4);
        bObstacles = NewBuffer(MaxObstacles, 144);
        spawnStage = new Vector3[maxParticles];
        spawnDensStage = new float[maxParticles];
        n = 0;

        // static uniforms
        cs.SetInt("_TableSize", tableSize);
        cs.SetFloat("_H", h);
        cs.SetFloat("_H2", h * h);
        cs.SetFloat("_Mass", mass);
        cs.SetFloat("_KernelSigma", kernelSigma);
        cs.SetFloat("_InvRho0", 1f / restDensity);
        cs.SetFloat("_LambdaEps", lambdaEps);
        cs.SetFloat("_SCorrK", sCorrK);
        cs.SetFloat("_InvSCorrDenom", invSCorrDenom);
        cs.SetFloat("_Xsph", xsph);
        cs.SetFloat("_DpClamp", dpClampFrac > 0f ? dpClampFrac * h : 0f);
        cs.SetFloat("_VelClamp", maxVelocity);
        cs.SetFloat("_WallDamp", wallDamp);
        BindAll();
        UploadObstacles();   // zero obstacles until told otherwise
    }

    static GraphicsBuffer NewBuffer(int count, int stride)
        => new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, stride);

    void BindAll()
    {
        foreach (int k in new[] { kIntegrate, kClearGrid, kBuildGrid, kLambda, kDelta, kApply,
                                  kFinalize, kXsphAccum, kXsphApply, kVmax })
        {
            cs.SetBuffer(k, "_Pos", bPos);
            cs.SetBuffer(k, "_Vel", bVel);
            cs.SetBuffer(k, "_Pred", bPred);
            cs.SetBuffer(k, "_Dp", bDp);
            cs.SetBuffer(k, "_Dens", bDens);
            cs.SetBuffer(k, "_Lambda", bLambda);
            cs.SetBuffer(k, "_Head", bHead);
            cs.SetBuffer(k, "_Next", bNext);
            cs.SetBuffer(k, "_VmaxBits", bVmax);
            cs.SetBuffer(k, "_Obstacles", bObstacles);
        }
    }

    public void Dispose()
    {
        foreach (var b in new[] { bPos, bVel, bPred, bDp, bDens, bLambda, bHead, bNext, bVmax, bObstacles })
            b?.Release();
        bPos = bVel = bPred = bDp = bDens = bLambda = bHead = bNext = bVmax = bObstacles = null;
    }

    // cubic spline, CPU copy for the sCorr denominator (same as LiveSphProvider.W)
    float W(float r)
    {
        float q = r / h;
        if (q >= 1f) return 0f;
        if (q <= 0.5f) { float q2 = q * q; return kernelSigma * (6f * (q2 * q - q2) + 1f); }
        float t = 1f - q;
        return kernelSigma * 2f * t * t * t;
    }

    // ---------- spawning ----------

    public void Reset() { n = 0; }

    // Lattice block at spacing 2r, spawn order = ix-major (identical to LiveSphProvider.SpawnBlock).
    // Returns particles actually added (clipped to maxParticles).
    // `shuffle` (067, data-gen only; null = today's order): seeded Fisher-Yates over the block's spawn order. The
    // training LR keeps every 25th particle BY INDEX; on an ix-major lattice that is a degenerate rod sub-lattice,
    // which a dam break mixes away and a coherent pool does not. Shuffling makes index-thinning a random subsample.
    public int SpawnBlock(Vector3 blockMin, Vector3Int count, System.Random shuffle = null)
    {
        float spacing = 2f * particleRadius;
        int added = 0, start = n;
        for (int ix = 0; ix < count.x; ix++)
            for (int iy = 0; iy < count.y; iy++)
                for (int iz = 0; iz < count.z; iz++)
                {
                    if (n >= maxParticles) goto done;
                    spawnStage[added] = blockMin + new Vector3(ix * spacing, iy * spacing, iz * spacing);
                    n++; added++;
                }
        done:
        if (shuffle != null)
            for (int i = added - 1; i > 0; i--)
            {
                int j = shuffle.Next(i + 1);
                (spawnStage[i], spawnStage[j]) = (spawnStage[j], spawnStage[i]);
            }
        if (added > 0)
        {
            bPos.SetData(spawnStage, 0, start, added);
            Array.Clear(spawnStage, 0, added);            // zero velocities from the same staging
            bVel.SetData(spawnStage, 0, start, added);
            for (int i = 0; i < added; i++) spawnDensStage[i] = restDensity * 0.8f;
            bDens.SetData(spawnDensStage, 0, start, added);   // pre-solve export parity with C#
        }
        return added;
    }

    // ---------- obstacles ----------

    // toLocal maps SIM space into the obstacle's local space where the unit primitives have
    // radius / half-extent 0.5 (same convention as LiveSphProvider.Obstacle).
    // `shape` is (int)LiveSphProvider.Obstacle.Shape — 0 sphere, 1 box, 2 torus — passed through
    // verbatim to ObstacleGpu.shape in GpuSph.compute.
    public void SetObstacle(int slot, Matrix4x4 toLocal, int shape, bool active)
    {
        Matrix4x4 fromLocal = toLocal.inverse;
        obstacleStage[slot] = new ObstacleGpu
        {
            t0 = toLocal.GetRow(0), t1 = toLocal.GetRow(1),      // see ObstacleGpu comment:
            t2 = toLocal.GetRow(2), t3 = toLocal.GetRow(3),      // explicit rows, no packing
            f0 = fromLocal.GetRow(0), f1 = fromLocal.GetRow(1),  // convention to get wrong
            f2 = fromLocal.GetRow(2), f3 = fromLocal.GetRow(3),
            shape = shape,
            active = active ? 1 : 0,
        };
        numObstacles = Mathf.Max(numObstacles, slot + 1);
    }

    // Resetting only the count is NOT enough: numObstacles is a high-water mark (SetObstacle
    // raises it to slot+1), so a slot that goes inactive while a LATER slot stays active is still
    // inside the GPU's loop. Unless its active flag is cleared here it keeps last frame's
    // transform and the fluid collides with a prop that is no longer in the scene — which is
    // exactly what toggling a demo prop off does. Clear the whole stage.
    public void ClearObstacles()
    {
        numObstacles = 0;
        for (int i = 0; i < MaxObstacles; i++) obstacleStage[i].active = 0;
    }

    public void UploadObstacles()
    {
        bObstacles.SetData(obstacleStage);
        cs.SetInt("_NumObstacles", numObstacles);
    }

    // ---------- stepping ----------

    int Groups(int count) => (count + 255) / 256;

    // One solver frame of frameDt seconds, CFL-substepped exactly like LiveSphProvider.StepFrame
    // (including its sqrt(max(0.5, |v|^2max)) floor — ported verbatim, oracle fidelity over sense).
    public void Step(float frameDt)
    {
        if (n == 0 || frameDt <= 0f) return;
        cs.SetInt("_N", n);

        vmaxStage[0] = 0;
        bVmax.SetData(vmaxStage);
        cs.Dispatch(kVmax, Groups(n), 1, 1);
        bVmax.GetData(vmaxStage);                                    // 4-byte sync readback
        float vmax = Mathf.Sqrt(Mathf.Max(0.5f, BitsToFloat(vmaxStage[0])));

        float dtMax = cfl * h / vmax;
        int steps = Mathf.Clamp(Mathf.CeilToInt(frameDt / dtMax), 1, maxSubsteps);
        float dt = frameDt / steps;
        for (int s = 0; s < steps; s++)
        {
            beforeSubstep?.Invoke((s + 1f) / steps);
            Substep(dt);
        }
    }

    /// <summary>067, data-gen only (null = today's behaviour): called before every CFL substep with the fraction of the
    /// solver frame that substep ENDS at. Obstacles are a position projection, so a prop moved once per frame ejects the
    /// fluid in its swept shell during the first substep at ~v_obs x steps; dense solves take 6-32 substeps, the live
    /// coarse solve 1-3. Moving the prop per substep keeps the imparted velocity ~v_obs in both.</summary>
    public Action<float> beforeSubstep;

    static float BitsToFloat(uint u) => BitConverter.Int32BitsToSingle((int)u);

    void Substep(float dt)
    {
        float wallMin = particleRadius;
        cs.SetFloat("_Dt", dt);
        cs.SetVector("_Gravity", gravity);
        cs.SetVector("_DomainMin", domainMin + Vector3.one * wallMin);
        cs.SetVector("_DomainMax", domainMax - Vector3.one * wallMin);

        int g = Groups(n);
        cs.Dispatch(kIntegrate, g, 1, 1);
        cs.Dispatch(kClearGrid, Groups(tableSize), 1, 1);
        cs.Dispatch(kBuildGrid, g, 1, 1);
        for (int it = 0; it < solverIters; it++)
        {
            cs.Dispatch(kLambda, g, 1, 1);
            cs.Dispatch(kDelta, g, 1, 1);
            cs.Dispatch(kApply, g, 1, 1);
        }
        cs.Dispatch(kFinalize, g, 1, 1);
        if (xsph > 0f)
        {
            cs.Dispatch(kXsphAccum, g, 1, 1);
            cs.Dispatch(kXsphApply, g, 1, 1);
        }
    }

    // ---------- export / readback ----------

    // Interleave pos/vel/density into the ParticleFrameProvider 7-float record contract.
    // dst must hold >= n*7 floats. Synchronous readback (export/live-seam path).
    public int ReadbackFrame(float[] dst, Vector3[] posStage, Vector3[] velStage, float[] densStage)
    {
        if (n == 0) return 0;
        bPos.GetData(posStage, 0, 0, n);
        bVel.GetData(velStage, 0, 0, n);
        bDens.GetData(densStage, 0, 0, n);
        for (int i = 0; i < n; i++)
        {
            int b = i * 7;
            dst[b] = posStage[i].x; dst[b + 1] = posStage[i].y; dst[b + 2] = posStage[i].z;
            dst[b + 3] = velStage[i].x; dst[b + 4] = velStage[i].y; dst[b + 5] = velStage[i].z;
            dst[b + 6] = densStage[i];
        }
        return n;
    }
}
