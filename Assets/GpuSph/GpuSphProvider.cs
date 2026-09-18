// GpuSphProvider.cs — GPU PBF solver behind the ParticleFrameProvider seam. Drop-in
// replacement for LiveSphProvider: same inspector surface, same defaults (the validated
// true-coarse dam-break distribution), same Tick/accumulator/obstacle semantics — but the
// solve runs in GpuSph.compute and scales to maxParticles ~512k instead of ~4k.
//
// The frame handed to FluidSceneMVP is still a CPU float[] (one synchronous readback per
// solver frame) so the existing CPU splat works unchanged. The GPU-resident buffers are
// exposed (Solver.PositionBuffer etc.) for the future compute-shader splat — at which point
// the readback disappears (FUTURE_PLAN.md contract).

using System;
using UnityEngine;
using UnityEngine.InputSystem;

public class GpuSphProvider : ParticleFrameProvider
{
    [Header("Compute")]
    [Tooltip("GpuSph.compute asset.")]
    public ComputeShader sphCompute;

    [Header("Fluid (defaults = validated coarse dam-break distribution)")]
    public float particleRadius = 0.0414f;
    public float restDensity = 1000f;
    [Tooltip("Hard cap. The GPU solver is sized for up to ~512k (dense-GT class); live class is 4-50k.")]
    public int maxParticles = 4096;
    public float simHz = 25f;

    [Header("Domain (sim space, y-up; [0,3]³ matches training dam breaks)")]
    public Vector3 domainMin = Vector3.zero;
    public Vector3 domainMax = new Vector3(3f, 3f, 3f);
    public Vector2 domainCenterXZ = new Vector2(1.5f, 1.5f);

    [Header("Initial dam block")]
    public Vector3 blockMin = new Vector3(0.52f, 1.33f, 0.40f);
    public Vector3Int blockCount = new Vector3Int(11, 11, 11);
    public float autoDropInterval = 0f;

    [Header("PBF solver")]
    public int solverIters = 3;
    public float cfl = 0.4f;
    public int maxSubsteps = 8;
    public float lambdaEps = 100f;
    public float sCorrK = 0.001f;
    public float xsph = 0.05f;
    [Tooltip("Uniform gravity in sim space. LiveSphProvider hardcodes (0,-9.81,0); the data-gen runner randomizes this.")]
    public Vector3 gravity = new Vector3(0f, -9.81f, 0f);

    [Header("Scene obstacles (sphere/box SDFs, unit primitives scaled by the transform)")]
    public LiveSphProvider.Obstacle[] obstacles;   // reuse the serialized class + its semantics

    GpuSphSolver solver;
    float[] records;
    Vector3[] posStage, velStage;
    float[] densStage;
    bool recordsDirty = true;
    int cachedCount;
    float simTimeAcc, timeSinceDrop, solverTime;
    float lastStepMs;
    bool inited;

    public float LastStepMs => lastStepMs;
    public int ActiveParticles => solver != null ? solver.Count : 0;
    public GpuSphSolver Solver { get { EnsureInit(); return solver; } }

    public override int FrameCount { get { EnsureInit(); return 1; } }
    public override float NativeFps => simHz;
    public override float ParticleRadius => particleRadius;

    void EnsureInit()
    {
        if (inited) return;
        inited = true;
        solver = new GpuSphSolver(sphCompute)
        {
            particleRadius = particleRadius,
            restDensity = restDensity,
            maxParticles = maxParticles,
            solverIters = solverIters,
            cfl = cfl,
            maxSubsteps = maxSubsteps,
            lambdaEps = lambdaEps,
            sCorrK = sCorrK,
            xsph = xsph,
            gravity = gravity,
            domainMin = domainMin,
            domainMax = domainMax,
        };
        solver.Init();
        records = new float[maxParticles * 7];
        posStage = new Vector3[maxParticles];
        velStage = new Vector3[maxParticles];
        densStage = new float[maxParticles];
        solver.SpawnBlock(blockMin, blockCount);
    }

    public override void GetFrame(int idx, out float[] data, out int offset, out int count)
    {
        EnsureInit();
        if (recordsDirty)
        {
            cachedCount = solver.ReadbackFrame(records, posStage, velStage, densStage);
            recordsDirty = false;
        }
        data = records; offset = 0; count = cachedCount;
    }

    public override void Tick(float dt)
    {
        EnsureInit();
        if (dt <= 0f) return;
        var kb = Keyboard.current;
        if (kb != null && kb.fKey.wasPressedThisFrame) { solver.SpawnBlock(blockMin, blockCount); recordsDirty = true; }
        if (autoDropInterval > 0f)
        {
            timeSinceDrop += dt;
            if (timeSinceDrop >= autoDropInterval)
            { solver.SpawnBlock(blockMin, blockCount); timeSinceDrop = 0f; recordsDirty = true; }
        }

        simTimeAcc = Mathf.Min(simTimeAcc + dt, 3f / simHz);   // drop time if hopelessly behind
        float step = 1f / simHz;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        while (simTimeAcc >= step)
        {
            simTimeAcc -= step;
            solverTime += step;
            AdvanceObstacles(step);
            solver.gravity = gravity;
            solver.Step(step);
            recordsDirty = true;
        }
        sw.Stop();
        lastStepMs = (float)sw.Elapsed.TotalMilliseconds;   // dispatch + vmax readback, not GPU time
    }

    public override void ResetSim()
    {
        EnsureInit();
        solver.Reset();
        solverTime = 0f;
        timeSinceDrop = 0f;
        solver.SpawnBlock(blockMin, blockCount);
        recordsDirty = true;
    }

    // Same sim<->world mapping and orbit semantics as LiveSphProvider.AdvanceObstacles.
    void AdvanceObstacles(float dt)
    {
        solver.ClearObstacles();
        if (obstacles != null)
        {
            Transform anchor = transform;
            Vector3 off = new Vector3(domainCenterXZ.x, 0f, domainCenterXZ.y);
            Matrix4x4 flip = Matrix4x4.Scale(new Vector3(1f, 1f, -1f));
            Matrix4x4 simToWorld = anchor.localToWorldMatrix * flip * Matrix4x4.Translate(-off);

            for (int s = 0; s < obstacles.Length && s < GpuSphSolver.MaxObstacles; s++)
            {
                var o = obstacles[s];
                o.active = o.transform != null && o.transform.gameObject.activeInHierarchy;
                // Inactive slots are left cleared by ClearObstacles above — do NOT skip without
                // that clear, or the slot keeps last frame's transform (ghost collider).
                if (!o.active) continue;
                if (!o.started) { o.startPos = o.transform.position; o.started = true; }
                Vector3 moved = LiveSphProvider.MotionOffset(o, solverTime);
                if (moved != Vector3.zero) o.transform.position = o.startPos + moved;
                Matrix4x4 toLocal = o.transform.worldToLocalMatrix * simToWorld;
                solver.SetObstacle(s, toLocal, (int)o.shape, true);
            }
        }
        solver.UploadObstacles();
    }

    void OnDestroy()
    {
        solver?.Dispose();
        solver = null;
    }
}
