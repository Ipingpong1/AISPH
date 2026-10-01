// DfsphProvider.cs — DFSPH1001 (2026-10-01): SPlisHSPlasH's DFSPH (GPU, DfsphSolver) behind the ParticleFrameProvider seam,
// interchangeable with GpuSphProvider (IGpuSimSource) for the GPU splat / U-Net / spray path.
//
// Scene: a SPlisHSPlasH scene JSON (the corpus format; `scenePath`, absolute or relative to the project folder) — fluid
// blocks, volume-map rigid bodies on analytic primitives, every DFSPH / CFL / viscosity parameter — so a corpus scene runs
// here with the reference's configuration. Unity props in `obstacles` (LiveSphProvider.Obstacle, the GpuSphProvider
// convention: unit primitives, transform scale shapes them) are added as KINEMATIC volume-map bodies: pose and velocity
// are updated once per rendered frame.
//
// Pacing: dt is chosen on the GPU (CFL), so the CPU does not know it when it records. Each rendered frame records
// round((simClock - tEstimate) / dtEstimate) steps (<= maxStepsPerFrame); the async-read stats ring corrects tEstimate
// and dtEstimate 1-3 frames later. OnSolverStepGpu fires once per 1/simHz of estimated sim time (the spray emitter's frame,
// the corpus' 25 fps export).
// Iterations: `exactIterations` records maxIterations slots per solve (bit-for-bit the reference loop, slower); otherwise
// the cap adapts to the observed iteration counts and a step that would have needed more is counted (status line).
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

public class DfsphProvider : ParticleFrameProvider, IGpuSimSource
{
    [Header("Solver")]
    [Tooltip("Dfsph.compute (Assets/Dfsph/Resources). Null = Resources.Load(\"Dfsph\").")]
    public ComputeShader dfsphCompute;
    [Tooltip("SPlisHSPlasH scene JSON (absolute, or relative to the Unity project folder). Empty = the built-in TC_V-like dam break.")]
    public string scenePath = "Assets/Dfsph/Scenes/tcv_pilot_0000.json";
    public int maxParticles = 65536;
    [Tooltip("Scene-file sphere / torus bodies use the corpus' tessellated stock meshes (exact SPlisHSPlasH geometry). Off = analytic primitives (up to ~7 mm larger). Props are always analytic.")]
    public bool corpusTessellation = true;

    [Header("Pacing")]
    [Tooltip("Solver-frame rate: OnSolverStepGpu fires once per 1/simHz of sim time (the corpus' dataExportFPS).")]
    public float simHz = 25f;
    public int maxStepsPerFrame = 8;
    [Tooltip("Record maxIterations iteration slots per solve: identical loop to the reference, more empty dispatches.")]
    public bool exactIterations = false;
    [Tooltip("Live mode: start caps (they adapt to the observed iteration counts).")]
    public int liveCapPressure = 16, liveCapDivergence = 8;

    [Header("Domain (sim space; used by the spray and the obstacle mapping)")]
    public Vector3 domainMin = Vector3.zero;
    public Vector3 domainMax = new Vector3(3f, 3f, 3f);
    public Vector2 domainCenterXZ = new Vector2(1.5f, 1.5f);

    [Header("Kinematic props (sphere / box / torus; unit primitives scaled by the transform)")]
    public LiveSphProvider.Obstacle[] obstacles;
    [Tooltip("Volume-map resolution of the prop maps (the corpus used 20^3 for obstacles).")]
    public Vector3Int propMapResolution = new Vector3Int(20, 20, 20);

    DfsphSolver solver;
    DfsphScene scene;
    int sceneBodyCount;
    readonly List<Vector3> propScaleKey = new List<Vector3>();
    readonly List<Matrix4x4> propPrevPose = new List<Matrix4x4>();
    float simClock, tEst, dtEst, nextFrameT, solverTime;
    long stepsRecorded, stepsKnown;
    int solverFrameIdx;
    bool inited, readbackPending;
    float lastStepMs;
    float[] records; Vector3[] posStage, velStage; float[] densStage;
    bool recordsDirty = true; int cachedCount;
    readonly uint[] ring = new uint[DfsphSolver.Ring * DfsphSolver.StatWords];
    DfsphSolver.StepStats last;
    uint maxItWindow, maxItVWindow, truncSeen, truncVSeen;
    int windowSteps, generation;
    const int MinCap = 8, MinCapV = 6;   // floors: a sudden splash needs more than the calm-flow counts

    public event Action<int, float> OnSolverStepGpu;
    public DfsphSolver Solver { get { EnsureInit(); return solver; } }
    public DfsphScene Scene { get { EnsureInit(); return scene; } }
    public DfsphSolver.StepStats LastStats => last;
    public int ActiveParticles => solver != null ? solver.Count : 0;

    // IGpuSimSource
    public GraphicsBuffer PositionBuffer => Solver.PositionBuffer;
    public GraphicsBuffer VelocityBuffer => Solver.VelocityBuffer;
    public GraphicsBuffer DensityBuffer => Solver.DensityBuffer;
    public int GpuCount => Solver.Count;
    public float SolverTime => solverTime;
    public float SimHz => simHz;
    public Vector3 Gravity => solver != null ? solver.gravity : Vector3.down * 9.81f;
    public Vector3 DomainMin => domainMin;
    public Vector3 DomainMax => domainMax;
    public Vector2 DomainCenterXZ => domainCenterXZ;
    public LiveSphProvider.Obstacle[] SceneObstacles => obstacles;
    public float LastStepMs => lastStepMs;

    public override int FrameCount { get { EnsureInit(); return 1; } }
    public override float NativeFps => simHz;
    public override float ParticleRadius => scene != null ? scene.particleRadius : 0.0424f;

    public static string ResolvePath(string p)
    {
        if (string.IsNullOrEmpty(p)) return null;
        if (Path.IsPathRooted(p)) return p;
        return Path.GetFullPath(Path.Combine(Application.dataPath, "..", p));
    }

    /// <summary>The built-in default when no scene file is given: TC_V PilotV3_4x sim_0000's tank and block.</summary>
    public static DfsphScene DefaultScene()
    {
        var s = new DfsphScene
        {
            particleRadius = 0.0424032509f, gravity = new Vector3(0.1427f, -10.8482f, 0.2978f), timeStepSize = 0.00153995235f,
            cflFactor = 1f, maxError = 0.0500000007f, maxErrorV = 0.100000001f, viscosity = 0.00999999978f,
        };
        s.fluidBlocks.Add(new DfsphScene.FluidBlock { start = new Vector3(0.70432375f, 0.17562375f, 1.27802375f), end = new Vector3(2.16723588f, 1.63853587f, 2.74093588f) });
        s.bodies.Add(new DfsphSolver.Body { shape = DfsphSolver.Shape.Box, scale = new Vector3(3, 3, 3), translation = new Vector3(1.5f, 1.5f, 1.5f), mapInvert = true, mapResolution = new Vector3Int(30, 30, 30) });
        return s;
    }

    void EnsureInit()
    {
        if (inited) return;
        inited = true;
        string path = ResolvePath(scenePath);
        scene = path != null && File.Exists(path) ? DfsphScene.FromFile(path) : DefaultScene();
        if (path != null && !File.Exists(path)) Debug.LogWarning($"DfsphProvider: scene '{scenePath}' not found — using the built-in TC_V default");
        foreach (var u in scene.unsupported) Debug.LogWarning($"DfsphProvider: not reproduced: {u}");
        scene.SetTessellated(corpusTessellation);
        if (dfsphCompute == null) dfsphCompute = Resources.Load<ComputeShader>("Dfsph");
        solver = new DfsphSolver(dfsphCompute) { maxParticles = maxParticles };
        scene.Configure(solver);
        ApplyCaps();
        solver.Init();
        sceneBodyCount = scene.bodies.Count;
        var all = new List<DfsphSolver.Body>(scene.bodies);
        AddProps(all);
        solver.SetBodies(all);
        solver.FitGridToBodies(solver.SupportRadius);
        records = new float[maxParticles * 7];
        posStage = new Vector3[maxParticles]; velStage = new Vector3[maxParticles]; densStage = new float[maxParticles];
        Spawn();
        Debug.Log($"DfsphProvider: {solver.Count} particles r {scene.particleRadius:G6}, {all.Count} bodies ({sceneBodyCount} scene, {all.Count - sceneBodyCount} props), " +
                  $"dt0 {scene.timeStepSize * 1e3f:F3} ms, maxError {scene.maxError}% / V {scene.maxErrorV}%, nu {scene.viscosity}, scene {(path ?? "default")}");
    }

    void ApplyCaps()
    {
        solver.kCap = exactIterations ? solver.maxIterations : Mathf.Clamp(Mathf.Max(liveCapPressure, MinCap), 2, solver.maxIterations);
        solver.kCapV = exactIterations ? solver.maxIterationsV : Mathf.Clamp(Mathf.Max(liveCapDivergence, MinCapV), 1, solver.maxIterationsV);
    }

    void Spawn()
    {
        var pos = new List<Vector3>(); var vel = new List<Vector3>();
        scene.SampleFluid(pos, vel);
        solver.AddParticles(pos, vel);
        simClock = tEst = 0f; solverTime = 0f; dtEst = scene.timeStepSize; nextFrameT = 0f;
        stepsRecorded = 0; stepsKnown = -1; solverFrameIdx = 0;
        generation++;                         // readbacks issued before this reset are ignored
        truncSeen = truncVSeen = 0; maxItWindow = maxItVWindow = 0; windowSteps = 0;
        last = default;
        recordsDirty = true;
    }

    // ---------- kinematic props -> volume-map bodies ----------
    Matrix4x4 SimToWorld()
    {
        Vector3 off = new Vector3(domainCenterXZ.x, 0f, domainCenterXZ.y);
        return transform.localToWorldMatrix * Matrix4x4.Scale(new Vector3(1f, 1f, -1f)) * Matrix4x4.Translate(-off);
    }

    /// <summary>Obstacle local (unit primitive) -> sim, split into a proper rotation, translation and per-axis scale.</summary>
    bool PropPose(LiveSphProvider.Obstacle o, out Matrix4x4 R, out Vector3 t, out Vector3 S)
    {
        R = Matrix4x4.identity; t = Vector3.zero; S = Vector3.one;
        if (o == null || o.transform == null || !o.transform.gameObject.activeInHierarchy) return false;
        Matrix4x4 M = SimToWorld().inverse * o.transform.localToWorldMatrix;   // local -> sim
        Vector3 c0 = M.GetColumn(0), c1 = M.GetColumn(1), c2 = M.GetColumn(2);
        S = new Vector3(c0.magnitude, c1.magnitude, c2.magnitude);
        c0 /= S.x; c1 /= S.y; c2 /= S.z;
        if (Vector3.Dot(Vector3.Cross(c0, c1), c2) < 0f) c2 = -c2;            // the z flip: primitives are symmetric in local z
        R.SetColumn(0, c0); R.SetColumn(1, c1); R.SetColumn(2, c2);
        t = M.GetColumn(3);
        return true;
    }

    DfsphSolver.Body PropBody(LiveSphProvider.Obstacle o, Matrix4x4 R, Vector3 t, Vector3 S)
    {
        var b = new DfsphSolver.Body { translation = t, mapResolution = propMapResolution, rotationMatrix = R, useRotationMatrix = true };
        switch (o.shape)
        {
            case LiveSphProvider.Obstacle.Shape.Sphere: b.shape = DfsphSolver.Shape.Sphere; b.scale = Vector3.one * (0.5f * S.x); break;   // unit sphere r 0.5 -> sphere.obj r 1
            case LiveSphProvider.Obstacle.Shape.Box: b.shape = DfsphSolver.Shape.Box; b.scale = S; break;                                      // half extent 0.5 -> UnitBox
            default: b.shape = DfsphSolver.Shape.Torus; b.scale = Vector3.one * (S.x / 3f); break;                                          // R 1/3, r 1/6 -> torus.obj R 1, r 0.5
        }
        return b;
    }

    void AddProps(List<DfsphSolver.Body> all)
    {
        propScaleKey.Clear(); propPrevPose.Clear();
        if (obstacles == null) return;
        foreach (var o in obstacles)
        {
            if (all.Count >= DfsphSolver.MaxBodies) { Debug.LogWarning("DfsphProvider: more than 8 bodies, extra props ignored"); break; }
            if (!PropPose(o, out var R, out var t, out var S)) { propScaleKey.Add(Vector3.zero); propPrevPose.Add(Matrix4x4.identity); continue; }
            all.Add(PropBody(o, R, t, S));
            propScaleKey.Add(S);
            var pose = R; pose.SetColumn(3, new Vector4(t.x, t.y, t.z, 1f));
            propPrevPose.Add(pose);
        }
    }

    /// <summary>Once per rendered frame: move the prop bodies (pose + velocity); rebuild their maps if a scale changed.</summary>
    void UpdateProps(float frameDt)
    {
        if (obstacles == null || obstacles.Length == 0) return;
        bool rebuild = false;
        var bodies = solver.Bodies;
        int bi = sceneBodyCount;
        for (int s = 0; s < obstacles.Length && s < propScaleKey.Count; s++)
        {
            var o = obstacles[s];
            if (o == null) continue;
            if (!o.started && o.transform != null) { o.startPos = o.transform.position; o.started = true; }
            Vector3 moved = o.transform != null ? LiveSphProvider.MotionOffset(o, solverTime) : Vector3.zero;
            if (moved != Vector3.zero) o.transform.position = o.startPos + moved;
            if (!PropPose(o, out var R, out var t, out var S)) continue;
            if (propScaleKey[s] == Vector3.zero || (S - propScaleKey[s]).sqrMagnitude > 1e-10f) { rebuild = true; break; }
            if (bi >= bodies.Count) break;
            var b = bodies[bi++];
            var prev = propPrevPose[s];
            Vector3 tPrev = prev.GetColumn(3);
            b.linearVelocity = frameDt > 0f ? (t - tPrev) / frameDt : Vector3.zero;
            Matrix4x4 dR = R * prev.transpose; dR.SetColumn(3, new Vector4(0, 0, 0, 1)); dR.SetRow(3, new Vector4(0, 0, 0, 1));
            b.angularVelocity = frameDt > 0f ? AxisAngle(dR) / frameDt : Vector3.zero;
            b.moving = b.linearVelocity.sqrMagnitude > 0f || b.angularVelocity.sqrMagnitude > 0f;
            b.translation = t; b.rotationMatrix = R; b.useRotationMatrix = true;
            var pose = R; pose.SetColumn(3, new Vector4(t.x, t.y, t.z, 1f));
            propPrevPose[s] = pose;
        }
        if (rebuild)
        {
            var all = new List<DfsphSolver.Body>(scene.bodies);
            AddProps(all);
            solver.SetBodies(all);
        }
        else solver.UploadBodies();
    }

    static Vector3 AxisAngle(Matrix4x4 R)
    {
        float tr = R.m00 + R.m11 + R.m22;
        float ang = Mathf.Acos(Mathf.Clamp((tr - 1f) * 0.5f, -1f, 1f));
        if (ang < 1e-6f) return Vector3.zero;
        var axis = new Vector3(R.m21 - R.m12, R.m02 - R.m20, R.m10 - R.m01) / (2f * Mathf.Sin(ang));
        return axis.normalized * ang;
    }

    // ---------- stepping ----------
    public override void Tick(float dt)
    {
        EnsureInit();
        PollStats();
        if (dt <= 0f || solver.Count == 0) return;
        simClock = Mathf.Min(simClock + dt, tEst + 0.25f);       // drop time when hopelessly behind
        int steps = Mathf.Clamp(Mathf.FloorToInt((simClock - tEst) / Mathf.Max(dtEst, 1e-5f)), 0, maxStepsPerFrame);
        if (steps == 0) return;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        UpdateProps(dt);
        solver.ExecuteSteps(steps);
        stepsRecorded += steps;
        tEst += steps * dtEst;
        solverTime = tEst;
        recordsDirty = true;
        while (tEst >= nextFrameT)
        {
            OnSolverStepGpu?.Invoke(solverFrameIdx++, nextFrameT);
            nextFrameT += 1f / simHz;
        }
        RequestStats();
        sw.Stop();
        lastStepMs = (float)sw.Elapsed.TotalMilliseconds;
    }

    void RequestStats()
    {
        if (readbackPending) return;
        readbackPending = true;
        long recordedAtRequest = stepsRecorded;
        int gen = generation;
        AsyncGPUReadback.Request(solver.StatsBuffer, r =>
        {
            readbackPending = false;
            if (r.hasError || solver == null || gen != generation) return;
            r.GetData<uint>().CopyTo(ring);
            ApplyStats(recordedAtRequest);
        });
    }

    void PollStats() { }

    void ApplyStats(long recordedAtRequest)
    {
        // newest valid entry = the step with the largest index
        int best = -1; uint bestStep = 0;
        for (int s = 0; s < DfsphSolver.Ring; s++)
        {
            var st = DfsphSolver.DecodeStats(ring, s);
            if (!st.valid) continue;
            if (best < 0 || st.step > bestStep) { best = s; bestStep = st.step; }
        }
        if (best < 0) return;
        // scan the entries newer than the last poll for iteration maxima / truncations
        for (int s = 0; s < DfsphSolver.Ring; s++)
        {
            var st = DfsphSolver.DecodeStats(ring, s);
            if (!st.valid || (long)st.step <= stepsKnown) continue;
            maxItWindow = Math.Max(maxItWindow, st.it); maxItVWindow = Math.Max(maxItVWindow, st.itV);
            windowSteps++;
        }
        last = DfsphSolver.DecodeStats(ring, best);
        stepsKnown = last.step;
        bool newTrunc = last.trunc > truncSeen || last.truncV > truncVSeen;
        truncSeen = last.trunc; truncVSeen = last.truncV;
        if (newTrunc && !exactIterations)
        {   // a step ran out of recorded iteration slots: grow at once (the reference loop would have continued)
            if (last.trunc > 0) solver.kCap = Mathf.Min(solver.kCap * 2, solver.maxIterations);
            if (last.truncV > 0) solver.kCapV = Mathf.Min(solver.kCapV * 2, solver.maxIterationsV);
            solver.MarkDirty();
        }
        dtEst = last.dtNew;
        // tEst = known time + the steps recorded since that one, at the current dt estimate
        long pending = Math.Max(0, stepsRecorded - (long)last.step - 1);
        tEst = last.t + pending * dtEst;
        solverTime = tEst;
        if (!exactIterations && windowSteps >= 50) AdaptCaps();
    }

    void AdaptCaps()
    {
        int want = Mathf.Clamp((int)maxItWindow + 4 + (int)maxItWindow / 2, Mathf.Max(MinCap, solver.minIterations + 2), solver.maxIterations);
        int wantV = Mathf.Clamp((int)maxItVWindow + 2 + (int)maxItVWindow / 2, MinCapV, solver.maxIterationsV);
        // shrink slowly (one slot per window), grow at once
        if (want < solver.kCap) want = solver.kCap - 1;
        if (wantV < solver.kCapV) wantV = solver.kCapV - 1;
        if (want != solver.kCap || wantV != solver.kCapV)
        {
            solver.kCap = want; solver.kCapV = wantV;
            solver.MarkDirty();
        }
        maxItWindow = maxItVWindow = 0; windowSteps = 0;
    }

    public override void GetFrame(int idx, out float[] data, out int offset, out int count)
    {
        EnsureInit();
        if (recordsDirty)
        {
            int n = solver.Count;
            solver.ReadParticles(posStage, velStage, densStage);
            for (int i = 0; i < n; i++)
            {
                int b = i * 7;
                records[b] = posStage[i].x; records[b + 1] = posStage[i].y; records[b + 2] = posStage[i].z;
                records[b + 3] = velStage[i].x; records[b + 4] = velStage[i].y; records[b + 5] = velStage[i].z;
                records[b + 6] = densStage[i];
            }
            cachedCount = n;
            recordsDirty = false;
        }
        data = records; offset = 0; count = cachedCount;
    }

    public override void ResetSim()
    {
        EnsureInit();
        solver.Reset();
        Spawn();
    }

    public string StatusLine()
    {
        if (solver == null) return "";
        return $"LIVE GPU-DFSPH {solver.Count}p t {solverTime:F2}s dt {last.dtNew * 1e3f:F2}ms it {last.it}/{solver.kCap} itV {last.itV}/{solver.kCapV}" +
               $" err {last.err:G2}{(truncSeen + truncVSeen > 0 ? $" TRUNC {truncSeen}/{truncVSeen}" : "")}{(last.nbrMax > DfsphSolver.MaxNbr ? " NBR-OVERFLOW" : "")}" +
               $"{(last.pushed > 0 ? $" push {last.pushed}" : "")} rec {lastStepMs:F1}ms   ";
    }

    void OnDestroy()
    {
        solver?.Dispose();
        solver = null;
    }
}
