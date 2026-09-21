// SimExportRunner.cs — headless training-data generation with the GPU PBF solver.
//
// Mirrors runner.py's domain randomization (sample_params / sample_obstacles) and dumps each
// sim as <name>.bytes (1LPS, the FluidLiveMVP streamable format) + <name>_scene.json into
// SimExport/<rootName>/ next to Assets/. unity_sim_to_bgeo.py then converts a dump root into
// the exact Simulations/<Root>/ layout (gzip BGEO + manifest) the training pipeline ingests.
//
// Randomization parity with runner.py, with three recorded deviations:
//   - obstacle shapes: sphere/box only. The solver itself gained a torus on 2026-08-20 (for the
//     live demo), but this randomizer deliberately still samples sphere/box: changing the
//     data-gen shape mix is a corpus decision, not a solver one, and GpuPbfV1 was generated
//     without it. runner.py samples sphere/box/torus.
//   - obstacles are STATIC only (SPlisHSPlasH dynamic rigid bodies are a physics engine we
//     don't have; 'dynamic' is always false in the scene json so the corpus is honest about it)
//   - frames are exported at t = k/25 for k = 1..round(stop_at*25) (SPlisHSPlasH exports from
//     t = 0; same frame count, half-frame phase offset, irrelevant to training pairs)
//
// Sphere scale semantics: local unit primitives have radius/half-extent 0.5 (LiveSphProvider
// convention), so a transform scale s gives world sphere DIAMETER s and box EDGE s. Both GT
// and coarse arms of any pair use this same solver, so the convention is self-consistent.
//
// Run: put this + a GpuSph.compute reference on a GameObject, set runOnPlay, enter Play — or
// drive it from MCP execute_code. Progress goes to the Console. One sim at a time; the
// coroutine yields per exported frame so the editor stays alive.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

public class SimExportRunner : MonoBehaviour
{
    [Header("Compute")]
    public ComputeShader sphCompute;

    [Header("Batch")]
    public string rootName = "GpuPbfV1";
    public int simCount = 10;
    public int seed = 1234;
    public bool runOnPlay = false;

    [Header("Fluid class (coarse live-class default; dense-GT class = 0.014-0.016)")]
    public float radiusMin = 0.0414f;
    public float radiusMax = 0.0414f;
    public float stopAt = 1.5f;
    public int exportFps = 25;

    public enum Family { DamBreak, PoolSlosh, PoolDrop, PoolStir, PoolStirDrop }
    [Header("Scene families (067). Sim s uses families[s % Length]; {DamBreak} = the GpuPbfV1 generator, unchanged.")]
    public Family[] families = { Family.DamBreak };
    [Tooltip("Pool families: clip length [s]. Every disturbance ends >= 1.5 s earlier, so each clip has a settle tail.")]
    public float poolStopAt = 5f;
    [Tooltip("Pool families: CFL substep cap (dense class needs 32, see Max Substeps).")]
    public int poolMaxSubsteps = 32;
    [Tooltip("Pool families: particle budget; the pool is made shallower until pool + drops fit.")]
    public int poolParticleCap = 200000;
    [Tooltip("Pool families: GpuSphSolver.wallDamp [1/s]. 0 reproduces the skating floor layer of the dense class.")]
    public float poolWallDamp = 0f;
    [Tooltip("Pool families: PBF constraint iterations per substep. 3 (the live / dam-break value) UNDER-CONVERGES a resting 7-13 layer dense column: measured 2026-09-21, median density 800-850 instead of 1000 and a floor monolayer skating at 7-8 m/s (14 % of the fluid above 6 m/s) under calm water. 10 fixes 27 of 30 pools (density 1000, 0.00 % above 6 m/s, FEWER CFL substeps); the 3 that stayed under-converged at 10 (density 674-740, vmax pinned at the cap all clip) converge at 20.")]
    public int poolSolverIters = 20;

    [Header("Domain (training convention: [0,3]³)")]
    public float domainSize = 3f;
    public float padding = 0.2f;

    [Header("Obstacles (sphere/box, static)")]
    public int obstaclesMin = 1;
    public int obstaclesMax = 3;
    public float obstaclePadding = 0.3f;

    [Header("Safety")]
    [Tooltip("Refuse a sim whose fluid block would exceed this particle count.")]
    public int maxParticlesCap = 524288;
    [Tooltip("CFL substep cap per 1/25s frame. 8 is fine for the coarse live class (r~0.041); the dense-GT class (r~0.009-0.016) VIOLATES CFL at 8 and gains unphysical energy — measured 2026-08-17, use 32 there.")]
    public int maxSubsteps = 8;

    public string OutDir => Path.Combine(Application.dataPath, "..", "SimExport", rootName);

    void Start()
    {
        if (runOnPlay) StartCoroutine(RunAll());
    }

    [ContextMenu("Run Export Batch (play mode only)")]
    public void RunBatch() => StartCoroutine(RunAll());

    // ---------------------------------------------------------------- batch

    public IEnumerator RunAll()
    {
        Directory.CreateDirectory(OutDir);
        Application.runInBackground = true;
        for (int s = 0; s < simCount; s++)
        {
            var rng = new System.Random(seed + s);
            Family fam = families != null && families.Length > 0 ? families[s % families.Length] : Family.DamBreak;
            if (File.Exists(Path.Combine(OutDir, $"sim_{s:0000}_scene.json"))) continue;   // resumable
            if (fam == Family.DamBreak) yield return RunOne($"sim_{s:0000}", rng);
            else yield return RunPool($"sim_{s:0000}", rng, fam);
        }
        Debug.Log($"[SimExportRunner] batch done: {simCount} sims -> {OutDir}\n" +
                  $"next: SSU-python unity_sim_to_bgeo.py --dump_dir {OutDir} --out Simulations/{rootName}");
    }

    IEnumerator RunOne(string name, System.Random rng)
    {
        // ---- sample_params (runner.py parity) ----
        float cube = Lerp(rng, 0.8f, 1.4f);
        float startMax = domainSize - padding - cube;
        Vector3 fluidStart = new Vector3(Lerp(rng, padding, startMax),
                                         Lerp(rng, padding, startMax),
                                         Lerp(rng, padding, startMax));
        float gMag = Lerp(rng, 8.5f, 11f);
        const float tilt = 0.08f;   // ~5 deg max lateral component
        Vector3 gravity = new Vector3(Lerp(rng, -tilt, tilt) * gMag, -gMag, Lerp(rng, -tilt, tilt) * gMag);
        float radius = Lerp(rng, radiusMin, radiusMax);

        float spacing = 2f * radius;
        Vector3Int count = new Vector3Int(Mathf.Max(1, Mathf.FloorToInt(cube / spacing)),
                                          Mathf.Max(1, Mathf.FloorToInt(cube / spacing)),
                                          Mathf.Max(1, Mathf.FloorToInt(cube / spacing)));
        long total = (long)count.x * count.y * count.z;
        if (total > maxParticlesCap)
        {
            Debug.LogError($"[SimExportRunner] {name}: {total} particles exceeds cap {maxParticlesCap} — skipped");
            yield break;
        }

        var obstacles = SampleObstacles(rng, fluidStart, cube);

        // ---- solver ----
        var solver = new GpuSphSolver(sphCompute)
        {
            particleRadius = radius,
            maxParticles = (int)total,
            gravity = gravity,
            domainMin = Vector3.zero,
            domainMax = Vector3.one * domainSize,
            maxSubsteps = maxSubsteps,
        };
        solver.Init();
        // Clamp: obstacleStage has exactly MaxObstacles slots, and obstaclesMax is an inspector
        // field — an over-set value would write past the end of the array.
        int nObs = Mathf.Min(obstacles.Count, GpuSphSolver.MaxObstacles);
        if (nObs < obstacles.Count)
            Debug.LogWarning($"[SimExportRunner] {obstacles.Count} obstacles sampled, " +
                             $"only {GpuSphSolver.MaxObstacles} supported — extras dropped.");
        for (int o = 0; o < nObs; o++)
            solver.SetObstacle(o, obstacles[o].trs.inverse,
                               (int)(obstacles[o].sphere ? LiveSphProvider.Obstacle.Shape.Sphere
                                                         : LiveSphProvider.Obstacle.Shape.Box), true);
        solver.UploadObstacles();
        int spawned = solver.SpawnBlock(fluidStart, count);

        var records = new float[spawned * 7];
        var posStage = new Vector3[spawned];
        var velStage = new Vector3[spawned];
        var densStage = new float[spawned];
        int frames = Mathf.RoundToInt(stopAt * exportFps);
        var dump = new List<float[]>(frames);
        float dt = 1f / exportFps;

        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int f = 0; f < frames; f++)
        {
            solver.Step(dt);
            int n = solver.ReadbackFrame(records, posStage, velStage, densStage);
            var copy = new float[n * 7];
            Array.Copy(records, copy, n * 7);
            dump.Add(copy);
            if (f % 10 == 0) yield return null;   // keep the editor alive
        }
        sw.Stop();
        solver.Dispose();

        Write1Lps(Path.Combine(OutDir, name + ".bytes"), dump);
        WriteSceneJson(Path.Combine(OutDir, name + "_scene.json"),
                       rng, radius, gravity, cube, fluidStart, spawned, obstacles);
        Debug.Log($"[SimExportRunner] {name}: {spawned} particles x {frames} frames " +
                  $"in {sw.Elapsed.TotalSeconds:0.1}s -> {OutDir}");
    }

    static float Lerp(System.Random rng, float a, float b) => a + (float)rng.NextDouble() * (b - a);

    // ---------------------------------------------------------------- pool families (067: the live demo's content)
    // A shallow pool in the [0,3]^3 tank (the live scene is 17-44 cm deep), disturbed the way the player disturbs it:
    // a tilted start (slosh), block drops into the pool (the live F key), a moving stir sphere — then left to settle.
    // Spawn order is SHUFFLED so the pipeline's index thinning (every 25th particle) is a random subsample, and the
    // stirrer moves per CFL substep (see GpuSphSolver.SpawnBlock / beforeSubstep for why).

    struct Drop { public float t; public Vector3 start; public Vector3Int count; public float cube; public int added; }

    IEnumerator RunPool(string name, System.Random rng, Family fam)
    {
        float radius = Lerp(rng, radiusMin, radiusMax);
        float spacing = 2f * radius;
        float depth = rng.NextDouble() < 0.7 ? Lerp(rng, 0.10f, 0.30f) : Lerp(rng, 0.30f, 0.40f);
        float fill = fam == Family.PoolSlosh ? Lerp(rng, 0.55f, 1f) : 1f;          // slab collapse = the slosh
        float gMag = Lerp(rng, 9.3f, 10.3f);
        Vector3 gRest = new Vector3(0f, -gMag, 0f);
        float tiltDeg = fam == Family.PoolSlosh ? Lerp(rng, 8f, 20f) : 0f, tiltT = Lerp(rng, 0.3f, 0.8f), tiltAz = Lerp(rng, 0f, 6.2832f);
        Vector3 gTilt = Quaternion.AngleAxis(tiltDeg, new Vector3(Mathf.Cos(tiltAz), 0f, Mathf.Sin(tiltAz))) * gRest;

        bool stir = fam == Family.PoolStir || fam == Family.PoolStirDrop;
        bool dropFam = fam == Family.PoolDrop || fam == Family.PoolStirDrop;
        StirPath.Params sp = stir ? StirPath.Sample(rng, depth, poolStopAt, domainSize) : default;

        // drops: 1-3 cubes, bottom clear of the pool, top <= 1.4 m (keeps blocks off the near cameras), >= 0.8 s apart
        var drops = new List<Drop>();
        if (dropFam)
        {
            int want = rng.Next(1, 4); float tPrev = -10f;
            for (int k = 0; k < want; k++)
            {
                float cube = Lerp(rng, 0.25f, 0.55f);
                float t = Lerp(rng, 0.4f, poolStopAt - 2.0f);
                float y0 = Lerp(rng, depth + 0.35f, Mathf.Max(depth + 0.36f, 1.4f - cube));
                var start = new Vector3(Lerp(rng, 0.1f, domainSize - 0.1f - cube), y0, Lerp(rng, 0.1f, domainSize - 0.1f - cube));
                if (Mathf.Abs(t - tPrev) < 0.8f) continue;
                if (stir)   // never spawn a block on top of the stir sphere
                {
                    Vector3 c = StirPath.Eval(sp, t), bc = start + Vector3.one * (cube * 0.5f);
                    if ((c - bc).magnitude < sp.diameter * 0.5f + cube * 0.87f + 0.1f) continue;
                }
                int n1 = Mathf.Max(1, Mathf.FloorToInt(cube / spacing));
                drops.Add(new Drop { t = t, start = start, count = new Vector3Int(n1, n1, n1), cube = cube }); tPrev = t;
            }
            drops.Sort((a, b) => a.t.CompareTo(b.t));
        }
        long dropTotal = 0; foreach (var d in drops) dropTotal += (long)d.count.x * d.count.y * d.count.z;

        int nx = Mathf.Max(2, Mathf.FloorToInt((domainSize * fill - 2f * radius) / spacing) + 1);
        int nz = Mathf.Max(2, Mathf.FloorToInt((domainSize - 2f * radius) / spacing) + 1);
        int ny = Mathf.Max(2, Mathf.RoundToInt(depth / spacing));
        while (ny > 2 && (long)nx * ny * nz + dropTotal > poolParticleCap) ny--;
        depth = ny * spacing;
        long capacity = (long)nx * ny * nz + dropTotal;
        if (capacity > maxParticlesCap) { Debug.LogError($"[SimExportRunner] {name}: {capacity} particles exceeds cap — skipped"); yield break; }

        var solver = new GpuSphSolver(sphCompute)
        {
            particleRadius = radius, maxParticles = (int)capacity, gravity = tiltDeg > 0f ? gTilt : gRest,
            domainMin = Vector3.zero, domainMax = Vector3.one * domainSize, maxSubsteps = poolMaxSubsteps, wallDamp = poolWallDamp, solverIters = poolSolverIters,
        };
        solver.Init();
        solver.ClearObstacles(); solver.UploadObstacles();
        int spawned = solver.SpawnBlock(new Vector3(radius, radius, radius), new Vector3Int(nx, ny, nz), rng);
        int initial = spawned;

        float dt = 1f / exportFps, tFrame = 0f;
        if (stir)
            solver.beforeSubstep = frac =>
            {
                float t = tFrame + frac * dt;
                solver.ClearObstacles();
                if (StirPath.Active(sp, t))
                    solver.SetObstacle(0, Matrix4x4.TRS(StirPath.Eval(sp, t), Quaternion.identity, Vector3.one * sp.diameter).inverse,
                                       (int)LiveSphProvider.Obstacle.Shape.Sphere, true);
                solver.UploadObstacles();
            };

        var records = new float[capacity * 7];                       // sized by CAPACITY: drops grow the count mid-sim
        var posStage = new Vector3[capacity]; var velStage = new Vector3[capacity]; var densStage = new float[capacity];
        int frames = Mathf.RoundToInt(poolStopAt * exportFps);
        var dump = new List<float[]>(frames);
        int nextDrop = 0;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int f = 0; f < frames; f++)
        {
            tFrame = f * dt;
            if (tiltDeg > 0f) solver.gravity = tFrame < tiltT ? gTilt : gRest;
            while (nextDrop < drops.Count && drops[nextDrop].t <= tFrame)
            {
                var d = drops[nextDrop];
                d.added = solver.SpawnBlock(d.start, d.count, rng);
                if (d.added != d.count.x * d.count.y * d.count.z)
                    Debug.LogError($"[SimExportRunner] {name}: drop {nextDrop} added {d.added} of {d.count.x * d.count.y * d.count.z} — capacity bug");
                drops[nextDrop] = d; spawned += d.added; nextDrop++;
            }
            solver.Step(dt);
            int n = solver.ReadbackFrame(records, posStage, velStage, densStage);
            var copy = new float[n * 7];
            Array.Copy(records, copy, n * 7);
            dump.Add(copy);
            if (f % 5 == 0) yield return null;   // keep the editor alive
        }
        sw.Stop();
        solver.Dispose();

        Write1Lps(Path.Combine(OutDir, name + ".bytes"), dump);
        var ci = CultureInfo.InvariantCulture;
        string V(float v) => v.ToString("0.#####", ci);
        string V3(Vector3 v) => $"[{V(v.x)}, {V(v.y)}, {V(v.z)}]";
        var sb = new StringBuilder();
        sb.Append("{\n");
        sb.Append($"  \"solver\": \"unity_gpu_pbf\",\n  \"family\": \"{fam}\",\n  \"seed\": {seed},\n");
        sb.Append($"  \"particle_radius\": {V(radius)},\n  \"stop_at\": {V(poolStopAt)},\n  \"export_fps\": {exportFps},\n");
        sb.Append($"  \"gravitation\": {V3(gRest)},\n  \"max_substeps\": {poolMaxSubsteps},\n  \"wall_damp\": {V(poolWallDamp)},\n  \"solver_iters\": {poolSolverIters},\n  \"lr_shuffle\": true,\n");
        sb.Append($"  \"pool_depth\": {V(depth)},\n  \"pool_fill\": {V(fill)},\n  \"pool_lattice\": [{nx}, {ny}, {nz}],\n");
        sb.Append($"  \"gravity_schedule\": [{{\"t0\": 0, \"t1\": {V(tiltDeg > 0f ? tiltT : 0f)}, \"g\": {V3(tiltDeg > 0f ? gTilt : gRest)}, \"tilt_deg\": {V(tiltDeg)}}}],\n");
        sb.Append("  \"drops\": [");
        for (int i = 0; i < drops.Count; i++)
            sb.Append((i > 0 ? ", " : "") + $"{{\"t\": {V(drops[i].t)}, \"cube\": {V(drops[i].cube)}, \"start\": {V3(drops[i].start)}, \"count\": {drops[i].added}}}");
        sb.Append("],\n");
        sb.Append(stir
            ? $"  \"stirrer\": {{\"diameter\": {V(sp.diameter)}, \"y_stir\": {V(sp.yStir)}, \"center\": [{V(sp.cx)}, {V(sp.cz)}], \"amp\": [{V(sp.ax)}, {V(sp.az)}], " +
              $"\"freq\": [{V(sp.fx)}, {V(sp.fz)}], \"phase\": [{V(sp.phx)}, {V(sp.phz)}], \"t_in\": {V(sp.tIn)}, \"t_out\": {V(sp.tOut)}, \"max_speed\": {V(sp.maxSpeed)}, \"substepped\": true}},\n"
            : "  \"stirrer\": null,\n");
        sb.Append($"  \"particle_count_initial\": {initial},\n  \"particle_count\": {spawned},\n  \"solve_sec\": {V((float)sw.Elapsed.TotalSeconds)},\n  \"obstacles\": []\n}}\n");
        File.WriteAllText(Path.Combine(OutDir, name + "_scene.json"), sb.ToString());
        Debug.Log($"[SimExportRunner] {name} ({fam}): {initial}->{spawned} particles x {frames} frames, depth {depth:0.00} r {radius:0.0000} " +
                  $"in {sw.Elapsed.TotalSeconds:0.0}s -> {OutDir}");
    }

    // ---------------------------------------------------------------- obstacles (runner.py parity)

    struct Obs
    {
        public bool sphere;
        public Vector3 center, scale;
        public float yawRad, half;
        public Matrix4x4 trs;
    }

    List<Obs> SampleObstacles(System.Random rng, Vector3 fluidStart, float cube)
    {
        Vector3 fluidC = fluidStart + Vector3.one * (cube / 2f);
        float fluidH = cube / 2f;
        var list = new List<Obs>();
        int want = rng.Next(obstaclesMin, obstaclesMax + 1);
        for (int k = 0; k < want; k++)
        {
            bool sphere = rng.NextDouble() < 0.5;
            float s = Lerp(rng, 0.25f, 0.6f);
            Vector3 scale = sphere ? Vector3.one * s
                : new Vector3(s * Lerp(rng, 0.8f, 1.25f), s * Lerp(rng, 0.8f, 1.25f), s * Lerp(rng, 0.8f, 1.25f));
            float half = sphere ? s : 0.71f * Mathf.Max(scale.x, Mathf.Max(scale.y, scale.z));
            float lo = obstaclePadding + half, hi = domainSize - obstaclePadding - half;
            if (lo >= hi) continue;

            bool placed = false;
            Vector3 c = Vector3.zero;
            for (int t = 0; t < 40 && !placed; t++)
            {
                c = new Vector3(Lerp(rng, lo, hi), Lerp(rng, half + 0.02f, 1.2f), Lerp(rng, lo, hi));
                if (Overlap(c, half, fluidC, fluidH)) continue;
                bool bad = false;
                foreach (var o in list) if (Overlap(c, half, o.center, o.half)) { bad = true; break; }
                placed = !bad;
            }
            if (!placed) continue;

            float yaw = sphere ? 0f : Lerp(rng, 0f, 3.14f);
            var rot = Quaternion.AngleAxis(yaw * Mathf.Rad2Deg, Vector3.up);
            list.Add(new Obs
            {
                sphere = sphere, center = c, scale = scale, yawRad = yaw, half = half,
                trs = Matrix4x4.TRS(c, rot, scale),
            });
        }
        return list;
    }

    static bool Overlap(Vector3 c1, float h1, Vector3 c2, float h2, float margin = 0.05f)
    {
        float d = h1 + h2 + margin;
        return Mathf.Abs(c1.x - c2.x) < d && Mathf.Abs(c1.y - c2.y) < d && Mathf.Abs(c1.z - c2.z) < d;
    }

    // ---------------------------------------------------------------- output

    public static void Write1Lps(string path, List<float[]> frames)
    {
        using var fh = new BinaryWriter(File.Create(path));
        fh.Write(0x53504C31); fh.Write(1); fh.Write(frames.Count);
        foreach (var f in frames) fh.Write(f.Length / 7);
        foreach (var f in frames)
        {
            var bytes = new byte[f.Length * 4];
            Buffer.BlockCopy(f, 0, bytes, 0, bytes.Length);
            fh.Write(bytes);
        }
    }

    void WriteSceneJson(string path, System.Random rng, float radius, Vector3 gravity,
                        float cube, Vector3 fluidStart, int particleCount, List<Obs> obstacles)
    {
        var ci = CultureInfo.InvariantCulture;
        string V(float v) => v.ToString("0.#####", ci);
        string V3(Vector3 v) => $"[{V(v.x)}, {V(v.y)}, {V(v.z)}]";
        var sb = new StringBuilder();
        sb.Append("{\n");
        sb.Append($"  \"solver\": \"unity_gpu_pbf\",\n");
        sb.Append($"  \"seed\": {seed},\n");
        sb.Append($"  \"particle_radius\": {V(radius)},\n");
        sb.Append($"  \"stop_at\": {V(stopAt)},\n");
        sb.Append($"  \"export_fps\": {exportFps},\n");
        sb.Append($"  \"gravitation\": {V3(gravity)},\n");
        sb.Append($"  \"cube_size\": {V(cube)},\n");
        sb.Append($"  \"fluid_start\": {V3(fluidStart)},\n");
        sb.Append($"  \"fluid_end\": {V3(fluidStart + Vector3.one * cube)},\n");
        sb.Append($"  \"particle_count\": {particleCount},\n");
        sb.Append("  \"obstacles\": [");
        for (int i = 0; i < obstacles.Count; i++)
        {
            var o = obstacles[i];
            sb.Append(i > 0 ? ",\n    " : "\n    ");
            sb.Append($"{{\"shape\": \"{(o.sphere ? "sphere" : "box")}\", " +
                      $"\"translation\": {V3(o.center)}, \"scale\": {V3(o.scale)}, " +
                      $"\"rotation_axis\": [0, 1, 0], \"rotation_angle\": {V(o.yawRad)}, " +
                      $"\"dynamic\": false}}");
        }
        sb.Append(obstacles.Count > 0 ? "\n  ]\n" : "]\n");
        sb.Append("}\n");
        File.WriteAllText(path, sb.ToString());
    }
}
