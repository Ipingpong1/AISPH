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
        for (int s = 0; s < simCount; s++)
        {
            var rng = new System.Random(seed + s);
            yield return RunOne($"sim_{s:0000}", rng);
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

    static void Write1Lps(string path, List<float[]> frames)
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
