// ReseedRunner.cs — 2026-09-30 overnight (PBF FT-RS): relaxed re-seed rollouts in Unity's OWN GPU PBF, driven by
// SSU_restart/Helpers/pbf_rs.py over the MCP bridge (execute_code -> ReseedRunner.Describe / Relax / Rollout).
// The DFSPH recipe (SSU_restart/Helpers/tc_reseed.py) ported to the deployment solver:
//   Relax(jobDir):   per seed, upload the k-means coarse state (positions, zero velocity) and pack it: gravity 0,
//                    velocities zeroed before EVERY substep, nRelax raw substeps of dtRelax, the stirrer frozen at its pose
//                    at the seed time -> relaxed positions + a per-substep log (max displacement, density p50/p99/max).
//   Rollout(jobDir): per seed, upload (positions, velocities) written by Python and advance `horizon` solver frames with
//                    the pool's own forcing, CFL-substepped exactly like SimExportRunner.RunPool (gravity schedule,
//                    per-substep stirrer, drops spawned as coarse lattice blocks at their times) -> the 7-float records
//                    after every frame. The same entry point runs the known-answer gate (Python feeds TC_POOL's own frame
//                    state and compares the output with TC_POOL's next frames).
// The pool's forcing is REPLAYED from RunPool's random stream (System.Random(batchSeed + sim), same draw order), because
// the scene json rounds to 5 decimals and stores the lattice-rounded depth rather than the depth StirPath.Sample saw.
// Describe(jobDir) returns the replay so pbf_rs.py can check it against the TC_POOL / dense scene json before any run.
// Time convention (RunPool): dump frame f (0-based) is the state after the step with tFrame = f * dt, i.e. dataset gt index
// F = f + 1 is time F * dt. A seed at gt index F0 is time F0 * dt; its rollout runs the steps tFrame = F0 * dt,
// (F0 + 1) * dt, ... and spawns every drop with (F0 - 1) * dt < t <= tFrame before that step.
// Files (little-endian float32, no header): init_FFFF.bin (K,3) | relaxed_FFFF.bin (K,3) | state_FFFF.bin (K,6) |
// roll_FFFF_hH.bin (n,7) records after H frames; logs relax_FFFF.json / rollout.log.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

public static class ReseedRunner
{
    [Serializable] public class Drop { public float t; public float cube; public float[] start; }

    [Serializable] public class Job
    {
        public string name;
        public int batchSeed, sim;                 // RunPool's stream: System.Random(batchSeed + sim)
        public string family;                      // SimExportRunner.Family name
        public float radiusMin, radiusMax;         // the batch's radius range (the coarse class)
        public float poolStopAt = 5f, domain = 3f, fps = 25f, dtRelax = 0.005f, wallDamp = 0f;
        public int iters = 10, maxSubsteps = 32, nRelax = 30;
        public int relaxGravity = 0;               // 0: gravity off while relaxing (the DFSPH recipe); 1: the pool's rest gravity
        public int settleSubsteps = 0;             // phase B (after the relax): quasi-static settle under the rest gravity,
        public float settleDt = 0.01f;             // velocities zeroed before every substep; phase-A positions -> relaxedA_FFFF.bin
        public int[] seeds = new int[0];           // gt indices F0
        public int[] horizons = new int[0];        // frames to roll out per seed (Rollout only)
    }

    public class Replay
    {
        public float radius, depthPre, gMag, tiltDeg, tiltT, tiltAz;
        public Vector3 gRest, gTilt;
        public bool stir;
        public StirPath.Params sp;
        public List<Drop> drops = new List<Drop>();
    }

    static Job Load(string dir) => JsonUtility.FromJson<Job>(File.ReadAllText(Path.Combine(dir, "job.json")));
    static Vector3 V(float[] a) => new Vector3(a[0], a[1], a[2]);
    static string F(float v) => v.ToString("R", CultureInfo.InvariantCulture);
    static float Lerp(System.Random rng, float a, float b) => a + (float)rng.NextDouble() * (b - a);   // == RunPool

    /// <summary>Exact replay of SimExportRunner.RunPool's draws (same order) for sim `j.sim` of the batch.</summary>
    public static Replay ReplayPool(Job j)
    {
        var fam = (SimExportRunner.Family)Enum.Parse(typeof(SimExportRunner.Family), j.family);
        var rng = new System.Random(j.batchSeed + j.sim);
        var r = new Replay();
        r.radius = Lerp(rng, j.radiusMin, j.radiusMax);
        float depth = rng.NextDouble() < 0.7 ? Lerp(rng, 0.10f, 0.30f) : Lerp(rng, 0.30f, 0.40f);
        r.depthPre = depth;
        float fill = fam == SimExportRunner.Family.PoolSlosh ? Lerp(rng, 0.55f, 1f) : 1f;
        r.gMag = Lerp(rng, 9.3f, 10.3f);
        r.gRest = new Vector3(0f, -r.gMag, 0f);
        r.tiltDeg = fam == SimExportRunner.Family.PoolSlosh ? Lerp(rng, 8f, 20f) : 0f;
        r.tiltT = Lerp(rng, 0.3f, 0.8f);
        r.tiltAz = Lerp(rng, 0f, 6.2832f);
        r.gTilt = Quaternion.AngleAxis(r.tiltDeg, new Vector3(Mathf.Cos(r.tiltAz), 0f, Mathf.Sin(r.tiltAz))) * r.gRest;
        r.stir = fam == SimExportRunner.Family.PoolStir || fam == SimExportRunner.Family.PoolStirDrop;
        bool dropFam = fam == SimExportRunner.Family.PoolDrop || fam == SimExportRunner.Family.PoolStirDrop;
        r.sp = r.stir ? StirPath.Sample(rng, depth, j.poolStopAt, j.domain) : default;
        if (dropFam)
        {
            int want = rng.Next(1, 4); float tPrev = -10f;
            for (int k = 0; k < want; k++)
            {
                float cube = Lerp(rng, 0.25f, 0.55f);
                float t = Lerp(rng, 0.4f, j.poolStopAt - 2.0f);
                float y0 = Lerp(rng, depth + 0.35f, Mathf.Max(depth + 0.36f, 1.4f - cube));
                var start = new Vector3(Lerp(rng, 0.1f, j.domain - 0.1f - cube), y0, Lerp(rng, 0.1f, j.domain - 0.1f - cube));
                if (Mathf.Abs(t - tPrev) < 0.8f) continue;
                if (r.stir)
                {
                    Vector3 c = StirPath.Eval(r.sp, t), bc = start + Vector3.one * (cube * 0.5f);
                    if ((c - bc).magnitude < r.sp.diameter * 0.5f + cube * 0.87f + 0.1f) continue;
                }
                r.drops.Add(new Drop { t = t, cube = cube, start = new[] { start.x, start.y, start.z } });
                tPrev = t;
            }
            r.drops.Sort((a, b) => a.t.CompareTo(b.t));
        }
        return r;
    }

    static int DropSide(Drop d, float radius) => Mathf.Max(1, Mathf.FloorToInt(d.cube / (2f * radius)));

    /// <summary>The replayed draws as json (pbf_rs.py checks them against the scene json before any run).</summary>
    public static string Describe(string dir)
    {
        var j = Load(dir); var r = ReplayPool(j);
        var sb = new StringBuilder("{");
        sb.Append($"\"radius\": {F(r.radius)}, \"depth_pre\": {F(r.depthPre)}, \"g_rest\": [{F(r.gRest.x)}, {F(r.gRest.y)}, {F(r.gRest.z)}], ");
        sb.Append($"\"g_tilt\": [{F(r.gTilt.x)}, {F(r.gTilt.y)}, {F(r.gTilt.z)}], \"tilt_deg\": {F(r.tiltDeg)}, \"tilt_t\": {F(r.tiltT)}, ");
        if (r.stir)
            sb.Append($"\"stirrer\": {{\"diameter\": {F(r.sp.diameter)}, \"y_stir\": {F(r.sp.yStir)}, \"y_above\": {F(r.sp.yAbove)}, " +
                      $"\"center\": [{F(r.sp.cx)}, {F(r.sp.cz)}], \"amp\": [{F(r.sp.ax)}, {F(r.sp.az)}], \"freq\": [{F(r.sp.fx)}, {F(r.sp.fz)}], " +
                      $"\"phase\": [{F(r.sp.phx)}, {F(r.sp.phz)}], \"t_in\": {F(r.sp.tIn)}, \"t_out\": {F(r.sp.tOut)}}}, ");
        else sb.Append("\"stirrer\": null, ");
        sb.Append("\"drops\": [");
        for (int i = 0; i < r.drops.Count; i++)
        {
            var d = r.drops[i]; int n1 = DropSide(d, r.radius);
            sb.Append((i > 0 ? ", " : "") + $"{{\"t\": {F(d.t)}, \"cube\": {F(d.cube)}, \"start\": [{F(d.start[0])}, {F(d.start[1])}, {F(d.start[2])}], \"count\": {n1 * n1 * n1}}}");
        }
        sb.Append("]}");
        return sb.ToString();
    }

    static ComputeShader Shader()
    {
#if UNITY_EDITOR
        return UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/GpuSph/GpuSph.compute");
#else
        return null;
#endif
    }

    static GpuSphSolver NewSolver(Job j, Replay r, int capacity)
    {
        var s = new GpuSphSolver(Shader())
        {
            particleRadius = r.radius, maxParticles = Math.Max(capacity, 16), gravity = r.gRest,
            domainMin = Vector3.zero, domainMax = Vector3.one * j.domain, maxSubsteps = j.maxSubsteps, wallDamp = j.wallDamp,
            solverIters = j.iters,
        };
        s.Init();
        s.ClearObstacles(); s.UploadObstacles();
        return s;
    }

    static void SetStir(GpuSphSolver s, Replay r, float t)   // == RunPool's beforeSubstep body
    {
        s.ClearObstacles();
        if (r.stir && StirPath.Active(r.sp, t))
            s.SetObstacle(0, Matrix4x4.TRS(StirPath.Eval(r.sp, t), Quaternion.identity, Vector3.one * r.sp.diameter).inverse,
                          (int)LiveSphProvider.Obstacle.Shape.Sphere, true);
        s.UploadObstacles();
    }

    static float[] ReadF(string path)
    {
        var b = File.ReadAllBytes(path);
        var f = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, f, 0, b.Length);
        return f;
    }

    static void WriteF(string path, float[] f, int count)
    {
        var b = new byte[count * 4];
        Buffer.BlockCopy(f, 0, b, 0, b.Length);
        string tmp = path + ".tmp";
        File.WriteAllBytes(tmp, b);
        if (File.Exists(path)) File.Delete(path);
        File.Move(tmp, path);
    }

    /// <summary>Relax every seed of the job (skips seeds whose relaxed file exists). -> one-line summary.</summary>
    public static string Relax(string dir, int maxSeeds = 100000)
    {
        var j = Load(dir);
        var r = ReplayPool(j);
        int done = 0, skipped = 0;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        foreach (int f0 in j.seeds)
        {
            if (done >= maxSeeds) break;
            string outPath = Path.Combine(dir, $"relaxed_{f0:0000}.bin");
            if (File.Exists(outPath)) { skipped++; continue; }
            var x = ReadF(Path.Combine(dir, $"init_{f0:0000}.bin"));
            int K = x.Length / 3;
            var pos = new Vector3[K]; var vel = new Vector3[K]; var prev = new Vector3[K]; var dens = new float[K];
            for (int i = 0; i < K; i++) pos[i] = new Vector3(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
            var s = NewSolver(j, r, K);
            try
            {
                s.gravity = j.relaxGravity != 0 ? r.gRest : Vector3.zero;
                SetStir(s, r, f0 * (1f / j.fps));
                s.SetState(pos, vel, K);
                var log = new StringBuilder("{\"step_disp_max\": [");
                var rho = new StringBuilder("\"rho_p50_p99_max\": [");
                Array.Copy(pos, prev, K);
                var sorted = new float[K];
                for (int k = 0; k < j.nRelax; k++)
                {
                    s.ZeroVelocities();
                    s.SubstepOnce(j.dtRelax);
                    s.PositionBuffer.GetData(pos, 0, 0, K);
                    s.DensityBuffer.GetData(dens, 0, 0, K);
                    float dmax = 0f;
                    for (int i = 0; i < K; i++) dmax = Mathf.Max(dmax, (pos[i] - prev[i]).magnitude);
                    Array.Copy(pos, prev, K);
                    Array.Copy(dens, sorted, K); Array.Sort(sorted);
                    log.Append((k > 0 ? ", " : "") + F(dmax));
                    rho.Append((k > 0 ? ", " : "") + $"[{F(sorted[K / 2])}, {F(sorted[(int)(0.99 * (K - 1))])}, {F(sorted[K - 1])}]");
                }
                if (j.settleSubsteps > 0)
                {
                    var oa = new float[K * 3];
                    for (int i = 0; i < K; i++) { oa[3 * i] = pos[i].x; oa[3 * i + 1] = pos[i].y; oa[3 * i + 2] = pos[i].z; }
                    WriteF(Path.Combine(dir, $"relaxedA_{f0:0000}.bin"), oa, oa.Length);
                    s.gravity = r.gRest;
                    rho.Append("], \"settle_rho_p50_p99_max\": [");
                    for (int k = 0; k < j.settleSubsteps; k++)
                    {
                        s.ZeroVelocities();
                        s.SubstepOnce(j.settleDt);
                        if (k == j.settleSubsteps - 1 || k % 10 == 0)
                        {
                            s.DensityBuffer.GetData(dens, 0, 0, K);
                            Array.Copy(dens, sorted, K); Array.Sort(sorted);
                            rho.Append((k > 0 ? ", " : "") + $"[{F(sorted[K / 2])}, {F(sorted[(int)(0.99 * (K - 1))])}, {F(sorted[K - 1])}]");
                        }
                    }
                    s.PositionBuffer.GetData(pos, 0, 0, K);
                }
                log.Append("], ").Append(rho).Append($"], \"K\": {K}, \"n_relax\": {j.nRelax}, \"dt_relax\": {F(j.dtRelax)}, \"relax_gravity\": {j.relaxGravity}, \"settle_substeps\": {j.settleSubsteps}, \"settle_dt\": {F(j.settleDt)}, \"radius\": {F(r.radius)}}}");
                File.WriteAllText(Path.Combine(dir, $"relax_{f0:0000}.json"), log.ToString());
                var o = new float[K * 3];
                for (int i = 0; i < K; i++) { o[3 * i] = pos[i].x; o[3 * i + 1] = pos[i].y; o[3 * i + 2] = pos[i].z; }
                WriteF(outPath, o, o.Length);
            }
            finally { s.Dispose(); }
            done++;
        }
        return $"relax {j.name}: {done} done, {skipped} skipped, {sw.Elapsed.TotalSeconds:0.0} s";
    }

    /// <summary>Roll out every seed of the job from state_FFFF.bin (skips seeds whose last output exists).</summary>
    public static string Rollout(string dir, int maxSeeds = 100000)
    {
        var j = Load(dir);
        var r = ReplayPool(j);
        int done = 0, skipped = 0;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        float dt = 1f / j.fps;                                                   // == RunPool: dt = 1f / exportFps
        var summary = new StringBuilder();
        for (int si = 0; si < j.seeds.Length; si++)
        {
            if (done >= maxSeeds) break;
            int f0 = j.seeds[si], H = j.horizons[si];
            if (H <= 0 || File.Exists(Path.Combine(dir, $"roll_{f0:0000}_h{H}.bin"))) { skipped++; continue; }
            var x = ReadF(Path.Combine(dir, $"state_{f0:0000}.bin"));
            int K = x.Length / 6;
            int capacity = K;
            var pending = new List<Drop>();
            foreach (var d in r.drops)
                if (d.t > (f0 - 1) * dt && d.t <= (f0 + H - 1) * dt)
                {
                    pending.Add(d);
                    int n1 = DropSide(d, r.radius);
                    capacity += n1 * n1 * n1;
                }
            var pos = new Vector3[K]; var vel = new Vector3[K];
            for (int i = 0; i < K; i++)
            {
                pos[i] = new Vector3(x[6 * i], x[6 * i + 1], x[6 * i + 2]);
                vel[i] = new Vector3(x[6 * i + 3], x[6 * i + 4], x[6 * i + 5]);
            }
            var s = NewSolver(j, r, capacity);
            try
            {
                s.SetState(pos, vel, K);
                float tFrame = 0f;
                if (r.stir) s.beforeSubstep = frac => SetStir(s, r, tFrame + frac * dt);
                var records = new float[capacity * 7];
                var ps = new Vector3[capacity]; var vs = new Vector3[capacity]; var ds = new float[capacity];
                int next = 0, added = 0;
                for (int h = 1; h <= H; h++)
                {
                    int f = f0 + h - 1;
                    tFrame = f * dt;                                               // == RunPool
                    if (r.tiltDeg > 0f) s.gravity = tFrame < r.tiltT ? r.gTilt : r.gRest;
                    while (next < pending.Count && pending[next].t <= tFrame)
                    {
                        var d = pending[next++];
                        int n1 = DropSide(d, r.radius);
                        added += s.SpawnBlock(V(d.start), new Vector3Int(n1, n1, n1));
                    }
                    s.Step(dt);
                    int n = s.ReadbackFrame(records, ps, vs, ds);
                    WriteF(Path.Combine(dir, $"roll_{f0:0000}_h{h}.bin"), records, n * 7);
                }
                summary.Append($"{f0}:{K}+{added} ");
            }
            finally { s.Dispose(); }
            done++;
        }
        File.AppendAllText(Path.Combine(dir, "rollout.log"), $"[{DateTime.Now:s}] {done} done, {skipped} skipped: {summary}\n");
        return $"rollout {j.name}: {done} done, {skipped} skipped, {sw.Elapsed.TotalSeconds:0.0} s";
    }
}
