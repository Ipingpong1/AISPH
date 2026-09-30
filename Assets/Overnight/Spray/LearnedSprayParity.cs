// LearnedSprayParity.cs — G2 (overnight 2026-09-30) known-answer / parity entry points for LearnedSprayLayer, meant to be
// called from the MCP bridge (execute_code) in EDIT mode; nothing here runs in a normal Play session.
//   DumpFrames  : features + emitter output + KNN + n^ of chosen frames of a frozen live clip (SPL1 records.bytes)
//                 -> one binary per frame for SSU_restart/Experiments/Overnight0930/G2_spray/parity.py
//   BirthStats  : `trials` independent birth rounds on one frame (same features / MLP) -> per-trial counts + per-event log
//   Sequence    : the full per-solver-frame loop (flight, culls, features, MLP, births) over a clip -> per-frame counts
//   SelfTest    : ballistic flight, wall / obstacle / re-entry / age culls, lattice blob on synthetic input
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using Unity.InferenceEngine;

public static class LearnedSprayParity
{
    public const int Magic = 0x47325350;   // 'G2SP'

    public static List<float[]> ReadBake(string path, out int[] counts)
    {
        var raw = File.ReadAllBytes(path);
        int magic = BitConverter.ToInt32(raw, 0), version = BitConverter.ToInt32(raw, 4), n = BitConverter.ToInt32(raw, 8);
        if (magic != 0x53504C31 || version != 1) throw new Exception($"not an SPL1 v1 bake: {path}");
        counts = new int[n];
        for (int f = 0; f < n; f++) counts[f] = BitConverter.ToInt32(raw, 12 + 4 * f);
        var frames = new List<float[]>(n);
        int off = 12 + 4 * n;
        for (int f = 0; f < n; f++)
        {
            var a = new float[counts[f] * 7];
            Buffer.BlockCopy(raw, off, a, 0, a.Length * 4);
            off += a.Length * 4;
            frames.Add(a);
        }
        return frames;
    }

    static LearnedSprayLayer MakeLayer(string modelResource, string backend, ulong seed)
    {
        var asset = Resources.Load<ModelAsset>(modelResource);
        if (asset == null) throw new Exception($"model resource '{modelResource}' not found");
        var L = new LearnedSprayLayer(seed);
        L.SetModel(asset, (BackendType)Enum.Parse(typeof(BackendType), backend));
        return L;
    }

    /// <summary>Dump features / packed output / knn / n^ for each frame -> outDir/&lt;tag&gt;_f&lt;frame&gt;.bin. Returns JSON timings.</summary>
    public static string DumpFrames(string recordsPath, int[] frames, string outDir, string tag, float rc,
                                    string modelResource = "spray_emitter_L_s0", string backend = "GPUCompute")
    {
        var data = ReadBake(recordsPath, out var counts);
        Directory.CreateDirectory(outDir);
        var sb = new StringBuilder("{\"frames\":[");
        using (var L = MakeLayer(modelResource, backend, 0))
        {
            for (int q = 0; q < frames.Length; q++)
            {
                int f = frames[q];
                L.ComputeFeatures(data[f], 0, counts[f], rc);
                var sw = System.Diagnostics.Stopwatch.StartNew();
                L.RunMlp();
                float msMlp = (float)sw.Elapsed.TotalMilliseconds;
                int N = L.Count;
                string path = Path.Combine(outDir, $"{tag}_f{f:D4}.bin");
                using (var bw = new BinaryWriter(File.Create(path)))
                {
                    bw.Write(Magic); bw.Write(1); bw.Write(N); bw.Write(LearnedSprayLayer.NFeat); bw.Write(LearnedSprayLayer.NPack); bw.Write(LearnedSprayLayer.KNN);
                    var X = L.Features; for (int i = 0; i < N * LearnedSprayLayer.NFeat; i++) bw.Write(X[i]);
                    var Pk = L.Packed; for (int i = 0; i < N * LearnedSprayLayer.NPack; i++) bw.Write(Pk[i]);
                    var K = L.Knn; for (int i = 0; i < N * LearnedSprayLayer.KNN; i++) bw.Write(K[i]);
                    var nh = L.NHat; for (int i = 0; i < N * 3; i++) bw.Write((float)nh[i]);
                    bw.Write((float)L.NFull); bw.Write((float)L.MinHeight); bw.Write(L.RC); bw.Write(L.MsFeatures);
                }
                sb.Append(q > 0 ? "," : "").Append($"{{\"frame\":{f},\"n\":{N},\"ms_features\":{L.MsFeatures:F3},\"ms_mlp\":{msMlp:F3},\"n_full\":{L.NFull:F3},\"path\":\"{path}\"}}");
            }
        }
        return sb.Append("]}").ToString();
    }

    /// <summary>`trials` independent birth rounds on frame `frame` (features + MLP once). Writes outPath (int32 trials,
    /// then per trial: int32 events, int32 droplets; then int32 nEvents and float32 [nEvents, 10] event log).</summary>
    public static string BirthStats(string recordsPath, int frame, int trials, string outPath, float rc,
                                    float rateScale = 2.9f, float launch = 3f, ulong seed = 12345,
                                    string modelResource = "spray_emitter_L_s0", string backend = "GPUCompute")
    {
        var data = ReadBake(recordsPath, out var counts);
        using (var L = MakeLayer(modelResource, backend, seed))
        {
            L.rateScale = rateScale; L.launchSpeed = launch; L.maxDroplets = int.MaxValue / 2;
            L.ComputeFeatures(data[frame], 0, counts[frame], rc);
            L.RunMlp();
            L.eventLog = new List<float>(1 << 16);
            var ev = new int[trials]; var dr = new int[trials];
            double rateSum = 0;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int t = 0; t < trials; t++)
            {
                L.Reset();
                L.RunBirths();
                ev[t] = L.LastEvents; dr[t] = L.LastBorn; rateSum = L.LastRateSum;
            }
            float ms = (float)sw.Elapsed.TotalMilliseconds / Math.Max(trials, 1);
            Directory.CreateDirectory(Path.GetDirectoryName(outPath));
            using (var bw = new BinaryWriter(File.Create(outPath)))
            {
                bw.Write(trials);
                for (int t = 0; t < trials; t++) { bw.Write(ev[t]); bw.Write(dr[t]); }
                int ne = L.eventLog.Count / 10;
                bw.Write(ne);
                foreach (var v in L.eventLog) bw.Write(v);
            }
            double me = 0; foreach (var e in ev) me += e; me /= Math.Max(trials, 1);
            return $"{{\"frame\":{frame},\"trials\":{trials},\"rate_sum\":{rateSum:F6},\"mean_events\":{me:F4},\"ms_per_birth_round\":{ms:F3},\"path\":\"{outPath}\"}}";
        }
    }

    /// <summary>The live loop over clip frames [f0, f1): Step() per frame. Obstacles OFF (the clip's stirrer pose is not
    /// replayed), domain [0,3]^3, gravity -9.81 y. Writes outPath as text rows: frame alive born events wall obst merge age
    /// rate_sum ms_feat ms_mlp ms_birth ms_fly.</summary>
    public static string Sequence(string recordsPath, int f0, int f1, string outPath, float rc, ulong seed,
                                  float rateScale = 2.9f, float launch = 3f,
                                  string modelResource = "spray_emitter_L_s0", string backend = "GPUCompute",
                                  int[] dumpStates = null)
    {
        var data = ReadBake(recordsPath, out var counts);
        var sb = new StringBuilder("frame alive born events wall obst merge age rate_sum ms_feat ms_mlp ms_birth ms_fly\n");
        double tf = 0, tm = 0, tb = 0, td = 0; int steps = 0;
        using (var L = MakeLayer(modelResource, backend, seed))
        {
            L.rateScale = rateScale; L.launchSpeed = launch; L.cullObstacles = false;
            for (int f = f0; f < Math.Min(f1, data.Count); f++)
            {
                L.Step(data[f], 0, counts[f], rc);
                if (dumpStates != null && Array.IndexOf(dumpStates, f) >= 0) WriteDroplets(L, outPath + $".state_f{f:D4}.bin");
                // after Step: survivors of the flight/cull = Visible (what is drawn with this frame), + LastBorn newborns
                sb.Append($"{f} {L.Visible} {L.LastBorn} {L.LastEvents} {L.KillWall} {L.KillObst} {L.KillMerge} {L.KillAge} {L.LastRateSum:F5} " +
                          $"{L.MsFeatures:F3} {L.MsMlp:F3} {L.MsBirths:F3} {L.MsDroplets:F3}\n");
                if (f > f0 + 2) { tf += L.MsFeatures; tm += L.MsMlp; tb += L.MsBirths; td += L.MsDroplets; steps++; }
            }
        }
        Directory.CreateDirectory(Path.GetDirectoryName(outPath));
        File.WriteAllText(outPath, sb.ToString());
        steps = Math.Max(steps, 1);
        string ms = $"\"features\":{tf / steps:F3},\"mlp\":{tm / steps:F3},\"births\":{tb / steps:F3},\"droplets\":{td / steps:F3}";
        return "{\"frames\":[" + f0 + "," + f1 + "],\"mean_ms\":{" + ms + "},\"path\":\"" + outPath + "\"}";
    }

    // droplet state after a Step: int32 n, then per droplet px py pz vx vy vz (float32) age (int32)
    static void WriteDroplets(LearnedSprayLayer L, string path)
    {
        using (var bw = new BinaryWriter(File.Create(path)))
        {
            int n = L.Alive; bw.Write(n);
            var p = L.DropletPositions; var v = L.DropletVelocities; var a = L.DropletAges;
            for (int i = 0; i < n; i++) { bw.Write(p[i].x); bw.Write(p[i].y); bw.Write(p[i].z); bw.Write(v[i].x); bw.Write(v[i].y); bw.Write(v[i].z); bw.Write(a[i]); }
        }
    }

    /// <summary>Synthetic known answers (no model needed except for none): returns JSON with each check and pass.</summary>
    public static string SelfTest()
    {
        var res = new StringBuilder("{");
        bool all = true;
        void Check(string name, bool ok, string detail) { all &= ok; res.Append($"\"{name}\":{{\"pass\":{(ok ? "true" : "false")},\"detail\":\"{detail}\"}},"); }

        // coarse state: a 6x6x6 lattice block at spacing 1.9 r_c (no pair at exactly h) resting on the floor, r_c 0.0414
        float rc = 0.0414f, sp = 1.9f * rc; int m = 6, N = m * m * m;
        var rec = new float[N * 7];
        for (int a = 0, i = 0; a < m; a++) for (int b = 0; b < m; b++) for (int c = 0; c < m; c++, i++)
        { rec[7 * i] = 1.0f + sp * a; rec[7 * i + 1] = rc + sp * b; rec[7 * i + 2] = 1.0f + sp * c; rec[7 * i + 6] = 1000f; }

        // 1. ballistic flight: one frame of 4 semi-implicit substeps, no culls in reach
        var L = new LearnedSprayLayer(1);
        L.cullMerge = false; L.cullObstacles = false;
        Vector3 p0 = new Vector3(2.5f, 2.0f, 2.5f), v0 = new Vector3(0.3f, 1.2f, -0.4f);
        L.AddDroplets(new[] { p0 }, new[] { v0 }, 1);
        L.ComputeFeatures(rec, 0, N, rc);                       // loads coarse + grid (no model needed)
        StepFlightOnly(L, rec, N, rc);
        double h = (1.0 / 25.0) / 4.0, px = p0.x, py = p0.y, pz = p0.z, vx = v0.x, vy = v0.y, vz = v0.z;
        for (int s = 0; s < 4; s++) { vy += -9.81 * h; px += vx * h; py += vy * h; pz += vz * h; }
        Vector3 got = L.DropletPositions[0];
        double err = Math.Sqrt((got.x - px) * (got.x - px) + (got.y - py) * (got.y - py) + (got.z - pz) * (got.z - pz));
        Check("ballistic_one_frame", L.Alive == 1 && err < 1e-5, $"err {err:E2} alive {L.Alive}");

        // 2. wall cull: droplet leaving the domain through x = 3 within the frame
        L = new LearnedSprayLayer(1); L.cullMerge = false; L.cullObstacles = false;
        L.AddDroplets(new[] { new Vector3(2.99f, 1.5f, 1.5f) }, new[] { new Vector3(1f, 0f, 0f) }, 1);
        L.ComputeFeatures(rec, 0, N, rc); StepFlightOnly(L, rec, N, rc);
        Check("wall_cull", L.Alive == 0 && L.KillWall == 1, $"alive {L.Alive} wall {L.KillWall}");

        // 3. obstacle cull: sphere of radius 0.5 (unit primitive) scaled to 0.4 at (2, 1, 2) in sim space
        L = new LearnedSprayLayer(1); L.cullMerge = false;
        Matrix4x4 toLocal = (Matrix4x4.Translate(new Vector3(2f, 1f, 2f)) * Matrix4x4.Scale(Vector3.one * 0.4f)).inverse;
        L.obstacles.Add(new LearnedSprayLayer.Obstacle { toLocal = toLocal, shape = 0, padLocal = 0.0142f / 0.4f });
        L.AddDroplets(new[] { new Vector3(2.1f, 1.0f, 2.0f), new Vector3(2.5f, 1.0f, 2.0f) }, new[] { Vector3.zero, Vector3.zero }, 2);
        L.ComputeFeatures(rec, 0, N, rc); StepFlightOnly(L, rec, N, rc);
        Check("obstacle_cull", L.Alive == 1 && L.KillObst == 1, $"alive {L.Alive} obst {L.KillObst}");

        // 4. re-entry: a droplet sitting 0.5 r_c above a coarse particle merges only once age >= 2
        L = new LearnedSprayLayer(1); L.cullObstacles = false; L.gravity = Vector3.zero;
        Vector3 top = new Vector3(rec[7 * (N - 1)], rec[7 * (N - 1) + 1] + 0.5f * rc, rec[7 * (N - 1) + 2]);
        L.AddDroplets(new[] { top }, new[] { Vector3.zero }, 1);
        L.ComputeFeatures(rec, 0, N, rc);
        StepFlightOnly(L, rec, N, rc); int a1 = L.Alive;       // age 1: grace
        StepFlightOnly(L, rec, N, rc); int a2 = L.Alive;       // age 2: merges
        Check("merge_after_grace", a1 == 1 && a2 == 0 && L.KillMerge == 1, $"alive after 1: {a1}, after 2: {a2}");

        // 5. max age: zero gravity, far from everything: alive through age 50 (2.0 s), culled at 51
        L = new LearnedSprayLayer(1); L.cullObstacles = false; L.gravity = Vector3.zero;
        L.AddDroplets(new[] { new Vector3(2.5f, 2.5f, 2.5f) }, new[] { Vector3.zero }, 1);
        L.ComputeFeatures(rec, 0, N, rc);
        int aliveAt50 = -1;
        for (int f = 1; f <= 51; f++) { StepFlightOnly(L, rec, N, rc); if (f == 50) aliveAt50 = L.Alive; }
        Check("max_age", aliveAt50 == 1 && L.Alive == 0 && L.KillAge == 1, $"alive@50 {aliveAt50} alive@51 {L.Alive}");

        // 6. lattice interior count = 32 at h = 4 r_c, spacing 1.9 r_c (|o|^2 <= 4 lattice shells)
        L = new LearnedSprayLayer(1);
        L.ComputeFeatures(rec, 0, N, rc);
        int centre = (2 * m + 2) * m + 2;   // (2,2,2)
        float cntFeat = L.Features[centre * LearnedSprayLayer.NFeat + 6] * (float)L.NFull;
        Check("lattice_count", Mathf.Abs(cntFeat - 32f) < 1e-3f, $"cnt {cntFeat} n_full {L.NFull}");

        res.Append($"\"all_pass\":{(all ? "true" : "false")}}}");
        return res.ToString();
    }

    // flight + culls only (the Step's first half) with the coarse frame loaded
    static void StepFlightOnly(LearnedSprayLayer L, float[] rec, int n, float rc) => L.AdvanceAndCull();
}
