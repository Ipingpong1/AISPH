// DfsphParity.cs — DFSPH1001 gates (SSU_restart/Experiments/DFSPH1001/PLAN.md), headless, callable from the Unity MCP
// (execute_code) in edit or play mode. Every function writes raw little-endian float32 / uint32 files + a JSON summary that
// Experiments/DFSPH1001/gates.py compares against the SPlisHSPlasH references.
//   K0  ProbeKernels(dir, radius, count, seed)        kernels at random r (W_avx, W_pre, gradW_avx, gradW_pre)
//   M0  DumpVolumeMaps(scene, dir)                     all nodes (field 0, field 1) of every body of the scene
//   M1  ProbeBoundary(scene, inFile, outFile)          interpolation (dist, vol, chk, grad) at points (x, body)
//   S1/S2 RunSteps(scene, steps, dumpEvery, dir)       per-step stats + particle dumps (pos, vel, density) in id order
//   S3  RunExport(scene, stopAt, dir, exact)          the reference's 25 fps export rule, one file per frame
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

public static class DfsphParity
{
    static ComputeShader Shader() => Resources.Load<ComputeShader>("Dfsph");

    static void WriteF(string path, float[] a)
    {
        var b = new byte[a.Length * 4];
        Buffer.BlockCopy(a, 0, b, 0, b.Length);
        File.WriteAllBytes(path, b);
    }
    static void WriteU(string path, uint[] a)
    {
        var b = new byte[a.Length * 4];
        Buffer.BlockCopy(a, 0, b, 0, b.Length);
        File.WriteAllBytes(path, b);
    }

    static DfsphSolver MakeSolver(DfsphScene scene, bool exact, int maxParticles = 0)
    {
        var s = new DfsphSolver(Shader());
        scene.Configure(s);
        var pos = new List<Vector3>(); var vel = new List<Vector3>();
        scene.SampleFluid(pos, vel);
        s.maxParticles = maxParticles > 0 ? maxParticles : Math.Max(pos.Count, 1);
        if (exact) { s.kCap = s.maxIterations; s.kCapV = s.maxIterationsV; }
        s.Init();
        s.SetBodies(scene.bodies);
        s.FitGridToBodies(s.SupportRadius);
        s.AddParticles(pos, vel);
        return s;
    }

    public static string ProbeKernels(string dir, float radius, int count, int seed)
    {
        Directory.CreateDirectory(dir);
        var scene = DfsphProvider.DefaultScene();
        scene.particleRadius = radius;
        scene.fluidBlocks.Clear();
        scene.bodies.Clear();
        var s = new DfsphSolver(Shader()) { maxParticles = 1 };
        scene.Configure(s);
        s.Init();
        float h = s.SupportRadius;
        var rng = new System.Random(seed);
        var input = new Vector4[count];
        for (int i = 0; i < count; i++)
        {
            float rl = (float)(rng.NextDouble() * 1.05 * h);
            if (i < 16) rl = i * h / 15f;                    // include 0, h/2, h exactly
            var d = new Vector3((float)(rng.NextDouble() * 2 - 1), (float)(rng.NextDouble() * 2 - 1), (float)(rng.NextDouble() * 2 - 1)).normalized;
            input[i] = new Vector4(d.x * rl, d.y * rl, d.z * rl, 0f);
        }
        var o = s.Probe(false, input);
        var fin = new float[count * 4]; var fout = new float[count * 8];
        for (int i = 0; i < count; i++) { fin[4 * i] = input[i].x; fin[4 * i + 1] = input[i].y; fin[4 * i + 2] = input[i].z; }
        for (int i = 0; i < 2 * count; i++) { fout[4 * i] = o[i].x; fout[4 * i + 1] = o[i].y; fout[4 * i + 2] = o[i].z; fout[4 * i + 3] = o[i].w; }
        WriteF(Path.Combine(dir, "k0_in.f32"), fin);
        WriteF(Path.Combine(dir, "k0_out.f32"), fout);
        File.WriteAllText(Path.Combine(dir, "k0_meta.json"), $"{{\"radius\": {radius:R}, \"h\": {h:R}, \"count\": {count}}}");
        s.Dispose();
        return $"K0 probes {count} -> {dir}";
    }

    public static string DumpVolumeMaps(string scenePath, string dir, bool tessellated = false)
    {
        Directory.CreateDirectory(dir);
        var scene = DfsphScene.FromFile(scenePath);
        scene.SetTessellated(tessellated);
        scene.fluidBlocks.Clear();
        var s = new DfsphSolver(Shader()) { maxParticles = 1 };
        scene.Configure(s);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        s.Init();
        s.SetBodies(scene.bodies);
        double ms = sw.Elapsed.TotalMilliseconds;
        var nodes = s.ReadNodes();
        var sb = new StringBuilder("{\"bodies\": [");
        for (int bi = 0; bi < s.Bodies.Count; bi++)
        {
            var b = s.Bodies[bi];
            var f = new float[b.nodeCount * 2];
            for (int k = 0; k < b.nodeCount; k++) { f[2 * k] = nodes[b.nodeOffset + k].x; f[2 * k + 1] = nodes[b.nodeOffset + k].y; }
            WriteF(Path.Combine(dir, $"m0_body{bi}.f32"), f);
            sb.Append($"{(bi > 0 ? "," : "")}{{\"shape\": \"{b.shape}\", \"scale\": [{b.scale.x:R}, {b.scale.y:R}, {b.scale.z:R}], " +
                      $"\"res\": [{b.mapResolution.x}, {b.mapResolution.y}, {b.mapResolution.z}], \"invert\": {(b.mapInvert ? "true" : "false")}, " +
                      $"\"dmin\": [{b.dmin[0]:R}, {b.dmin[1]:R}, {b.dmin[2]:R}], \"dmax\": [{b.dmax[0]:R}, {b.dmax[1]:R}, {b.dmax[2]:R}], \"nodes\": {b.nodeCount}}}");
        }
        sb.Append($"], \"radius\": {scene.particleRadius:R}, \"h\": {s.SupportRadius:R}, \"build_ms\": {ms:F1} }}");
        File.WriteAllText(Path.Combine(dir, "m0_meta.json"), sb.ToString());
        s.Dispose();
        return $"M0 {s.Bodies.Count} bodies, {nodes.Length} nodes, build {ms:F0} ms -> {dir}";
    }

    /// <summary>inFile: float32 (x, y, z, body) per point (sim space). outFile: float32 (dist, vol, chk, 0, gx, gy, gz, 0).</summary>
    public static string ProbeBoundary(string scenePath, string inFile, string outFile, bool tessellated = false)
    {
        var scene = DfsphScene.FromFile(scenePath);
        scene.SetTessellated(tessellated);
        scene.fluidBlocks.Clear();
        var s = new DfsphSolver(Shader()) { maxParticles = 1 };
        scene.Configure(s);
        s.Init();
        s.SetBodies(scene.bodies);
        var raw = File.ReadAllBytes(inFile);
        var f = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, f, 0, raw.Length);
        int n = f.Length / 4;
        var input = new Vector4[n];
        for (int i = 0; i < n; i++) input[i] = new Vector4(f[4 * i], f[4 * i + 1], f[4 * i + 2], f[4 * i + 3]);
        var o = s.Probe(true, input);
        var outF = new float[n * 8];
        for (int i = 0; i < 2 * n; i++) { outF[4 * i] = o[i].x; outF[4 * i + 1] = o[i].y; outF[4 * i + 2] = o[i].z; outF[4 * i + 3] = o[i].w; }
        WriteF(outFile, outF);
        s.Dispose();
        return $"M1 {n} points -> {outFile}";
    }

    static void DumpParticles(DfsphSolver s, string prefix, Vector3[] pos, Vector3[] vel, float[] dens)
    {
        s.ReadParticles(pos, vel, dens);
        int n = s.Count;
        var a = new float[n * 7];
        for (int i = 0; i < n; i++)
        {
            a[7 * i] = pos[i].x; a[7 * i + 1] = pos[i].y; a[7 * i + 2] = pos[i].z;
            a[7 * i + 3] = vel[i].x; a[7 * i + 4] = vel[i].y; a[7 * i + 5] = vel[i].z; a[7 * i + 6] = dens[i];
        }
        WriteF(prefix + ".f32", a);
    }

    /// <summary>S1/S2: `steps` reference steps; after each step the stats record (16 words) is appended to steps.u32 and,
    /// every `dumpEvery` steps (and for steps 1-5), the particles (pos, vel, density; id order) to step_XXXXX.f32.</summary>
    public static string RunSteps(string scenePath, int steps, int dumpEvery, string dir, bool exact = true, bool tessellated = false)
    {
        Directory.CreateDirectory(dir);
        var scene = DfsphScene.FromFile(scenePath);
        scene.SetTessellated(tessellated);
        var s = MakeSolver(scene, exact);
        int n = s.Count;
        var pos = new Vector3[n]; var vel = new Vector3[n]; var dens = new float[n];
        DumpParticles(s, Path.Combine(dir, "step_00000"), pos, vel, dens);
        var ring = new uint[DfsphSolver.Ring * DfsphSolver.StatWords];
        var stats = new List<uint>();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int k = 1; k <= steps; k++)
        {
            s.ExecuteSteps(1);
            s.StatsBuffer.GetData(ring);
            int slot = (k - 1) % DfsphSolver.Ring;
            for (int w = 0; w < DfsphSolver.StatWords; w++) stats.Add(ring[slot * DfsphSolver.StatWords + w]);
            if (k <= 5 || (dumpEvery > 0 && k % dumpEvery == 0)) DumpParticles(s, Path.Combine(dir, $"step_{k:D5}"), pos, vel, dens);
        }
        WriteU(Path.Combine(dir, "steps.u32"), stats.ToArray());
        var last = DfsphSolver.DecodeStats(ring, (steps - 1) % DfsphSolver.Ring);
        File.WriteAllText(Path.Combine(dir, "run_meta.json"),
            $"{{\"scene\": \"{scenePath}\", \"n\": {n}, \"steps\": {steps}, \"exact\": {(exact ? "true" : "false")}, \"wall_ms\": {sw.Elapsed.TotalMilliseconds:F1}, " +
            $"\"unsupported\": [{string.Join(",", scene.unsupported.ConvertAll(u => $"\"{u}\""))}]}}");
        s.Dispose();
        return $"S {steps} steps, n {n}, last: {last} -> {dir}";
    }

    /// <summary>S3: run until t > stopAt with the reference's export rule (after every step: if t >= next, next += 1/fps
    /// (float), write frame_XXXX.f32 = pos, vel, density in id order). Also steps.u32 (all stats).</summary>
    public static string RunExport(string scenePath, float stopAt, string dir, bool exact = true, bool tessellated = false)
    {
        Directory.CreateDirectory(dir);
        var scene = DfsphScene.FromFile(scenePath);
        scene.SetTessellated(tessellated);
        var s = MakeSolver(scene, exact);
        int n = s.Count;
        var pos = new Vector3[n]; var vel = new Vector3[n]; var dens = new float[n];
        var ring = new uint[DfsphSolver.Ring * DfsphSolver.StatWords];
        var stats = new List<uint>();
        float next = 0f, fps = scene.dataExportFPS > 0f ? scene.dataExportFPS : 25f;
        int frame = 1, k = 0;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        while (true)
        {
            s.ExecuteSteps(1);
            k++;
            s.StatsBuffer.GetData(ring);
            int slot = (k - 1) % DfsphSolver.Ring;
            for (int w = 0; w < DfsphSolver.StatWords; w++) stats.Add(ring[slot * DfsphSolver.StatWords + w]);
            var st = DfsphSolver.DecodeStats(ring, slot);
            if (st.t >= next)
            {
                next += 1f / fps;
                DumpParticles(s, Path.Combine(dir, $"frame_{frame:D4}"), pos, vel, dens);
                frame++;
            }
            if (st.t > stopAt || k > 200000) break;
        }
        WriteU(Path.Combine(dir, "steps.u32"), stats.ToArray());
        File.WriteAllText(Path.Combine(dir, "run_meta.json"),
            $"{{\"scene\": \"{scenePath}\", \"n\": {n}, \"steps\": {k}, \"frames\": {frame - 1}, \"exact\": {(exact ? "true" : "false")}, \"wall_ms\": {sw.Elapsed.TotalMilliseconds:F1} }}");
        s.Dispose();
        return $"S3 {k} steps, {frame - 1} frames, {sw.Elapsed.TotalSeconds:F1} s -> {dir}";
    }

    /// <summary>P: GPU throughput. Warm up `warm` steps, then time `steps` steps (submit all, one sync at the end) for each
    /// (kCap, kCapV) setting; wall ms per step ~ GPU ms per step (the CPU only submits a pre-recorded command buffer).
    /// `scaleBlock` > 1 rescales the scene's fluid blocks' extent (more particles at the same radius).</summary>
    public static string Bench(string scenePath, int warm, int steps, int[] caps, int[] capsV, float scaleBlock = 1f, bool tessellated = true)
    {
        var scene = DfsphScene.FromFile(scenePath);
        scene.SetTessellated(tessellated);
        if (scaleBlock != 1f)
            foreach (var fb in scene.fluidBlocks)
            {
                Vector3 c = 0.5f * (fb.start + fb.end), e = 0.5f * (fb.end - fb.start);
                e = new Vector3(e.x * scaleBlock, e.y, e.z * scaleBlock);
                fb.start = c - e; fb.end = c + e;
                fb.start.y = Mathf.Max(fb.start.y, 0.05f);
            }
        var sb = new StringBuilder();
        var probe = new uint[DfsphSolver.StateWords];
        for (int c = 0; c < caps.Length; c++)
        {
            var s = MakeSolver(scene, false);
            s.kCap = caps[c]; s.kCapV = capsV[c]; s.MarkDirty();
            s.ExecuteSteps(warm); s.ReadState(probe);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            s.ExecuteSteps(steps); s.ReadState(probe);
            double ms = sw.Elapsed.TotalMilliseconds / steps;
            var ring = new uint[DfsphSolver.Ring * DfsphSolver.StatWords];
            s.StatsBuffer.GetData(ring);
            uint maxIt = 0, maxItV = 0; double sumIt = 0, sumItV = 0; int cnt = 0; uint trunc = 0, truncV = 0;
            for (int k = 0; k < DfsphSolver.Ring; k++)
            {
                var st = DfsphSolver.DecodeStats(ring, k);
                if (!st.valid) continue;
                maxIt = Math.Max(maxIt, st.it); maxItV = Math.Max(maxItV, st.itV); sumIt += st.it; sumItV += st.itV; cnt++;
                trunc = Math.Max(trunc, st.trunc); truncV = Math.Max(truncV, st.truncV);
            }
            sb.AppendLine($"n {s.Count} caps {caps[c]}/{capsV[c]}: {ms:F3} ms/step (t {BitConverter.ToSingle(BitConverter.GetBytes(probe[2]), 0):F2} s), it mean {sumIt / Math.Max(cnt, 1):F1} max {maxIt}, itV mean {sumItV / Math.Max(cnt, 1):F1} max {maxItV}, truncated steps {trunc}/{truncV}");
            s.Dispose();
        }
        return sb.ToString();
    }
}
