// FluidSceneMVP.GpuSprayParity.cs — GPU1001 gates P1-P5 (spray), definitions frozen in
// SSU_restart/Experiments/GPU1001/PLAN.md. Play mode, GPU provider: StartGpuSprayParityRun(outDir, times, rounds) pauses
// at each solver time and compares a FRESH C# LearnedSprayLayer (the parity-gated G2 reference) with a FRESH
// GpuSprayLayer on the same coarse state: P1 features, P2 MLP, P3 birth events (dumped for p3_births.py), P4 flight +
// culls (droplet by droplet), P5 raster (Overlay field + normals) against the same bulk field.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.InferenceEngine;

public partial class FluidSceneMVP
{
    public bool GpuSprayParityRunning { get; private set; }

    public void StartGpuSprayParityRun(string outDir, float[] times, int[] birthRounds)
    {
        if (GpuSprayParityRunning) throw new InvalidOperationException("spray parity run already in progress");
        StartCoroutine(GpuSprayParityRoutine(outDir, times, birthRounds));
    }

    IEnumerator GpuSprayParityRoutine(string outDir, float[] times, int[] rounds)
    {
        GpuSprayParityRunning = true;
        Directory.CreateDirectory(outDir);
        var gpu = provider as GpuSphProvider;
        bool wasPaused = paused;
        for (int q = 0; q < times.Length; q++)
        {
            paused = false;
            while (gpu.SolverTime < times[q]) yield return null;
            paused = true;
            yield return null;
            string tag = $"t{times[q]:F2}";
            string line;
            try { line = GpuSprayParityCase(outDir, tag, rounds[q]); }
            catch (Exception e) { line = $"{{\"case\":\"{tag}\",\"error\":\"{e.Message.Replace("\"", "'")}\"}}"; Debug.LogException(e); }
            File.AppendAllText(Path.Combine(outDir, "p1245.jsonl"), line + "\n");
        }
        paused = wasPaused;
        GpuSprayParityRunning = false;
        Debug.Log($"GPU1001 spray parity done -> {outDir}");
    }

    void ConfigureSpray(LearnedSprayLayer L, GpuSprayLayer G)
    {
        var gpu = provider as GpuSphProvider;
        float rdm = SprayCoarseRadius / Mathf.Pow(25f, 1f / 3f);
        var obs = new List<LearnedSprayLayer.Obstacle>();
        if (gpu.obstacles != null)
        {
            Vector3 off = new Vector3(gpu.domainCenterXZ.x, 0f, gpu.domainCenterXZ.y);
            Matrix4x4 simToWorld = gpu.transform.localToWorldMatrix * Matrix4x4.Scale(new Vector3(1f, 1f, -1f)) * Matrix4x4.Translate(-off);
            foreach (var o in gpu.obstacles)
            {
                if (o == null || o.transform == null || !o.transform.gameObject.activeInHierarchy) continue;
                Matrix4x4 toLocal = o.transform.worldToLocalMatrix * simToWorld;
                float s = Mathf.Max(new Vector3(toLocal.m00, toLocal.m01, toLocal.m02).magnitude,
                                    new Vector3(toLocal.m10, toLocal.m11, toLocal.m12).magnitude,
                                    new Vector3(toLocal.m20, toLocal.m21, toLocal.m22).magnitude);
                obs.Add(new LearnedSprayLayer.Obstacle { toLocal = toLocal, shape = (int)o.shape, padLocal = rdm * s });
            }
        }
        L.rateScale = G.rateScale = sprayRateScale;
        L.launchSpeed = G.launchSpeed = sprayLaunchSpeed;
        L.dropletRadius = G.dropletRadius = sprayDropletRadius;
        L.maxDroplets = G.maxDroplets = Mathf.Max(sprayMaxDroplets, 0);
        L.frameDt = G.frameDt = 1.0 / gpu.simHz;
        L.gravity = G.gravity = gpu.gravity;
        L.domainMin = G.domainMin = gpu.domainMin; L.domainMax = G.domainMax = gpu.domainMax;
        L.obstacles.Clear(); L.obstacles.AddRange(obs);
        G.obstacles.Clear(); G.obstacles.AddRange(obs);
        L.multithreaded = true;
    }

    static void WriteF32(string path, float[] a, int count)
    {
        var bytes = new byte[count * 4];
        Buffer.BlockCopy(a, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }

    string GpuSprayParityCase(string outDir, string tag, int rounds)
    {
        var inv = CultureInfo.InvariantCulture;
        var gp = provider as GpuSphProvider;
        var solver = gp.Solver;
        provider.GetFrame(frameIdx, out var rec, out int off, out int n);
        float rc = SprayCoarseRadius;
        var asset = sprayModel != null ? sprayModel : Resources.Load<ModelAsset>(sprayModelResource);
        var shader = Resources.Load<ComputeShader>("GpuSpray");
        var L = new LearnedSprayLayer((ulong)(uint)spraySeed);
        var G = new GpuSprayLayer(shader) { seed = (uint)spraySeed };
        var sb = new StringBuilder();
        sb.Append($"{{\"case\":\"{tag}\",\"n\":{n}");
        try
        {
            L.SetModel(asset, BackendType.GPUCompute);
            G.SetModel(asset);
            var metaTA = Resources.Load<TextAsset>(asset.name + "_meta");
            if (metaTA != null) { var m = JsonUtility.FromJson<SprayMeta>(metaTA.text); if (m != null && m.sig_in > 0f) { L.sigIn = m.sig_in; G.sigIn = m.sig_in; } }
            ConfigureSpray(L, G);

            // ---------------- P1 features
            L.ComputeFeatures(rec, off, n, rc);
            var Xc = L.Features; var Kc = L.Knn;
            var fd = G.ParityFeatures(solver.PositionBuffer, solver.VelocityBuffer, n, rc);
            bool nfullSame = Math.Abs(fd.NFull - L.NFull) == 0.0;
            double minhD = Math.Abs(fd.MinH - L.MinHeight);
            int ownBadRows = 0; double ownMax = 0;
            int knnSetDiff = 0, knnOrderDiff = 0, nonTie = 0; double tieWorst = 0;
            int nbBadRows = 0; double nbMax = 0;
            var setA = new HashSet<int>();
            var ownCols = new int[12]; var ownEx = new StringBuilder();
            for (int i = 0; i < n; i++)
            {
                bool bad = false;
                for (int c = 0; c < 12; c++)
                {
                    double d = Math.Abs(Xc[i * 124 + c] - fd.X[i * 124 + c]); ownMax = Math.Max(ownMax, d);
                    if (d > 1e-4)
                    {
                        bad = true; ownCols[c]++;
                        if (ownEx.Length < 600)
                        {
                            int b0 = off + 7 * i;
                            double vx = rec[b0 + 3], vy = rec[b0 + 4], vz = rec[b0 + 5];
                            ownEx.Append($"row {i} col {c}: cpu {Xc[i * 124 + c].ToString("G6", inv)} gpu {fd.X[i * 124 + c].ToString("G6", inv)} |v| {Math.Sqrt(vx * vx + vy * vy + vz * vz).ToString("G3", inv)} |v_h| {Math.Sqrt(vx * vx + vz * vz).ToString("G3", inv)}; ");
                        }
                    }
                }
                if (bad) ownBadRows++;
                bool orderSame = true;
                setA.Clear();
                for (int k = 0; k < 16; k++) { if (Kc[i * 16 + k] != fd.Knn[i * 16 + k]) orderSame = false; setA.Add(Kc[i * 16 + k]); }
                bool setSame = true;
                for (int k = 0; k < 16; k++) if (!setA.Contains(fd.Knn[i * 16 + k])) setSame = false;
                if (!setSame) knnSetDiff++;
                if (!orderSame)
                {
                    knnOrderDiff++;
                    for (int k = 0; k < 16; k++)
                    {
                        int a = Kc[i * 16 + k], b = fd.Knn[i * 16 + k];
                        if (a == b) continue;
                        if (a < 0 || b < 0) { nonTie++; continue; }
                        double da = Dist(rec, off, i, a), db = Dist(rec, off, i, b);
                        double t = Math.Abs(da - db) / rc;
                        tieWorst = Math.Max(tieWorst, t);
                        if (t > 1e-5) nonTie++;
                    }
                }
                else
                {
                    bool nb = false;
                    for (int c = 12; c < 124; c++) { double d = Math.Abs(Xc[i * 124 + c] - fd.X[i * 124 + c]); nbMax = Math.Max(nbMax, d); if (d > 1e-4) nb = true; }
                    if (nb) nbBadRows++;
                }
            }
            bool p1 = nfullSame && minhD <= 1e-5 && ownBadRows <= 0.001 * n && knnSetDiff <= 0.01 * n && nonTie == 0 && nbBadRows == 0;
            sb.Append($",\"P1\":{{\"pass\":{(p1 ? "true" : "false")},\"nfull_cpu\":{L.NFull.ToString("R", inv)},\"nfull_gpu\":{fd.NFull.ToString("R", inv)}," +
                      $"\"minh_abs\":{minhD.ToString("G4", inv)},\"own_max_abs\":{ownMax.ToString("G4", inv)},\"own_rows_over_1e-4\":{ownBadRows}," +
                      $"\"knn_set_diff_rows\":{knnSetDiff},\"knn_order_diff_rows\":{knnOrderDiff},\"knn_nontie_slots\":{nonTie},\"knn_tie_worst_rc\":{tieWorst.ToString("G4", inv)}," +
                      $"\"nb_max_abs\":{nbMax.ToString("G4", inv)},\"nb_rows_over_1e-4\":{nbBadRows},\"own_cols_over_1e-4\":[{string.Join(",", ownCols)}],\"own_examples\":\"{ownEx}\"}}");

            // ---------------- P2 MLP on the C# rows
            L.RunMlp();
            var Pc = L.Packed;
            var Pg = G.ParityMlp(Xc, n);
            double[] headMax = new double[5];
            int[] h0 = { 0, 1, 5, 29, 53 }, h1 = { 1, 5, 29, 53, 63 };
            for (int i = 0; i < n; i++)
                for (int hd = 0; hd < 5; hd++)
                    for (int c = h0[hd]; c < h1[hd]; c++)
                        headMax[hd] = Math.Max(headMax[hd], Math.Abs(Pc[i * 63 + c] - Pg[i * 63 + c]));
            bool p2 = true; foreach (var v in headMax) p2 &= v <= 1e-5;
            sb.Append($",\"P2\":{{\"pass\":{(p2 ? "true" : "false")},\"head_max_abs\":[{string.Join(",", Array.ConvertAll(headMax, v => v.ToString("G4", inv)))}]}}");

            // ---------------- P3 births: dumps for p3_births.py
            if (rounds > 0)
            {
                float[] ev = G.ParityBirthRounds(solver.PositionBuffer, solver.VelocityBuffer, n, rc, rounds, Math.Max(400000, rounds * 400), out int ng, out float gsum);
                WriteF32(Path.Combine(outDir, $"{tag}_events_gpu.f32"), ev, ng * 12);
                L.maxDroplets = 0;                    // events are logged before the cap: no droplet placement work
                L.eventLog = new List<float>(rounds * 160 * 10);
                double csum = 0;
                for (int r = 0; r < rounds; r++) { L.RunBirths(); csum += L.LastRateSum; }
                var ec = L.eventLog.ToArray();
                WriteF32(Path.Combine(outDir, $"{tag}_events_cpu.f32"), ec, ec.Length);
                // calibration: a second C# sample with another seed (reference-vs-reference values for every P3 statistic)
                L.Seed(0x5EED777UL);
                L.eventLog = new List<float>(rounds * 160 * 10);
                for (int r = 0; r < rounds; r++) L.RunBirths();
                var ec2 = L.eventLog.ToArray();
                WriteF32(Path.Combine(outDir, $"{tag}_events_cpu2.f32"), ec2, ec2.Length);
                // parents: P xyz, V xyz, nhat xyz, lambda (C#), per particle
                var par = new float[n * 10];
                double sR = sprayRateScale, cap = 50.0 * Math.Max(sR, 1.0);
                for (int i = 0; i < n; i++)
                {
                    int b = off + 7 * i;
                    for (int c = 0; c < 6; c++) par[i * 10 + c] = rec[b + c];
                    par[i * 10 + 6] = (float)L.NHat[3 * i]; par[i * 10 + 7] = (float)L.NHat[3 * i + 1]; par[i * 10 + 8] = (float)L.NHat[3 * i + 2];
                    double lam = Math.Exp((double)Pc[i * 63]) * sR; if (lam > cap) lam = cap;
                    par[i * 10 + 9] = (float)lam;
                }
                WriteF32(Path.Combine(outDir, $"{tag}_parents.f32"), par, par.Length);
                File.WriteAllText(Path.Combine(outDir, $"{tag}_births_meta.json"),
                    $"{{\"n\":{n},\"rounds\":{rounds},\"rc\":{rc.ToString("R", inv)},\"sR\":{sprayRateScale.ToString("R", inv)},\"launch\":{sprayLaunchSpeed.ToString("R", inv)}," +
                    $"\"rate_sum_cpu\":{(csum / rounds).ToString("R", inv)},\"rate_sum_gpu\":{gsum.ToString("R", inv)},\"events_gpu\":{ng},\"events_cpu\":{ec.Length / 10}}}");
                sb.Append($",\"P3_dumped\":{{\"rounds\":{rounds},\"events_gpu\":{ng},\"events_cpu\":{ec.Length / 10},\"rate_sum_cpu\":{(csum / rounds).ToString("G6", inv)},\"rate_sum_gpu\":{gsum.ToString("G6", inv)}}}");
            }

            // ---------------- P4 flight + culls, droplet by droplet
            // a realistic droplet set: C# births at this frame (20 rounds, uncapped), ages spread 0..54 so the merge grace and
            // the 2 s age cull are both exercised
            var L2 = new LearnedSprayLayer(12345UL);
            L2.SetModel(asset, BackendType.GPUCompute);
            ConfigureSpray(L2, G);
            L2.ComputeFeatures(rec, off, n, rc); L2.RunMlp();
            for (int r = 0; r < 20 && L2.Alive < 20000; r++) L2.RunBirths();
            int nd = L2.Alive;
            var dp = new Vector3[nd]; var dv = new Vector3[nd]; var da_ = new int[nd];
            Array.Copy(L2.DropletPositions, dp, nd); Array.Copy(L2.DropletVelocities, dv, nd);
            for (int i = 0; i < nd; i++) da_[i] = i % 55;
            G.ParitySetPool(dp, dv, da_, nd);
            var surv = G.ParityAdvance(solver.PositionBuffer, n, rc, out int ns, out var gst);
            var gpuById = new Dictionary<int, GpuSprayLayer.DropGpu>(ns);
            foreach (var d in surv) gpuById[(int)d.id] = d;
            // C#: one droplet at a time against the same loaded state (its compaction does not keep ids)
            L2.ComputeFeatures(rec, off, n, rc);
            int cpuAlive = 0, setDiff = 0; double dxMax = 0, dvMax = 0;
            var cpuSurvP = new List<Vector3>(); var cpuSurvV = new List<Vector3>(); var cpuSurvA = new List<int>();
            var oneP = new Vector3[1]; var oneV = new Vector3[1];
            var diffs = new StringBuilder();
            for (int i = 0; i < nd; i++)
            {
                L2.Reset();
                oneP[0] = dp[i]; oneV[0] = dv[i];
                L2.AddDroplets(oneP, oneV, 1, da_[i]);
                L2.AdvanceAndCull();
                bool ca = L2.Alive == 1, ga = gpuById.TryGetValue(i, out var gd);
                if (ca) { cpuAlive++; cpuSurvP.Add(L2.DropletPositions[0]); cpuSurvV.Add(L2.DropletVelocities[0]); cpuSurvA.Add(L2.DropletAges[0]); }
                if (ca != ga) { setDiff++; if (diffs.Length < 400) diffs.Append($"{i}:{(ca ? "cpu" : "gpu")}-only;"); continue; }
                if (ca)
                {
                    dxMax = Math.Max(dxMax, (L2.DropletPositions[0] - (Vector3)gd.xa).magnitude);
                    dvMax = Math.Max(dvMax, (L2.DropletVelocities[0] - gd.v).magnitude);
                }
            }
            bool p4 = setDiff == 0 && dxMax <= 1e-5 && dvMax <= 1e-5;
            sb.Append($",\"P4\":{{\"pass\":{(p4 ? "true" : "false")},\"droplets\":{nd},\"survivors_cpu\":{cpuAlive},\"survivors_gpu\":{ns},\"set_diff\":{setDiff}," +
                      $"\"dx_max_m\":{dxMax.ToString("G4", inv)},\"dv_max\":{dvMax.ToString("G4", inv)},\"gpu_kills_wall_obst_merge_age\":[{gst[4]},{gst[5]},{gst[6]},{gst[7]}],\"diffs\":\"{diffs}\"}}");

            // ---------------- P5 raster (Overlay) of the C# survivors against the same bulk field
            if (!EnsureGpuPath()) throw new Exception("GPU path unavailable");
            ComputeSimCamera();
            gpuCb.Clear(); RecordGpuInference(); Graphics.ExecuteCommandBuffer(gpuCb); gpuCb.Clear();
            var bulk = new Color[HW]; gpuSplat.Field.GetData(bulk);
            int m5 = cpuSurvP.Count;
            G.ParitySetPool(cpuSurvP.ToArray(), cpuSurvV.ToArray(), cpuSurvA.ToArray(), m5);
            float ts = useV2 ? v2TS : SplatV2.ThicknessScaleLegacy;
            var rcb = new CommandBuffer { name = "GPU1001.parity.raster" };
            G.RecordRaster(rcb, gpuSplat.Field, W, H, eyeSim, rightSim, upSim, fwdSim, focalM, winAspect, ts, sprayMinRadiusPx, sprayRadiusScale, false);
            Graphics.ExecuteCommandBuffer(rcb); rcb.Release();
            var gF = new Vector4[HW]; var gN = new Vector4[HW];
            G.ReadRaster(W, H, gF, gN);
            var L3 = new LearnedSprayLayer(1UL);
            for (int i = 0; i < m5; i++) { oneP[0] = cpuSurvP[i]; oneV[0] = cpuSurvV[i]; L3.AddDroplets(oneP, oneV, 1, cpuSurvA[i]); }
            L3.dropletRadius = sprayDropletRadius;
            L3.ComputeFeatures(rec, off, n, rc);   // sets r_d from r_c (raster uses rd)
            var cF = new Color[HW]; var cN = new Color[HW];
            L3.CompositeOverlay(bulk, cF, W, H, eyeSim, rightSim, upSim, fwdSim, focalM, winAspect, ts, sprayMinRadiusPx, sprayRadiusScale, cN);
            int drawnC = 0, drawnG = 0, drawnDiff = 0; double fMax = 0, tMax = 0, nMax = 0;
            int badPx = 0; double badRelChordMax = 0, goodRelChordMin = 1e9;
            float fullChord = 2f * L3.RD * sprayRadiusScale * ts;   // chord through the centre of one droplet (x thickness scale)
            for (int i = 0; i < HW; i++)
            {
                bool a = cF[i].b > 0f, b = gF[i].z > 0f;
                if (a) drawnC++; if (b) drawnG++;
                if (a != b) { drawnDiff++; continue; }
                if (!a) continue;
                fMax = Math.Max(fMax, Math.Abs(cF[i].r - gF[i].x));
                tMax = Math.Max(tMax, Math.Abs(cF[i].g - gF[i].y));
                nMax = Math.Max(nMax, Math.Max(Math.Abs(cN[i].r - gN[i].x), Math.Max(Math.Abs(cN[i].g - gN[i].y), Math.Abs(cN[i].b - gN[i].z))));
                bool pxBad = Math.Abs(cF[i].r - gF[i].x) > 1e-5 || Math.Abs(cF[i].g - gF[i].y) > 1e-4;
                double relChord = cF[i].g / Math.Max(fullChord, 1e-12);   // ~ 2 hh / (2 r) for a single droplet: 0 at the rim
                if (pxBad) { badPx++; badRelChordMax = Math.Max(badRelChordMax, relChord); }
            }
            bool p5 = drawnDiff <= 0.001 * Math.Max(drawnC, 1) && fMax <= 1e-5 && tMax <= 1e-4 && nMax <= 1e-3;
            sb.Append($",\"P5\":{{\"pass\":{(p5 ? "true" : "false")},\"droplets\":{m5},\"drawn_cpu\":{drawnC},\"drawn_gpu\":{drawnG},\"drawn_diff\":{drawnDiff}," +
                      $"\"front_max_m\":{fMax.ToString("G4", inv)},\"chord_max\":{tMax.ToString("G4", inv)},\"normal_max\":{nMax.ToString("G4", inv)}," +
                      $"\"px_over_tol\":{badPx},\"over_tol_max_chord_over_full\":{badRelChordMax.ToString("G4", inv)}}}");
            L2.Dispose(); L3.Dispose();
        }
        finally
        {
            L.Dispose(); G.Dispose();
        }
        sb.Append("}");
        return sb.ToString();
    }

    static double Dist(float[] rec, int off, int i, int j)
    {
        int a = off + 7 * i, b = off + 7 * j;
        double x = (double)rec[a] - rec[b], y = (double)rec[a + 1] - rec[b + 1], z = (double)rec[a + 2] - rec[b + 2];
        return Math.Sqrt(x * x + y * y + z * z);
    }
}
