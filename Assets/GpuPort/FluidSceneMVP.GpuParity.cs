// FluidSceneMVP.GpuParity.cs — GPU1001 gates S1 (splat), S2 (decode), S3 (end to end); definitions frozen in
// SSU_restart/Experiments/GPU1001/PLAN.md before any number. Play mode: StartGpuParityRun(outPath) pauses the scene at
// fixed solver times, runs every gate from three cameras on the SAME particle frame, restores the camera, resumes, and
// appends one JSON line per case to outPath; "GPU1001 parity done" is logged at the end.
using System;
using System.Collections;
using System.Globalization;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.InferenceEngine;

public partial class FluidSceneMVP
{
    public bool GpuParityRunning { get; private set; }

    public void StartGpuParityRun(string outPath, float[] solverTimes)
    {
        if (GpuParityRunning) throw new InvalidOperationException("parity run already in progress");
        StartCoroutine(GpuParityRoutine(outPath, solverTimes));
    }

    IEnumerator GpuParityRoutine(string outPath, float[] times)
    {
        GpuParityRunning = true;
        var gpu = provider as GpuSphProvider;
        Transform cam = targetCamera.transform;
        Vector3 cam0 = cam.position; Quaternion rot0 = cam.rotation;
        bool wasPaused = paused;
        int pass = 0, total = 0;
        foreach (float t in times)
        {
            paused = false;
            while ((gpu != null ? gpu.SolverTime : simTime) < t) yield return null;
            paused = true;
            yield return null;   // one rendered frame with the solver held: every buffer is at this state
            Vector3 c = ParticleCentroidWorld();
            float s = transform.lossyScale.x;
            var poses = new (string name, Vector3 pos, Quaternion rot)[]
            {
                ("scene", cam0, rot0),
                ("close", c + new Vector3(0.9f, 0.55f, -1.1f).normalized * 1.5f * s, Quaternion.identity),
                ("grazing", c + new Vector3(-3.0f, 0.15f, -1.2f).normalized * 3.2f * s, Quaternion.identity),
            };
            foreach (var p in poses)
            {
                cam.position = p.pos;
                cam.rotation = p.name == "scene" ? p.rot : Quaternion.LookRotation(c - p.pos, Vector3.up);
                string line;
                bool ok;
                try { line = GpuParityCase($"t{t:F2}_{p.name}", out ok); }
                catch (Exception e) { line = $"{{\"case\":\"t{t:F2}_{p.name}\",\"error\":\"{e.Message.Replace("\"", "'")}\"}}"; ok = false; Debug.LogException(e); }
                System.IO.File.AppendAllText(outPath, line + "\n");
                total++; if (ok) pass++;
            }
            cam.position = cam0; cam.rotation = rot0;
        }
        paused = wasPaused;
        GpuParityRunning = false;
        Debug.Log($"GPU1001 parity done: {pass}/{total} cases pass all of S1-S3 -> {outPath}");
    }

    Vector3 ParticleCentroidWorld()
    {
        provider.GetFrame(frameIdx, out var data, out int off, out int count);
        if (count == 0) return transform.position;
        double x = 0, y = 0, z = 0;
        for (int i = 0; i < count; i++) { int b = off + 7 * i; x += data[b]; y += data[b + 1]; z += data[b + 2]; }
        var sim = new Vector3((float)(x / count), (float)(y / count), (float)(z / count));
        return transform.TransformPoint(FlipZ(sim - simOffset));
    }

    /// <summary>S1 + S2 + S3 for the current particle frame and camera. Returns one JSON object.</summary>
    public string GpuParityCase(string tag, out bool allOk)
    {
        if (!EnsureGpuPath()) throw new Exception("GPU path unavailable: " + gpuError);
        ComputeSimCamera();
        var inv = CultureInfo.InvariantCulture;
        var sb = new StringBuilder();
        sb.Append("{\"case\":\"").Append(tag).Append("\"");

        // ---------------- S1: splat (normalised in7), CPU SplatV2 vs GPU
        SplatToInput(frameIdx);
        var cpuIn = (float[])in7.Clone();
        var cb = new CommandBuffer { name = "GPU1001.parity" };
        var tmpIn = new ComputeBuffer(7 * HW, 4);
        GpuParticles(out var pos, out var vel, out var dens, out int n);
        gpuSplat.RecordSplat(cb, pos, vel, dens, n, GpuCam(), GpuSplatParams(), tmpIn, meta.mean, meta.std, thickScale);
        Graphics.ExecuteCommandBuffer(cb);
        var gpuIn = new float[7 * HW];
        tmpIn.GetData(gpuIn);
        tmpIn.Release();
        int maskDiff = 0, maskOn = 0;
        for (int i = 0; i < HW; i++) { if (cpuIn[6 * HW + i] != gpuIn[6 * HW + i]) maskDiff++; if (cpuIn[6 * HW + i] > 0f) maskOn++; }
        var chMax = new double[6]; var chBadFrac = new double[6];
        for (int ch = 1; ch < 6; ch++)
        {
            int bad = 0; double mx = 0;
            for (int i = 0; i < HW; i++) { double d = Math.Abs(cpuIn[ch * HW + i] - gpuIn[ch * HW + i]); if (d > mx) mx = d; if (!(d < 1e-3)) bad++; }
            chMax[ch] = mx; chBadFrac[ch] = bad / (double)HW;
        }
        int depthBad = 0, depthFlip = 0; double depthMax = 0, depthMaxNonFlip = 0;
        for (int i = 0; i < HW; i++)
        {
            if (cpuIn[6 * HW + i] != gpuIn[6 * HW + i]) continue;
            double d = Math.Abs(cpuIn[i] - gpuIn[i]);
            if (d > depthMax) depthMax = d;
            if (d < 1e-4) continue;
            depthBad++;
            if (Math.Abs(cpuIn[HW + i] - gpuIn[HW + i]) < 1e-3) depthFlip++;
            else if (d > depthMaxNonFlip) depthMaxNonFlip = d;
        }
        bool s1 = maskDiff <= 5 && depthBad <= 5 && depthFlip == depthBad;
        for (int ch = 1; ch < 6; ch++) s1 &= chBadFrac[ch] <= 1e-4;
        sb.Append(",\"particles\":").Append(n).Append(",\"mask_px\":").Append(maskOn);
        sb.Append(",\"S1\":{\"pass\":").Append(s1 ? "true" : "false")
          .Append(",\"mask_diff_px\":").Append(maskDiff)
          .Append(",\"depth_max_abs\":").Append(depthMax.ToString("G4", inv))
          .Append(",\"depth_px_over_1e-4\":").Append(depthBad).Append(",\"depth_boundary_flips\":").Append(depthFlip)
          .Append(",\"ch_max_abs\":[");
        for (int ch = 1; ch < 6; ch++) sb.Append(ch > 1 ? "," : "").Append(chMax[ch].ToString("G4", inv));
        sb.Append("],\"ch_frac_over_1e-3\":[");
        for (int ch = 1; ch < 6; ch++) sb.Append(ch > 1 ? "," : "").Append(chBadFrac[ch].ToString("G4", inv));
        sb.Append("]}");

        // ---------------- S3 reference: CPU path end to end (CPU splat -> RunModel -> FillFieldCpu)
        Array.Copy(cpuIn, in7, in7.Length);
        float[] predCpu = RunModel();
        FillFieldCpu(predCpu);
        var fieldCpuPath = (Color[])px.Clone();

        // ---------------- GPU chain (splat -> net -> decode), then S2 on its own prediction
        cb.Release();
        gpuCb.Clear();
        var predBuf = RecordGpuInference();   // records into gpuCb
        Graphics.ExecuteCommandBuffer(gpuCb);
        gpuCb.Clear();
        var predGpu = new float[7 * HW];
        predBuf.GetData(predGpu, 0, 0, 7 * HW);
        var fieldGpu = new Color[HW];
        gpuSplat.Field.GetData(fieldGpu);
        gpuIn7.GetData(in7);              // FillFieldCpu's temporal speed reads in7
        FillFieldCpu(predGpu);            // CPU decode of the GPU prediction
        double s2Max = 0;
        for (int i = 0; i < HW; i++)
        {
            Color a = px[i], b = fieldGpu[i];
            s2Max = Math.Max(s2Max, Math.Max(Math.Max(Math.Abs(a.r - b.r), Math.Abs(a.g - b.g)), Math.Max(Math.Abs(a.b - b.b), Math.Abs(a.a - b.a))));
        }
        bool s2 = s2Max <= 1e-6;
        sb.Append(",\"S2\":{\"pass\":").Append(s2 ? "true" : "false").Append(",\"max_abs\":").Append(s2Max.ToString("G4", inv)).Append("}");

        // ---------------- S3: GPU chain vs CPU path
        int inter = 0, uni = 0; var dd = new System.Collections.Generic.List<double>(HW / 4); double tMax = 0;
        for (int i = 0; i < HW; i++)
        {
            bool a = fieldCpuPath[i].b > 0f, b = fieldGpu[i].b > 0f;
            if (a || b) uni++;
            if (a && b) { inter++; dd.Add(Math.Abs(fieldCpuPath[i].r - fieldGpu[i].r)); tMax = Math.Max(tMax, Math.Abs(fieldCpuPath[i].g - fieldGpu[i].g)); }
        }
        dd.Sort();
        double iou = uni > 0 ? inter / (double)uni : 1.0;
        double p99 = dd.Count > 0 ? dd[Math.Min(dd.Count - 1, (int)(0.99 * dd.Count))] : 0, dmax = dd.Count > 0 ? dd[dd.Count - 1] : 0;
        bool s3 = iou >= 0.999 && p99 <= 1e-3;
        sb.Append(",\"S3\":{\"pass\":").Append(s3 ? "true" : "false").Append(",\"mask_iou\":").Append(iou.ToString("F6", inv))
          .Append(",\"depth_p99_m\":").Append(p99.ToString("G4", inv)).Append(",\"depth_max_m\":").Append(dmax.ToString("G4", inv))
          .Append(",\"thick_max\":").Append(tMax.ToString("G4", inv)).Append("}");
        sb.Append("}");
        allOk = s1 && s2 && s3;
        return sb.ToString();
    }
}
