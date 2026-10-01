// GpuPortBench.cs — GPU1001 timing harness (Play mode). Added at runtime next to FluidSceneMVP; cycles configurations
// (CPU vs GPU path, spray on/off, foam on/off) in alternating windows of the SAME session and writes one JSON file.
// Per frame it records: unscaled frame time, FluidSceneMVP.LateUpdateMs (main-thread cost of solver tick + splat +
// inference + shading setup), the GPU stage times of the GPU path (CustomSampler recorders) and, where supported,
// FrameTimingManager's CPU/GPU frame times. Editor numbers include editor overhead (an unfocused editor also throttles
// the frame rate, so read LateUpdateMs and the GPU times, not fps).
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;

public class GpuPortBench : MonoBehaviour
{
    [Serializable] public struct Config { public string name; public bool gpu, spray, foam; }

    public string outPath = "";
    public float warmupSec = 2f, measureSec = 5f;
    public int rounds = 2;
    public bool resetEachWindow = true;   // every window replays the same drop from t = 0
    public Config[] configs;
    public bool Done { get; private set; }

    FluidSceneMVP f;
    readonly FrameTiming[] ft = new FrameTiming[1];

    public void Run(FluidSceneMVP target, string path, Config[] cfgs)
    {
        f = target; outPath = path; configs = cfgs;
        StartCoroutine(Routine());
    }

    class Acc
    {
        public List<float> frame = new(), late = new(), gSplat = new(), gNet = new(), gDecode = new(), gSpray = new(), ftCpu = new(), ftGpu = new();
    }

    IEnumerator Routine()
    {
        var res = new Dictionary<string, Acc>();
        bool s0 = f.SprayEnabled, g0 = f.GpuPathEnabled, fo0 = f.FoamEnabled;
        for (int r = 0; r < rounds; r++)
            foreach (var c in configs)
            {
                if (f.GpuPathEnabled != c.gpu) f.ToggleGpuPath();
                if (f.SprayEnabled != c.spray) f.ToggleSpray();
                if (f.FoamEnabled != c.foam) f.ToggleFoam();
                if (resetEachWindow) f.ResetSim();
                float t0 = Time.realtimeSinceStartup;
                while (Time.realtimeSinceStartup - t0 < warmupSec) yield return null;
                if (!res.TryGetValue(c.name, out var a)) res[c.name] = a = new Acc();
                t0 = Time.realtimeSinceStartup;
                while (Time.realtimeSinceStartup - t0 < measureSec)
                {
                    yield return null;
                    a.frame.Add(Time.unscaledDeltaTime * 1000f);
                    a.late.Add(f.LateUpdateMs);
                    if (c.gpu) { a.gSplat.Add(f.GpuMsSplat); a.gNet.Add(f.GpuMsNet); a.gDecode.Add(f.GpuMsDecode); a.gSpray.Add(f.GpuMsSpray); }
                    FrameTimingManager.CaptureFrameTimings();
                    if (FrameTimingManager.GetLatestTimings(1, ft) > 0) { a.ftCpu.Add((float)ft[0].cpuFrameTime); a.ftGpu.Add((float)ft[0].gpuFrameTime); }
                }
            }
        if (f.GpuPathEnabled != g0) f.ToggleGpuPath();
        if (f.SprayEnabled != s0) f.ToggleSpray();
        if (f.FoamEnabled != fo0) f.ToggleFoam();

        var inv = CultureInfo.InvariantCulture;
        var sb = new StringBuilder("{");
        bool first = true;
        foreach (var kv in res)
        {
            if (!first) sb.Append(","); first = false;
            var a = kv.Value;
            sb.Append($"\"{kv.Key}\":{{\"frames\":{a.frame.Count}");
            void S(string n, List<float> v)
            {
                if (v.Count == 0) return;
                v.Sort();
                double m = 0; foreach (var x in v) m += x; m /= v.Count;
                sb.Append($",\"{n}\":{{\"median\":{v[v.Count / 2].ToString("F3", inv)},\"mean\":{m.ToString("F3", inv)},\"p95\":{v[Math.Min(v.Count - 1, (int)(0.95 * v.Count))].ToString("F3", inv)}}}");
            }
            S("frame_ms", a.frame); S("late_update_ms", a.late); S("gpu_splat_ms", a.gSplat); S("gpu_unet_ms", a.gNet);
            S("gpu_decode_ms", a.gDecode); S("gpu_spray_ms", a.gSpray); S("ftm_cpu_ms", a.ftCpu); S("ftm_gpu_ms", a.ftGpu);
            sb.Append("}");
        }
        sb.Append("}");
        System.IO.File.WriteAllText(outPath, sb.ToString());
        Debug.Log($"GPU1001 bench done -> {outPath}");
        Done = true;
    }
}
