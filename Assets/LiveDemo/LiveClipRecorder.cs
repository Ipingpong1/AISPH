// LiveClipRecorder.cs — 067: freeze a true-coarse live clip so every splat / focal / model / blur
// configuration can be replayed OFFLINE on the identical particle stream (Helpers/live_gap_probe.py
// in the SSU_restart repo). Nothing here changes what the scene renders.
//
// Writes <project>/<outRoot>/<clipName>/ :
//   records.bytes      1LPS (SimExportRunner.Write1Lps): ONE record-set per SOLVER frame, 7 float32 per
//                      particle (px py pz vx vy vz density, sim space), per-frame counts in the header.
//   clip.json          per-solver-frame sim-space camera (eye, right, up, fwd, focalM = 13 floats), solver
//                      time, fov/aspect, stirrer world position, plus the scene's splat/model settings.
//   in7_fKKKK.bytes    the normalized 7x512x512 model input Unity built   } on dumpFrames (clip-relative solver
//   pred_fKKKK.bytes   the raw 7x512x512 network output Unity produced    } frame K) — splat + backend parity
//
// fixedStep pins Time.deltaTime to 1/simHz (Time.captureFramerate) so a take is one solver frame per
// rendered frame regardless of editor frame rate — reproducible, and the dumps line up with the records.

using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class LiveClipRecorder : MonoBehaviour
{
    public FluidSceneMVP scene;
    public GpuSphProvider provider;
    [Tooltip("Optional: a prop whose world position is logged per solver frame (the stir sphere).")]
    public Transform stirrer;

    public string clipName = "clip";
    [Tooltip("Project-relative output root.")]
    public string outRoot = "LiveClips";
    [Tooltip("Number of SOLVER frames to record.")]
    public int frames = 150;
    [Tooltip("Start once the provider's solver time reaches this (s).")]
    public float startAtSolverTime = 0f;
    public bool recordOnPlay = true;
    [Tooltip("Pin Time.deltaTime to 1/simHz while recording (deterministic, one solver frame per rendered frame).")]
    public bool fixedStep = true;
    [Tooltip("Clip-relative solver frames on which the model input and raw output are also dumped.")]
    public int[] dumpFrames = { 30, 75, 120 };
    [Tooltip("Editor only: leave Play mode when the clip is written.")]
    public bool exitPlayWhenDone = false;

    [Serializable]
    class ClipJson
    {
        public string clip, unityVersion, recordedAtUtc;
        public int frames, firstSolverFrame, screenW, screenH;
        public float simHz, particleRadius;
        public float[] gravity;
        public float[] cam;            // 13 per frame: eye xyz, right xyz, up xyz, fwd xyz, focalM (sim space)
        public float[] solverTime, fovV, aspect;
        public float[] stirPosWorld;   // 3 per frame (zeros when no stirrer / inactive)
        public int[] count, dumped;
        public FluidSceneMVP.ClipSettings settings;
    }

    readonly List<float[]> recs = new List<float[]>();
    readonly List<float> cam = new List<float>(), tSolver = new List<float>(), fov = new List<float>(),
                         asp = new List<float>(), stir = new List<float>();
    readonly List<int> counts = new List<int>(), dumped = new List<int>();
    bool recording, done;
    int firstSolverFrame = -1, lastRelThisTick = -1;
    int prevCaptureFramerate;

    public bool Done => done;

    void OnEnable()
    {
        if (scene == null) scene = GetComponent<FluidSceneMVP>();
        if (provider == null) provider = GetComponent<GpuSphProvider>();
        if (scene == null || provider == null) { Debug.LogError("[LiveClipRecorder] needs a FluidSceneMVP and a GpuSphProvider"); enabled = false; return; }
        provider.OnSolverFrame += HandleSolverFrame;
        scene.OnInferred += HandleInferred;
        scene.OnShaded += HandleShaded;
        if (recordOnPlay) Begin();
    }

    void OnDisable()
    {
        if (provider != null) provider.OnSolverFrame -= HandleSolverFrame;
        if (scene != null) { scene.OnInferred -= HandleInferred; scene.OnShaded -= HandleShaded; }
        if (recording && fixedStep) Time.captureFramerate = prevCaptureFramerate;
        recording = false;
    }

    public void Begin()
    {
        if (recording) return;
        recs.Clear(); cam.Clear(); tSolver.Clear(); fov.Clear(); asp.Clear(); stir.Clear(); counts.Clear(); dumped.Clear();
        firstSolverFrame = -1; lastRelThisTick = -1; done = false; recording = true;
        if (fixedStep)
        {
            prevCaptureFramerate = Time.captureFramerate;
            Time.captureFramerate = Mathf.Max(1, Mathf.RoundToInt(provider.simHz));
        }
        Debug.Log($"[LiveClipRecorder] armed: '{clipName}', {frames} solver frames from t={startAtSolverTime:0.00}s, fixedStep={fixedStep}");
    }

    void HandleSolverFrame(int idx, float solverTime, float[] records, int count)
    {
        if (!recording || recs.Count >= frames) return;
        if (solverTime < startAtSolverTime - 1e-4f) return;
        if (firstSolverFrame < 0) firstSolverFrame = idx;

        var copy = new float[count * 7];
        Array.Copy(records, copy, count * 7);
        recs.Add(copy);
        counts.Add(count);
        tSolver.Add(solverTime);

        scene.GetSimCamera(out Vector3 e, out Vector3 r, out Vector3 u, out Vector3 f, out float focal);
        cam.Add(e.x); cam.Add(e.y); cam.Add(e.z);
        cam.Add(r.x); cam.Add(r.y); cam.Add(r.z);
        cam.Add(u.x); cam.Add(u.y); cam.Add(u.z);
        cam.Add(f.x); cam.Add(f.y); cam.Add(f.z);
        cam.Add(focal);
        Camera c = scene.targetCamera != null ? scene.targetCamera : Camera.main;
        fov.Add(c != null ? c.fieldOfView : 0f);
        asp.Add(c != null ? c.aspect : 0f);
        Vector3 sp = stirrer != null && stirrer.gameObject.activeInHierarchy ? stirrer.position : Vector3.zero;
        stir.Add(sp.x); stir.Add(sp.y); stir.Add(sp.z);

        lastRelThisTick = recs.Count - 1;
    }

    // Runs right after inference in FluidSceneMVP.LateUpdate: in7/pred belong to the LAST solver frame of
    // this rendered frame's Tick (the splat read that frame's records and the camera has not moved since).
    void HandleInferred()
    {
        if (!recording) return;
        int k = lastRelThisTick;
        lastRelThisTick = -1;
        if (k >= 0 && dumpFrames != null && Array.IndexOf(dumpFrames, k) >= 0 && !dumped.Contains(k))
        {
            string dir = OutDir();
            Directory.CreateDirectory(dir);
            WriteFloats(Path.Combine(dir, $"in7_f{k:D4}.bytes"), scene.LastInput);
            WriteFloats(Path.Combine(dir, $"pred_f{k:D4}.bytes"), scene.LastPred);
            dumped.Add(k);
            pendingTemporalDump = k;
        }
        if (recs.Count >= frames) finishAfterShade = true;
    }

    int pendingTemporalDump = -1;
    bool finishAfterShade;

    // After the shading blits: the temporal stage's input / history / output for the frame just dumped (067-EMA3 shader-vs-mirror parity).
    void HandleShaded()
    {
        if (pendingTemporalDump >= 0 && scene.TemporalOut != null)
        {
            string dir = OutDir(); int k = pendingTemporalDump;
            DumpRT(Path.Combine(dir, $"tin_f{k:D4}.bytes"), scene.TemporalIn);
            DumpRT(Path.Combine(dir, $"thist_f{k:D4}.bytes"), scene.TemporalHist);
            DumpRT(Path.Combine(dir, $"tout_f{k:D4}.bytes"), scene.TemporalOut);
        }
        pendingTemporalDump = -1;
        if (finishAfterShade) { finishAfterShade = false; Finish(); }
    }

    static void DumpRT(string path, RenderTexture rt)
    {
        var prev = RenderTexture.active; RenderTexture.active = rt;
        var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGBAFloat, false, true);
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0); tex.Apply(false);
        RenderTexture.active = prev;
        WriteFloats(path, tex.GetRawTextureData<float>().ToArray());
        Destroy(tex);
    }

    string OutDir() => Path.GetFullPath(Path.Combine(Application.dataPath, "..", outRoot, clipName));

    static void WriteFloats(string path, float[] a)
    {
        if (a == null) return;
        var bytes = new byte[a.Length * 4];
        Buffer.BlockCopy(a, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }

    void Finish()
    {
        recording = false;
        if (fixedStep) Time.captureFramerate = prevCaptureFramerate;
        string dir = OutDir();
        Directory.CreateDirectory(dir);
        SimExportRunner.Write1Lps(Path.Combine(dir, "records.bytes"), recs);
        Camera c = scene.targetCamera != null ? scene.targetCamera : Camera.main;
        var j = new ClipJson
        {
            clip = clipName, unityVersion = Application.unityVersion, recordedAtUtc = DateTime.UtcNow.ToString("o"),
            frames = recs.Count, firstSolverFrame = firstSolverFrame,
            screenW = c != null ? c.pixelWidth : 0, screenH = c != null ? c.pixelHeight : 0,
            simHz = provider.simHz, particleRadius = provider.particleRadius,
            gravity = new[] { provider.gravity.x, provider.gravity.y, provider.gravity.z },
            cam = cam.ToArray(), solverTime = tSolver.ToArray(), fovV = fov.ToArray(), aspect = asp.ToArray(),
            stirPosWorld = stir.ToArray(), count = counts.ToArray(), dumped = dumped.ToArray(),
            settings = scene.GetClipSettings(),
        };
        File.WriteAllText(Path.Combine(dir, "clip.json"), JsonUtility.ToJson(j, false));
        done = true;
        Debug.Log($"[LiveClipRecorder] done: '{clipName}' {recs.Count} solver frames, dumps [{string.Join(",", dumped)}] -> {dir}");
#if UNITY_EDITOR
        if (exitPlayWhenDone) UnityEditor.EditorApplication.isPlaying = false;
#endif
    }
}
