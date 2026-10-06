using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

/// <summary>Repeatable integration snapshots and a scene-only player build for the demo.</summary>
public static class LiquidRelayQA
{
    [Serializable]
    public sealed class Snapshot
    {
        public string timestampUtc, scene, state, level, model, renderSource, solverStatus;
        public bool rendererReady, playing, paused, autopilot;
        public int levelIndex, round, particles, invalidParticles, outsideTank, scoreSamples;
        public float solverSeconds, elapsed, left, right, waste, remaining, angle, gate;
        public float rendererMainThreadMs, observedFrameMs;
        public Vector3 boundsMin, boundsMax, meanPosition, meanVelocity;
        public int sourceParticles, belowFloorParticles, airborneParticles;
    }

    public static string Capture(string path = null)
    {
        var game = UnityEngine.Object.FindFirstObjectByType<LiquidRelayGame>();
        if (game == null) throw new InvalidOperationException("Open the Liquid Relay scene first.");
        if (!EditorApplication.isPlaying) throw new InvalidOperationException("Enter Play mode before capturing solver state.");
        var p = game.provider;
        var f = game.fluidRenderer;
        var s = new Snapshot
        {
            timestampUtc = DateTime.UtcNow.ToString("o"),
            scene = game.gameObject.scene.path, state = game.State.ToString(), level = game.LevelTitle,
            levelIndex = game.LevelIndex, round = game.RoundNumber, particles = p.ActiveParticles,
            playing = EditorApplication.isPlaying, paused = f.Paused, rendererReady = f.Ready,
            model = f.ActiveModelName, renderSource = f.showRawInput ? "Raw" : f.surfaceSource.ToString(),
            solverStatus = p.StatusLine(), solverSeconds = p.SolverTime, elapsed = game.ElapsedSeconds,
            left = game.LeftFill01, right = game.RightFill01, waste = game.Waste01, remaining = game.Remaining01,
            angle = game.DeflectorAngle, gate = game.GateFraction, autopilot = game.Autopilot,
            scoreSamples = game.ScoreSampleCount, rendererMainThreadMs = f.LateUpdateMs,
            observedFrameMs = Time.unscaledDeltaTime * 1000,
            boundsMin = Vector3.one * float.MaxValue, boundsMax = Vector3.one * float.MinValue
        };
        p.GetFrame(0, out var records, out var offset, out var count);
        int finite = 0;
        for (int i = 0; i < count; i++)
        {
            int b = offset + 7 * i;
            var position = new Vector3(records[b], records[b + 1], records[b + 2]);
            var velocity = new Vector3(records[b + 3], records[b + 4], records[b + 5]);
            if (!Finite(position) || !Finite(velocity)) { s.invalidParticles++; continue; }
            finite++;
            s.boundsMin = Vector3.Min(s.boundsMin, position); s.boundsMax = Vector3.Max(s.boundsMax, position);
            s.meanPosition += position; s.meanVelocity += velocity;
            if (position.x < -.1f || position.x > 3.1f || position.y < -.1f || position.y > 3.1f || position.z < -.1f || position.z > 3.1f) s.outsideTank++;
            if (position.y < -.05f) s.belowFloorParticles++;
            if (position.x < 1.1f && position.y > 1.65f) s.sourceParticles++;
            if (position.y > .65f && position.y < 1.60f) s.airborneParticles++;
        }
        if (finite > 0) { s.meanPosition /= finite; s.meanVelocity /= finite; }
        string json = JsonUtility.ToJson(s, true);
        if (!string.IsNullOrEmpty(path))
        {
            string dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
            File.WriteAllText(path, json);
        }
        return json;
    }

    static bool Finite(Vector3 v) => float.IsFinite(v.x) && float.IsFinite(v.y) && float.IsFinite(v.z);

    public static string BuildLinux(string output)
    {
        if (EditorApplication.isPlaying) throw new InvalidOperationException("Stop Play mode before building.");
        const string scene = "Assets/LiquidRelay/Scenes/LiquidRelay.unity";
        if (!File.Exists(scene)) throw new FileNotFoundException("Build the demo scene first.", scene);
        Directory.CreateDirectory(Path.GetDirectoryName(output));
        string oldName = PlayerSettings.productName;
        int oldWidth = PlayerSettings.defaultScreenWidth, oldHeight = PlayerSettings.defaultScreenHeight;
        var oldFullscreen = PlayerSettings.fullScreenMode;
        bool oldResizable = PlayerSettings.resizableWindow;
        try
        {
            PlayerSettings.productName = "Liquid Relay";
            PlayerSettings.defaultScreenWidth = 1600;
            PlayerSettings.defaultScreenHeight = 900;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.resizableWindow = true;
            var report = BuildPipeline.BuildPlayer(new[] { scene }, output, BuildTarget.StandaloneLinux64,
                BuildOptions.CompressWithLz4 | BuildOptions.StrictMode | BuildOptions.DetailedBuildReport);
            var summary = report.summary;
            string result = $"{summary.result}: {summary.totalErrors} errors, {summary.totalWarnings} warnings, {summary.totalSize} bytes, {summary.totalTime.TotalSeconds:F1}s";
            File.WriteAllText(Path.Combine(Path.GetDirectoryName(output), "build-result.txt"), result);
            if (summary.result != BuildResult.Succeeded) throw new InvalidOperationException(result);
            return result;
        }
        finally
        {
            PlayerSettings.productName = oldName;
            PlayerSettings.defaultScreenWidth = oldWidth;
            PlayerSettings.defaultScreenHeight = oldHeight;
            PlayerSettings.fullScreenMode = oldFullscreen;
            PlayerSettings.resizableWindow = oldResizable;
        }
    }
}
