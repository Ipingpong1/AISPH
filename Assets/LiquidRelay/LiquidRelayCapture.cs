using System;
using System.Collections;
using System.IO;
using UnityEngine;

/// <summary>Captures the composed game frame including the HUD; used by the demo's verification tools.</summary>
public sealed class LiquidRelayCapture : MonoBehaviour
{
    public bool Busy { get; private set; }
    public string LastPath { get; private set; }
    int previousCaptureRate;
    bool capturingSequence;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void BootstrapFromCommandLine()
    {
        string[] args = Environment.GetCommandLineArgs();
        int index = Array.IndexOf(args, "--relay-capture");
        if (index < 0 || index + 1 >= args.Length) return;
        var capture = new GameObject("Liquid Relay Capture").AddComponent<LiquidRelayCapture>();
        capture.StartCoroutine(capture.CommandLineCapture(args[index + 1], args));
    }

    IEnumerator CommandLineCapture(string folder, string[] args)
    {
        float deadline = Time.realtimeSinceStartup + 70;
        LiquidRelayGame game = null;
        while (Time.realtimeSinceStartup < deadline)
        {
            game = FindFirstObjectByType<LiquidRelayGame>();
            if (game != null && game.InitialParticles > 0 && game.fluidRenderer.Ready) break;
            yield return null;
        }
        bool quit = Array.IndexOf(args, "--relay-quit-after-capture") >= 0;
        if (game == null || game.InitialParticles == 0 || !game.fluidRenderer.Ready)
        {
            Debug.LogError("Liquid Relay capture: renderer startup timed out.");
            if (quit && !Application.isEditor) Application.Quit(1);
            yield break;
        }
        yield return SequenceRoutine(folder, 360, 24, 2);
        if (quit && !Application.isEditor) Application.Quit(0);
    }

    public void FreezeAt(float simulationSeconds, int level, string path)
    {
        if (Busy) return;
        StartCoroutine(FreezeRoutine(simulationSeconds, level, path));
    }

    IEnumerator FreezeRoutine(float seconds, int level, string path)
    {
        Busy = true;
        var game = FindFirstObjectByType<LiquidRelayGame>();
        game.StartAutopilot(level);
        float deadline = Time.realtimeSinceStartup + 70;
        while (game.ElapsedSeconds < seconds && Time.realtimeSinceStartup < deadline
            && (game.State == LiquidRelayGame.RoundState.Ready || game.State == LiquidRelayGame.RoundState.Running))
            yield return null;
        if (!game.fluidRenderer.Paused) game.fluidRenderer.TogglePause();
        yield return StillRoutine(path);
    }

    public void Still(string path)
    {
        if (Busy) return;
        StartCoroutine(StillRoutine(path));
    }

    IEnumerator StillRoutine(string path)
    {
        Busy = true;
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        // Let changed UI text, material variants and the fluid composite reach the camera.
        for (int i = 0; i < 4; i++) yield return null;
        yield return new WaitForEndOfFrame();
        ScreenCapture.CaptureScreenshot(path);
        LastPath = path;
        yield return null;
        yield return null;
        Busy = false;
    }

    public void Sequence(string folder, int frames = 240, int fps = 24, int level = 2)
    {
        if (Busy) return;
        StartCoroutine(SequenceRoutine(folder, frames, fps, level));
    }

    IEnumerator SequenceRoutine(string folder, int frames, int fps, int level)
    {
        Busy = true;
        Directory.CreateDirectory(folder);
        var game = GetComponent<LiquidRelayGame>();
        if (game == null) game = FindFirstObjectByType<LiquidRelayGame>();
        previousCaptureRate = Time.captureFramerate;
        capturingSequence = true;
        Time.captureFramerate = fps;
        game.StartAutopilot(level);
        // Let the level text and UI transition tints settle before the first frame.
        for (int i = 0; i < 8; i++) yield return null;
        for (int i = 0; i < frames; i++)
        {
            yield return new WaitForEndOfFrame();
            string path = Path.Combine(folder, $"frame_{i:D5}.png");
            ScreenCapture.CaptureScreenshot(path);
            LastPath = path;
        }
        yield return null;
        yield return null;
        Time.captureFramerate = previousCaptureRate;
        capturingSequence = false;
        Busy = false;
        Debug.Log($"Liquid Relay capture: {frames} frames at {fps}fps -> {folder}");
    }

    void OnDisable()
    {
        if (capturingSequence) Time.captureFramerate = previousCaptureRate;
        capturingSequence = false;
        Busy = false;
    }
}
