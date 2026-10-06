using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Runtime integration acceptance for the real GPU solver and the playable demo.
/// Add this component in Play mode and call Run with a report file or directory.
/// No mock physics, score injection, editor APIs, or screenshot capture is used.
/// </summary>
public sealed class LiquidRelayAcceptance : MonoBehaviour
{
    [Serializable]
    public sealed class ParticleSample
    {
        public string timestampUtc, label, state, renderSource, failureReason;
        public int levelIndex, phase, round, particles, initialParticles, scoreSamples;
        public int invalidParticles, outsideDomainParticles, escapedReservoirParticles;
        public int leftParticles, rightParticles, wasteParticles;
        public bool paused, rendererReady;
        public float wallSeconds, solverSeconds, elapsedSeconds, left, right, waste, gate, angle;
        public float maxSpeed, gpuLeft, gpuRight, gpuWaste;
        public Vector3 boundsMin, boundsMax;
    }

    [Serializable]
    public sealed class CaseResult
    {
        public string name, startedUtc, finishedUtc, expected, status = "running";
        public float wallSeconds;
        public List<string> failures = new List<string>();
        public List<ParticleSample> samples = new List<ParticleSample>();
    }

    [Serializable]
    public sealed class AcceptanceReport
    {
        public string startedUtc, finishedUtc, status = "running", scene, model, fatalError;
        public int initialParticles, passedCases, failedCases;
        public float simulationSpeed, angleSpeed, caseTimeoutSeconds;
        public List<CaseResult> cases = new List<CaseResult>();
    }

    public LiquidRelayGame game;
    public float caseTimeoutSeconds = 70;
    public bool Busy { get; private set; }
    public bool Done { get; private set; }
    public bool Passed => Done && report != null && report.status == "passed";
    public string LastReportPath { get; private set; }
    public string CurrentCase => activeCase == null ? "" : activeCase.name;

    AcceptanceReport report;
    CaseResult activeCase;
    double caseStarted, runStarted;
    bool previousKeyboard, previousRendererKeyboard;
    bool quitAfterCompletion;
    FluidSceneMVP.SurfaceSource? restoreSourceAfterCase;
    Vector3[] positions, velocities;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void BootstrapFromCommandLine()
    {
        string[] args = Environment.GetCommandLineArgs();
        if (Array.IndexOf(args, "--relay-acceptance") < 0) return;
        string output = Path.Combine(Application.persistentDataPath, "LiquidRelayAcceptance", "acceptance-report.json");
        int reportArgument = Array.IndexOf(args, "--relay-report");
        if (reportArgument >= 0 && reportArgument + 1 < args.Length) output = args[reportArgument + 1];
        var runner = new GameObject("Liquid Relay Acceptance (command line)").AddComponent<LiquidRelayAcceptance>();
        runner.quitAfterCompletion = Array.IndexOf(args, "--relay-quit-after-acceptance") >= 0;
        runner.StartCoroutine(runner.StartRequestedRun(output));
    }

    IEnumerator StartRequestedRun(string output)
    {
        double deadline = Time.realtimeSinceStartupAsDouble + caseTimeoutSeconds;
        while (Time.realtimeSinceStartupAsDouble < deadline)
        {
            if (game == null) game = FindAnyObjectByType<LiquidRelayGame>();
            if (game != null && game.provider != null && game.fluidRenderer != null
                && game.InitialParticles > 0 && game.fluidRenderer.Ready)
            {
                Run(output);
                yield break;
            }
            yield return null;
        }
        LastReportPath = Path.GetFullPath(output.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
            ? output : Path.Combine(output, "acceptance-report.json"));
        Directory.CreateDirectory(Path.GetDirectoryName(LastReportPath));
        report = new AcceptanceReport
        {
            startedUtc = DateTime.UtcNow.ToString("o"), finishedUtc = DateTime.UtcNow.ToString("o"),
            status = "failed", fatalError = "Game, GPU particles, or renderer did not become ready within the startup timeout."
        };
        Done = true;
        Flush();
        Debug.LogError("LiquidRelay acceptance startup failed -> " + LastReportPath);
        if (quitAfterCompletion && !Application.isEditor) Application.Quit(1);
    }

    public void Run(string outputPath)
    {
        if (Busy) throw new InvalidOperationException("Liquid Relay acceptance is already running.");
        if (!Application.isPlaying) throw new InvalidOperationException("Run acceptance in Play mode.");
        if (game == null) game = GetComponent<LiquidRelayGame>();
        if (game == null) game = FindAnyObjectByType<LiquidRelayGame>();
        if (game == null || game.provider == null || game.fluidRenderer == null)
            throw new InvalidOperationException("Liquid Relay game, GPU provider and renderer are required.");
        if (string.IsNullOrWhiteSpace(outputPath))
            outputPath = Path.Combine(Application.persistentDataPath, "LiquidRelayAcceptance");
        LastReportPath = Path.GetFullPath(outputPath.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
            ? outputPath : Path.Combine(outputPath, "acceptance-report.json"));
        Directory.CreateDirectory(Path.GetDirectoryName(LastReportPath));
        Done = false;
        Busy = true;
        runStarted = Time.realtimeSinceStartupAsDouble;
        previousKeyboard = game.keyboardControls;
        previousRendererKeyboard = game.fluidRenderer.keyboardShortcuts;
        game.keyboardControls = false;
        game.fluidRenderer.keyboardShortcuts = false;
        report = new AcceptanceReport
        {
            startedUtc = DateTime.UtcNow.ToString("o"),
            scene = game.gameObject.scene.path,
            model = game.fluidRenderer.ActiveModelName,
            initialParticles = game.provider.Solver.Count,
            simulationSpeed = game.simulationSpeed,
            angleSpeed = game.angleSpeed,
            caseTimeoutSeconds = caseTimeoutSeconds
        };
        Flush();
        StartCoroutine(Guarded(Suite()));
    }

    // Flatten nested enumerators so failures in any case are recorded, not silently lost.
    IEnumerator Guarded(IEnumerator suite)
    {
        var stack = new Stack<IEnumerator>();
        stack.Push(suite);
        while (stack.Count > 0)
        {
            object next = null;
            bool moved = false;
            Exception error = null;
            try
            {
                moved = stack.Peek().MoveNext();
                if (moved) next = stack.Peek().Current;
            }
            catch (Exception ex) { error = ex; }
            if (error != null)
            {
                report.fatalError = error.ToString();
                if (activeCase != null)
                {
                    Fail("Unhandled acceptance error: " + error.Message);
                    EndCase();
                }
                report.status = "failed";
                break;
            }
            if (!moved) { stack.Pop(); continue; }
            if (next is IEnumerator child) { stack.Push(child); continue; }
            yield return next;
        }
        RestoreCaseSurfaceSource();
        game.keyboardControls = previousKeyboard;
        game.fluidRenderer.keyboardShortcuts = previousRendererKeyboard;
        report.finishedUtc = DateTime.UtcNow.ToString("o");
        report.status = report.failedCases == 0 && string.IsNullOrEmpty(report.fatalError) ? "passed" : "failed";
        Busy = false;
        Done = true;
        Flush();
        Debug.Log($"LiquidRelay acceptance {report.status}: {report.passedCases} passed, {report.failedCases} failed -> {LastReportPath}");
        if (quitAfterCompletion && !Application.isEditor) Application.Quit(Passed ? 0 : 1);
    }

    IEnumerator Suite()
    {
        // Let all scene Start methods finish when invoked immediately after entering Play mode.
        yield return null;
        yield return null;
        BeginCase("configuration", "Three production levels, 60% A / 60% B / A-first then 35% each.");
        Check(game.LevelCount == 3, "Expected exactly three levels.");
        Check(Mathf.Abs(game.simulationSpeed - .42f) < .0001f, "Simulation speed must be 0.42.");
        Check(Mathf.Abs(game.angleSpeed - 84f) < .0001f, "Angle speed must be 84.");
        if (game.LevelCount >= 3)
        {
            Check(Near(game.levels[0].leftTarget, .60f), "Level 1 A target must be 60%.");
            Check(Near(game.levels[1].rightTarget, .60f), "Level 2 B target must be 60%.");
            var l = game.levels[2];
            Check(l.stagedSequence && Near(l.firstLeftTarget, .30f) && Near(l.firstRightLimit, .14f)
                && Near(l.leftTarget, .35f) && Near(l.rightTarget, .35f), "Level 3 must require A 30% before B 14%, then both 35%.");
        }
        Check(report.initialParticles > 0, "The GPU solver has no particles.");
        EndCase();

        for (int level = 0; level < 3 && level < game.LevelCount; level++)
        {
            BeginCase("autopilot_level_" + (level + 1), "Won with the unmodified production autopilot.");
            game.StartAutopilot(level);
            yield return WaitForResult();
            Check(game.State == LiquidRelayGame.RoundState.Won, "Autopilot did not win: " + game.LastResult);
            if (level == 2) Check(game.Phase == 2, "Staged level did not reach phase 2.");
            Check(game.fluidRenderer.Paused, "Terminal result did not pause the fluid renderer.");
            EndCase();
        }

        BeginCase("frozen_result_renderer_toggle", "Both renderer toggles preserve every particle, velocity, score and the result.");
        Check(IsTerminal(), "Expected a completed round for the renderer comparison.");
        // Allow the final asynchronous scoring callback to settle before comparing frozen frames.
        for (int i = 0; i < 6; i++) yield return null;
        var before = Sample("before_toggle");
        var frozenPositions = (Vector3[])positions.Clone();
        var frozenVelocities = (Vector3[])velocities.Clone();
        var originalSource = game.fluidRenderer.surfaceSource;
        for (int mode = 0; mode < 2; mode++)
        {
            game.fluidRenderer.ToggleSurfaceSource();
            for (int i = 0; i < 8; i++) yield return null;
            var after = Sample("toggle_" + (mode + 1));
            Check(after.paused, "Renderer toggle resumed a completed round.");
            Check(before.state == after.state && before.phase == after.phase, "Renderer toggle changed result or phase.");
            Check(Near(before.solverSeconds, after.solverSeconds), "Renderer toggle advanced solver time.");
            Check(Near(before.left, after.left) && Near(before.right, after.right) && Near(before.waste, after.waste), "Renderer toggle changed scoring.");
            Check(positions.Length == frozenPositions.Length, "Renderer toggle changed particle count.");
            int moved = 0;
            for (int i = 0; i < positions.Length && i < frozenPositions.Length; i++)
                if ((positions[i] - frozenPositions[i]).sqrMagnitude > 1e-12f || (velocities[i] - frozenVelocities[i]).sqrMagnitude > 1e-12f) moved++;
            Check(moved == 0, "Renderer toggle changed " + moved + " GPU particle positions or velocities.");
        }
        Check(game.fluidRenderer.surfaceSource.Equals(originalSource), "Two toggles did not restore the original renderer.");
        EndCase();

        BeginCase("neutral_level_1", "A neutral deflector fails the 60% A target.");
        game.SelectLevel(0);
        game.SetDeflectorAngle(0);
        yield return WaitSimulated(.75f);
        game.Release();
        yield return WaitForResult();
        Check(game.State == LiquidRelayGame.RoundState.Lost, "Neutral level 1 unexpectedly won or failed to finish.");
        EndCase();

        BeginCase("reset_after_failure", "Reset returns Ready with a closed gate, unchanged particle count, zero scores, and no leak.");
        Check(game.State == LiquidRelayGame.RoundState.Lost, "The preceding round did not fail.");
        yield return VerifyReset();
        EndCase();

        BeginCase("neutral_level_3", "B reaches 14% during phase 1 and loses the A-first sequence.");
        game.SelectLevel(2);
        game.SetDeflectorAngle(0);
        yield return WaitSimulated(.75f);
        game.Release();
        yield return WaitForResult();
        Check(game.State == LiquidRelayGame.RoundState.Lost, "Neutral level 3 unexpectedly won or failed to finish.");
        Check(game.Phase == 1, "Neutral level 3 passed the initial A-first phase.");
        Check(game.FailureReason.Contains("B filled too soon"), "Neutral level 3 did not fail for filling B first: " + game.FailureReason);
        EndCase();

        BeginCase("midpour_reset", "Reset during a live pour restores the initial closed, unscored, leak-free charge.");
        game.StartAutopilot(0);
        yield return WaitSimulated(2.10f);
        Check(game.State == LiquidRelayGame.RoundState.Running, "Midpour precondition was not Running.");
        Check(game.GateFraction > .5f, "Midpour precondition did not open the gate.");
        Sample("pour_before_reset");
        yield return VerifyReset();
        EndCase();

        BeginCase("classical_autopilot_level_3", "Win the staged puzzle while Classical rendering remains active throughout the live GPU simulation.");
        var previousSource = game.fluidRenderer.surfaceSource;
        restoreSourceAfterCase = previousSource;
        if (!game.fluidRenderer.ClassicalActive) game.fluidRenderer.ToggleSurfaceSource();
        game.StartAutopilot(2);
        yield return WaitForResult();
        Check(game.State == LiquidRelayGame.RoundState.Won, "Classical-rendered autopilot did not win: " + game.LastResult);
        Check(game.Phase == 2, "Classical-rendered staged puzzle did not reach phase 2.");
        Check(game.fluidRenderer.Paused, "Classical-rendered result did not pause the simulation.");
        Check(game.fluidRenderer.Classical != null, "The Classical surface renderer was not initialized.");
        foreach (var sample in activeCase.samples)
            Check(sample.renderSource == FluidSceneMVP.SurfaceSource.Classical.ToString() && sample.rendererReady,
                "Classical rendering was not active and ready throughout the live puzzle.");
        var classicalResult = activeCase.samples[activeCase.samples.Count - 1];
        Check(classicalResult.gpuLeft + .002f >= game.levels[2].leftTarget
            && classicalResult.gpuRight + .002f >= game.levels[2].rightTarget,
            "Final GPU particle counts did not independently meet both staged puzzle targets.");
        RestoreCaseSurfaceSource();
        Check(game.fluidRenderer.surfaceSource == previousSource, "Classical test did not restore the previous surface renderer.");
        EndCase();

        // Leave a clean, immediately playable first puzzle after the suite.
        game.SelectLevel(0);
    }

    void RestoreCaseSurfaceSource()
    {
        if (!restoreSourceAfterCase.HasValue) return;
        if (game != null && game.fluidRenderer != null && game.fluidRenderer.surfaceSource != restoreSourceAfterCase.Value)
            game.fluidRenderer.ToggleSurfaceSource();
        restoreSourceAfterCase = null;
    }

    IEnumerator VerifyReset()
    {
        game.ResetLevel();
        Check(game.State == LiquidRelayGame.RoundState.Ready, "Reset did not return Ready.");
        Check(game.Phase == 1 && !game.Autopilot, "Reset retained an advanced phase or autopilot.");
        Check(game.provider.Solver.Count == report.initialParticles && game.InitialParticles == report.initialParticles, "Reset changed the particle count.");
        Check(Near(game.GateFraction, 0), "Reset did not close the gate.");
        Check(Near(game.LeftFill01, 0) && Near(game.RightFill01, 0) && Near(game.Waste01, 0), "Reset did not clear scores immediately.");
        Check(game.ScoreSampleCount == 0 && game.ScoredParticles == 0, "Reset retained scoring counters.");
        Check(string.IsNullOrEmpty(game.FailureReason) && string.IsNullOrEmpty(game.LastResult), "Reset retained result text.");
        Check(!game.fluidRenderer.Paused, "Reset left simulation paused.");
        Sample("reset_immediate");
        yield return WaitSimulated(.50f);
        var sample = Sample("reset_after_half_sim_second");
        Check(game.State == LiquidRelayGame.RoundState.Ready && Near(game.GateFraction, 0), "Idle reset opened or released the gate.");
        Check(game.ScoreSampleCount > 0, "Reset did not resume asynchronous score sampling.");
        Check(Near(game.LeftFill01, 0) && Near(game.RightFill01, 0) && Near(game.Waste01, 0), "Reset scores increased with the gate closed.");
        Check(sample.leftParticles + sample.rightParticles + sample.wasteParticles == 0, "GPU particles reached a scoring zone with the gate closed.");
        Check(sample.escapedReservoirParticles == 0, "Particles escaped the closed reservoir after reset: " + sample.escapedReservoirParticles);
    }

    IEnumerator WaitForResult()
    {
        double nextSample = 0;
        while (!IsTerminal() && Time.realtimeSinceStartupAsDouble - caseStarted < caseTimeoutSeconds)
        {
            if (Time.realtimeSinceStartupAsDouble >= nextSample)
            {
                Sample("running");
                nextSample = Time.realtimeSinceStartupAsDouble + 1;
            }
            yield return null;
        }
        // Flush outstanding GPU commands/readbacks before the final terminal snapshot.
        for (int i = 0; i < 4; i++) yield return null;
        Sample("result");
        Check(IsTerminal(), "Round exceeded " + caseTimeoutSeconds + " wall seconds without a result.");
    }

    IEnumerator WaitSimulated(float seconds)
    {
        float start = game.provider.SolverTime;
        double deadline = caseStarted + caseTimeoutSeconds;
        while (game.provider.SolverTime - start < seconds && Time.realtimeSinceStartupAsDouble < deadline && !IsTerminal())
            yield return null;
        Check(game.provider.SolverTime - start >= seconds, "Simulation did not advance " + seconds + " seconds before its timeout or terminal result.");
    }

    ParticleSample Sample(string label)
    {
        var p = game.provider;
        int count = p.Solver.Count;
        if (positions == null || positions.Length != count)
        {
            positions = new Vector3[count];
            velocities = new Vector3[count];
        }
        // Read the live GPU buffers directly; GetFrame caching cannot hide a change.
        p.Solver.ReadParticles(positions, velocities, null);
        var s = new ParticleSample
        {
            timestampUtc = DateTime.UtcNow.ToString("o"), label = label,
            wallSeconds = (float)(Time.realtimeSinceStartupAsDouble - runStarted),
            state = game.State.ToString(), levelIndex = game.LevelIndex, phase = game.Phase, round = game.RoundNumber,
            particles = count, initialParticles = game.InitialParticles, scoreSamples = game.ScoreSampleCount,
            solverSeconds = p.SolverTime, elapsedSeconds = game.ElapsedSeconds,
            left = game.LeftFill01, right = game.RightFill01, waste = game.Waste01,
            gate = game.GateFraction, angle = game.DeflectorAngle, failureReason = game.FailureReason,
            paused = game.fluidRenderer.Paused, rendererReady = game.fluidRenderer.Ready,
            renderSource = game.fluidRenderer.surfaceSource.ToString(),
            boundsMin = Vector3.one * float.MaxValue, boundsMax = Vector3.one * float.MinValue
        };
        Vector3 lo = p.DomainMin - Vector3.one * .15f, hi = p.DomainMax + Vector3.one * .15f;
        for (int i = 0; i < count; i++)
        {
            Vector3 x = positions[i], v = velocities[i];
            if (!Finite(x) || !Finite(v)) { s.invalidParticles++; continue; }
            s.boundsMin = Vector3.Min(s.boundsMin, x);
            s.boundsMax = Vector3.Max(s.boundsMax, x);
            s.maxSpeed = Mathf.Max(s.maxSpeed, v.magnitude);
            if (x.x < lo.x || x.y < lo.y || x.z < lo.z || x.x > hi.x || x.y > hi.y || x.z > hi.z) s.outsideDomainParticles++;
            // The sloped reservoir floor reaches y ~1.61 at its low edge; its side walls are z=.70/2.30.
            if (x.x > 1.22f || x.y < 1.50f || x.z < .65f || x.z > 2.35f) s.escapedReservoirParticles++;
            if (x.y < -.05f || x.x < -.05f || x.x > 3.05f || x.z < -.05f || x.z > 3.05f) { s.wasteParticles++; continue; }
            if (x.y > .62f) continue;
            if (x.x >= 1.55f && x.x <= 3.05f)
            {
                if (x.z <= 1.42f && x.z >= -.05f) s.leftParticles++;
                else if (x.z >= 1.58f && x.z <= 3.05f) s.rightParticles++;
            }
            else if (x.x < 1.39f) s.wasteParticles++;
        }
        float inv = 1f / Mathf.Max(1, count);
        s.gpuLeft = s.leftParticles * inv; s.gpuRight = s.rightParticles * inv; s.gpuWaste = s.wasteParticles * inv;
        Check(count == report.initialParticles && count == game.InitialParticles, "GPU particle count changed in " + label + ".");
        Check(s.invalidParticles == 0, "Non-finite GPU positions/velocities in " + label + ": " + s.invalidParticles);
        Check(s.outsideDomainParticles == 0, "GPU particles outside domain +/-0.15m in " + label + ": " + s.outsideDomainParticles);
        Check(Finite(s.left) && Finite(s.right) && Finite(s.waste) && s.left >= 0 && s.right >= 0 && s.waste >= 0
            && s.left + s.right + s.waste <= 1.00001f, "Invalid score fractions in " + label + ".");
        Check(game.ScoredParticles <= count, "Scored particle count exceeds available particles.");
        activeCase.samples.Add(s);
        Flush();
        return s;
    }

    void BeginCase(string name, string expected)
    {
        activeCase = new CaseResult { name = name, expected = expected, startedUtc = DateTime.UtcNow.ToString("o") };
        report.cases.Add(activeCase);
        caseStarted = Time.realtimeSinceStartupAsDouble;
        Flush();
    }

    void EndCase()
    {
        activeCase.finishedUtc = DateTime.UtcNow.ToString("o");
        activeCase.wallSeconds = (float)(Time.realtimeSinceStartupAsDouble - caseStarted);
        activeCase.status = activeCase.failures.Count == 0 ? "passed" : "failed";
        if (activeCase.failures.Count == 0) report.passedCases++; else report.failedCases++;
        string casePath = Path.Combine(Path.GetDirectoryName(LastReportPath), activeCase.name + ".json");
        File.WriteAllText(casePath, JsonUtility.ToJson(activeCase, true));
        Debug.Log($"LiquidRelay acceptance {activeCase.name}: {activeCase.status} ({activeCase.wallSeconds:F1}s)");
        activeCase = null;
        Flush();
    }

    void Check(bool condition, string message) { if (!condition) Fail(message); }
    void Fail(string message)
    {
        if (activeCase != null && !activeCase.failures.Contains(message)) activeCase.failures.Add(message);
    }
    bool IsTerminal() => game.State == LiquidRelayGame.RoundState.Won || game.State == LiquidRelayGame.RoundState.Lost;
    static bool Near(float a, float b) => Mathf.Abs(a - b) <= .00001f;
    static bool Finite(float x) => !float.IsNaN(x) && !float.IsInfinity(x);
    static bool Finite(Vector3 x) => Finite(x.x) && Finite(x.y) && Finite(x.z);
    void Flush()
    {
        string temp = LastReportPath + ".tmp";
        File.WriteAllText(temp, JsonUtility.ToJson(report, true));
        File.Copy(temp, LastReportPath, true);
        File.Delete(temp);
    }
}
