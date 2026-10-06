using System;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Rendering;

/// <summary>A finite-volume fluid puzzle. Scores measure the solver's actual particles.</summary>
[DefaultExecutionOrder(-200)]
public sealed class LiquidRelayGame : MonoBehaviour
{
    public enum RoundState { Ready, Running, Won, Lost }

    [Serializable]
    public sealed class Level
    {
        public string title, instruction;
        [Range(0, 1)] public float leftTarget, rightTarget;
        public float initialAngle, demoAngle, demoSwitchTime = -1, demoSecondAngle;
        public float timeLimit = 12;
        public bool stagedSequence;
        [Range(0, 1)] public float firstLeftTarget = .30f, firstRightLimit = .14f;
    }

    public DfsphProvider provider;
    public FluidSceneMVP fluidRenderer;
    public TextAsset solverScene;
    public Transform gate, deflector;
    public bool keyboardControls = true;
    public bool autoplayOnStart;
    [Range(.1f, 2)] public float simulationSpeed = .42f;
    public float angleSpeed = 84;
    public float gateOpenSeconds = .90f;
    public float scoreInterval = .075f;
    public Level[] levels = DefaultLevels();

    public static readonly Vector3 GateCenter = new Vector3(1.17f, 2.15f, 1.5f);
    public static readonly Vector3 DeflectorCenter = new Vector3(1.77f, 1.06f, 1.5f);
    public static readonly Vector3 FluidStart = new Vector3(.12f, 1.80f, .94f);
    public static readonly Vector3 FluidEnd = new Vector3(1.04f, 2.72f, 2.06f);
    public const float MaxAngle = 32;

    public RoundState State { get; private set; } = RoundState.Ready;
    public int LevelIndex { get; private set; }
    public int LevelCount => levels == null ? 0 : levels.Length;
    public string LevelTitle => Current.title;
    public string LevelInstruction => !Current.stagedSequence ? Current.instruction
        : Phase == 1 ? $"First fill A to {Mathf.RoundToInt(Current.firstLeftTarget * 100)}% before B reaches {Mathf.RoundToInt(Current.firstRightLimit * 100)}%."
        : $"Now steer toward B. Finish with at least {Mathf.RoundToInt(Current.leftTarget * 100)}% in A and {Mathf.RoundToInt(Current.rightTarget * 100)}% in B.";
    public int Phase { get; private set; } = 1;
    public string PhaseTitle => !Current.stagedSequence ? "" : Phase == 1 ? "01 / A FIRST" : "02 / SHARE THE CHARGE";
    public string FailureReason { get; private set; } = "";
    public float LeftFill01 { get; private set; }
    public float RightFill01 { get; private set; }
    public float Waste01 { get; private set; }
    public float Remaining01 => Mathf.Clamp01(1 - LeftFill01 - RightFill01 - Waste01);
    public float LeftTarget01 => Current.stagedSequence && Phase == 1 ? Current.firstLeftTarget : Current.leftTarget;
    public float RightTarget01 => Current.rightTarget;
    public float RightLimit01 => Current.stagedSequence && Phase == 1 ? Current.firstRightLimit : 0;
    public float DeflectorAngle { get; private set; }
    public float ElapsedSeconds => State == RoundState.Ready ? 0 : Mathf.Max(0, Clock - releaseClock);
    public float TimeLimit => Current.timeLimit;
    public float GateFraction { get; private set; }
    public bool Autopilot { get; private set; }
    public int InitialParticles { get; private set; }
    public int ScoredParticles { get; private set; }
    public int ScoreSampleCount { get; private set; }
    public int RoundNumber { get; private set; }
    public string LastResult { get; private set; } = "";
    public event Action StateChanged;

    Level Current => levels[Mathf.Clamp(LevelIndex, 0, levels.Length - 1)];
    float Clock => provider != null ? provider.SolverTime : 0;
    float resetClock, releaseClock, nextScoreAt, successSince = -1, settledSince = -1;
    float previousLeft, previousRight, previousWaste;
    float desiredAngle;
    bool ready, readbackPending, resetting;
    bool previousRunInBackground;
    int previousTargetFrameRate;
    int generation;

    public static Level[] DefaultLevels() => new[]
    {
        new Level { title = "FIRST TRANSFER", instruction = "Catch 60% of the charge in receiver A.", leftTarget = .60f, demoAngle = -28 },
        new Level { title = "CHANGE OF COURSE", instruction = "Catch 60% of the charge in receiver B.", rightTarget = .60f, demoAngle = 28 },
        new Level { title = "BALANCING ACT", instruction = "Fill A first, then steer toward B.", leftTarget = .35f, rightTarget = .35f,
            stagedSequence = true, firstLeftTarget = .30f, firstRightLimit = .14f, demoAngle = -28, demoSecondAngle = 28 }
    };

    public static Vector3 SimToWorld(Vector3 p) => new Vector3(p.x - 1.5f, p.y, 1.5f - p.z);
    public Vector3 SimPointToWorld(Vector3 p) => provider != null ? provider.transform.TransformPoint(SimToWorld(p)) : SimToWorld(p);
    public static Quaternion DeflectorRotation(float angle) => Quaternion.AngleAxis(-25, Vector3.forward) * Quaternion.AngleAxis(-angle, Vector3.right);

    void Awake()
    {
        previousRunInBackground = Application.runInBackground;
        previousTargetFrameRate = Application.targetFrameRate;
        Application.runInBackground = true;
        Application.targetFrameRate = 60;
        if (levels == null || levels.Length == 0) levels = DefaultLevels();
        if (provider == null) provider = GetComponent<DfsphProvider>();
        if (fluidRenderer == null) fluidRenderer = GetComponent<FluidSceneMVP>();
        if (solverScene != null && provider != null) provider.sceneAsset = solverScene;
        if (fluidRenderer != null) fluidRenderer.speed = simulationSpeed;
        ApplyPose(0, Current.initialAngle);
    }

    void Start()
    {
        if (provider == null || gate == null || deflector == null)
        {
            Debug.LogError("Liquid Relay: provider, gate and deflector must be assigned.");
            enabled = false;
            return;
        }
        InitialParticles = provider.Solver.Count;
        ready = true;
        ResetLevel();
        if (autoplayOnStart || Array.IndexOf(Environment.GetCommandLineArgs(), "--relay-autoplay") >= 0) StartAutopilot();
    }

    void Update()
    {
        if (!ready || resetting) return;
        if (keyboardControls) ReadKeyboard();
        if (fluidRenderer != null && fluidRenderer.Paused) return;

        if (Autopilot && (State == RoundState.Ready || State == RoundState.Running))
        {
            desiredAngle = Current.demoAngle;
            if (State == RoundState.Ready && Clock - resetClock >= .75f) Release();
            if (State == RoundState.Running && ((Current.stagedSequence && Phase >= 2)
                || (!Current.stagedSequence && Current.demoSwitchTime >= 0 && ElapsedSeconds >= Current.demoSwitchTime)))
                desiredAngle = Current.demoSecondAngle;
        }

        // Transform motion is smooth and bounded. The DFSPH provider derives its kinematic velocities.
        float movementDt = Mathf.Min(Time.deltaTime, .05f) * simulationSpeed;
        if (State == RoundState.Ready || State == RoundState.Running)
            DeflectorAngle = Mathf.MoveTowards(DeflectorAngle, desiredAngle, angleSpeed * movementDt);
        float open = State == RoundState.Ready ? 0 : Mathf.SmoothStep(0, 1, ElapsedSeconds / Mathf.Max(.1f, gateOpenSeconds));
        ApplyPose(open, DeflectorAngle);

        if (Clock >= nextScoreAt && !readbackPending)
        {
            nextScoreAt = Clock + scoreInterval;
            RequestScore();
        }
        if (State == RoundState.Running && ElapsedSeconds >= Current.timeLimit) Finish(false, "Time is up. Retry with a different tilt.");
    }

    void ReadKeyboard()
    {
        var k = Keyboard.current;
        if (k == null) return;
        float dir = (k.dKey.isPressed || k.rightArrowKey.isPressed ? 1 : 0) - (k.aKey.isPressed || k.leftArrowKey.isPressed ? 1 : 0);
        if (dir != 0) SetDeflectorAngle(desiredAngle + dir * angleSpeed * Time.deltaTime);
        if (k.spaceKey.wasPressedThisFrame) Release();
        if (k.rKey.wasPressedThisFrame) ResetLevel();
        if (k.nKey.wasPressedThisFrame) NextLevel();
        if (k.hKey.wasPressedThisFrame) ToggleAutopilot();
        if (k.cKey.wasPressedThisFrame && fluidRenderer != null) fluidRenderer.ToggleSurfaceSource();
        if (k.digit1Key.wasPressedThisFrame) SelectLevel(0);
        if (k.digit2Key.wasPressedThisFrame) SelectLevel(1);
        if (k.digit3Key.wasPressedThisFrame) SelectLevel(2);
        if (k.escapeKey.wasPressedThisFrame && !Application.isEditor) Application.Quit();
    }

    void ApplyPose(float open, float angle)
    {
        GateFraction = Mathf.Clamp01(open);
        if (gate != null) gate.position = SimPointToWorld(GateCenter + Vector3.up * (1.62f * GateFraction));
        if (deflector != null)
        {
            deflector.position = SimPointToWorld(DeflectorCenter);
            deflector.rotation = (provider != null ? provider.transform.rotation : Quaternion.identity) * DeflectorRotation(angle);
        }
    }

    public void SetDeflectorAngle(float degrees)
    {
        if (State == RoundState.Won || State == RoundState.Lost) return;
        Autopilot = false;
        desiredAngle = Mathf.Clamp(degrees, -MaxAngle, MaxAngle);
    }

    public void Release()
    {
        if (!ready || State != RoundState.Ready) return;
        State = RoundState.Running;
        releaseClock = Clock;
        successSince = -1;
        StateChanged?.Invoke();
    }

    public void ResetLevel()
    {
        if (!ready) return;
        resetting = true;
        generation++;
        RoundNumber++;
        Autopilot = false;
        State = RoundState.Ready;
        LeftFill01 = RightFill01 = Waste01 = 0;
        ScoreSampleCount = ScoredParticles = 0;
        LastResult = "";
        FailureReason = "";
        Phase = 1;
        successSince = -1;
        settledSince = -1;
        previousLeft = previousRight = previousWaste = 0;
        desiredAngle = DeflectorAngle = Current.initialAngle;
        ApplyPose(0, DeflectorAngle);
        provider.SyncPropsAfterTeleport();
        if (fluidRenderer != null)
        {
            if (fluidRenderer.Paused) fluidRenderer.TogglePause();
            fluidRenderer.speed = simulationSpeed;
            fluidRenderer.ResetSim();
        }
        else provider.ResetSim();
        InitialParticles = provider.Solver.Count;
        resetClock = Clock;
        releaseClock = 0;
        nextScoreAt = .10f;
        resetting = false;
        StateChanged?.Invoke();
    }

    public void SelectLevel(int index)
    {
        LevelIndex = Mathf.Clamp(index, 0, LevelCount - 1);
        ResetLevel();
    }

    public void NextLevel() => SelectLevel((LevelIndex + 1) % LevelCount);
    public void ToggleAutopilot() { if (Autopilot) Autopilot = false; else StartAutopilot(LevelIndex); StateChanged?.Invoke(); }
    public void StartAutopilot(int level = 0)
    {
        SelectLevel(level);
        Autopilot = true;
        desiredAngle = DeflectorAngle = Current.demoAngle;
        ApplyPose(0, DeflectorAngle);
        provider.SyncPropsAfterTeleport();
        StateChanged?.Invoke();
    }

    void RequestScore()
    {
        int n = provider.Solver.Count;
        if (n == 0) return;
        readbackPending = true;
        int gen = generation;
        // Only particle positions are read asynchronously, at the scoring interval.
        AsyncGPUReadback.Request(provider.PositionBuffer, n * 12, 0, request =>
        {
            readbackPending = false;
            if (this == null || gen != generation || request.hasError || !ready) return;
            var positions = request.GetData<Vector3>();
            int left = 0, right = 0, waste = 0;
            for (int i = 0; i < n; i++)
            {
                Vector3 p = positions[i];
                if (p.y < -.05f || p.x < -.05f || p.x > 3.05f || p.z < -.05f || p.z > 3.05f) { waste++; continue; }
                if (p.y > .62f) continue;
                if (p.x >= 1.55f && p.x <= 3.05f)
                {
                    if (p.z <= 1.42f && p.z >= -.05f) left++;
                    else if (p.z >= 1.58f && p.z <= 3.05f) right++;
                }
                else if (p.x < 1.39f) waste++;
            }
            float inv = 1f / Mathf.Max(1, InitialParticles);
            LeftFill01 = left * inv; RightFill01 = right * inv; Waste01 = waste * inv;
            ScoredParticles = left + right + waste;
            ScoreSampleCount++;
            if (State != RoundState.Running) return;
            if (Current.stagedSequence && Phase == 1)
            {
                if (RightFill01 >= Current.firstRightLimit)
                {
                    Finish(false, $"B filled too soon. Fill A to {Mathf.RoundToInt(Current.firstLeftTarget * 100)}% before steering toward B.");
                    return;
                }
                if (LeftFill01 >= Current.firstLeftTarget)
                {
                    Phase = 2;
                    StateChanged?.Invoke();
                }
            }
            bool met = (!Current.stagedSequence || Phase >= 2)
                && LeftFill01 + .002f >= LeftTarget01 && RightFill01 + .002f >= RightTarget01;
            if (met)
            {
                if (successSince < 0) successSince = Clock;
                if (Clock - successSince >= .65f) Finish(true);
            }
            else successSince = -1;
            // Once nearly all fluid has landed and the counts stop changing, give an immediate retry.
            float change = Mathf.Abs(LeftFill01 - previousLeft) + Mathf.Abs(RightFill01 - previousRight) + Mathf.Abs(Waste01 - previousWaste);
            previousLeft = LeftFill01; previousRight = RightFill01; previousWaste = Waste01;
            if (!met && ElapsedSeconds >= 3.5f && Remaining01 < .08f && change < .006f)
            {
                if (settledSince < 0) settledSince = Clock;
                if (Clock - settledSince >= 1.15f) Finish(false, "The charge has landed below the targets. Try a stronger tilt.");
            }
            else settledSince = -1;
        });
    }

    void Finish(bool won, string reason = "")
    {
        if (State != RoundState.Running) return;
        State = won ? RoundState.Won : RoundState.Lost;
        Autopilot = false;
        FailureReason = won ? "" : reason;
        LastResult = $"{State}: A {LeftFill01:P1}, B {RightFill01:P1}, waste {Waste01:P1}; phase {Phase}, {ElapsedSeconds:F2}s, {InitialParticles} particles. {FailureReason}";
        Debug.Log($"LiquidRelay round {RoundNumber} level {LevelIndex + 1}: {LastResult}");
        if (fluidRenderer != null && !fluidRenderer.Paused) fluidRenderer.TogglePause();
        StateChanged?.Invoke();
    }

    void OnDestroy()
    {
        generation++; ready = false;
        Application.runInBackground = previousRunInBackground;
        Application.targetFrameRate = previousTargetFrameRate;
    }
}
