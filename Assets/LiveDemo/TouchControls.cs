// TouchControls.cs — phone/tablet controls for the live demo. iOS has no Mouse or Keyboard device,
// so the bindings in FluidSceneMVP / MouseStirrer / ObstacleSpawner / GpuSphProvider never fire;
// this drives the same public hooks from touches.
//
//   1-finger drag   stir (the MouseStirrer sphere follows the finger on its height plane)
//   2-finger drag   orbit the camera around the fluid          pinch   zoom
//   buttons         Drop / Box / Ball / Undo / Stir up / Stir down / Raw / Pause / Foam / Reset / Sync|Spread N
//
// Box / Ball arm a placement: the next 1-finger tap drops the obstacle there instead of stirring.
// Buttons are hit-tested here from touch positions (not IMGUI events), so a finger that lands on a
// button never also stirs, and a finger that was part of a 2-finger gesture never stirs either.
// Created automatically on phones/tablets (no scene edit needed). To try it in the editor, add it
// by hand and tick Enable In Editor: the mouse becomes one simulated finger (no orbit/pinch).
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem.EnhancedTouch;
using ETouch = UnityEngine.InputSystem.EnhancedTouch.Touch;

public class TouchControls : MonoBehaviour
{
    public FluidSceneMVP scene;
    public MouseStirrer stirrer;
    public ObstacleSpawner spawner;
    public GpuSphProvider provider;
    [Tooltip("Also run in the editor / desktop players (the mouse becomes one simulated finger; no orbit/pinch).")]
    public bool enableInEditor = false;
    [Tooltip("Orbit speed, degrees per point of two-finger drag.")]
    public float orbitDegPerPt = 0.35f;
    [Tooltip("Pinch-zoom limits: camera distance to the orbit pivot (world units).")]
    public float minDistance = 1.2f, maxDistance = 14f;
    [Tooltip("Camera elevation limits above the pivot (degrees).")]
    public float minElevation = 4f, maxElevation = 85f;
    [Tooltip("Orbit pivot height above the fluid anchor, in anchor-scale units.")]
    public float pivotHeight = 0.5f;
    [Tooltip("Stirrer height change per Stir up / Stir down tap (world units).")]
    public float heightStep = 0.15f;

    static readonly int[] SpreadCycle = { 0, 2, 3, 4, 6 };
    const float BtnW = 92f, BtnH = 44f, Gap = 6f, Margin = 10f;   // logical points

    enum Arm { None, Box, Ball }
    struct Btn { public string label; public System.Action act; public System.Func<bool> on; }

    Btn[] buttons;
    Rect[] rects;                                   // GUI space: pixels, top-left origin
    GUIStyle style;
    Arm armed;
    readonly HashSet<int> seen = new HashSet<int>();       // touch ids already handled as "began"
    readonly HashSet<int> consumed = new HashSet<int>();   // fingers that hit a button / placed an obstacle
    bool multiGesture;                              // a 2-finger gesture happened since all fingers were up
    bool touchEnabled, simEnabled;
    float s = 1f;

    bool Active => Application.isMobilePlatform || enableInEditor;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void AutoCreate()
    {
        if (!Application.isMobilePlatform || FindAnyObjectByType<TouchControls>() != null) return;
        if (FindAnyObjectByType<FluidSceneMVP>() == null) return;
        new GameObject("TouchControls (auto)").AddComponent<TouchControls>();
    }

    /// <summary>IMGUI scale: 1 on desktop, ~2x on a phone so text is readable.</summary>
    public static float UiScale =>
        Application.isMobilePlatform ? Mathf.Max(1f, Mathf.Min(Screen.width, Screen.height) / 540f) : 1f;

    /// <summary>Set GUI.matrix to UiScale, offset into the safe area (notch / Dynamic Island). Returns the scale.</summary>
    public static float BeginScaledGUI()
    {
        float sc = UiScale;
        Rect sa = Screen.safeArea;
        GUI.matrix = Matrix4x4.TRS(new Vector3(sa.xMin, Screen.height - sa.yMax, 0f), Quaternion.identity, new Vector3(sc, sc, 1f));
        return sc;
    }

    void OnEnable()
    {
        if (!Active) return;
        EnhancedTouchSupport.Enable(); touchEnabled = true;
        if (!Application.isMobilePlatform) { TouchSimulation.Enable(); simEnabled = true; }
    }

    void OnDisable()
    {
        if (simEnabled) { TouchSimulation.Disable(); simEnabled = false; }
        if (touchEnabled) { EnhancedTouchSupport.Disable(); touchEnabled = false; }
        if (stirrer != null) stirrer.mouseInput = true;
    }

    void Start()
    {
        if (!Active) { enabled = false; return; }
        if (scene == null) scene = FindAnyObjectByType<FluidSceneMVP>();
        if (stirrer == null) stirrer = FindAnyObjectByType<MouseStirrer>();
        if (spawner == null) spawner = FindAnyObjectByType<ObstacleSpawner>();
        if (provider == null) provider = FindAnyObjectByType<GpuSphProvider>();
        if (stirrer != null) stirrer.mouseInput = false;    // the (simulated) finger drives it now
        if (spawner != null) spawner.showHelp = false;      // keyboard legend is meaningless here

        buttons = new[]
        {
            new Btn { label = "Drop",      act = () => { if (provider != null) provider.DropBlock(); } },
            new Btn { label = "Box",       act = () => armed = armed == Arm.Box ? Arm.None : Arm.Box,   on = () => armed == Arm.Box },
            new Btn { label = "Ball",      act = () => armed = armed == Arm.Ball ? Arm.None : Arm.Ball, on = () => armed == Arm.Ball },
            new Btn { label = "Undo",      act = () => { if (spawner != null) spawner.RemoveLast(); } },
            new Btn { label = "Reset",     act = () => { if (scene != null) scene.ResetSim(); } },
            new Btn { label = "Stir up",   act = () => { if (stirrer != null) stirrer.NudgeHeight(heightStep); } },
            new Btn { label = "Stir down", act = () => { if (stirrer != null) stirrer.NudgeHeight(-heightStep); } },
            new Btn { label = "Raw",       act = () => { if (scene != null) scene.ToggleRawInput(); }, on = () => scene != null && scene.ShowingRawInput },
            new Btn { label = "Pause",     act = () => { if (scene != null) scene.TogglePause(); },    on = () => scene != null && scene.Paused },
            new Btn { label = "Foam",      act = () => { if (scene != null) scene.ToggleFoam(); },     on = () => scene != null && scene.FoamEnabled },
            new Btn { label = null,        act = CycleSpread },   // label = current inference scheduling
        };
        rects = new Rect[buttons.Length];
    }

    void CycleSpread()
    {
        if (scene == null) return;
        int i = System.Array.IndexOf(SpreadCycle, scene.ActiveSpread);   // -1 (off-cycle value) -> first entry
        int next = SpreadCycle[(i + 1) % SpreadCycle.Length];
        if (Application.isMobilePlatform) scene.spreadFramesMobile = next; else scene.spreadFrames = next;
    }

    string SpreadLabel => scene == null ? "-" : scene.ActiveSpread == 0 ? "Sync" : $"Spread {scene.ActiveSpread}";

    // Two columns of buttons at the right edge of the safe area.
    void Layout()
    {
        s = UiScale;
        Rect sa = Screen.safeArea;                          // pixels, bottom-left origin
        float top = Screen.height - sa.yMax + Margin * s;   // GUI y
        float right = sa.xMax - Margin * s;
        int rows = (buttons.Length + 1) / 2;
        for (int i = 0; i < buttons.Length; i++)
        {
            int col = i / rows, row = i % rows;
            float x = right - (2 - col) * BtnW * s - (1 - col) * Gap * s;
            float y = top + row * (BtnH + Gap) * s;
            rects[i] = new Rect(x, y, BtnW * s, BtnH * s);
        }
    }

    bool HitButton(Vector2 screenPos, out int idx)
    {
        var g = new Vector2(screenPos.x, Screen.height - screenPos.y);
        for (idx = 0; idx < rects.Length; idx++)
            if (rects[idx].Contains(g)) return true;
        idx = -1;
        return false;
    }

    void Update()
    {
        if (buttons == null || scene == null) return;
        Layout();

        var touches = ETouch.activeTouches;
        if (touches.Count == 0) { seen.Clear(); consumed.Clear(); multiGesture = false; return; }

        // taps: a finger's first frame either presses a button or places an armed obstacle
        foreach (var t in touches)
        {
            if (!seen.Add(t.touchId)) continue;
            if (HitButton(t.screenPosition, out int b)) { buttons[b].act(); consumed.Add(t.touchId); }
            else if (armed != Arm.None && touches.Count == 1)
            {
                if (spawner != null && spawner.ScreenToGround(t.screenPosition, out Vector3 p))
                    spawner.SpawnAt(armed == Arm.Box ? LiveSphProvider.Obstacle.Shape.Box
                                                     : LiveSphProvider.Obstacle.Shape.Sphere, p);
                armed = Arm.None;
                consumed.Add(t.touchId);
            }
        }

        // gestures from the fingers that are still free
        int n = 0; ETouch a = default, c = default;
        foreach (var t in touches)
        {
            if (t.ended || consumed.Contains(t.touchId)) continue;
            if (n == 0) a = t; else if (n == 1) c = t;
            n++;
        }
        if (n >= 2) { multiGesture = true; Orbit(a, c); }
        else if (n == 1 && !multiGesture && stirrer != null) stirrer.DriveTo(a.screenPosition);
    }

    void Orbit(ETouch a, ETouch b)
    {
        Camera cam = scene.TargetCamera != null ? scene.TargetCamera : Camera.main;
        if (cam == null) return;
        Transform ct = cam.transform;
        Vector3 pivot = scene.transform.position + Vector3.up * (pivotHeight * scene.transform.lossyScale.x);
        Vector2 drag = (a.delta + b.delta) * (0.5f / s);   // points

        // yaw about world up (no roll), then pitch about the horizontal axis through the pivot;
        // RotateAround turns the camera with its position, so an off-pivot framing is kept.
        ct.RotateAround(pivot, Vector3.up, drag.x * orbitDegPerPt);
        Vector3 rel = ct.position - pivot;
        float dist = rel.magnitude;
        if (dist < 1e-3f) return;
        float el = Mathf.Asin(Mathf.Clamp(rel.y / dist, -1f, 1f)) * Mathf.Rad2Deg;
        float elNew = Mathf.Clamp(el - drag.y * orbitDegPerPt, minElevation, maxElevation);
        Vector3 axis = Vector3.Cross(rel, Vector3.up);     // +angle about this raises the camera
        if (axis.sqrMagnitude > 1e-8f) ct.RotateAround(pivot, axis.normalized, elNew - el);

        // pinch: scale the distance to the pivot by the change in finger span
        float spanNow = (a.screenPosition - b.screenPosition).magnitude;
        float spanPrev = ((a.screenPosition - a.delta) - (b.screenPosition - b.delta)).magnitude;
        if (spanNow > 1f && spanPrev > 1f)
        {
            rel = ct.position - pivot;
            dist = rel.magnitude;
            ct.position = pivot + rel / dist * Mathf.Clamp(dist * spanPrev / spanNow, minDistance, maxDistance);
        }
    }

    void OnGUI()
    {
        if (buttons == null || rects == null) return;
        if (style == null) style = new GUIStyle(GUI.skin.box) { alignment = TextAnchor.MiddleCenter };
        style.fontSize = Mathf.RoundToInt(15f * s);

        Color prev = GUI.color;
        for (int i = 0; i < buttons.Length; i++)
        {
            bool on = buttons[i].on != null && buttons[i].on();
            GUI.color = on ? new Color(0.55f, 0.9f, 1f, 1f) : Color.white;
            GUI.Box(rects[i], buttons[i].label ?? SpreadLabel, style);
        }
        GUI.color = prev;

        float sc = BeginScaledGUI();
        string hint = armed != Arm.None
            ? $"Tap the ground to place the {(armed == Arm.Box ? "box" : "ball")}"
            : "1 finger: stir    2 fingers: orbit    pinch: zoom";
        GUI.Label(new Rect(10, Screen.safeArea.height / sc - 30, 600, 24), hint);
        GUI.matrix = Matrix4x4.identity;
    }
}
