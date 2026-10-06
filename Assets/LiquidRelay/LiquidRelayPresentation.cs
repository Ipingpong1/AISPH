// The Liquid Relay interface is built at runtime so the demo has no package, font,
// prefab, or scene-serialization dependency beyond Unity's standard UGUI package.
// Scoring and control authority live in LiquidRelayGame; this class only presents them.
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.UI;
using UnityEngine.UI;

[DisallowMultipleComponent]
[DefaultExecutionOrder(100)]
public sealed class LiquidRelayPresentation : MonoBehaviour
{
    public LiquidRelayGame game;
    public FluidSceneMVP fluidRenderer;
    [Tooltip("Small receiver identifiers projected over the machine.")]
    public bool showWorldLabels = true;
    public LiquidRelayAudio sound;

    public static readonly Color Navy = Hex("0B1820");
    public static readonly Color Panel = Hex("102630");
    public static readonly Color Ivory = Hex("EAF0E9");
    public static readonly Color Muted = Hex("8199A2");
    public static readonly Color Teal = Hex("57DEC3");
    public static readonly Color Brass = Hex("DFB56A");
    public static readonly Color Coral = Hex("F28A73");
    static readonly Color Line = Hex("29414A");

    Font font;
    Sprite roundSprite;
    Texture2D roundTexture;
    Canvas canvas;
    RectTransform root;
    GameObject uiObject, helpOverlay;
    Text levelTitle, levelNumber, objective, stateLabel, stateDetail;
    Text angleValue, releaseLabel, nextLabel, modeLabel, demoLabel, soundLabel;
    Text chargeValue, wasteValue, timerLabel, resultTitle, resultDetail, footerHint;
    Image chargeFill, wasteFill, stateDot, resultRule;
    Slider angleSlider;
    Button releaseButton, nextButton, demoButton;
    CanvasGroup resultGroup;
    RectTransform resultRoot, labelA, labelB;
    Gauge leftGauge, rightGauge;
    readonly List<Button> levelButtons = new List<Button>();
    readonly List<Text> levelButtonTexts = new List<Text>();
    bool previousKeyboardControls, pausedForHelp;
    LiquidRelayGame.RoundState lastState;
    int lastLevel = -1;
    bool stateKnown, overlayOpen;
    float displayCharge = 1f, displayWaste, resultAlpha;
    float nextTextRefresh;

    sealed class Gauge
    {
        public Text amount, requirement, completed, name;
        public Image accent;
        public Image[] ticks;
        public float displayed;
        public Color color;
    }

    public bool IsOverlayOpen => overlayOpen;

    void Start()
    {
        if (game == null) game = GetComponent<LiquidRelayGame>();
        if (game == null) game = FindFirstObjectByType<LiquidRelayGame>();
        if (fluidRenderer == null) fluidRenderer = FindFirstObjectByType<FluidSceneMVP>();
        if (sound == null) sound = FindFirstObjectByType<LiquidRelayAudio>();
        if (game == null)
        {
            Debug.LogError("Liquid Relay presentation needs a LiquidRelayGame.", this);
            enabled = false;
            return;
        }
        if (fluidRenderer != null) fluidRenderer.showStatus = false;
        font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        MakeRoundedSprite();
        BuildCanvas();
        BuildHeader();
        BuildReceivers();
        BuildMission();
        BuildControls();
        BuildResult();
        BuildHelp();
        BuildWorldLabels();
        RefreshText(true);
    }

    void BuildCanvas()
    {
        uiObject = new GameObject("Presentation", typeof(RectTransform), typeof(Canvas),
            typeof(CanvasScaler), typeof(GraphicRaycaster));
        uiObject.transform.SetParent(transform, false);
        canvas = uiObject.GetComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 80;
        var scaler = uiObject.GetComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1600, 900);
        scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.Expand;
        root = uiObject.GetComponent<RectTransform>();
        if (FindFirstObjectByType<EventSystem>() == null)
        {
            var events = new GameObject("UI input", typeof(EventSystem), typeof(InputSystemUIInputModule));
            events.transform.SetParent(uiObject.transform, false);
        }
    }

    void BuildHeader()
    {
        var title = Box("Title", root, 38, 33, 610, 95, null);
        var emblem = Box("Emblem", title, 0, 9, 34, 45, null);
        Rect("Left stream", emblem, 0, 15, 7, 23, Teal, true);
        Rect("Middle stream", emblem, 12, 4, 7, 34, Teal, true);
        Rect("Right stream", emblem, 24, 10, 7, 28, Teal, true);
        Label("Title", title, "LIQUID RELAY", 47, 0, 550, 53, 40, Ivory, FontStyle.Bold);
        Label("Eyebrow", title, "A  P U Z Z L E  I N  M O T I O N", 49, 58, 440, 22, 12, Muted);
        Rect("Header rule", root, 38, 140, 228, 1, Line);

        var stage = Box("Level title", root, -330, 36, 292, 106, null, true);
        levelNumber = Label("Level number", stage, "LEVEL 01 / 03", 0, 0, 292, 22, 13, Brass,
            FontStyle.Bold, TextAnchor.UpperRight);
        levelTitle = Label("Level name", stage, "FIRST CONTACT", 0, 30, 292, 45, 27, Ivory,
            FontStyle.Bold, TextAnchor.UpperRight);
        timerLabel = Label("Timer", stage, "", 0, 77, 292, 24, 14, Muted,
            FontStyle.Normal, TextAnchor.UpperRight);
    }

    void BuildReceivers()
    {
        var rail = Box("Receiver instruments", root, 38, 178, 228, 470, null);
        Label("Goals caption", rail, "COLLECTION GOALS", 0, 0, 228, 24, 13, Muted, FontStyle.Bold);
        leftGauge = MakeGauge(rail, "A", 38, Teal);
        rightGauge = MakeGauge(rail, "B", 202, Brass);
        Rect("Charge rule", rail, 0, 377, 228, 1, Line);
        Label("Charge label", rail, "IN PLAY", 0, 393, 160, 24, 12, Muted);
        chargeValue = Label("Charge value", rail, "100%", 159, 391, 69, 24, 17, Ivory,
            FontStyle.Bold, TextAnchor.UpperRight);
        var track = Box("Charge track", rail, 0, 423, 228, 4, Line, false, true);
        chargeFill = Rect("Charge fill", track, 0, 0, 228, 4, Ivory, true);
        Label("Waste label", rail, "SPILLED", 0, 444, 155, 24, 12, Muted);
        wasteValue = Label("Waste value", rail, "0%", 160, 442, 68, 24, 17, Coral,
            FontStyle.Bold, TextAnchor.UpperRight);
        track = Box("Waste track", rail, 0, 474, 228, 4, Line, false, true);
        wasteFill = Rect("Waste fill", track, 0, 0, 0, 4, Coral, true);
    }

    Gauge MakeGauge(RectTransform parent, string id, float y, Color color)
    {
        var g = new Gauge { color = color, ticks = new Image[16] };
        var card = Box("Receiver " + id, parent, 0, y, 228, 140, Panel, false, true);
        g.accent = Rect("Accent", card, 0, 14, 3, 110, color, true);
        g.name = Label("Receiver name", card, "RECEIVER " + id, 19, 14, 172, 24, 13, color, FontStyle.Bold);
        g.completed = Label("Complete", card, "", 165, 12, 47, 24, 13, color,
            FontStyle.Bold, TextAnchor.UpperRight);
        g.amount = Label("Collected", card, "0%", 18, 40, 195, 40, 33, Ivory, FontStyle.Bold);
        for (int i = 0; i < g.ticks.Length; i++)
            g.ticks[i] = Rect("Measure " + i, card, 19 + i * 12, 91, 8, 9, Alpha(color, 0.13f), true);
        g.requirement = Label("Target", card, "TARGET 18% OF CHARGE", 19, 112, 195, 20, 11, Muted);
        return g;
    }

    void BuildMission()
    {
        var rail = Box("Mission", root, -272, 184, 234, 449, null, true);
        Label("Objective caption", rail, "THE OBJECTIVE", 0, 0, 234, 24, 13, Muted, FontStyle.Bold);
        objective = Label("Objective", rail, "Fill both receivers from one charge.", 0, 36, 234, 122,
            19, Ivory);
        objective.lineSpacing = 1.12f;
        Rect("Mission rule", rail, 0, 177, 234, 1, Line);
        stateDot = Rect("State light", rail, 0, 203, 7, 7, Teal, true);
        stateLabel = Label("Status", rail, "READY TO RELEASE", 19, 197, 215, 26, 13, Teal, FontStyle.Bold);
        stateDetail = Label("Status hint", rail, "Choose an angle, then open the gate.", 0, 236, 234, 90, 15, Muted);
        stateDetail.lineSpacing = 1.12f;

        var levels = Box("Level selection", rail, 0, 338, 234, 45, null);
        for (int i = 0; i < game.LevelCount; i++)
        {
            int index = i;
            float width = (234f - 10 * (game.LevelCount - 1)) / Mathf.Max(1, game.LevelCount);
            Text text;
            var button = MakeButton("Level " + (i + 1), levels, (i + 1).ToString("00"),
                i * (width + 10), 0, width, 39, () => game.SelectLevel(index), Panel, Ivory, out text, 14);
            levelButtons.Add(button);
            levelButtonTexts.Add(text);
        }
        demoButton = MakeButton("Watch solution", rail, "WATCH A SOLUTION   H", 0, 405, 234, 42,
            () => game.ToggleAutopilot(), Panel, Muted, out demoLabel, 12);
        MakeButton("Help", rail, "HOW TO PLAY   ?", 0, 453, 234, 24,
            () => SetHelp(true), Alpha(Panel, 0f), Muted, out _, 12);
    }

    void BuildControls()
    {
        // The square neural surface occupies the center of a wide screen. Keep
        // the controls on its outside rails, including at the bottom of the view.
        var instrument = BottomBox("Deflector instrument", 38, 38, 228, 180, false, Panel);
        Rect("Instrument accent", instrument, 16, 0, 50, 2, Teal, true);
        Label("Deflector heading", instrument, "DEFLECTOR ANGLE", 16, 17, 196, 22, 12, Muted, FontStyle.Bold);
        angleValue = Label("Deflector value", instrument, "0°", 16, 49, 196, 42, 35, Ivory,
            FontStyle.Bold, TextAnchor.UpperCenter);
        angleSlider = MakeSlider(instrument, 28, 109, 172, 27);
        Label("A key", instrument, "A", 6, 110, 20, 24, 14, Teal, FontStyle.Bold, TextAnchor.MiddleCenter);
        Label("D key", instrument, "D", 202, 110, 20, 24, 14, Brass, FontStyle.Bold, TextAnchor.MiddleCenter);
        Label("Left route", instrument, "TOWARD A", 16, 151, 95, 18, 10, Muted);
        Label("Right route", instrument, "TOWARD B", 117, 151, 95, 18, 10, Muted,
            FontStyle.Normal, TextAnchor.UpperRight);

        var actions = BottomBox("Release controls", -272, 38, 234, 180, true, null);
        releaseButton = MakeButton("Release", actions, "RELEASE CHARGE", 0, 0, 234, 60,
            () => game.Release(), Teal, Navy, out releaseLabel, 18);
        Label("Release shortcut", actions, "SPACE  TO OPEN THE GATE", 0, 69, 234, 22, 10, Muted,
            FontStyle.Normal, TextAnchor.UpperCenter);
        MakeButton("Retry", actions, "RETRY   R", 0, 101, 112, 43,
            () => game.ResetLevel(), Panel, Ivory, out _, 12);
        nextButton = MakeButton("Next level", actions, "NEXT   N", 122, 101, 112, 43,
            () => game.NextLevel(), Panel, Ivory, out nextLabel, 12);
        MakeButton("Surface comparison", actions, "NEURAL WATER   C", 0, 157, 234, 24,
            () => { if (fluidRenderer != null) fluidRenderer.ToggleSurfaceSource(); },
            Alpha(Panel, 0f), Muted, out modeLabel, 10);

        var footer = BottomBox("Keyboard hints", 0, 4, 760, 24, false, null);
        footer.anchorMin = footer.anchorMax = new Vector2(0.5f, 0);
        footer.pivot = new Vector2(0.5f, 0);
        footer.anchoredPosition = new Vector2(0, 4);
        footerHint = Label("Footer hint", footer, "A / D  STEER      SPACE  RELEASE      R  RETRY      1 / 2 / 3  LEVEL", 0, 0, 760, 24,
            10, Muted, FontStyle.Normal, TextAnchor.MiddleCenter);
        var soundRoot = BottomBox("Sound control", -272, 4, 234, 24, true, null);
        MakeButton("Sound", soundRoot, "SOUND ON   M", 0, 0, 234, 24,
            () => { if (sound != null) sound.ToggleSound(); },
            Alpha(Panel, 0f), Muted, out soundLabel, 10);
    }

    RectTransform BottomBox(string name, float x, float bottom, float w, float h, bool right, Color? color)
    {
        var rt = Box(name, root, x, 0, w, h, color, right, color.HasValue);
        rt.anchorMin = rt.anchorMax = new Vector2(right ? 1 : 0, 0);
        rt.pivot = new Vector2(0, 0);
        rt.anchoredPosition = new Vector2(x, bottom);
        return rt;
    }

    Slider MakeSlider(RectTransform parent, float x, float y, float width, float height)
    {
        var holder = Box("Angle slider", parent, x, y, width, height, null);
        var slider = holder.gameObject.AddComponent<Slider>();
        slider.minValue = -32f;
        slider.maxValue = 32f;
        slider.direction = Slider.Direction.LeftToRight;
        var hit = Rect("Hit area", holder, 0, 0, width, height, Color.clear);
        hit.raycastTarget = true;
        Rect("Rail", holder, 0, height / 2 - 2, width, 4, Line, true);
        for (int i = 0; i <= 8; i++)
            Rect("Tick " + i, holder, i * width / 8 - 1, height / 2 + 7, 2, i == 4 ? 9 : 4,
                i == 4 ? Muted : Line);
        var area = Box("Handle area", holder, 9, 0, width - 18, height, null);
        var handle = Rect("Handle", area, 0, height / 2 - 10, 20, 20, Ivory, true).rectTransform;
        handle.pivot = new Vector2(0.5f, 0.5f);
        handle.anchoredPosition = Vector2.zero;
        // Slider stretches the handle across the non-sliding axis. Compensate
        // in sizeDelta so it remains a compact 20-pixel control.
        handle.sizeDelta = new Vector2(20, 20 - height);
        slider.handleRect = handle;
        slider.targetGraphic = handle.GetComponent<Image>();
        var colors = slider.colors;
        colors.normalColor = Ivory;
        colors.highlightedColor = Teal;
        colors.pressedColor = Brass;
        colors.selectedColor = Ivory;
        colors.disabledColor = Alpha(Muted, 0.45f);
        slider.colors = colors;
        slider.navigation = new Navigation { mode = Navigation.Mode.None };
        slider.onValueChanged.AddListener(v => game.SetDeflectorAngle(v));
        return slider;
    }

    void BuildResult()
    {
        resultRoot = new GameObject("Round result", typeof(RectTransform), typeof(CanvasGroup)).GetComponent<RectTransform>();
        resultRoot.SetParent(root, false);
        resultRoot.anchorMin = resultRoot.anchorMax = new Vector2(0.5f, 1);
        resultRoot.pivot = new Vector2(0.5f, 1);
        resultRoot.anchoredPosition = new Vector2(0, -39);
        resultRoot.sizeDelta = new Vector2(568, 98);
        var bg = resultRoot.gameObject.AddComponent<Image>();
        bg.sprite = roundSprite;
        bg.type = Image.Type.Sliced;
        bg.color = Alpha(Navy, 0.96f);
        bg.raycastTarget = false;
        resultGroup = resultRoot.GetComponent<CanvasGroup>();
        resultGroup.alpha = 0;
        resultGroup.interactable = false;
        resultGroup.blocksRaycasts = false;
        resultRule = Rect("Result accent", resultRoot, 23, 20, 4, 58, Teal, true);
        resultTitle = Label("Result title", resultRoot, "RELAY COMPLETE", 44, 17, 505, 34, 27, Ivory, FontStyle.Bold);
        resultDetail = Label("Result detail", resultRoot, "Both receivers filled. Beautifully routed.", 45, 57, 505, 26, 14, Muted);
    }

    void BuildHelp()
    {
        helpOverlay = new GameObject("How to play", typeof(RectTransform), typeof(Image));
        var overlay = helpOverlay.GetComponent<RectTransform>();
        overlay.SetParent(root, false);
        overlay.anchorMin = Vector2.zero;
        overlay.anchorMax = Vector2.one;
        overlay.offsetMin = overlay.offsetMax = Vector2.zero;
        helpOverlay.GetComponent<Image>().color = Alpha(Navy, 0.84f);
        var panel = new GameObject("Instructions", typeof(RectTransform), typeof(Image)).GetComponent<RectTransform>();
        panel.SetParent(overlay, false);
        panel.anchorMin = panel.anchorMax = panel.pivot = new Vector2(0.5f, 0.5f);
        panel.sizeDelta = new Vector2(700, 540);
        var panelImage = panel.GetComponent<Image>();
        panelImage.color = Panel;
        panelImage.sprite = roundSprite;
        panelImage.type = Image.Type.Sliced;
        Label("Help eyebrow", panel, "LIQUID RELAY  /  FIELD GUIDE", 42, 33, 616, 26, 12, Teal, FontStyle.Bold);
        Label("Help title", panel, "Make every drop count.", 40, 75, 620, 47, 35, Ivory, FontStyle.Bold);
        HelpRow(panel, "01", "SET YOUR ANGLE", "Drag the slider or hold A / D to aim the deflector.", 151, Teal);
        HelpRow(panel, "02", "OPEN THE GATE", "Press Space or Release Charge. You get one reservoir.", 235, Brass);
        HelpRow(panel, "03", "REACH THE TARGETS", "Steer while the water flows. Fill each marked receiver to its target.", 319, Coral);
        Label("Help note", panel, "Retry instantly with R. Switch levels with 1 / 2 / 3.\nWatch a solution with H. Toggle sound with M.",
            42, 404, 616, 49, 14, Muted);
        MakeButton("Close instructions", panel, "LET'S ROUTE", 442, 465, 216, 47,
            () => SetHelp(false), Teal, Navy, out _, 15);
        MakeButton("Close cross", panel, "×", 620, 22, 38, 38,
            () => SetHelp(false), Alpha(Panel, 0), Muted, out _, 24);
        helpOverlay.SetActive(false);
    }

    void HelpRow(RectTransform parent, string number, string heading, string detail, float y, Color color)
    {
        Label("Step " + number, parent, number, 42, y + 1, 46, 42, 28, color, FontStyle.Bold);
        Label("Heading " + number, parent, heading, 110, y, 530, 24, 14, Ivory, FontStyle.Bold);
        Label("Detail " + number, parent, detail, 110, y + 32, 530, 42, 15, Muted);
    }

    void BuildWorldLabels()
    {
        labelA = WorldLabel("A", Teal);
        labelB = WorldLabel("B", Brass);
    }

    RectTransform WorldLabel(string name, Color color)
    {
        var rt = Box("Receiver marker " + name, root, 0, 0, 28, 28, Alpha(Navy, 0.88f), false, true);
        rt.pivot = new Vector2(0.5f, 0.5f);
        Label("Identifier", rt, name, 0, 0, 28, 28, 14, color, FontStyle.Bold, TextAnchor.MiddleCenter);
        rt.SetSiblingIndex(0);
        return rt;
    }

    void Update()
    {
        if (root == null || game == null) return;
        if (Keyboard.current != null)
        {
            if (Keyboard.current.escapeKey.wasPressedThisFrame && overlayOpen) SetHelp(false);
            if (Keyboard.current.slashKey.wasPressedThisFrame) SetHelp(!overlayOpen);
        }
        RefreshText(false);
        float ease = 1f - Mathf.Exp(-9f * Time.unscaledDeltaTime);
        UpdateGauge(leftGauge, game.LeftFill01, game.LeftTarget01, ease);
        UpdateGauge(rightGauge, game.RightFill01, game.RightTarget01, ease, game.RightLimit01);
        displayCharge = Mathf.Lerp(displayCharge, game.Remaining01, ease);
        displayWaste = Mathf.Lerp(displayWaste, game.Waste01, ease);
        chargeFill.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, 228 * Mathf.Clamp01(displayCharge));
        wasteFill.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, 228 * Mathf.Clamp01(displayWaste));
        angleSlider.SetValueWithoutNotify(game.DeflectorAngle);
        resultAlpha = Mathf.MoveTowards(resultAlpha,
            game.State == LiquidRelayGame.RoundState.Won || game.State == LiquidRelayGame.RoundState.Lost ? 1f : 0f,
            Time.unscaledDeltaTime * 3f);
        resultGroup.alpha = resultAlpha;
        UpdateStateFeedback();
        UpdateWorldLabels();
    }

    void RefreshText(bool force)
    {
        bool newLevel = lastLevel != game.LevelIndex;
        if (!force && !newLevel && Time.unscaledTime < nextTextRefresh) return;
        nextTextRefresh = Time.unscaledTime + 0.1f;
        if (newLevel || force)
        {
            lastLevel = game.LevelIndex;
            SetText(levelNumber, "LEVEL " + (game.LevelIndex + 1).ToString("00") + " / " + game.LevelCount.ToString("00"));
            SetText(levelTitle, game.LevelTitle.ToUpperInvariant());
            SetText(objective, game.LevelInstruction);
            leftGauge.displayed = rightGauge.displayed = 0;
            displayCharge = 1f;
            displayWaste = 0;
            stateKnown = false;
            for (int i = 0; i < levelButtons.Count; i++)
            {
                Color c = i == game.LevelIndex ? Brass : Panel;
                ApplyButtonColors(levelButtons[i], c);
                levelButtonTexts[i].color = i == game.LevelIndex ? Navy : Muted;
            }
        }
        SetText(objective, game.LevelInstruction);
        SetText(angleValue, (game.DeflectorAngle > 0.5f ? "+" : "") + Mathf.RoundToInt(game.DeflectorAngle) + "°");
        SetText(chargeValue, Mathf.RoundToInt(game.Remaining01 * 100) + "%");
        SetText(wasteValue, Mathf.RoundToInt(game.Waste01 * 100) + "%");
        SetText(leftGauge.amount, Mathf.RoundToInt(game.LeftFill01 * 100) + "%");
        SetText(rightGauge.amount, Mathf.RoundToInt(game.RightFill01 * 100) + "%");
        SetText(leftGauge.requirement, game.LeftTarget01 > 0 ? "TARGET " + Mathf.RoundToInt(game.LeftTarget01 * 100) + "% OF CHARGE" : "NO TARGET THIS ROUND");
        SetText(rightGauge.requirement, game.RightLimit01 > 0 ? "KEEP BELOW " + Mathf.RoundToInt(game.RightLimit01 * 100) + "% FOR NOW" :
            game.RightTarget01 > 0 ? "TARGET " + Mathf.RoundToInt(game.RightTarget01 * 100) + "% OF CHARGE" : "NO TARGET THIS ROUND");
        SetText(modeLabel, fluidRenderer == null ? "FLUID SIMULATION" :
            fluidRenderer.ClassicalActive ? "CLASSICAL WATER   C" : "NEURAL WATER   C");
        SetText(demoLabel, game.Autopilot ? "TAKE CONTROL   H" : "WATCH A SOLUTION   H");
        SetText(soundLabel, sound != null && sound.soundEnabled ? "SOUND ON   M" : "SOUND OFF   M");
        SetText(footerHint, game.Autopilot ? "DEMO RUNNING  /  H TO TAKE CONTROL" : "A / D  STEER      SPACE  RELEASE      R  RETRY      1 / 2 / 3  LEVEL");
        var state = game.State;
        bool running = state == LiquidRelayGame.RoundState.Running;
        bool ready = state == LiquidRelayGame.RoundState.Ready;
        releaseButton.interactable = ready;
        releaseLabel.color = ready ? Navy : Muted;
        angleSlider.interactable = ready || running;
        SetText(releaseLabel, ready ? "RELEASE CHARGE" : running ? "GATE OPEN" : "CHARGE RELEASED");
        SetText(nextLabel, game.LevelIndex + 1 >= game.LevelCount ? "LEVEL 01   N" : "NEXT   N");
        SetText(timerLabel, running ? "TIME LEFT  " + FormatTime(Mathf.Max(0, game.TimeLimit - game.ElapsedSeconds)) :
            ready ? "ONE CHARGE. TWO RECEIVERS." : "ROUND  " + FormatTime(game.ElapsedSeconds));
        switch (state)
        {
            case LiquidRelayGame.RoundState.Ready:
                SetState(string.IsNullOrEmpty(game.PhaseTitle) ? "READY TO RELEASE" : game.PhaseTitle,
                    "Choose an angle, then open the gate. You can steer during the pour.", Teal);
                break;
            case LiquidRelayGame.RoundState.Running:
                SetState(game.Autopilot ? "DEMONSTRATION" : string.IsNullOrEmpty(game.PhaseTitle) ? "WATER IN MOTION" : game.PhaseTitle, game.Autopilot ?
                    "Watch the deflector route the charge. Press H to take control." :
                    "Watch the receivers. Change the angle to route the remaining water.", Brass);
                break;
            case LiquidRelayGame.RoundState.Won:
                SetState("TARGETS REACHED", "Your collection targets are reached. Try the next puzzle.", Teal);
                SetText(resultTitle, "RELAY COMPLETE");
                SetText(resultDetail, "Targets reached. " + Mathf.RoundToInt((game.LeftFill01 + game.RightFill01) * 100) + "% of the charge collected.");
                resultRule.color = Teal;
                break;
            case LiquidRelayGame.RoundState.Lost:
                SetState("ANOTHER ANGLE?", string.IsNullOrEmpty(game.FailureReason) ?
                    "Try a different release angle, then steer earlier during the pour." : game.FailureReason, Coral);
                SetText(resultTitle, "TRY ANOTHER ROUTE");
                SetText(resultDetail, "One charge, another chance. Press R to reset the machine.");
                resultRule.color = Coral;
                break;
        }
        ApplyButtonColors(nextButton, state == LiquidRelayGame.RoundState.Won ? Brass : Panel);
        nextLabel.color = state == LiquidRelayGame.RoundState.Won ? Navy : Ivory;
    }

    void SetState(string title, string detail, Color color)
    {
        SetText(stateLabel, title);
        SetText(stateDetail, detail);
        stateLabel.color = stateDot.color = color;
    }

    void UpdateGauge(Gauge gauge, float collected, float target, float ease, float limit = 0)
    {
        Color color = limit > 0 ? Coral : gauge.color;
        gauge.accent.color = gauge.name.color = gauge.completed.color = color;
        float ratio = limit > 0 ? Mathf.Clamp01(collected / limit) : target > 0 ? Mathf.Clamp01(collected / target) : 0;
        gauge.displayed = Mathf.Lerp(gauge.displayed, ratio, ease);
        for (int i = 0; i < gauge.ticks.Length; i++)
        {
            float filled = Mathf.Clamp01(gauge.displayed * gauge.ticks.Length - i);
            gauge.ticks[i].color = Alpha(color, Mathf.Lerp(0.13f, 1f, filled));
        }
        SetText(gauge.completed, limit > 0 ? "HOLD" : target > 0 && (collected >= target || game.State == LiquidRelayGame.RoundState.Won) ? "MET" : "");
    }

    void UpdateStateFeedback()
    {
        if (!stateKnown || lastState != game.State)
        {
            if (game.State == LiquidRelayGame.RoundState.Ready)
                leftGauge.displayed = rightGauge.displayed = 0;
            lastState = game.State;
            stateKnown = true;
            nextTextRefresh = 0;
        }
    }

    void UpdateWorldLabels()
    {
        var camera = fluidRenderer != null ? fluidRenderer.targetCamera : Camera.main;
        if (camera == null || !showWorldLabels)
        {
            labelA.gameObject.SetActive(false);
            labelB.gameObject.SetActive(false);
            return;
        }
        PositionWorldLabel(labelA, camera, game.SimPointToWorld(new Vector3(2.45f, 0.15f, 0.85f)));
        PositionWorldLabel(labelB, camera, game.SimPointToWorld(new Vector3(2.45f, 0.15f, 2.15f)));
    }

    void PositionWorldLabel(RectTransform label, Camera camera, Vector3 world)
    {
        Vector3 screen = camera.WorldToScreenPoint(world);
        bool visible = screen.z > 0 && !overlayOpen;
        label.gameObject.SetActive(visible);
        if (!visible) return;
        RectTransformUtility.ScreenPointToLocalPointInRectangle(root, screen, null, out Vector2 local);
        label.anchorMin = label.anchorMax = new Vector2(0.5f, 0.5f);
        label.anchoredPosition = local;
    }

    void SetHelp(bool visible)
    {
        if (overlayOpen == visible) return;
        overlayOpen = visible;
        helpOverlay.SetActive(visible);
        if (visible)
        {
            helpOverlay.transform.SetAsLastSibling();
            previousKeyboardControls = game.keyboardControls;
            game.keyboardControls = false;
            pausedForHelp = fluidRenderer != null && !fluidRenderer.Paused;
            if (pausedForHelp) fluidRenderer.TogglePause();
        }
        else
        {
            game.keyboardControls = previousKeyboardControls;
            if (pausedForHelp && fluidRenderer != null && fluidRenderer.Paused) fluidRenderer.TogglePause();
            pausedForHelp = false;
        }
    }

    Button MakeButton(string name, RectTransform parent, string caption, float x, float y, float w, float h,
        Action callback, Color background, Color foreground, out Text text, int size = 15)
    {
        var rt = Box(name, parent, x, y, w, h, background, false, true);
        var image = rt.GetComponent<Image>();
        image.raycastTarget = true;
        var button = rt.gameObject.AddComponent<Button>();
        button.targetGraphic = image;
        button.navigation = new Navigation { mode = Navigation.Mode.None };
        ApplyButtonColors(button, background);
        image.CrossFadeColor(background, 0f, true, true);
        button.onClick.AddListener(() => callback?.Invoke());
        text = Label("Caption", rt, caption, 8, 1, w - 16, h - 2, size, foreground,
            FontStyle.Bold, TextAnchor.MiddleCenter);
        return button;
    }

    static void ApplyButtonColors(Button button, Color background)
    {
        // Unity multiplies a button's transition tint by its Image color.
        button.image.color = Color.white;
        var colors = button.colors;
        colors.normalColor = background;
        colors.highlightedColor = background.a < 0.01f ? Alpha(Muted, 0.1f) : Color.Lerp(background, Color.white, 0.12f);
        colors.pressedColor = background.a < 0.01f ? Alpha(Muted, 0.16f) : Color.Lerp(background, Navy, 0.15f);
        colors.selectedColor = background;
        colors.disabledColor = Alpha(Panel, 0.75f);
        colors.fadeDuration = 0.12f;
        colors.colorMultiplier = 1;
        button.colors = colors;
    }

    RectTransform Box(string name, RectTransform parent, float x, float y, float w, float h,
        Color? color, bool right = false, bool rounded = false)
    {
        var rt = new GameObject(name, typeof(RectTransform)).GetComponent<RectTransform>();
        rt.SetParent(parent, false);
        rt.anchorMin = rt.anchorMax = new Vector2(right ? 1 : 0, 1);
        rt.pivot = new Vector2(0, 1);
        rt.anchoredPosition = new Vector2(x, -y);
        rt.sizeDelta = new Vector2(w, h);
        if (color.HasValue)
        {
            var image = rt.gameObject.AddComponent<Image>();
            image.color = color.Value;
            image.raycastTarget = false;
            if (rounded) { image.sprite = roundSprite; image.type = Image.Type.Sliced; }
        }
        return rt;
    }

    Image Rect(string name, RectTransform parent, float x, float y, float w, float h, Color color, bool rounded = false)
        => Box(name, parent, x, y, w, h, color, false, rounded).GetComponent<Image>();

    Text Label(string name, RectTransform parent, string value, float x, float y, float w, float h,
        int size, Color color, FontStyle style = FontStyle.Normal, TextAnchor align = TextAnchor.UpperLeft)
    {
        var rt = Box(name, parent, x, y, w, h, null);
        var text = rt.gameObject.AddComponent<Text>();
        text.font = font;
        text.fontSize = size;
        text.fontStyle = style;
        text.alignment = align;
        text.color = color;
        text.text = value;
        text.raycastTarget = false;
        text.horizontalOverflow = HorizontalWrapMode.Wrap;
        text.verticalOverflow = VerticalWrapMode.Truncate;
        text.supportRichText = false;
        return text;
    }

    void MakeRoundedSprite()
    {
        const int size = 32;
        const float radius = 8f;
        roundTexture = new Texture2D(size, size, TextureFormat.RGBA32, false)
        { name = "Liquid Relay UI corners", filterMode = FilterMode.Bilinear, wrapMode = TextureWrapMode.Clamp };
        var pixels = new Color[size * size];
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                float dx = Mathf.Max(Mathf.Abs(x + 0.5f - size * 0.5f) - (size * 0.5f - radius), 0);
                float dy = Mathf.Max(Mathf.Abs(y + 0.5f - size * 0.5f) - (size * 0.5f - radius), 0);
                float a = Mathf.Clamp01(radius + 0.5f - Mathf.Sqrt(dx * dx + dy * dy));
                pixels[y * size + x] = new Color(1, 1, 1, a);
            }
        roundTexture.SetPixels(pixels);
        roundTexture.Apply(false, true);
        roundSprite = Sprite.Create(roundTexture, new Rect(0, 0, size, size), new Vector2(0.5f, 0.5f),
            100, 0, SpriteMeshType.FullRect, new Vector4(radius, radius, radius, radius));
        roundSprite.name = "Liquid Relay rounded rectangle";
    }

    static Color Hex(string hex) { ColorUtility.TryParseHtmlString("#" + hex, out Color color); return color; }
    static Color Alpha(Color c, float alpha) { c.a = alpha; return c; }
    static string FormatTime(float seconds) => Mathf.FloorToInt(seconds / 60).ToString("00") + ":" + Mathf.FloorToInt(seconds % 60).ToString("00");
    static void SetText(Text text, string value) { if (text != null && text.text != value) text.text = value; }

    void OnDestroy()
    {
        if (overlayOpen && game != null) game.keyboardControls = previousKeyboardControls;
        if (roundSprite != null) Destroy(roundSprite);
        if (roundTexture != null) Destroy(roundTexture);
        if (uiObject != null) Destroy(uiObject);
    }
}
