// SSFRViewer.cs — shades the baked LR / NEURAL / GT frames as liquid using screen-space fluid
// rendering (Hidden/FluidSSFR, a port of the ML repo's fluid_shader.py) and draws them side by side.
//
// Two modes:
//  - animate=false: single frame from the three TextAsset slots (the original MVP frame).
//  - animate=true:  cycles a whole sim baked into <project>/FrameData/ (lr/pred/gt_sim{S}_f{N}.bytes,
//    read straight from disk — kept outside Assets/ so Unity doesn't import ~800 MB of TextAssets).
//
// No inference at play time: NEURAL frames are pre-baked once via CPU backend.
// Channels per frame (7,512,512) float32 CHW, normalized; ch0 depth, ch1 thickness, ch6 mask
// (0/1 for LR/GT, occupancy LOGIT for model output — threshold at 0).
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class SSFRViewer : MonoBehaviour
{
    [Header("Single frame (TextAssets, used when animate = false)")]
    [Tooltip("Fallback single-frame LR input tensor (lr_sim46_f20.bytes), used only when Animate is off or FrameData/ has no sequence loaded.")]
    public TextAsset lrFrame;
    [Tooltip("Fallback single-frame model prediction tensor (pred_sim46_f20.bytes, ch6 = occupancy logit), used only when Animate is off.")]
    public TextAsset predFrame;
    [Tooltip("Fallback single-frame ground-truth tensor (gt_sim46_f20.bytes), used only when Animate is off.")]
    public TextAsset gtFrame;

    [Header("Animation (streams <project>/FrameData/)")]
    [Tooltip("When true, streams a whole baked sequence from Frames Folder on disk and cycles through it. When false, shows one static frame from the TextAsset slots above. Recommended default: true.")]
    public bool animate = true;
    [Tooltip("Which baked simulation index to stream (matches the {S} in FrameData/lr_sim{S}_f{N}.bytes filenames). Recommended default: 46.")]
    public int animSim = 46;
    [Tooltip("Folder (relative to the project root, outside Assets/) containing the baked lr/pred/gt_sim{S}_f{N}.bytes frame files. Recommended default: FrameData.")]
    public string framesFolder = "FrameData";
    [Tooltip("Animation playback rate in frames/sec. Higher = faster playback through the sequence; lower = slower playback. Recommended default: 15.")]
    public float playbackFps = 15f;
    [Tooltip("If non-empty, dumps one PNG screenshot per frame during the first loop through the sequence (for video export). Empty = disabled. Recommended default: empty (disabled).")]
    public string captureDir = "";

    [Header("Shading")]
    [Tooltip("Shading shader (Hidden/FluidSSFR). Required — unlike FluidLiveMVP, this component has no Shader.Find fallback if left unassigned.")]
    public Shader ssfrShader;
    [Tooltip("Camera field of view in degrees, used for the shader's focal length. Higher = wider view; lower = narrower/more zoomed-in view. Must match the camera the sim46 data was rendered with (dam_breaks_v3 manifest.json). Recommended default: 49.88 (do not change without re-baking).")]
    public float fovDeg = 49.88038f;
    [Tooltip("Gaussian pre-blur sigma (pixels) applied to the LR panel before shading. Higher = smoother/softer; lower/0 = sharper, raw. Recommended default: 0 (reference renders LR raw, unsmoothed).")]
    public float presmoothLR = 0f;
    [Tooltip("Gaussian pre-blur sigma (pixels) applied to the NEURAL prediction panel before shading. Higher = smoother/softer; lower/0 = sharper, raw. Recommended default: 0 (reference renders NEURAL raw, unsmoothed).")]
    public float presmoothPred = 0f;
    [Tooltip("Gaussian pre-blur sigma (pixels) applied to the GT panel before shading. Higher = smoother/softer; lower/0 = sharper, raw. Recommended default: 2.0 (matches the reference render's presmooth_sigma).")]
    public float presmoothGT = 2.0f;
    [Tooltip("Refraction distortion strength in the SSFR shader. Higher = more warped/bent refraction through the fluid; lower = flatter, less distorted refraction. Recommended default: 22.")]
    public float refrStrength = 22f;
    [Tooltip("Specular reflectance coefficient. Higher = brighter, more prominent specular highlights (shinier surface); lower = dimmer highlights (more matte surface). Recommended default: 0.7.")]
    public float ks = 0.7f;
    [Tooltip("Specular highlight exponent (Blinn-Phong). Higher = smaller, tighter, sharper highlights (glossier); lower = broader, softer highlights (more diffuse-looking). Recommended default: 120.")]
    public float shininess = 120f;

    const int H = 512, W = 512, HW = H * W;
    // Normalization stats for depth/thickness (meta_sim46_f20.json).
    const float DEPTH_MEAN = 3.9124022f, DEPTH_STD = 1.06759f;
    const float THICK_MEAN = 3.67851f, THICK_STD = 5.0205793f;

    Material mat;
    readonly Texture2D[] fields = new Texture2D[3];
    readonly RenderTexture[] smoothed = new RenderTexture[3];
    readonly RenderTexture[] shaded = new RenderTexture[3];
    readonly string[] labels = { "LR input — shaded", "NEURAL 014 — shaded", "GT — shaded" };
    float kThick;
    string info = "initializing...";

    // animation state
    readonly List<Color[][]> frameData = new List<Color[][]>();   // [frame][source] -> pixels
    readonly List<int> frameNums = new List<int>();
    int curFrame = -1;
    float playT;
    bool capturing;

    void Start()
    {
        try
        {
            mat = new Material(ssfrShader);
            for (int i = 0; i < 3; i++)
            {
                fields[i] = new Texture2D(W, H, TextureFormat.RGBAFloat, false) { filterMode = FilterMode.Point };
                smoothed[i] = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBFloat) { filterMode = FilterMode.Point };
                shaded[i] = new RenderTexture(W, H, 0, RenderTextureFormat.ARGB32);
            }

            if (animate) LoadSequence();

            if (frameData.Count > 0)
            {
                // k_thick from the middle frame's GT, held constant so brightness doesn't flicker.
                kThick = 1.2f / Mathf.Max(MedianFgThickness(GtPath(frameNums[frameNums.Count / 2])), 1e-3f);
                capturing = captureDir.Length > 0;
                if (capturing) Directory.CreateDirectory(captureDir);
                SetFrame(0);
                if (capturing)
                    ScreenCapture.CaptureScreenshot(Path.Combine(captureDir, "frame_000.png"));
                info = $"sim{animSim}  {frameData.Count} frames  fov={fovDeg:F2}°  k_thick={kThick:F3}";
            }
            else
            {
                // single-frame fallback (original MVP assets)
                animate = false;
                SetPixelsFor(0, Parse(lrFrame.bytes, false));
                SetPixelsFor(1, Parse(predFrame.bytes, true));
                SetPixelsFor(2, Parse(gtFrame.bytes, false));
                kThick = 1.2f / Mathf.Max(MedianFgThickness(gtFrame.bytes), 1e-3f);
                info = $"single frame  fov={fovDeg:F2}°  k_thick={kThick:F3}";
            }
        }
        catch (Exception e) { info = $"ERROR: {e.Message}"; Debug.LogException(e); }
    }

    string GtPath(int n) => Path.Combine(framesFolder, $"gt_sim{animSim}_f{n}.bytes");

    void LoadSequence()
    {
        if (!Directory.Exists(framesFolder)) { Debug.LogWarning($"no {framesFolder}/ folder"); return; }
        var nums = new List<int>();
        foreach (var f in Directory.GetFiles(framesFolder, $"lr_sim{animSim}_f*.bytes"))
        {
            var stem = Path.GetFileNameWithoutExtension(f);
            int n = int.Parse(stem.Substring(stem.LastIndexOf('f') + 1));
            string pred = Path.Combine(framesFolder, $"pred_sim{animSim}_f{n}.bytes");
            if (File.Exists(pred) && File.Exists(GtPath(n))) nums.Add(n);
        }
        nums.Sort();
        foreach (int n in nums)
        {
            frameData.Add(new[]
            {
                Parse(File.ReadAllBytes(Path.Combine(framesFolder, $"lr_sim{animSim}_f{n}.bytes")), false),
                Parse(File.ReadAllBytes(Path.Combine(framesFolder, $"pred_sim{animSim}_f{n}.bytes")), true),
                Parse(File.ReadAllBytes(GtPath(n)), false),
            });
            frameNums.Add(n);
        }
        Debug.Log($"SSFRViewer: loaded {frameData.Count} frames for sim{animSim}");
    }

    // R = denormalized depth (bg 0), G = denormalized thickness (>=0, bg 0), B = binary alpha.
    Color[] Parse(byte[] raw, bool logitMask)
    {
        float[] d = new float[7 * HW];
        Buffer.BlockCopy(raw, 0, d, 0, raw.Length);
        var px = new Color[HW];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                int i = y * W + x;
                int flipped = (H - 1 - y) * W + x;    // tensor row 0 = top; texture row 0 = bottom
                float m = d[6 * HW + i];
                float a = (logitMask ? m > 0f : m > 0.5f) ? 1f : 0f;
                float depth = a > 0f ? d[i] * DEPTH_STD + DEPTH_MEAN : 0f;
                float thick = a > 0f ? Mathf.Max(d[HW + i] * THICK_STD + THICK_MEAN, 0f) : 0f;
                px[flipped] = new Color(depth, thick, a, 1f);
            }
        return px;
    }

    void SetPixelsFor(int src, Color[] px)
    {
        fields[src].SetPixels(px);
        fields[src].Apply(false, false);
    }

    void SetFrame(int idx)
    {
        curFrame = idx;
        for (int s = 0; s < 3; s++) SetPixelsFor(s, frameData[idx][s]);
    }

    float MedianFgThickness(string path) => MedianFgThickness(File.ReadAllBytes(path));

    float MedianFgThickness(byte[] raw)
    {
        float[] d = new float[7 * HW];
        Buffer.BlockCopy(raw, 0, d, 0, raw.Length);
        var vals = new List<float>(HW / 2);
        for (int i = 0; i < HW; i++)
            if (d[6 * HW + i] > 0.5f) vals.Add(d[HW + i] * THICK_STD + THICK_MEAN);
        if (vals.Count == 0) return 1f;
        vals.Sort();
        return vals[vals.Count / 2];
    }

    void Update()
    {
        if (mat == null) return;

        if (animate && frameData.Count > 1)
        {
            int idx;
            if (capturing)
            {
                idx = curFrame + 1;                       // deterministic: one anim frame per editor frame
                if (idx >= frameData.Count) { capturing = false; idx = 0; playT = 0f; }
            }
            else
            {
                playT += Time.deltaTime * playbackFps;
                idx = (int)playT % frameData.Count;
            }
            if (idx != curFrame)
            {
                SetFrame(idx);
                if (capturing)
                    ScreenCapture.CaptureScreenshot(Path.Combine(captureDir, $"frame_{idx:D3}.png"));
            }
        }

        mat.SetFloat("_Focal", 1f / Mathf.Tan(fovDeg * Mathf.Deg2Rad / 2f));
        mat.SetFloat("_KThick", kThick);
        mat.SetFloat("_RefrStrength", refrStrength);
        mat.SetFloat("_F0", 0.02f);
        mat.SetFloat("_Shininess", shininess);
        mat.SetFloat("_Ks", ks);
        mat.SetFloat("_FrMax", 1.0f);
        mat.SetVector("_SpecTint", new Vector4(1f, 1f, 1f, 0f));
        mat.SetVector("_AbsorbSigma", new Vector4(0.45f, 0.16f, 0.10f, 0f));
        mat.SetVector("_LightDir", new Vector4(-0.5f, 0.8f, 0.6f, 0f));
        mat.SetVector("_SkyTop", new Vector4(0.27f, 0.47f, 0.78f, 0f));
        mat.SetVector("_SkyHor", new Vector4(0.80f, 0.88f, 0.95f, 0f));
        mat.SetVector("_FloorA", new Vector4(0.30f, 0.34f, 0.40f, 0f));
        mat.SetVector("_FloorB", new Vector4(0.16f, 0.19f, 0.24f, 0f));
        mat.SetVector("_BodyTint", new Vector4(0.04f, 0.10f, 0.14f, 0f));
        float[] sigmas = { presmoothLR, presmoothPred, presmoothGT };
        for (int i = 0; i < 3; i++)
        {
            mat.SetFloat("_BlurSigma", sigmas[i]);
            Graphics.Blit(fields[i], smoothed[i], mat, 0);
            Graphics.Blit(smoothed[i], shaded[i], mat, 1);
        }
    }

    void OnGUI()
    {
        string frameTag = animate && curFrame >= 0 ? $"  frame {frameNums[curFrame]}/{frameNums[frameNums.Count - 1]}" : "";
        GUI.Label(new Rect(10, 8, 1400, 24),
            $"Screen-space fluid rendering (van der Laan) — port of fluid_shader.py   {info}{frameTag}");
        if (shaded[0] == null) return;
        int s = Mathf.Min(Screen.width / 3 - 12, Screen.height - 70);
        for (int i = 0; i < 3; i++)
        {
            GUI.Label(new Rect(8 + i * (s + 8), 34, s, 20), labels[i]);
            GUI.DrawTexture(new Rect(8 + i * (s + 8), 56, s, s), shaded[i], ScaleMode.ScaleToFit);
        }
    }

    void OnDestroy()
    {
        foreach (var t in fields) if (t != null) Destroy(t);
        foreach (var rt in smoothed) if (rt != null) rt.Release();
        foreach (var rt in shaded) if (rt != null) rt.Release();
    }
}
