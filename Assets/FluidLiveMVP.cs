// FluidLiveMVP.cs — live fluid MVP: baked LR particle playback → in-engine splat (bit-parity
// with the ML repo's splatRender.py) → ONNX upsampler (Inference Engine) → SSFR liquid shading,
// fullscreen, with a user-flyable camera.
//
// The fluid never touches a Unity Camera: all math stays in sim space (right-handed, y-up,
// exact Python conventions), a virtual SimCamera is driven by input, and the shaded result is
// drawn via OnGUI. Vector3 is used purely as a float3 container (Cross/Dot are component math).
//
// Controls: WASD fly (+Q/E world down/up, Shift ×3), hold RMB + mouse to look,
// O auto-orbit, B bilateral smoothing, P pause sim clock, R restart, [ / ] playback fps −/+5,
// 1-9 switch baked sim slot, V toggle the side-by-side RAW INPUT panel.
//
// Sim Slots (inspector): optional list of (label, particleData, metaJson) — when non-empty it
// overrides the single Assets fields and 1-9 switch between them at runtime (camera jumps to
// each bake's own parity pose). The raw panel shades the model's INPUT fields (the true
// low-res SPH splat, pre-normalization) through the exact same smoothing+SSFR path as the
// prediction, so the side-by-side isolates what the network adds.
//
// Compare Models (inspector): optional list of extra ONNX upsamplers. Every model runs on the
// SAME splatted input tensor each frame and gets its own panel, laid out left→right as
// [RAW] | model 0 | model 1 | ... — same camera, same sim frame, same shading settings, so any
// visible difference is the network's. Cost is linear: N models = N inferences per frame.
//
// parityDump mode: splats the reference frame (sim46 f20 camera) and writes the normalized
// 7×512×512 input tensor to disk for comparison against Assets/lr_sim46_f20.bytes — run
// compare_unity_splat.py in the ML repo. Gate before trusting anything downstream.

using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.InputSystem;
using Unity.InferenceEngine;

public class FluidLiveMVP : MonoBehaviour
{
    // ---------- particle frame source (swap for a live Unity SPH later) ----------

    public interface IParticleFrameSource
    {
        int FrameCount { get; }
        float NativeFps { get; }
        // Interleaved records of 7 floats: px py pz vx vy vz density (world space).
        // offset is a float index into data; the frame spans count records from there.
        void GetFrame(int idx, out float[] data, out int offset, out int count);
    }

    public sealed class BakedParticleFrames : IParticleFrameSource
    {
        readonly float[] all;
        readonly int[] starts, counts;
        readonly float fps;

        public BakedParticleFrames(byte[] raw, float nativeFps)
        {
            using var br = new BinaryReader(new MemoryStream(raw));
            int magic = br.ReadInt32();
            if (magic != 0x53504C31) throw new Exception($"sim13_lr.bytes: bad magic 0x{magic:X8}");
            int version = br.ReadInt32();
            if (version != 1) throw new Exception($"sim13_lr.bytes: unsupported version {version}");
            int n = br.ReadInt32();
            counts = new int[n];
            starts = new int[n];
            for (int i = 0; i < n; i++) counts[i] = br.ReadInt32();
            int total = 0;
            for (int i = 0; i < n; i++) { starts[i] = total * 7; total += counts[i]; }
            all = new float[total * 7];
            Buffer.BlockCopy(raw, 12 + 4 * n, all, 0, total * 7 * 4);
            fps = nativeFps;
        }

        public int FrameCount => counts.Length;
        public float NativeFps => fps;
        public void GetFrame(int idx, out float[] data, out int offset, out int count)
        {
            data = all; offset = starts[idx]; count = counts[idx];
        }
    }

    // ---------- virtual camera in sim space ----------

    struct SimCamera
    {
        public Vector3 eye;
        public float yawDeg, pitchDeg, fovDeg;
        public Vector3 right, trueUp, fwd;   // valid after UpdateBasisFromYawPitch/SetLookAt

        // look_at(): right = cross(fwd, up) normalized; trueUp = cross(right, fwd) NOT normalized.
        static void BasisFromForward(Vector3 f, out Vector3 right, out Vector3 trueUp)
        {
            right = Vector3.Cross(f, Vector3.up);
            right /= right.magnitude;
            trueUp = Vector3.Cross(right, f);
        }

        public void UpdateBasisFromYawPitch()
        {
            float p = pitchDeg * Mathf.Deg2Rad, y = yawDeg * Mathf.Deg2Rad;
            fwd = new Vector3(Mathf.Cos(p) * Mathf.Cos(y), Mathf.Sin(p), Mathf.Cos(p) * Mathf.Sin(y));
            BasisFromForward(fwd, out right, out trueUp);
        }

        public void SetLookAt(Vector3 e, Vector3 target)
        {
            eye = e;
            Vector3 f = target - e;
            f /= f.magnitude;
            fwd = f;
            BasisFromForward(f, out right, out trueUp);
            pitchDeg = Mathf.Asin(Mathf.Clamp(f.y, -1f, 1f)) * Mathf.Rad2Deg;
            yawDeg = Mathf.Atan2(f.z, f.x) * Mathf.Rad2Deg;
        }
    }

    // ---------- meta sidecar ----------

    [Serializable] class MetaCam { public double azimuth, elevation, distance, fov; }
    [Serializable] class Meta
    {
        public string source;
        public float fps;
        public int frameCount, factor;
        public float[] target, mean, std;
        public MetaCam parityCam;
        public int parityFrame, height, width, splatRadiusPx;
        public float sigmaPx;
        // renderer v2 fields (Helpers/patch_unity_meta_v2.py); absent in legacy metas
        public string renderer;
        public float coarseRadius, thicknessScale, minRadiusPx, maxRadiusPx;
    }

    // ---------- inspector ----------

    [Serializable]
    public class SimSlot
    {
        [Tooltip("Display name shown in the status line (e.g. \"emitter lowres 1\").")]
        public string label;
        [Tooltip("Baked particle sequence (FluidData/<name>_lr.bytes).")]
        public TextAsset particleData;
        [Tooltip("Sidecar metadata (FluidData/<name>_meta.json).")]
        public TextAsset metaJson;
    }

    [Serializable]
    public class ModelSlot
    {
        [Tooltip("Display name drawn over this model's panel (e.g. \"017a\"). Empty = the asset's own name.")]
        public string label;
        [Tooltip("Upsampler ONNX model to compare (FluidModels/model_XXX.onnx). Must take the same normalized 7×512×512 input and produce the same 7×512×512 output layout as the primary model — i.e. trained with the same normalization stats as the bake's meta json.")]
        public ModelAsset model;
    }

    [Header("Assets")]
    [Tooltip("Baked low-res particle sequence to play back (FluidData/sim13_lr.bytes). Required unless Sim Slots is non-empty.")]
    public TextAsset particleData;
    [Tooltip("Sidecar metadata for particleData — camera/target/normalization stats (FluidData/sim13_meta.json). Required unless Sim Slots is non-empty.")]
    public TextAsset metaJson;
    [Tooltip("Optional multi-sim playlist. When non-empty it overrides the two fields above; number keys 1-9 switch slots at runtime. Slot 0 loads on Play.")]
    public SimSlot[] simSlots;
    [Tooltip("Upsampler ONNX model to run each frame (model_017a.onnx). Required — this is the primary/reference model and always gets the rightmost-but-one panel position (first of the model panels).")]
    public ModelAsset modelAsset;
    [Tooltip("Display name drawn over the primary model's panel. Empty = \"NEURAL UPSAMPLE\" when it is the only model, otherwise the asset's own name.")]
    public string modelLabel = "";
    [Tooltip("Optional extra models to compare against Model Asset. Each one gets its own panel, fed the identical input tensor from the identical camera every frame, and is shaded through the identical SSFR path — so differences on screen are the networks'. Leave empty for the single-model view. WARNING: every model runs a full inference per frame, so N models means roughly N× the frame time (~56 ms each for 017a on GPUPixel).")]
    public ModelSlot[] compareModels;
    [Tooltip("Shading shader (Hidden/FluidSSFR). Optional — auto-resolved via Shader.Find if left empty.")]
    public Shader ssfrShader;

    public enum SplatMode { Auto, Legacy, V2 }
    [Header("Splat (input renderer version)")]
    [Tooltip("Which in-engine splatter builds the model input. Auto = follow the slot meta's `renderer` field (v2 metas → V2, old metas → legacy). Legacy = 5 px square Gaussian stamp, integer placement, constant depth (splatRender.splat_render, what 040-053 were trained on). V2 = subpixel spherical splats with the physical projected radius and volume-mode thickness (splatRender.splat_render_v2, what 058a+ were trained on). The model and the splatter MUST match, exactly like model and stats.")]
    public SplatMode splatMode = SplatMode.Auto;
    [Tooltip("Parity-dump mode only: which Sim Slot to dump (index into Sim Slots).")]
    public int paritySlot = 0;

    [Header("Foam / whitewater (059 compositor layer, key F)")]
    [Tooltip("Draw Ihmsen-style spray/foam/bubbles generated from the coarse particle frame (FoamLayer.cs, the in-engine port of the 059 probe) over every panel. Pure compositor: no network output involved; the layer is depth-tested against each panel's own fluid depth. Toggle at runtime with F.")]
    public bool foamEnabled = false;
    [Tooltip("Spawn rate multiplier (the probe's mass factor). Higher = more whitewater. Recommended default: 30.")]
    public float foamSpawnScale = 30f;
    [Tooltip("Per-particle disk footprint scale (fraction of the projected coarse radius). 0 = single pixel (the probe's look). Recommended default: 1.")]
    public float foamSpriteScale = 1f;
    [Tooltip("Trapped-air / wave-crest spawn constants per second per unit potential (probe defaults 8 / 12).")]
    public float foamKTa = 8f, foamKWc = 12f;
    [Tooltip("Multiplier on the running potential calibration. Lower = potentials saturate sooner = more, earlier foam. Recommended default: 1.")]
    public float foamTauScale = 1f;
    [Tooltip("Coverage constant: alpha = 1 - exp(-k * density). Recommended default: 2.")]
    public float foamCoverageK = 2.0f;
    [Tooltip("Cap on live diffuse particles; the oldest are dropped past it. CPU cost scales with this. Recommended default: 40000.")]
    public int foamMaxDiffuse = 40000;
    [Tooltip("Whitewater colour (sRGB, composited before the shader's gamma conversion).")]
    public Color foamColor = new Color(0.96f, 0.98f, 1f, 1f);
    [Tooltip("Gravity for ballistic spray / buoyant bubbles, sim units. Dam sims use (0,-9.81,0); tilted-gravity bakes differ slightly.")]
    public Vector3 foamGravity = new Vector3(0f, -9.81f, 0f);

    [Header("Playback")]
    [Tooltip("Simulation playback rate in frames/sec. Higher = the baked sequence advances faster (more motion per real second); lower = slower motion. Recommended default: 25.")]
    public float playbackFps = 25f;
    [Tooltip("Global time multiplier applied on top of Playback Fps. Higher = fast-forward; lower (<1) = slow motion; 1 = normal speed. Recommended default: 1.")]
    public float speed = 1f;

    [Header("Camera")]
    [Tooltip("Vertical field of view in degrees, shared by the particle splat and the SSFR shader's focal length. Higher = wider view (more scene visible, more distortion); lower = narrower/more zoomed-in view. Recommended default: 49.88 (matches the baked parity camera — changing it breaks the parity gate).")]
    public float fovDeg = 49.880380582f;
    [Tooltip("WASD fly speed in world units/sec (×3 while holding Shift). Higher = faster movement; lower = slower, more precise movement. Recommended default: 2.")]
    public float moveSpeed = 2f;
    [Tooltip("Mouse-look sensitivity in degrees per pixel of mouse movement (RMB drag). Higher = faster/more sensitive turning; lower = slower, finer aiming. Recommended default: 0.15.")]
    public float lookSpeedDegPerPx = 0.15f;
    [Tooltip("Seconds for one full 360° auto-orbit revolution (O key). Higher = slower orbit; lower = faster orbit. Recommended default: 12.")]
    public float orbitPeriodSec = 12f;
    [Tooltip("Auto-orbit camera elevation angle above the target, in degrees. Higher = camera looks down more steeply from above; lower (including negative) = camera sits closer to eye-level or below the target. Recommended default: 23.42.")]
    public float orbitElevationDeg = 23.4168303f;
    [Tooltip("Auto-orbit camera distance from the target. Higher = camera further away (more zoomed out); lower = camera closer (more zoomed in). Recommended default: 4.91.")]
    public float orbitDistance = 4.9055318f;

    [Header("Inference")]
    [Tooltip("Inference Engine backend used to run the model. WARNING: GPUCompute is silently wrong for model_017a (bad foreground mask, no exception thrown — verified against onnxruntime). GPUPixel + fp32 is the validated, shipped config. Recommended default: GPUPixel.")]
    public BackendType backend = BackendType.GPUPixel;
    [Tooltip("Attempt fp16 weight quantization for a smaller/faster model. Currently fp16 compiles but throws at Schedule time on every backend for model_017a, so Init() smoke-tests it and silently rebuilds fp32 on failure — this toggle has no real effect until that's fixed upstream. Recommended default: true (harmless; auto-falls-back).")]
    public bool useFp16 = true;
    [Tooltip("Manual override for the shader's thickness scale (_KThick). 0 = auto-derive each frame from the model's own predicted thickness (recommended). Any value >0 locks it: higher values make the fluid read as thicker/more light-absorbing for the same geometry, lower values make it read as thinner/more transparent. Auto-derived values are typically ~0.06-0.63 depending on scene scale. Recommended default: 0 (auto).")]
    public float kThickOverride = 0f;

    [Header("View")]
    [Tooltip("Side-by-side view: adds a LEFTMOST panel showing the raw true low-res SPH input (the model's own input fields, shaded through the identical smoothing+SSFR path) next to the model panel(s). Toggle at runtime with V. Recommended default: true for the low-res demo.")]
    public bool showRawSideBySide = true;
    [Tooltip("When comparing several models, shade every panel with the PRIMARY model's auto-derived thickness scale instead of each model deriving its own. On = a like-for-like comparison (a model that predicts thicker fluid actually looks thicker/darker). Off = each panel self-normalizes its brightness, which flatters a model whose thickness is off-scale. Ignored when K Thick Override > 0 (that value is used everywhere). Recommended default: true.")]
    public bool shareKThickAcrossModels = true;

    [Header("Shading")]
    [Tooltip("Gaussian blur sigma (in pixels) applied to the masked depth buffer before shading. Higher = smoother/softer surface with less high-frequency noise but less fine detail; lower = sharper, noisier surface; 0 = no blur (raw prediction). Recommended default: 2 (matches the ground-truth reference render).")]
    public float presmoothSigma = 2f;
    [Tooltip("Use an edge-preserving bilateral blur (toggle with B) instead of the plain Gaussian above. Reduces the checkerboard artifact a plain Gaussian can amplify while keeping edges sharp. Recommended default: false (off) — enable if you see checkerboarding.")]
    public bool bilateralSmoothing = false;
    [Tooltip("Bilateral blur spatial sigma (pixel-space falloff, kernel radius = 2×this). Higher = wider blur footprint (more smoothing); lower = tighter/narrower blur. Only used when Bilateral Smoothing is on. Recommended default: 3 (reference sigma_s).")]
    public float bilateralSigmaS = 3f;
    [Tooltip("Number of bilateral blur passes. Higher = more smoothing (softer result, more GPU cost); lower = less smoothing, closer to the raw signal. Only used when Bilateral Smoothing is on. Recommended default: 4 (reference iteration count).")]
    public int bilateralIters = 4;
    [Tooltip("Multiplier on the per-frame foreground-depth std-dev used to derive the bilateral range sigma (edge sensitivity). Higher = blurs across larger depth discontinuities (smooths over more edges); lower = preserves edges more strictly (sharper depth transitions). Only used when Bilateral Smoothing is on. Recommended default: 0.12 (reference default).")]
    public float bilateralRangeScale = 0.12f;
    [Tooltip("Refraction distortion strength in the SSFR shader. Higher = more warped/bent refraction through the fluid; lower = flatter, less distorted refraction. Recommended default: 22.")]
    public float refrStrength = 22f;
    [Tooltip("Specular reflectance coefficient. Higher = brighter, more prominent specular highlights (shinier surface); lower = dimmer highlights (more matte surface). Recommended default: 0.7.")]
    public float ks = 0.7f;
    [Tooltip("Specular highlight exponent (Blinn-Phong). Higher = smaller, tighter, sharper highlights (glossier); lower = broader, softer highlights (more diffuse-looking). Recommended default: 120.")]
    public float shininess = 120f;

    [Header("Color / Look")]
    [Tooltip("Hand-tuned blood look: red-dominant Beer-Lambert absorption strong enough that even thin/airy splash regions read as red (not clear/transparent), a dark near-black-red pooled color for thick regions, a dark moody background/reflection palette instead of blue sky, a capped Fresnel reflection and a dim warm-tinted specular so grazing-angle splashes/foam don't flash white/shiny. Overrides Default Look / Absorb Color / Body Tint below while on. Recommended default: true for the blood-fountain use case; false restores the original validated water look.")]
    public bool bloodMode = true;
    [Tooltip("When on (and Blood Mode is off), shades the fluid with the original validated water look and ignores Absorb Color / Body Tint below. Turn off to use the custom colors below. Recommended default: true.")]
    public bool defaultLook = true;
    [Tooltip("Per-channel light absorption through the fluid (Beer-Lambert). This is SUBTRACTIVE: a HIGHER value in a channel removes more of that color as the fluid gets thicker, a LOWER value lets that channel survive — so the fluid's visible color is roughly the inverse/complement of this. Only used when Default Look is off. Recommended default (water, for reference): R 0.45, G 0.16, B 0.10 (removes red fastest → blue tint). Preset here is tuned for a blood look (low red absorption, high green/blue absorption → red survives).")]
    public Color absorbColor = new Color(0.12f, 0.85f, 0.80f);
    [Tooltip("Color the fluid fades toward once fully opaque (thick/pooled regions where Beer-Lambert transmission drops to ~0) — its 'interior' color. Higher per-channel values brighten that channel in deep areas; lower darkens it. Only used when Default Look is off. Recommended default (water, for reference): a dark blue-grey (0.04, 0.10, 0.14). Preset here is a dark pooled-blood red.")]
    public Color bodyTintColor = new Color(0.35f, 0.02f, 0.02f);

    [Header("Parity dump (writes input tensor + quits; no inference)")]
    [Tooltip("Debug/validation mode. When true, Play writes one reference input tensor to disk and quits instead of running the live loop — used to bit-compare the in-engine splat against the ML repo's reference via compare_unity_splat.py. Recommended default: false (leave off for normal use).")]
    public bool parityDump = false;
    [Tooltip("Output file path for the parity dump tensor (relative to the project root). Only used when Parity Dump is enabled. Recommended default: ParityDump/unity_lr_sim46_f20.bytes.")]
    public string parityOutPath = "ParityDump/unity_lr_sim46_f20.bytes";

    [Header("Deterministic capture (fixed-step orbit -> PNG per frame; smooth regardless of runtime fps)")]
    [Tooltip("If non-empty, enables deterministic capture: a fixed-step auto-orbit that writes one PNG per frame to this folder (encode with ffmpeg afterward), producing smooth video regardless of actual editor framerate. Empty = normal interactive playback. Recommended default: empty (disabled).")]
    public string captureDir = "";
    [Tooltip("Fixed simulation/render step rate used during capture, and the frame rate of the resulting video. Higher = smoother output video but more frames to render and encode; lower = choppier video, faster to capture. Recommended default: 30.")]
    public float captureFps = 30f;
    [Tooltip("Total capture duration in seconds before it stops automatically. Higher = longer output video; lower = shorter. Recommended default: 18 (used for the shipped orbit_mvp.mp4).")]
    public float captureSeconds = 18f;

    // ---------- state ----------

    const int H = 512, W = 512, HW = H * W;
    const int SplatR = 5;                  // LR splat: radius 5 px square stamp, sigma 2.0
    const float SigmaPx = 2f;
    bool useV2;                                      // resolved per slot in LoadSim
    float v2Radius, v2ThickScale, v2MinR = 1f, v2MaxR = 24f;
    FoamLayer foam;                                  // 059 whitewater layer (CPU)
    Texture2D foamTexNet, foamTexRaw;                // RFloat, texture rows (row 0 = bottom)
    float[] foamDenNet, foamDenRaw, foamTmp, foamStage, rawDepthCHW;
    float lastSimTime; int lastFrameIdx = -1;

    // Everything one model needs to go from the shared input tensor to its own shaded panel.
    // The shading resources are per-model (not shared/reused) so all panels stay live at once.
    sealed class ModelPanel
    {
        public string label;
        public Worker worker;
        public bool fp16Active;
        public Color[] px;                                   // shaded-field staging
        public Texture2D fieldTex;
        public RenderTexture smoothedRT, smoothedRT2, shadedRT;   // RT2 = bilateral ping-pong
        public float bilateralSigmaR;
        public float kThick;
        public bool kThickLocked;
        public float inferMs;
        public float[] depthCHW;                             // predicted front depth, world units, 0 = bg (foam depth test)
    }

    IParticleFrameSource src;
    Meta meta;
    Vector3 target;
    SimCamera cam;

    float[] kernel;                                  // 11×11 gaussian, row-major (oy, ox)
    float[] depthBuf, thickBuf, velXBuf, velYBuf, velZBuf, denBuf;
    float[] in7;                                     // normalized (7,H,W) CHW model input

    ModelPanel[] panels = Array.Empty<ModelPanel>();

    Material mat;
    Texture2D rawFieldTex;
    RenderTexture rawSmoothedRT, rawSmoothedRT2, rawShadedRT;
    float rawBilateralSigmaR;
    Color[] rawPx;
    int currentSlot = -1;
    string simLabel = "";

    bool paused, orbit;
    bool capturing;
    int capFrame;
    float simTime, orbitAzDeg;
    int frameIdx;
    float smoothedFps;
    string status = "initializing...";
    bool ready;

    void Start()
    {
        try { Init(); }
        catch (Exception e) { status = $"ERROR: {e.Message}"; Debug.LogException(e); }
    }

    void Init()
    {
        LoadSim(parityDump ? paritySlot : 0);

        kernel = new float[(2 * SplatR + 1) * (2 * SplatR + 1)];
        int k = 0;
        for (int oy = -SplatR; oy <= SplatR; oy++)
            for (int ox = -SplatR; ox <= SplatR; ox++, k++)
                kernel[k] = (float)Math.Exp(-(ox * ox + oy * oy) / (2.0 * SigmaPx * SigmaPx));

        depthBuf = new float[HW]; thickBuf = new float[HW];
        velXBuf = new float[HW]; velYBuf = new float[HW]; velZBuf = new float[HW];
        denBuf = new float[HW];
        in7 = new float[7 * HW];
        rawPx = new Color[HW];
        foam = new FoamLayer();
        foamDenNet = new float[HW]; foamDenRaw = new float[HW]; foamTmp = new float[HW]; foamStage = new float[HW];
        rawDepthCHW = new float[HW];
        foamTexNet = new Texture2D(W, H, TextureFormat.RFloat, false) { filterMode = FilterMode.Bilinear };
        foamTexRaw = new Texture2D(W, H, TextureFormat.RFloat, false) { filterMode = FilterMode.Bilinear };

        if (parityDump) { RunParityDump(); return; }

        if (ssfrShader == null) ssfrShader = Shader.Find("Hidden/FluidSSFR");
        if (ssfrShader == null) throw new Exception("Hidden/FluidSSFR shader not found");
        mat = new Material(ssfrShader);
        rawFieldTex = new Texture2D(W, H, TextureFormat.RGBAFloat, false) { filterMode = FilterMode.Point };
        rawSmoothedRT = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBFloat) { filterMode = FilterMode.Point };
        rawSmoothedRT2 = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBFloat) { filterMode = FilterMode.Point };
        rawShadedRT = new RenderTexture(W, H, 0, RenderTextureFormat.ARGB32);

        SplatToInput(0, in cam);   // gives BuildPanel a real tensor for its smoke-test inference
        BuildPanels();

        capturing = captureDir.Length > 0;
        if (capturing) { Directory.CreateDirectory(captureDir); orbit = true; }
        ready = true;
    }

    // modelAsset first, then every assigned Compare Models entry, in inspector order.
    void BuildPanels()
    {
        if (modelAsset == null) throw new Exception("modelAsset not assigned");
        var list = new List<ModelPanel>();
        bool solo = compareModels == null || compareModels.Length == 0;
        list.Add(BuildPanel(modelAsset, string.IsNullOrEmpty(modelLabel)
                                        ? (solo ? "NEURAL UPSAMPLE" : modelAsset.name)
                                        : modelLabel));
        if (compareModels != null)
            foreach (var s in compareModels)
            {
                if (s == null || s.model == null) continue;
                list.Add(BuildPanel(s.model, string.IsNullOrEmpty(s.label) ? s.model.name : s.label));
            }
        panels = list.ToArray();
    }

    ModelPanel BuildPanel(ModelAsset asset, string label)
    {
        var p = new ModelPanel
        {
            label = label,
            px = new Color[HW],
            fieldTex = new Texture2D(W, H, TextureFormat.RGBAFloat, false) { filterMode = FilterMode.Point },
            smoothedRT = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBFloat) { filterMode = FilterMode.Point },
            smoothedRT2 = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBFloat) { filterMode = FilterMode.Point },
            shadedRT = new RenderTexture(W, H, 0, RenderTextureFormat.ARGB32),
            kThick = kThickOverride,
        };

        var model = ModelLoader.Load(asset);
        if (useFp16)
        {
            try
            {
                ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model);
                p.fp16Active = true;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[{label}] fp16 weight quantization failed ({e.Message}) — falling back to fp32");
                model = ModelLoader.Load(asset);
            }
        }
        // Smoke-test one inference now: a broken fp16 graph only throws at schedule time,
        // and the first schedule is the slow backend compile anyway.
        p.worker = new Worker(model, backend);
        try { RunModel(p.worker); }
        catch (Exception e) when (p.fp16Active)
        {
            Debug.LogWarning($"[{label}] fp16 model failed at schedule time ({e.Message}) — rebuilding fp32");
            p.worker.Dispose();
            p.fp16Active = false;
            p.worker = new Worker(ModelLoader.Load(asset), backend);
            RunModel(p.worker);
        }
        return p;
    }

    // Loads slot idx of simSlots (or the single particleData/metaJson fields when the list is
    // empty), resets the sim clock + kThick lock, and jumps the camera to the bake's own
    // parity pose (each bake_unity_live_sim.py output carries a freshly sampled TRAIN_CAM).
    void LoadSim(int idx)
    {
        TextAsset pd = particleData, mj = metaJson;
        simLabel = "";
        if (simSlots != null && simSlots.Length > 0)
        {
            idx = Mathf.Clamp(idx, 0, simSlots.Length - 1);
            pd = simSlots[idx].particleData;
            mj = simSlots[idx].metaJson;
            simLabel = string.IsNullOrEmpty(simSlots[idx].label) ? $"slot {idx + 1}" : simSlots[idx].label;
        }
        if (pd == null || mj == null) throw new Exception($"sim slot {idx}: particleData/metaJson not assigned");
        meta = JsonUtility.FromJson<Meta>(mj.text);
        target = new Vector3(meta.target[0], meta.target[1], meta.target[2]);
        src = new BakedParticleFrames(pd.bytes, meta.fps);
        currentSlot = idx;
        useV2 = splatMode == SplatMode.V2 || (splatMode == SplatMode.Auto && meta.renderer == "v2");
        if (useV2)
        {
            if (!(meta.coarseRadius > 0f))
                throw new Exception($"{mj.name}: v2 splat needs coarseRadius > 0 in the meta (run Helpers/patch_unity_meta_v2.py)");
            v2Radius = meta.coarseRadius;
            v2ThickScale = meta.thicknessScale > 0f ? meta.thicknessScale : SplatV2.ThicknessScaleLegacy;
            v2MinR = meta.minRadiusPx > 0f ? meta.minRadiusPx : 1f;
            v2MaxR = meta.maxRadiusPx > 0f ? meta.maxRadiusPx : 24f;
        }
        Debug.Log($"FluidLiveMVP: slot {idx} '{simLabel}' splat={(useV2 ? $"v2 r={v2Radius:F5} m, thicknessScale={v2ThickScale:F4}" : "legacy 5px/σ2")}");

        simTime = 0f;
        foreach (var p in panels) { p.kThickLocked = false; p.kThick = kThickOverride; }
        fovDeg = (float)meta.parityCam.fov;
        orbitAzDeg = (float)meta.parityCam.azimuth;
        orbitElevationDeg = (float)meta.parityCam.elevation;
        orbitDistance = (float)meta.parityCam.distance;
        cam.fovDeg = fovDeg;
        cam.SetLookAt(SphericalEye(target, orbitAzDeg, orbitElevationDeg, orbitDistance), target);
    }

    float[] RunModel(Worker w)
    {
        using var input = new Tensor<float>(new TensorShape(1, 7, H, W), in7);
        w.Schedule(input);
        using var t = (w.PeekOutput() as Tensor<float>).ReadbackAndClone();
        return t.DownloadToArray();
    }

    static Vector3 SphericalEye(Vector3 target, double azDeg, double elDeg, double dist)
    {
        double az = azDeg * Math.PI / 180.0, el = elDeg * Math.PI / 180.0;
        return new Vector3(
            (float)(target.x + dist * Math.Cos(el) * Math.Cos(az)),
            (float)(target.y + dist * Math.Sin(el)),
            (float)(target.z + dist * Math.Cos(el) * Math.Sin(az)));
    }

    void RunParityDump()
    {
        var pc = meta.parityCam;
        var pcam = new SimCamera { fovDeg = (float)pc.fov };
        pcam.SetLookAt(SphericalEye(target, pc.azimuth, pc.elevation, pc.distance), target);
        SplatToInput(Math.Max(meta.parityFrame - 1, 0), in pcam);   // sim13: dataset f20 = export index 19; parityFrame 0 metas -> index 0

        var bytes = new byte[in7.Length * 4];
        Buffer.BlockCopy(in7, 0, bytes, 0, bytes.Length);
        string dir = Path.GetDirectoryName(parityOutPath);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllBytes(parityOutPath, bytes);
        status = $"parity dump written: {parityOutPath}";
        Debug.Log($"parity dump written: {parityOutPath} ({bytes.Length} bytes)  " +
                  $"eye={pcam.eye.x:F6},{pcam.eye.y:F6},{pcam.eye.z:F6} fov={pcam.fovDeg:F6}");
    }

    // ---------- splat: exact port of splatRender.splat_render (LR settings) ----------

    void SplatToInput(int frame, in SimCamera c)
    {
        Array.Fill(depthBuf, 1e6f);
        Array.Clear(thickBuf, 0, HW);
        Array.Clear(velXBuf, 0, HW);
        Array.Clear(velYBuf, 0, HW);
        Array.Clear(velZBuf, 0, HW);
        Array.Clear(denBuf, 0, HW);

        src.GetFrame(frame, out var data, out int off, out int count);
        float focal = (float)(1.0 / Math.Tan(c.fovDeg * Math.PI / 180.0 / 2.0));   // aspect = 1
        Vector3 right = c.right, up = c.trueUp, fwd = c.fwd, eye = c.eye;
        if (useV2)
        {
            SplatV2.Splat(data, off, count, eye, right, up, fwd, focal, H, W, v2Radius, v2ThickScale, v2MinR, v2MaxR,
                          depthBuf, thickBuf, velXBuf, velYBuf, velZBuf, denBuf);
        }
        else
        {
        const int pad = SplatR + 1;

        for (int pi = 0; pi < count; pi++)
        {
            int b = off + pi * 7;
            float rx = data[b] - eye.x, ry = data[b + 1] - eye.y, rz = data[b + 2] - eye.z;
            float cx = rx * right.x + ry * right.y + rz * right.z;
            float cy = rx * up.x + ry * up.y + rz * up.z;
            float cz = -(rx * fwd.x + ry * fwd.y + rz * fwd.z);   // camera looks -z
            if (!(cz < -1e-3f)) continue;                          // near clip

            float ndcX = focal * cx / (-cz);
            float ndcY = focal * cy / (-cz);
            float pxf = (ndcX + 1f) * 0.5f * W;
            float pyf = (1f - ndcY) * 0.5f * H;                    // row 0 = top
            if (!(pxf > -pad && pxf < W + pad && pyf > -pad && pyf < H + pad)) continue;

            int cxi = (int)Math.Round(pxf, MidpointRounding.ToEven);
            int cyi = (int)Math.Round(pyf, MidpointRounding.ToEven);
            float depthVal = -cz;
            float wx = data[b + 3], wy = data[b + 4], wz = data[b + 5], den = data[b + 6];
            float vcx = wx * right.x + wy * right.y + wz * right.z;   // v_world @ R.T
            float vcy = wx * up.x + wy * up.y + wz * up.z;
            float vcz = -(wx * fwd.x + wy * fwd.y + wz * fwd.z);

            int kk = 0;
            for (int oy = -SplatR; oy <= SplatR; oy++)
            {
                int ty = cyi + oy;
                for (int ox = -SplatR; ox <= SplatR; ox++, kk++)
                {
                    int tx = cxi + ox;
                    if (tx < 0 || tx >= W || ty < 0 || ty >= H) continue;
                    int i = ty * W + tx;
                    float w = kernel[kk];
                    thickBuf[i] += w;
                    if (depthVal < depthBuf[i]) depthBuf[i] = depthVal;  // min over whole stamp
                    velXBuf[i] += vcx * w;
                    velYBuf[i] += vcy * w;
                    velZBuf[i] += vcz * w;
                    denBuf[i] += den * w;
                }
            }
        }
        }   // legacy

        // finalize + normalize into CHW (row 0 = top — no flip on the input side)
        float m0 = meta.mean[0], m1 = meta.mean[1], m2 = meta.mean[2], m3 = meta.mean[3], m4 = meta.mean[4], m5 = meta.mean[5];
        float s0 = 1f / meta.std[0], s1 = 1f / meta.std[1], s2 = 1f / meta.std[2], s3 = 1f / meta.std[3], s4 = 1f / meta.std[4], s5 = 1f / meta.std[5];
        for (int i = 0; i < HW; i++)
        {
            float t = thickBuf[i];
            float mask = useV2 ? (t > 0f ? 1f : 0f) : (t > 1e-4f ? 1f : 0f);   // v2: compact support, exact > 0
            float inv = useV2 ? (t > 0f ? 1f / t : 0f) : 1f / (t + 1e-6f);
            float d = mask > 0f ? depthBuf[i] : 0f;                 // sentinel 1e6 → 0 in bg
            in7[i] = (d - m0) * s0;
            in7[HW + i] = (t - m1) * s1;
            in7[2 * HW + i] = (velXBuf[i] * inv - m2) * s2;
            in7[3 * HW + i] = (velYBuf[i] * inv - m3) * s3;
            in7[4 * HW + i] = (velZBuf[i] * inv - m4) * s4;
            in7[5 * HW + i] = (denBuf[i] * inv - m5) * s5;
            in7[6 * HW + i] = mask;
        }
    }

    // ---------- per-frame loop ----------

    void Update()
    {
        if (!ready) return;

        float dt = capturing ? 1f / captureFps : Time.deltaTime;   // fixed step while capturing

        HandleInput();
        if (orbit)
        {
            orbitAzDeg += 360f * dt / Mathf.Max(orbitPeriodSec, 0.1f);
            cam.SetLookAt(SphericalEye(target, orbitAzDeg, orbitElevationDeg, orbitDistance), target);
        }
        cam.fovDeg = fovDeg;

        if (!paused) simTime += dt * speed;
        frameIdx = (int)(simTime * playbackFps) % src.FrameCount;   // auto-loop

        // One splat, one input tensor — every model sees byte-identical input from the same camera.
        SplatToInput(frameIdx, in cam);
        if (showRawSideBySide) FillRawFieldTex();

        // 059 whitewater: advance the diffuse particles in this frame's coarse field (sim time, not wall time)
        if (foamEnabled)
        {
            if (frameIdx < lastFrameIdx || simTime < lastSimTime) foam.Reset();      // playback looped / restarted
            float fdt = Mathf.Clamp(simTime - lastSimTime, 0f, 0.2f);
            foam.kTa = foamKTa; foam.kWc = foamKWc; foam.spawnScale = foamSpawnScale; foam.tauScale = foamTauScale;
            foam.maxDiffuse = Mathf.Max(foamMaxDiffuse, 1000); foam.gravity = foamGravity; foam.spriteScale = foamSpriteScale;
            src.GetFrame(frameIdx, out var fdata, out int foff, out int fcount);
            foam.Step(fdata, foff, fcount, FoamRadius(), fdt);
        }
        lastSimTime = simTime; lastFrameIdx = frameIdx;

        foreach (var pan in panels)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            float[] pred = RunModel(pan.worker);
            sw.Stop();
            float ms = (float)sw.Elapsed.TotalMilliseconds;
            pan.inferMs = Mathf.Lerp(pan.inferMs <= 0f ? ms : pan.inferMs, ms, 0.1f);

            // No GT live: derive kThick from the model's own thickness, but don't lock it until a
            // mid-sequence frame — frame 0 is the un-collapsed dam block, ~10× thicker than typical.
            if (kThickOverride > 0f) pan.kThick = kThickOverride;
            else if (!pan.kThickLocked)
            {
                pan.kThick = 1.2f / Mathf.Max(MedianFgThickness(pred), 1e-3f);
                if (frameIdx >= src.FrameCount / 2) pan.kThickLocked = true;
            }

            FillPanelField(pan, pred);
        }
        // Like-for-like: shade every model with the primary's scale so a thicker prediction
        // actually reads as thicker instead of each panel self-normalizing its own brightness.
        if (shareKThickAcrossModels)
            for (int i = 1; i < panels.Length; i++) panels[i].kThick = panels[0].kThick;

        mat.SetFloat("_Focal", 1f / Mathf.Tan(fovDeg * Mathf.Deg2Rad / 2f));
        mat.SetFloat("_RefrStrength", refrStrength);
        if (bloodMode)
        {
            mat.SetFloat("_F0", 0.02f);
            mat.SetFloat("_Shininess", 90f);
            mat.SetFloat("_Ks", 0.18f);
            mat.SetFloat("_FrMax", 0.18f);
            mat.SetVector("_SpecTint", new Vector4(1.0f, 0.45f, 0.4f, 0f));
            mat.SetVector("_AbsorbSigma", new Vector4(0.6f, 4.0f, 4.5f, 0f));
            mat.SetVector("_LightDir", new Vector4(-0.5f, 0.8f, 0.6f, 0f));
            mat.SetVector("_SkyTop", new Vector4(0.05f, 0.04f, 0.05f, 0f));
            mat.SetVector("_SkyHor", new Vector4(0.14f, 0.08f, 0.08f, 0f));
            mat.SetVector("_FloorA", new Vector4(0.10f, 0.07f, 0.07f, 0f));
            mat.SetVector("_FloorB", new Vector4(0.04f, 0.03f, 0.03f, 0f));
            mat.SetVector("_BodyTint", new Vector4(0.15f, 0.01f, 0.01f, 0f));
        }
        else
        {
            mat.SetFloat("_F0", 0.02f);
            mat.SetFloat("_Shininess", shininess);
            mat.SetFloat("_Ks", ks);
            mat.SetFloat("_FrMax", 1.0f);
            mat.SetVector("_SpecTint", new Vector4(1f, 1f, 1f, 0f));
            mat.SetVector("_AbsorbSigma", defaultLook ? new Vector4(0.45f, 0.16f, 0.10f, 0f) : (Vector4)absorbColor);
            mat.SetVector("_LightDir", new Vector4(-0.5f, 0.8f, 0.6f, 0f));
            mat.SetVector("_SkyTop", new Vector4(0.27f, 0.47f, 0.78f, 0f));
            mat.SetVector("_SkyHor", new Vector4(0.80f, 0.88f, 0.95f, 0f));
            mat.SetVector("_FloorA", new Vector4(0.30f, 0.34f, 0.40f, 0f));
            mat.SetVector("_FloorB", new Vector4(0.16f, 0.19f, 0.24f, 0f));
            mat.SetVector("_BodyTint", defaultLook ? new Vector4(0.04f, 0.10f, 0.14f, 0f) : (Vector4)bodyTintColor);
        }
        mat.SetFloat("_BlurSigma", bilateralSmoothing ? bilateralSigmaS : presmoothSigma);

        // foam density maps: one for the primary model's depth, one for the raw input's depth
        mat.SetFloat("_FoamOn", foamEnabled ? 1f : 0f);
        mat.SetFloat("_FoamK", foamCoverageK);
        mat.SetVector("_FoamColor", new Vector4(foamColor.r, foamColor.g, foamColor.b, 0f));
        if (foamEnabled)
        {
            float focalF = 1f / Mathf.Tan(fovDeg * Mathf.Deg2Rad / 2f);
            foam.Splat(cam.eye, cam.right, cam.trueUp, cam.fwd, focalF, H, W, panels[0].depthCHW, FoamRadius(), foamDenNet, foamTmp);
            UploadFoam(foamTexNet, foamDenNet);
            if (showRawSideBySide)
            {
                for (int i = 0; i < HW; i++) rawDepthCHW[i] = thickBuf[i] > 0f ? depthBuf[i] : 0f;
                foam.Splat(cam.eye, cam.right, cam.trueUp, cam.fwd, focalF, H, W, rawDepthCHW, FoamRadius(), foamDenRaw, foamTmp);
                UploadFoam(foamTexRaw, foamDenRaw);
            }
        }

        // identical smoothing+shading chain for every model; only the adaptive bilateral range
        // sigma is per-panel (same formula, each panel's own depth stats)
        foreach (var pan in panels)
        {
            mat.SetTexture("_FoamTex", foamTexNet);
            mat.SetFloat("_KThick", pan.kThick);
            mat.SetFloat("_BilateralRangeSigma", bilateralSmoothing ? pan.bilateralSigmaR : 0f);
            Graphics.Blit(pan.fieldTex, pan.smoothedRT, mat, 0);
            if (bilateralSmoothing)
                for (int it = 1; it < bilateralIters; it++)
                {
                    Graphics.Blit(pan.smoothedRT, pan.smoothedRT2, mat, 0);
                    (pan.smoothedRT, pan.smoothedRT2) = (pan.smoothedRT2, pan.smoothedRT);
                }
            Graphics.Blit(pan.smoothedRT, pan.shadedRT, mat, 1);
        }

        if (showRawSideBySide)
        {
            // same chain again on the raw input fields, at the primary model's thickness scale
            mat.SetTexture("_FoamTex", foamTexRaw);
            mat.SetFloat("_KThick", panels[0].kThick);
            mat.SetFloat("_BilateralRangeSigma", bilateralSmoothing ? rawBilateralSigmaR : 0f);
            Graphics.Blit(rawFieldTex, rawSmoothedRT, mat, 0);
            if (bilateralSmoothing)
                for (int it = 1; it < bilateralIters; it++)
                {
                    Graphics.Blit(rawSmoothedRT, rawSmoothedRT2, mat, 0);
                    (rawSmoothedRT, rawSmoothedRT2) = (rawSmoothedRT2, rawSmoothedRT);
                }
            Graphics.Blit(rawSmoothedRT, rawShadedRT, mat, 1);
        }

        if (capturing)
        {
            // warm up (no screenshots) until kThick has locked mid-sequence, so brightness is constant
            if (panels[0].kThickLocked || kThickOverride > 0f)
            {
                ScreenCapture.CaptureScreenshot(Path.Combine(captureDir, $"frame_{capFrame:D4}.png"));
                capFrame++;
                if (capFrame >= (int)(captureFps * captureSeconds))
                {
                    capturing = false;
                    Debug.Log($"capture done: {capFrame} frames in {captureDir}");
                }
            }
        }

        float fps = 1f / Mathf.Max(Time.deltaTime, 1e-5f);
        smoothedFps = smoothedFps <= 0f ? fps : Mathf.Lerp(smoothedFps, fps, 0.1f);
        var infer = new System.Text.StringBuilder();
        foreach (var pan in panels)
        {
            if (infer.Length > 0) infer.Append(" + ");
            infer.Append(panels.Length > 1 ? $"{pan.label} {pan.inferMs:F0}" : $"{pan.inferMs:F1}");
            if (pan.fp16Active) infer.Append("*");
        }
        status = $"{(simLabel.Length > 0 ? $"[{currentSlot + 1}: {simLabel}]   " : "")}" +
                 $"frame {frameIdx + 1}/{src.FrameCount}   sim {playbackFps:F0} fps ×{speed:F1}   " +
                 $"render {smoothedFps:F1} fps   infer {infer} ms   backend={backend}   splat={(useV2 ? "v2" : "legacy")}   " +
                 $"k_thick={panels[0].kThick:F3}{(bilateralSmoothing ? $"   BILATERAL σs={bilateralSigmaS:F0}×{bilateralIters} σr={panels[0].bilateralSigmaR:F3}" : "")}" +
                 $"{(foamEnabled ? $"   FOAM {foam.Alive} ({foam.Spray}s/{foam.Foam}f/{foam.Bubble}b) +{foam.SpawnedTa}/{foam.SpawnedWc} {foam.LastStepMs + 2f * foam.LastSplatMs:F1} ms" : "")}" +
                 $"{(paused ? "   PAUSED" : "")}{(orbit ? "   ORBIT" : "")}";
    }

    // world radius the foam layer uses for its support (4 r) and spawn cylinder: the slot's coarse radius
    float FoamRadius() => useV2 ? v2Radius : (meta != null && meta.coarseRadius > 0f ? meta.coarseRadius : 0.0414f);

    void UploadFoam(Texture2D tex, float[] denCHW)
    {
        for (int y = 0; y < H; y++)
            Array.Copy(denCHW, y * W, foamStage, (H - 1 - y) * W, W);   // tensor row 0 = top; texture row 0 = bottom
        tex.SetPixelData(foamStage, 0);
        tex.Apply(false, false);
    }

    // SSFRViewer pred semantics: ch6 = occupancy logit (>0 fluid), denorm depth/thickness, row flip.
    // Also derives this panel's own adaptive bilateral range sigma from its foreground depth stats.
    void FillPanelField(ModelPanel pan, float[] pred)
    {
        float dm = meta.mean[0], ds = meta.std[0], tm = meta.mean[1], ts = meta.std[1];
        double dSum = 0.0, dSumSq = 0.0;   // fg depth stats for the reference bilateral's adaptive sigma_r
        int fgCount = 0;
        var px = pan.px;
        if (pan.depthCHW == null) pan.depthCHW = new float[HW];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                int i = y * W + x;
                int flipped = (H - 1 - y) * W + x;   // tensor row 0 = top; texture row 0 = bottom
                float a = pred[6 * HW + i] > 0f ? 1f : 0f;
                float depth = a > 0f ? pred[i] * ds + dm : 0f;
                float thick = a > 0f ? Mathf.Max(pred[HW + i] * ts + tm, 0f) : 0f;
                px[flipped] = new Color(depth, thick, a, 1f);
                pan.depthCHW[i] = depth;
                if (a > 0f) { dSum += depth; dSumSq += (double)depth * depth; fgCount++; }
            }
        // sigma_r = max(0.12 * std(fg depth), 1e-3), unbiased std like torch.std() (n-1).
        pan.bilateralSigmaR = 0.05f;   // reference fallback when no fg pixels
        if (fgCount > 1)
        {
            double var = (dSumSq - dSum * dSum / fgCount) / (fgCount - 1);
            pan.bilateralSigmaR = Mathf.Max(bilateralRangeScale * (float)Math.Sqrt(Math.Max(var, 0.0)), 1e-3f);
        }
        pan.fieldTex.SetPixels(px);
        pan.fieldTex.Apply(false, false);
    }

    // Stages the model's INPUT depth/thickness (the true low-res SPH splat, world units,
    // pre-normalization, same buffers SplatToInput just filled) into rawFieldTex with the
    // (depth, thick, mask) layout the shader expects, and derives the raw panel's own
    // adaptive bilateral range sigma with the reference formula.
    void FillRawFieldTex()
    {
        double dSum = 0.0, dSumSq = 0.0;
        int fgCount = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                int i = y * W + x;
                int flipped = (H - 1 - y) * W + x;   // tensor row 0 = top; texture row 0 = bottom
                float t = thickBuf[i];
                float a = t > 1e-4f ? 1f : 0f;
                rawPx[flipped] = new Color(a > 0f ? depthBuf[i] : 0f, a > 0f ? t : 0f, a, 1f);
                if (a > 0f) { dSum += depthBuf[i]; dSumSq += (double)depthBuf[i] * depthBuf[i]; fgCount++; }
            }
        rawBilateralSigmaR = 0.05f;
        if (fgCount > 1)
        {
            double var = (dSumSq - dSum * dSum / fgCount) / (fgCount - 1);
            rawBilateralSigmaR = Mathf.Max(bilateralRangeScale * (float)Math.Sqrt(Math.Max(var, 0.0)), 1e-3f);
        }
        rawFieldTex.SetPixels(rawPx);
        rawFieldTex.Apply(false, false);
    }

    float MedianFgThickness(float[] pred)
    {
        var vals = new List<float>(HW / 4);
        float tm = meta.mean[1], ts = meta.std[1];
        for (int i = 0; i < HW; i++)
            if (pred[6 * HW + i] > 0f) vals.Add(Mathf.Max(pred[HW + i] * ts + tm, 0f));
        if (vals.Count == 0) return 1f;
        vals.Sort();
        return vals[vals.Count / 2];
    }

    // ---------- input (new Input System; devices may be absent in batch mode) ----------

    void HandleInput()
    {
        var kb = Keyboard.current;
        var ms = Mouse.current;
        float dt = Time.deltaTime;

        if (kb != null)
        {
            if (kb.fKey.wasPressedThisFrame) { foamEnabled = !foamEnabled; if (foamEnabled) foam.Reset(); }
            if (kb.oKey.wasPressedThisFrame)
            {
                orbit = !orbit;
                if (orbit)
                {
                    // seamless takeover: seed the full spherical pose from the current eye
                    Vector3 rel = cam.eye - target;
                    orbitDistance = Mathf.Max(rel.magnitude, 0.2f);
                    orbitElevationDeg = Mathf.Asin(Mathf.Clamp(rel.y / orbitDistance, -1f, 1f)) * Mathf.Rad2Deg;
                    orbitAzDeg = Mathf.Atan2(rel.z, rel.x) * Mathf.Rad2Deg;
                }
            }
            if (kb.pKey.wasPressedThisFrame) paused = !paused;
            if (kb.bKey.wasPressedThisFrame) bilateralSmoothing = !bilateralSmoothing;
            if (kb.vKey.wasPressedThisFrame) showRawSideBySide = !showRawSideBySide;
            if (kb.rKey.wasPressedThisFrame) simTime = 0f;
            if (simSlots != null)
                for (int s = 0; s < simSlots.Length && s < 9; s++)
                    if (kb[(Key)((int)Key.Digit1 + s)].wasPressedThisFrame && s != currentSlot)
                        LoadSim(s);
            if (kb.leftBracketKey.wasPressedThisFrame) playbackFps = Mathf.Max(1f, playbackFps - 5f);
            if (kb.rightBracketKey.wasPressedThisFrame) playbackFps += 5f;

            float mx = (kb.dKey.isPressed ? 1f : 0f) - (kb.aKey.isPressed ? 1f : 0f);
            float mz = (kb.wKey.isPressed ? 1f : 0f) - (kb.sKey.isPressed ? 1f : 0f);
            float my = (kb.eKey.isPressed ? 1f : 0f) - (kb.qKey.isPressed ? 1f : 0f);
            if (mx != 0f || mz != 0f || my != 0f)
            {
                orbit = false;
                float sp = moveSpeed * (kb.leftShiftKey.isPressed ? 3f : 1f);
                cam.UpdateBasisFromYawPitch();
                cam.eye += (cam.fwd * mz + cam.right * mx + Vector3.up * my) * (sp * dt);
            }
        }

        if (ms != null && ms.rightButton.isPressed)
        {
            Vector2 d = ms.delta.ReadValue();
            if (d.sqrMagnitude > 0f)
            {
                orbit = false;
                cam.yawDeg += d.x * lookSpeedDegPerPx;
                cam.pitchDeg = Mathf.Clamp(cam.pitchDeg - d.y * lookSpeedDegPerPx, -89f, 89f);
            }
        }

        if (!orbit) cam.UpdateBasisFromYawPitch();
    }

    void OnGUI()
    {
        GUI.Label(new Rect(10, 8, 1600, 24), status);
        if (!ready || panels.Length == 0) return;

        // [RAW] | model 0 | model 1 | ... — equal-width columns across the window
        bool raw = showRawSideBySide && rawShadedRT != null;
        int cols = panels.Length + (raw ? 1 : 0);
        float colW = Screen.width / (float)cols;
        int c = 0;
        // the long-form raw caption doesn't fit once the columns get narrow
        if (raw) DrawPanel(c++, colW, rawShadedRT,
                           cols > 2 ? "RAW LOW-RES SPH" : "RAW LOW-RES SPH (model input, same shading)");
        foreach (var pan in panels) DrawPanel(c++, colW, pan.shadedRT, pan.label);

        if (!capturing)
            GUI.Label(new Rect(10, Screen.height - 24, 1600, 22),
                "WASD/QE fly (Shift fast)   RMB look   O orbit   B bilateral   V raw panel   " +
                "1-9 sim slot   P pause   R restart   [ ] sim fps   F foam");
    }

    void DrawPanel(int col, float colW, RenderTexture rt, string label)
    {
        GUI.DrawTexture(new Rect(col * colW, 30, colW, Screen.height - 30), rt, ScaleMode.ScaleToFit);
        GUI.Label(new Rect(col * colW + 10, 34, colW - 20, 22), label);
    }

    void OnDestroy()
    {
        foreach (var pan in panels)
        {
            pan.worker?.Dispose();
            if (pan.fieldTex != null) Destroy(pan.fieldTex);
            if (pan.smoothedRT != null) pan.smoothedRT.Release();
            if (pan.smoothedRT2 != null) pan.smoothedRT2.Release();
            if (pan.shadedRT != null) pan.shadedRT.Release();
        }
        if (rawFieldTex != null) Destroy(rawFieldTex);
        if (foamTexNet != null) Destroy(foamTexNet);
        if (foamTexRaw != null) Destroy(foamTexRaw);
        if (rawSmoothedRT != null) rawSmoothedRT.Release();
        if (rawSmoothedRT2 != null) rawSmoothedRT2.Release();
        if (rawShadedRT != null) rawShadedRT.Release();
        if (mat != null) Destroy(mat);
    }
}
