// FluidSceneMVP.cs — scene-space version of FluidLiveMVP: the neural-upsampled fluid is
// composited into the real 3D Unity scene instead of the box with the procedural floor/sky.
//
// Per frame:
//   1. The target Camera's pose is transformed into sim space through this GameObject's
//      transform (the "anchor": position / yaw / uniform scale place the fluid in the world;
//      sim is right-handed y-up, Unity left-handed, so the mapping flips z).
//   2. Particles from the ParticleFrameProvider are splatted from that pose into the
//      7×512×512 model input (identical math to FluidLiveMVP / splatRender.py).
//   3. The ONNX upsampler predicts depth/thickness/occupancy.
//   4. A 512² shading pass (FluidSSFRScene pass 0) shades the fluid against the REAL scene:
//      _CameraOpaqueTexture as the refraction background, _CameraDepthTexture for occlusion
//      by scene geometry, the scene reflection probe/skybox for reflections, and the scene's
//      main directional light for specular.
//   5. A camera-child quad (pass 1, premultiplied alpha) composites the result into the frame.
//
// Focal Mode = CoverFrustum (default): the splat uses a square frustum COVERING the camera's full
// (possibly wide) frustum, so the model stays at its native 512² aspect-1 input and the composite
// samples the sub-window. FitVertical: the square fits the VERTICAL fov (focal = 1/tanV, the training
// camera model); the model window is then the central square of the screen and the side strips of a
// wide view carry no fluid (067: r_px*depth = r_w*focal*256 must sit in the training band 24-33 px.m).
//
// MVP caveats:
//  - _CameraOpaqueTexture/_CameraDepthTexture are read during LateUpdate, i.e. they hold the
//    previous frame — refraction/occlusion lag one frame (imperceptible at normal motion).
//    Requires Opaque Texture + Depth Texture enabled on the URP asset.
//  - Keep the anchor upright (yaw-only rotation) and uniformly scaled: the model never saw
//    rolled cameras or tilted gravity. Scale itself is free (the camera pose is expressed in
//    sim units before splatting, and predicted depth is scaled back for occlusion).
//  - Camera framing should stay roughly in the training envelope (~2–7 sim units from the
//    fluid); extremely close/wide shots degrade the prediction exactly as in FluidLiveMVP.
//
// Controls (when flyControls is on): WASD fly (+Q/E world down/up, Shift ×3), hold RMB +
// mouse to look, P pause, R restart, V shade raw model input instead of the prediction,
// B bilateral smoothing, [ / ] playback fps −/+5.

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using Unity.InferenceEngine;

public class FluidSceneMVP : MonoBehaviour
{
    // ---------- inspector ----------

    [Header("Source")]
    [Tooltip("Particle source (e.g. a BakedParticleProvider). Any ParticleFrameProvider works — this is the seam for future live SPH solvers.")]
    public ParticleFrameProvider provider;
    [Tooltip("Meta json carrying the model's normalization stats (mean/std) and the sim domain center (target) — any FluidData/<name>_meta.json baked against the training stats. Optional if the provider is a BakedParticleProvider with its metaJson assigned (that file is reused).")]
    public TextAsset statsJson;
    [Tooltip("Upsampler ONNX model to run each frame (model_019a.onnx). Required.")]
    public ModelAsset modelAsset;
    [Tooltip("Scene shading shader (Hidden/FluidSSFRScene). Optional — auto-resolved via Shader.Find.")]
    public Shader sceneShader;
    [Tooltip("Smoothing shader (Hidden/FluidSSFR, pass 0 only). Optional — auto-resolved via Shader.Find.")]
    public Shader smoothShader;

    [Header("Scene placement")]
    [Tooltip("Camera to render the fluid for. Empty = Camera.main. The fluid follows this camera like any scene object — no special fluid camera exists.")]
    public Camera targetCamera;
    [Tooltip("Shift the sim horizontally so meta 'target' x/z sits at this GameObject's origin (y is NOT shifted, so the sim floor y=0 stays on the anchor's y=0 plane — put the anchor on your scene floor). Recommended: true.")]
    public bool centerXZOnTarget = true;

    [Header("Playback")]
    [Tooltip("Simulation playback rate in frames/sec. Recommended default: 25.")]
    public float playbackFps = 25f;
    [Tooltip("Global time multiplier on top of Playback Fps. 1 = normal speed. Recommended default: 1.")]
    public float speed = 1f;

    [Header("Inference")]
    [Tooltip("Inference Engine backend. GPUPixel + fp32 is the validated config (GPUCompute is silently wrong for these models). Recommended default: GPUPixel.")]
    public BackendType backend = BackendType.GPUPixel;
    [Tooltip("Attempt fp16 weight quantization (auto-falls back to fp32 on failure). Recommended default: true.")]
    public bool useFp16 = true;
    [Tooltip("Manual override for the shader thickness scale (_KThick). 0 = auto-derive from the model's own predicted thickness. Recommended default: 0 (auto).")]
    public float kThickOverride = 0f;

    [Header("Input calibration")]
    [Tooltip("Rescale the splatted THICKNESS channel so it is invariant to the solver's particle " +
             "radius. The splat sums kernel weights, so thickness is proportional to particle " +
             "COUNT (measured on the 023 ratio probe: exactly 1/factor, within 1%), and count for " +
             "a given fluid volume goes as 1/r^3. Without this, any solver whose radius differs " +
             "from the training LR point set feeds the model an out-of-distribution channel. " +
             "Requires the provider to report ParticleRadius; a provider that returns 0 disables " +
             "it. OFF by default so existing baked slots stay bit-identical. Recommended: ON for " +
             "any live solver.")]
    public bool thicknessCountNormalize = false;
    [Tooltip("Non-empty = on Play, write this component's splatted 7x512x512 input tensor to this " +
             "path (raw <f4 CHW) and log the pose, then continue. Gate it offline with the repo's " +
             "compare_unity_splat.py against splatRender.py. FluidLiveMVP's splat was parity- " +
             "checked this way in July; THIS component's copy of the same math never was, so " +
             "nothing it renders has ever been verified against the training renderer. Leave " +
             "empty for normal play. Example: ParityDump/scene_splat.bytes")]
    public string parityDumpPath = "";
    [Tooltip("Effective particle radius of the TRAINING LR point set, in sim units. The training " +
             "LR is the dense solve subsampled by `factor`, so its effective spacing is " +
             "r_gt * factor^(1/3): dam_breaks_v3 median r_gt 0.01471 x 25^(1/3) = 0.04302. Only " +
             "used when Thickness Count Normalize is on.")]
    public float refLrParticleRadius = 0.04302f;

    public enum SplatMode { Legacy, V2 }
    [Header("Splat (input renderer version)")]
    [Tooltip("Which in-engine splatter builds the model input. Legacy = 5 px square Gaussian stamp, integer placement, constant depth (what 040-053 were trained on; pair with those models). V2 = subpixel spherical splats with the physical projected radius and volume-mode thickness (splatRender.splat_render_v2; what 058a+ were trained on). Model and splatter MUST match, exactly like model and stats. Thickness Count Normalize applies to Legacy only (v2 thickness is physical and count-invariant by construction).")]
    public SplatMode splatMode = SplatMode.Legacy;
    [Tooltip("V2 only: particle world radius [m] used for the projected footprint and the sphere depth. 0 = provider.ParticleRadius (GpuSphProvider 0.0414, BakedParticleProvider from its meta), then the stats meta's coarseRadius.")]
    public float v2WorldRadius = 0f;
    [Tooltip("V2 only: thickness scale (Python THICKNESS_SCALE_LEGACY = 3.9448). 0 = stats meta thicknessScale, else the constant.")]
    public float v2ThicknessScale = 0f;

    public enum FocalMode { CoverFrustum, FitVertical }
    [Tooltip("How the square 512² model window maps onto the camera. CoverFrustum (default, today's behaviour) = focal 1/max(tanV,tanH): covers the whole view, but at FOV 60 / 16:9 that is focal 0.97 vs the training cameras' 2.15-2.75, so every particle is 2.5-3x smaller on screen than the network saw at that depth. FitVertical = focal 1/tanV: the training camera model when the camera's vertical FOV is 40-50; the window is the central square of the screen.")]
    public FocalMode focalMode = FocalMode.CoverFrustum;
    [Tooltip("Model window width in px (height is 512). 512 = today's square window. 896 = a 16:9-class window at the SAME focal: FitVertical then covers the full width of a 16:9 view instead of its central 56 %. Needs the V2 splat and a model exported for a 1x7x512x896 input with a NON-square core (Unity_Models/*_wide896x512.onnx, SSU_restart/Helpers/export_onnx_wide.py) — a square-window model will not load into a wide tensor. Offline (067): same weights, central window identical to the square network (mask IoU 0.98-0.999, depth |d| median < 1 mm), side strips as healthy.")]
    public int windowWidth = 512;

    [Header("Shading")]
    [Tooltip("Gaussian blur sigma (px) on the masked depth buffer before shading. Recommended default: 2.")]
    public float presmoothSigma = 2f;
    [Tooltip("Edge-preserving bilateral blur instead of the plain Gaussian (toggle with B). Recommended default: false.")]
    public bool bilateralSmoothing = false;
    [Tooltip("Bilateral spatial sigma. Recommended default: 3.")]
    public float bilateralSigmaS = 3f;
    [Tooltip("Bilateral iterations. Recommended default: 4.")]
    public int bilateralIters = 4;
    [Tooltip("Multiplier on per-frame fg-depth std-dev for the bilateral range sigma. Recommended default: 0.12.")]
    public float bilateralRangeScale = 0.12f;
    [Tooltip("Screen-space refraction distortion strength, in screen texels of background offset. Recommended default: 22.")]
    public float refrStrength = 22f;
    [Tooltip("Specular coefficient. Recommended default: 0.7.")]
    public float ks = 0.7f;
    [Tooltip("Blinn-Phong specular exponent. Recommended default: 120.")]
    public float shininess = 120f;
    [Tooltip("World-units slack added to the scene-depth occlusion test in the composite (clip(sceneEye - fluidDepth + bias)). 0 = today's unbiased test, which z-fights wherever the fluid lies on scene geometry (the pool on the Ground plane).")]
    public float depthBias = 0f;

    [Header("Color / Look")]
    [Tooltip("Blood look (see FluidLiveMVP). Overrides Default Look / colors below while on. Recommended default: false in a real scene (water reads best against real backgrounds).")]
    public bool bloodMode = false;
    [Tooltip("When on (and Blood Mode off): the original validated water absorption/tint, ignoring the two colors below. Recommended default: true.")]
    public bool defaultLook = true;
    [Tooltip("Per-channel Beer-Lambert absorption (SUBTRACTIVE: high channel = that color removed in thick fluid). Only used when Default Look is off.")]
    public Color absorbColor = new Color(0.45f, 0.16f, 0.10f);
    [Tooltip("Interior color fully-opaque regions fade toward. Only used when Default Look is off.")]
    public Color bodyTintColor = new Color(0.04f, 0.10f, 0.14f);
    [Tooltip("Reflection fallback sky (zenith) when no reflection probe/skybox cubemap is available.")]
    public Color fallbackSkyTop = new Color(0.27f, 0.47f, 0.78f);
    [Tooltip("Reflection fallback sky (horizon) when no reflection probe/skybox cubemap is available.")]
    public Color fallbackSkyHorizon = new Color(0.80f, 0.88f, 0.95f);
    [Tooltip("Reflection fallback ground color for downward rays when no probe is available.")]
    public Color fallbackGround = new Color(0.16f, 0.19f, 0.24f);

    [Header("Controls / debug")]
    [Tooltip("Built-in fly controls for the target camera (WASD/QE + RMB look). Turn off if the scene has its own camera controller.")]
    public bool flyControls = true;
    [Tooltip("Fly speed in world units/sec (×3 with Shift). Recommended default: 2.")]
    public float moveSpeed = 2f;
    [Tooltip("Mouse look sensitivity, degrees per pixel. Recommended default: 0.15.")]
    public float lookSpeedDegPerPx = 0.15f;
    [Tooltip("Shade the raw model INPUT fields instead of the prediction (toggle with V) — in-place A/B of what the network adds.")]
    public bool showRawInput = false;
    [Tooltip("Draw the status line (frame, fps, inference ms).")]
    public bool showStatus = true;

    [Header("Deterministic capture (fixed-step world-space orbit -> PNG per frame)")]
    [Tooltip("If non-empty, Play runs a fixed-step orbit of the target camera around the fluid anchor and writes one PNG per frame here (project-relative; encode with ffmpeg). Starts once kThick has locked so brightness is constant. Empty = normal interactive mode.")]
    public string captureDir = "";
    [Tooltip("Fixed step rate during capture = frame rate of the resulting video. Recommended default: 30.")]
    public float captureFps = 30f;
    [Tooltip("Capture duration in seconds. Recommended default: 12.")]
    public float captureSeconds = 12f;
    [Tooltip("Seconds per full orbit revolution during capture. The orbit radius/elevation are taken from the camera's placed pose at Play. Recommended default: 12.")]
    public float orbitPeriodSec = 12f;

    // ---------- state ----------

    const int H = 512;
    int W = 512, HW = H * 512;            // W = windowWidth, fixed at Init (067: a wide model window)
    float winAspect = 1f;                  // W / H
    const int SplatR = 5;                  // LR splat: radius 5 px square stamp, sigma 2.0
    const float SigmaPx = 2f;

    [Serializable] class Meta { public float fps = 25f; public float[] target, mean, std; public string renderer; public float coarseRadius, thicknessScale, minRadiusPx, maxRadiusPx; }

    Meta meta;
    Vector3 simOffset;                     // added to camera pos in sim space (centerXZOnTarget)

    float[] kernel;
    float[] depthBuf, thickBuf, velXBuf, velYBuf, velZBuf, denBuf;
    float[] in7;
    Color[] px;

    Worker worker;
    bool fp16Active;
    float kThick;
    float thickScale = 1f;                 // particle-count normalization for the thickness channel (legacy only)
    bool useV2; float v2R, v2TS, v2MinR = 1f, v2MaxR = 24f;
    bool kThickLocked;

    Material smoothMat, shadeMat, compositeMat;
    Texture2D fieldTex;
    RenderTexture smoothedRT, smoothedRT2, cRT, mRT, nRT;
    float bilateralSigmaR;
    GameObject quadGO;
    Light mainLight;

    // camera pose in sim space, rebuilt each frame
    Vector3 eyeSim, rightSim, upSim, fwdSim;
    float focalM, tanH, tanV;

    bool paused;
    bool capturing;
    int capFrame;
    float orbitAzDeg, orbitElDeg, orbitDist;
    Vector3 orbitCenter;
    float simTime;
    int frameIdx;
    float inferMs, smoothedFps;
    string status = "initializing...";
    bool ready;

    static Vector3 FlipZ(Vector3 v) => new Vector3(v.x, v.y, -v.z);

    // ---------- read-only taps for LiveClipRecorder (067) ----------
    float[] lastPred;
    /// <summary>Raised every rendered frame right after inference; LastInput/LastPred are current.</summary>
    public event Action OnInferred;
    public bool Ready => ready;
    public float[] LastInput => in7;
    public float[] LastPred => lastPred;
    public void GetSimCamera(out Vector3 eye, out Vector3 right, out Vector3 up, out Vector3 fwd, out float focal)
    {
        ComputeSimCamera();   // pure function of the transforms; the camera is final by LateUpdate
        eye = eyeSim; right = rightSim; up = upSim; fwd = fwdSim; focal = focalM;
    }

    [Serializable]
    public class ClipSettings
    {
        public string splatMode, focalMode, model, stats, backend;
        public bool fp16Active, thicknessCountNormalize;
        public float v2R, v2TS, v2MinR, v2MaxR, thickScale, refLrParticleRadius, presmoothSigma, depthBias, simScale, kThick;
        public float[] mean, std, target, simOffset;
        public int winW, winH;
    }

    public ClipSettings GetClipSettings() => new ClipSettings
    {
        splatMode = splatMode.ToString(), focalMode = focalMode.ToString(),
        model = modelAsset != null ? modelAsset.name : "", stats = statsJson != null ? statsJson.name : "",
        backend = backend.ToString(), fp16Active = fp16Active, thicknessCountNormalize = thicknessCountNormalize,
        v2R = v2R, v2TS = v2TS, v2MinR = v2MinR, v2MaxR = v2MaxR, thickScale = thickScale,
        refLrParticleRadius = refLrParticleRadius, presmoothSigma = presmoothSigma, depthBias = depthBias,
        simScale = transform.lossyScale.x, kThick = kThick,
        mean = meta.mean, std = meta.std, target = meta.target,
        simOffset = new[] { simOffset.x, simOffset.y, simOffset.z },
        winW = W, winH = H,
    };

    void Start()
    {
        try { Init(); }
        catch (Exception e) { status = $"ERROR: {e.Message}"; Debug.LogException(e); }
    }

    void Init()
    {
        if (provider == null) provider = GetComponent<ParticleFrameProvider>();
        if (provider == null) throw new Exception("FluidSceneMVP: no ParticleFrameProvider assigned");
        if (provider.FrameCount <= 0) throw new Exception("FluidSceneMVP: provider has no frames");

        TextAsset stats = statsJson != null ? statsJson : (provider as BakedParticleProvider)?.metaJson;
        if (stats == null) throw new Exception("FluidSceneMVP: statsJson not assigned (and provider carries no metaJson)");
        meta = JsonUtility.FromJson<Meta>(stats.text);
        if (meta.mean == null || meta.mean.Length < 6 || meta.std == null || meta.std.Length < 6)
            throw new Exception($"FluidSceneMVP: {stats.name} has no mean/std normalization stats");
        simOffset = centerXZOnTarget && meta.target != null && meta.target.Length >= 3
            ? new Vector3(meta.target[0], 0f, meta.target[2]) : Vector3.zero;

        if (targetCamera == null) targetCamera = Camera.main;
        if (targetCamera == null) throw new Exception("FluidSceneMVP: no target camera (assign one or tag a camera MainCamera)");
        if (Mathf.Abs(transform.lossyScale.x - transform.lossyScale.y) > 1e-4f ||
            Mathf.Abs(transform.lossyScale.x - transform.lossyScale.z) > 1e-4f)
            Debug.LogWarning("FluidSceneMVP: anchor scale is non-uniform — using x; make it uniform");

        kernel = new float[(2 * SplatR + 1) * (2 * SplatR + 1)];
        int k = 0;
        for (int oy = -SplatR; oy <= SplatR; oy++)
            for (int ox = -SplatR; ox <= SplatR; ox++, k++)
                kernel[k] = (float)Math.Exp(-(ox * ox + oy * oy) / (2.0 * SigmaPx * SigmaPx));

        useV2 = splatMode == SplatMode.V2;
        W = windowWidth > 0 ? windowWidth : 512; HW = H * W; winAspect = (float)W / H;
        if (W != 512 && (W % 32 != 0 || !useV2))
            throw new Exception($"FluidSceneMVP: windowWidth {W} needs the V2 splat and a multiple of 32 (the legacy stamp path is square-only)");
        if (useV2)
        {
            v2R = v2WorldRadius > 0f ? v2WorldRadius : (provider.ParticleRadius > 0f ? provider.ParticleRadius : meta.coarseRadius);
            if (!(v2R > 0f)) throw new Exception("FluidSceneMVP: V2 splat needs a particle radius (v2WorldRadius, provider.ParticleRadius or meta coarseRadius)");
            v2TS = v2ThicknessScale > 0f ? v2ThicknessScale : (meta.thicknessScale > 0f ? meta.thicknessScale : SplatV2.ThicknessScaleLegacy);
            v2MinR = meta.minRadiusPx > 0f ? meta.minRadiusPx : 1f;
            v2MaxR = meta.maxRadiusPx > 0f ? meta.maxRadiusPx : 24f;
            Debug.Log($"FluidSceneMVP: splat V2 — r={v2R:F5} m, thicknessScale={v2TS:F4}, footprint clamp [{v2MinR},{v2MaxR}] px (count-normalize ignored)");
        }

        // Thickness ~ particle count ~ 1/r^3, so match the training LR point set's density (LEGACY only).
        thickScale = 1f;
        if (!useV2 && thicknessCountNormalize)
        {
            float r = provider.ParticleRadius;
            if (r > 0f && refLrParticleRadius > 0f)
            {
                float ratio = r / refLrParticleRadius;
                thickScale = ratio * ratio * ratio;
                Debug.Log($"FluidSceneMVP: thickness count-normalize ON — provider r={r:F5}, " +
                          $"ref r={refLrParticleRadius:F5}, scale={thickScale:F4}");
            }
            else
            {
                Debug.LogWarning("FluidSceneMVP: Thickness Count Normalize is on but the provider " +
                                 "reports no ParticleRadius — leaving thickness uncorrected.");
            }
        }

        depthBuf = new float[HW]; thickBuf = new float[HW];
        velXBuf = new float[HW]; velYBuf = new float[HW]; velZBuf = new float[HW];
        denBuf = new float[HW];
        in7 = new float[7 * HW];
        px = new Color[HW];

        if (smoothShader == null) smoothShader = Shader.Find("Hidden/FluidSSFR");
        if (smoothShader == null) throw new Exception("Hidden/FluidSSFR shader not found");
        if (sceneShader == null) sceneShader = Shader.Find("Hidden/FluidSSFRScene");
        if (sceneShader == null) throw new Exception("Hidden/FluidSSFRScene shader not found");
        smoothMat = new Material(smoothShader);
        shadeMat = new Material(sceneShader);
        compositeMat = new Material(sceneShader);
        compositeMat.renderQueue = 3100;   // after all standard transparents

        fieldTex = new Texture2D(W, H, TextureFormat.RGBAFloat, false) { filterMode = FilterMode.Point };
        smoothedRT = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBFloat) { filterMode = FilterMode.Point };
        smoothedRT2 = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBFloat) { filterMode = FilterMode.Point };
        cRT = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBHalf) { filterMode = FilterMode.Bilinear };
        mRT = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBHalf) { filterMode = FilterMode.Bilinear };
        nRT = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBHalf) { filterMode = FilterMode.Bilinear };

        mainLight = RenderSettings.sun;
        if (mainLight == null)
            foreach (var l in FindObjectsByType<Light>())
                if (l.type == LightType.Directional) { mainLight = l; break; }

        // fullscreen composite quad, child of the camera (pass 1 ignores its exact placement —
        // it just needs to cover the frustum and survive culling)
        quadGO = GameObject.CreatePrimitive(PrimitiveType.Quad);
        quadGO.name = "FluidSceneMVP Composite";
        Destroy(quadGO.GetComponent<Collider>());
        float qz = Mathf.Max(1f, targetCamera.nearClipPlane * 2f);
        quadGO.transform.SetParent(targetCamera.transform, false);
        quadGO.transform.localPosition = new Vector3(0, 0, qz);
        quadGO.transform.localScale = new Vector3(5f * qz, 5f * qz, 1f);
        var mr = quadGO.GetComponent<MeshRenderer>();
        mr.sharedMaterial = compositeMat;
        mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        mr.receiveShadows = false;
        mr.lightProbeUsage = UnityEngine.Rendering.LightProbeUsage.Off;

        var model = ModelLoader.Load(modelAsset);
        fp16Active = false;
        if (useFp16)
        {
            try
            {
                ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model);
                fp16Active = true;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"fp16 weight quantization failed ({e.Message}) — falling back to fp32");
                model = ModelLoader.Load(modelAsset);
            }
        }
        worker = new Worker(model, backend);
        ComputeSimCamera();
        SplatToInput(0);
        try { RunModel(); }
        catch (Exception e) when (fp16Active)
        {
            Debug.LogWarning($"fp16 model failed at schedule time ({e.Message}) — rebuilding fp32");
            worker.Dispose();
            fp16Active = false;
            worker = new Worker(ModelLoader.Load(modelAsset), backend);
            RunModel();
        }

        if (parityDumpPath.Length > 0) RunParityDump();

        kThick = kThickOverride;
        capturing = captureDir.Length > 0;
        if (capturing)
        {
            System.IO.Directory.CreateDirectory(captureDir);
            orbitCenter = transform.position + Vector3.up * (0.9f * transform.lossyScale.x);
            Vector3 rel = targetCamera.transform.position - orbitCenter;
            orbitDist = Mathf.Max(rel.magnitude, 0.5f);
            orbitElDeg = Mathf.Asin(Mathf.Clamp(rel.y / orbitDist, -1f, 1f)) * Mathf.Rad2Deg;
            orbitAzDeg = Mathf.Atan2(rel.z, rel.x) * Mathf.Rad2Deg;
        }
        ready = true;
    }

    // Dump the splatted model input so it can be diffed against splatRender.py offline.
    // Init() has already called SplatToInput(0) with the camera's placed pose, so in7 is current.
    void RunParityDump()
    {
        var bytes = new byte[in7.Length * 4];
        Buffer.BlockCopy(in7, 0, bytes, 0, bytes.Length);
        string dir = System.IO.Path.GetDirectoryName(parityDumpPath);
        if (!string.IsNullOrEmpty(dir)) System.IO.Directory.CreateDirectory(dir);
        System.IO.File.WriteAllBytes(parityDumpPath, bytes);
        Debug.Log($"FluidSceneMVP parity dump -> {parityDumpPath} ({bytes.Length} bytes)  " +
                  $"eye={eyeSim.x:F6},{eyeSim.y:F6},{eyeSim.z:F6}  fwd={fwdSim.x:F6},{fwdSim.y:F6}," +
                  $"{fwdSim.z:F6}  focalM={focalM:F6}  thickScale={thickScale:F6}");
    }

    float[] RunModel()
    {
        using var input = new Tensor<float>(new TensorShape(1, 7, H, W), in7);
        worker.Schedule(input);
        using var t = (worker.PeekOutput() as Tensor<float>).ReadbackAndClone();
        return t.DownloadToArray();
    }

    // ---------- camera pose: Unity world -> sim space ----------

    void ComputeSimCamera()
    {
        Transform a = transform, c = targetCamera.transform;
        eyeSim = FlipZ(a.InverseTransformPoint(c.position)) + simOffset;
        rightSim = FlipZ(a.InverseTransformDirection(c.right));
        upSim = FlipZ(a.InverseTransformDirection(c.up));
        fwdSim = FlipZ(a.InverseTransformDirection(c.forward));

        tanV = Mathf.Tan(targetCamera.fieldOfView * Mathf.Deg2Rad * 0.5f);
        tanH = tanV * targetCamera.aspect;
        focalM = focalMode == FocalMode.FitVertical
            ? 1f / tanV                                  // training camera model: the window's HEIGHT fits the vertical fov
            : 1f / Mathf.Max(tanV, tanH / winAspect);    // window covering the full camera frustum (winAspect 1 = the square of before)
    }

    // ---------- splat: identical math to FluidLiveMVP.SplatToInput ----------

    void SplatToInput(int frame)
    {
        Array.Fill(depthBuf, 1e6f);
        Array.Clear(thickBuf, 0, HW);
        Array.Clear(velXBuf, 0, HW);
        Array.Clear(velYBuf, 0, HW);
        Array.Clear(velZBuf, 0, HW);
        Array.Clear(denBuf, 0, HW);

        provider.GetFrame(frame, out var data, out int off, out int count);
        float focal = focalM;
        Vector3 right = rightSim, up = upSim, fwd = fwdSim, eye = eyeSim;
        if (useV2)
        {
            SplatV2.Splat(data, off, count, eye, right, up, fwd, focal, H, W, v2R, v2TS, v2MinR, v2MaxR,
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
            float vcx = wx * right.x + wy * right.y + wz * right.z;
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
                    if (depthVal < depthBuf[i]) depthBuf[i] = depthVal;
                    velXBuf[i] += vcx * w;
                    velYBuf[i] += vcy * w;
                    velZBuf[i] += vcz * w;
                    denBuf[i] += den * w;
                }
            }
        }
        }   // legacy

        float m0 = meta.mean[0], m1 = meta.mean[1], m2 = meta.mean[2], m3 = meta.mean[3], m4 = meta.mean[4], m5 = meta.mean[5];
        float s0 = 1f / meta.std[0], s1 = 1f / meta.std[1], s2 = 1f / meta.std[2], s3 = 1f / meta.std[3], s4 = 1f / meta.std[4], s5 = 1f / meta.std[5];
        for (int i = 0; i < HW; i++)
        {
            float t = thickBuf[i];
            float inv = useV2 ? (t > 0f ? 1f / t : 0f) : 1f / (t + 1e-6f);
            // Mask and the velocity/density weight-normalization both use the RAW accumulation:
            // occupancy is a geometric property of the splat footprint and the weighted means are
            // already count-invariant, so only the thickness VALUE is rescaled. Scaling before the
            // mask test would move the silhouette by a few boundary pixels for no reason.
            float mask = useV2 ? (t > 0f ? 1f : 0f) : (t > 1e-4f ? 1f : 0f);
            float d = mask > 0f ? depthBuf[i] : 0f;
            in7[i] = (d - m0) * s0;
            in7[HW + i] = (t * thickScale - m1) * s1;
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
        if (ready) HandleInput();
    }

    void LateUpdate()   // after any camera controllers have moved the camera this frame
    {
        if (!ready) return;
        float dt = capturing ? 1f / captureFps : Time.deltaTime;   // fixed step while capturing

        if (capturing)
        {
            orbitAzDeg += 360f * dt / Mathf.Max(orbitPeriodSec, 0.1f);
            float az = orbitAzDeg * Mathf.Deg2Rad, el = orbitElDeg * Mathf.Deg2Rad;
            Vector3 p = orbitCenter + new Vector3(
                orbitDist * Mathf.Cos(el) * Mathf.Cos(az),
                orbitDist * Mathf.Sin(el),
                orbitDist * Mathf.Cos(el) * Mathf.Sin(az));
            targetCamera.transform.SetPositionAndRotation(
                p, Quaternion.LookRotation(orbitCenter - p, Vector3.up));
        }

        if (!paused) simTime += dt * speed;
        provider.Tick(paused ? 0f : dt * speed);   // live solvers advance in lockstep
        frameIdx = (int)(simTime * playbackFps) % provider.FrameCount;

        ComputeSimCamera();
        SplatToInput(frameIdx);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        float[] pred = RunModel();
        sw.Stop();
        lastPred = pred;
        OnInferred?.Invoke();
        inferMs = Mathf.Lerp(inferMs <= 0f ? (float)sw.Elapsed.TotalMilliseconds : inferMs,
                             (float)sw.Elapsed.TotalMilliseconds, 0.1f);

        // kThick: as in FluidLiveMVP — derive from predicted thickness, lock mid-sequence.
        if (kThickOverride > 0f) kThick = kThickOverride;
        else if (!kThickLocked)
        {
            kThick = 1.2f / Mathf.Max(MedianFgThickness(pred), 1e-3f);
            // baked: lock mid-sequence; live (FrameCount<=1): lock once the flow has developed
            if (provider.FrameCount <= 1 ? simTime >= 2.5f : frameIdx >= provider.FrameCount / 2)
                kThickLocked = true;
        }

        // Stage (depth, thickness, alpha) — prediction, or the raw model input when showRawInput.
        float dm = meta.mean[0], ds = meta.std[0], tm = meta.mean[1], ts = meta.std[1];
        double dSum = 0.0, dSumSq = 0.0;
        int fgCount = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                int i = y * W + x;
                int flipped = (H - 1 - y) * W + x;   // tensor row 0 = top; texture row 0 = bottom
                float a, depth, thick;
                if (showRawInput)
                {
                    // Same thickScale the model input gets, so the raw panel and the neural panel
                    // differ only by the network (the V-toggle's whole purpose).
                    a = thickBuf[i] > 1e-4f ? 1f : 0f;
                    thick = thickBuf[i] * thickScale;
                    depth = a > 0f ? depthBuf[i] : 0f;
                    if (a == 0f) thick = 0f;
                }
                else
                {
                    a = pred[6 * HW + i] > 0f ? 1f : 0f;
                    depth = a > 0f ? pred[i] * ds + dm : 0f;
                    thick = a > 0f ? Mathf.Max(pred[HW + i] * ts + tm, 0f) : 0f;
                }
                px[flipped] = new Color(depth, thick, a, 1f);
                if (a > 0f) { dSum += depth; dSumSq += (double)depth * depth; fgCount++; }
            }
        bilateralSigmaR = 0.05f;
        if (fgCount > 1)
        {
            double var = (dSumSq - dSum * dSum / fgCount) / (fgCount - 1);
            bilateralSigmaR = Mathf.Max(bilateralRangeScale * (float)Math.Sqrt(Math.Max(var, 0.0)), 1e-3f);
        }
        fieldTex.SetPixels(px);
        fieldTex.Apply(false, false);

        // smoothing chain (FluidSSFR pass 0, unchanged semantics)
        smoothMat.SetFloat("_BlurSigma", bilateralSmoothing ? bilateralSigmaS : presmoothSigma);
        smoothMat.SetFloat("_BilateralRangeSigma", bilateralSmoothing ? bilateralSigmaR : 0f);
        Graphics.Blit(fieldTex, smoothedRT, smoothMat, 0);
        if (bilateralSmoothing)
            for (int it = 1; it < bilateralIters; it++)
            {
                Graphics.Blit(smoothedRT, smoothedRT2, smoothMat, 0);
                (smoothedRT, smoothedRT2) = (smoothedRT2, smoothedRT);
            }

        // scene-independent shading packs at 512² (premultiplied by coverage)
        SetShadeParams(shadeMat);
        Graphics.Blit(smoothedRT, cRT, shadeMat, 0);
        Graphics.Blit(smoothedRT, mRT, shadeMat, 1);
        Graphics.Blit(smoothedRT, nRT, shadeMat, 2);

        // composite quad (samples scene color/depth in-render, pass 3)
        compositeMat.SetTexture("_CTex", cRT);
        compositeMat.SetTexture("_MTex", mRT);
        compositeMat.SetTexture("_NTex", nRT);
        compositeMat.SetFloat("_FocalM", focalM);
        compositeMat.SetFloat("_WinAspect", winAspect);
        compositeMat.SetFloat("_SimScale", transform.lossyScale.x);
        compositeMat.SetFloat("_DepthBias", depthBias);
        Transform c = targetCamera.transform;
        compositeMat.SetVector("_CamRightWS", c.right);
        compositeMat.SetVector("_CamUpWS", c.up);
        compositeMat.SetVector("_CamFwdWS", c.forward);

        // warm up (no screenshots) until kThick locks, so capture brightness is constant
        if (capturing && (kThickLocked || kThickOverride > 0f))
        {
            ScreenCapture.CaptureScreenshot(System.IO.Path.Combine(captureDir, $"frame_{capFrame:D4}.png"));
            capFrame++;
            if (capFrame >= (int)(captureFps * captureSeconds))
            {
                capturing = false;
                Debug.Log($"capture done: {capFrame} frames in {captureDir}");
            }
        }

        float fps = 1f / Mathf.Max(Time.deltaTime, 1e-5f);
        smoothedFps = smoothedFps <= 0f ? fps : Mathf.Lerp(smoothedFps, fps, 0.1f);
        string live = provider is LiveSphProvider sph
            ? $"LIVE SPH {sph.ActiveParticles}p solver {sph.LastStepMs:F0}ms   "
            : provider is GpuSphProvider gpu
                ? $"LIVE GPU-PBF {gpu.ActiveParticles}p solver {gpu.LastStepMs:F1}ms   " : "";
        status = live + $"frame {frameIdx + 1}/{provider.FrameCount}   sim {playbackFps:F0} fps ×{speed:F1}   " +
                 $"render {smoothedFps:F1} fps   infer {inferMs:F1} ms   backend={backend} fp16={fp16Active}   " +
                 $"k_thick={kThick:F3}{(showRawInput ? "   RAW INPUT" : "")}" +
                 $"{(bilateralSmoothing ? "   BILATERAL" : "")}{(paused ? "   PAUSED" : "")}" +
                 $"{(capturing ? "   CAPTURING" : "")}";
    }

    void SetShadeParams(Material m)
    {
        Transform c = targetCamera.transform;
        m.SetFloat("_FocalM", focalM);
        m.SetFloat("_WinAspect", winAspect);
        m.SetVector("_CamRightWS", c.right);
        m.SetVector("_CamUpWS", c.up);
        m.SetVector("_CamFwdWS", c.forward);
        m.SetFloat("_KThick", kThick);
        m.SetFloat("_RefrStrength", refrStrength);

        Texture env = ReflectionProbe.defaultTexture;
        bool useEnv = env != null && env.dimension == UnityEngine.Rendering.TextureDimension.Cube;
        m.SetFloat("_UseEnvCube", useEnv ? 1f : 0f);
        if (useEnv) m.SetTexture("_EnvCube", env);
        m.SetColor("_SkyTopFallback", fallbackSkyTop);
        m.SetColor("_SkyHorFallback", fallbackSkyHorizon);
        m.SetColor("_GroundFallback", fallbackGround);

        Vector3 lDir = mainLight != null ? -mainLight.transform.forward : new Vector3(-0.3f, 0.8f, -0.5f).normalized;
        Color lCol = mainLight != null ? mainLight.color * mainLight.intensity : Color.white;
        m.SetVector("_LightDirWS", lDir);
        m.SetVector("_LightColor", new Vector4(lCol.r, lCol.g, lCol.b, 0f));

        if (bloodMode)
        {
            m.SetFloat("_F0", 0.02f);
            m.SetFloat("_Shininess", 90f);
            m.SetFloat("_Ks", 0.18f);
            m.SetFloat("_FrMax", 0.18f);
            m.SetVector("_SpecTint", new Vector4(1.0f, 0.45f, 0.4f, 0f));
            m.SetVector("_AbsorbSigma", new Vector4(0.6f, 4.0f, 4.5f, 0f));
            m.SetColor("_BodyTint", new Color(0.15f, 0.01f, 0.01f));
        }
        else
        {
            m.SetFloat("_F0", 0.02f);
            m.SetFloat("_Shininess", shininess);
            m.SetFloat("_Ks", ks);
            m.SetFloat("_FrMax", 1.0f);
            m.SetVector("_SpecTint", new Vector4(1f, 1f, 1f, 0f));
            m.SetVector("_AbsorbSigma", defaultLook ? new Vector4(0.45f, 0.16f, 0.10f, 0f) : (Vector4)absorbColor);
            m.SetColor("_BodyTint", defaultLook ? new Color(0.04f, 0.10f, 0.14f) : bodyTintColor);
        }
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

    // ---------- input ----------

    void HandleInput()
    {
        var kb = Keyboard.current;
        var ms = Mouse.current;
        float dt = Time.deltaTime;

        if (kb != null)
        {
            if (kb.pKey.wasPressedThisFrame) paused = !paused;
            if (kb.bKey.wasPressedThisFrame) bilateralSmoothing = !bilateralSmoothing;
            if (kb.vKey.wasPressedThisFrame) showRawInput = !showRawInput;
            if (kb.rKey.wasPressedThisFrame) { simTime = 0f; provider?.ResetSim(); }
            if (kb.leftBracketKey.wasPressedThisFrame) playbackFps = Mathf.Max(1f, playbackFps - 5f);
            if (kb.rightBracketKey.wasPressedThisFrame) playbackFps += 5f;

            if (flyControls)
            {
                float mx = (kb.dKey.isPressed ? 1f : 0f) - (kb.aKey.isPressed ? 1f : 0f);
                float mz = (kb.wKey.isPressed ? 1f : 0f) - (kb.sKey.isPressed ? 1f : 0f);
                float my = (kb.eKey.isPressed ? 1f : 0f) - (kb.qKey.isPressed ? 1f : 0f);
                if (mx != 0f || mz != 0f || my != 0f)
                {
                    float sp = moveSpeed * (kb.leftShiftKey.isPressed ? 3f : 1f);
                    Transform ct = targetCamera.transform;
                    ct.position += (ct.forward * mz + ct.right * mx + Vector3.up * my) * (sp * dt);
                }
            }
        }

        if (flyControls && ms != null && ms.rightButton.isPressed)
        {
            Vector2 d = ms.delta.ReadValue();
            if (d.sqrMagnitude > 0f)
            {
                Transform ct = targetCamera.transform;
                Vector3 e = ct.eulerAngles;
                float pitch = e.x > 180f ? e.x - 360f : e.x;
                pitch = Mathf.Clamp(pitch + d.y * lookSpeedDegPerPx * -1f, -89f, 89f);
                float yaw = e.y + d.x * lookSpeedDegPerPx;
                ct.rotation = Quaternion.Euler(pitch, yaw, 0f);
            }
        }
    }

    void OnGUI()
    {
        if (showStatus) GUI.Label(new Rect(10, 8, 1600, 24), status);
    }

    void OnDrawGizmosSelected()
    {
        // rough sim-domain guide: box of size 2*target (dam-break convention), on the anchor
        try
        {
            TextAsset stats = statsJson != null ? statsJson : (provider as BakedParticleProvider)?.metaJson;
            if (stats == null) return;
            var mt = JsonUtility.FromJson<Meta>(stats.text);
            if (mt.target == null || mt.target.Length < 3) return;
            Gizmos.color = new Color(0.2f, 0.6f, 1f, 0.8f);
            Gizmos.matrix = transform.localToWorldMatrix;
            Vector3 size = new Vector3(2f * mt.target[0], 2f * mt.target[1], 2f * mt.target[2]);
            Vector3 center = centerXZOnTarget ? new Vector3(0f, mt.target[1], 0f)
                                              : new Vector3(mt.target[0], mt.target[1], -mt.target[2]);
            Gizmos.DrawWireCube(center, size);
        }
        catch { }
    }

    void OnDestroy()
    {
        worker?.Dispose();
        if (fieldTex != null) Destroy(fieldTex);
        if (smoothedRT != null) smoothedRT.Release();
        if (smoothedRT2 != null) smoothedRT2.Release();
        if (cRT != null) cRT.Release();
        if (mRT != null) mRT.Release();
        if (nRT != null) nRT.Release();
        if (smoothMat != null) Destroy(smoothMat);
        if (shadeMat != null) Destroy(shadeMat);
        if (compositeMat != null) Destroy(compositeMat);
        if (quadGO != null) Destroy(quadGO);
    }
}
