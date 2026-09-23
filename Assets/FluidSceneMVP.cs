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
// B bilateral smoothing, [ / ] playback fps −/+5, G whitewater layer (FoamLayer), H whitewater only.

using System;
using System.Collections;
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
    [Tooltip("Inference Engine backend. 2026-09-21 (067, ledger A4): GPUCompute fp32 AND GPUPixel fp32 both match onnxruntime on 053a / 060a / 058c and the wide models (max |d| < 1e-3, identical occupancy mask) on Inference Engine 2.6.1 — the older 'GPUCompute is silently wrong' warning no longer holds for these models.")]
    public BackendType backend = BackendType.GPUPixel;
    [Tooltip("Attempt fp16 weight quantization (auto-falls back to fp32 on failure). NOTE (IE 2.6.1 source, QuantizeConstantsPass): this only STORES conv weights as fp16 and inserts a Cast back to float in front of each conv — the math stays fp32, so it shrinks the model but does not make inference faster.")]
    public bool useFp16 = true;
    [Tooltip("0 = synchronous: every rendered frame splats, runs the whole network and blocks on the readback (the research/recording default). N > 0 = spread each inference over ~N rendered frames (Worker.ScheduleIterable, ~1/N of the layers per frame) and fetch the result with an async readback: the scene, camera and stirrer render at the full frame rate and the fluid updates whenever a prediction lands (the composite keeps it world-locked through camera rotation). Forced to 0 while capturing or while LiveClipRecorder listens.")]
    [Min(0)] public int spreadFrames = 0;
    [Tooltip("Spread Frames used instead on phones/tablets (Application.isMobilePlatform). iPhone 16 Pro / 067b: ~84 ms per synchronous inference, so 3 = ~28 ms of network per frame.")]
    [Min(0)] public int spreadFramesMobile = 3;
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

    public enum TemporalMode { Off, Adaptive }
    [Header("Temporal (067-EMA3) — off by default")]
    [Tooltip("Adaptive = a temporal stage between the presmooth and the shading: the history is reprojected through the camera motion, depth/thickness get an EMA whose alpha rises to 1 where the INPUT's own speed channel says the fluid moves, and the occupancy mask gets hysteresis. Offline on frozen live clips (067): static camera -41..-43 % normal flicker after sigma 2 and -40..-45 % silhouette crawl with <= 0.5 cm ghosts; orbiting camera -50..-59 % frame-to-frame normal change (only -19..-53 % without the reprojection). Mirror + known-answer gate: SSU_restart/Helpers/live_gap_temporal.py.")]
    public TemporalMode temporalMode = TemporalMode.Off;
    [Tooltip("EMA alpha where the fluid is at rest (memory ~ 1/alpha frames). 1 = no smoothing.")]
    [Range(0.05f, 1f)] public float temporalAlphaRest = 0.3f;
    [Tooltip("Input speed [m/s] below which alpha = Alpha Rest / above which alpha = 1 (smoothstep between).")]
    public Vector2 temporalSpeedRamp = new Vector2(0.1f, 0.5f);
    [Tooltip("History is ignored where it disagrees with the current depth by more than this [sim m] (disocclusion).")]
    public float temporalRejectM = 0.10f;
    public bool temporalReproject = true;
    [Tooltip("Depth gate (x, y) in sim metres: alpha also rises to 1 as |current depth - reprojected history| goes from x to y. (0, 0) = off (default). (0.01, 0.04) cut the ghost tail behind a 3.3 m/s stirrer from 34 to 9.5 mm (p90) offline with no loss of the calm-pool flicker gain, at a few points of the moving-camera gain.")]
    public Vector2 temporalDepthGate = Vector2.zero;
    public bool temporalMaskHysteresis = true;

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

    [Header("Foam / whitewater (059 compositor layer, key G) — off by default")]
    [Tooltip("Draw Ihmsen-style spray/foam/bubbles generated from the provider's particle frame (FoamLayer.cs, the in-engine port of the 059 probe) over the fluid. Pure compositor: no network output involved. Diffuse particles are drawn in the model window from the splat camera, depth-tested against the predicted fluid depth, and (where they have their own depth) against the scene depth. Toggle at runtime with G.")]
    public bool foamEnabled = false;
    [Tooltip("Show ONLY the whitewater layer (the fluid is not composited) — for tuning, or to show the audience what the layer adds. Toggle at runtime with H.")]
    public bool foamOnlyView = false;
    [Tooltip("Particle radius the layer uses for its neighbour support (h = 4 r), spawn cylinder and sprite size. 0 = provider.ParticleRadius (GpuSphProvider 0.0414), then the stats meta's coarseRadius.")]
    public float foamRadiusOverride = 0f;

    [Header("Foam — spawning (Ihmsen potentials)")]
    [Tooltip("Spawn rate multiplier (the probe's mass factor). Higher = more whitewater. Tested: 20.")]
    public float foamSpawnScale = 20f;
    [Tooltip("Trapped-air rate k_ta (spawns inside turbulent / colliding flow). Tested: 8.")]
    public float foamKTa = 8f;
    [Tooltip("Wave-crest rate k_wc (spawns on convex crests moving outward). Tested: 12.")]
    public float foamKWc = 12f;
    [Tooltip("Multiplier on the running potential calibration tau. Lower = potentials saturate sooner = more, earlier foam. Tested: 1.")]
    public float foamTauScale = 1f;
    [Tooltip("Per-step decay of the running tau (tau = max(tau * decay, this frame's percentile)). On the live GPU solver one step = one solver frame (1/simHz); on baked playback one rendered frame. Closer to 1 = the biggest splash keeps setting the scale longer (less foam from later, smaller events). Tested: 0.98.")]
    [Range(0.8f, 1f)] public float foamTauDecay = 0.98f;
    [Tooltip("Percentile of this frame's potentials that feeds the running tau. Lower = tau lower = more particles spawn. Tested: 0.995.")]
    [Range(0.5f, 1f)] public float foamTauPercentile = 0.995f;
    [Tooltip("Potential clamp: Phi ramps from (this x tau) to tau. Lower = weaker potentials spawn too. Tested: 0.25.")]
    [Range(0f, 0.95f)] public float foamTauMinFrac = 0.25f;
    [Tooltip("Kinetic gate (m/s): no spawning below x, full rate above y (ramped over 0.5 v^2). Lower it if stirring does not produce foam; the dam drop reaches ~5 m/s. Tested: (1, 3).")]
    public Vector2 foamKineticSpeed = new Vector2(1f, 3f);
    [Tooltip("Wave crest only counts where the particle moves along its surface normal: v^.n >= this. Tested: 0.6.")]
    [Range(-1f, 1f)] public float foamCrestMinVelDotN = 0.6f;
    [Tooltip("A fluid particle is a SURFACE particle (eligible for wave crests) below this fraction of n_full neighbours. Tested: 0.75.")]
    [Range(0f, 1f)] public float foamSurfaceFrac = 0.75f;
    [Tooltip("n_full = this percentile of the fluid neighbour counts (the 'fully immersed' count all fractions refer to). Tested: 0.9.")]
    [Range(0.5f, 1f)] public float foamNFullPercentile = 0.9f;

    [Header("Foam — diffuse particles")]
    [Tooltip("Cap on live diffuse particles; the oldest are dropped when full. Tested: 30000.")]
    public int foamMaxDiffuse = 30000;
    [Tooltip("Foam particle lifetime range in seconds (scaled by 0.5 + 0.5 Phi_k at spawn; ticks only while classified as foam). Tested: (1, 4).")]
    public Vector2 foamLifetime = new Vector2(1f, 4f);
    [Tooltip("Classified as SPRAY (ballistic) below this fraction of n_full fluid neighbours. Tested: 0.15.")]
    [Range(0f, 1f)] public float foamSprayFrac = 0.15f;
    [Tooltip("Classified as BUBBLE (buoyant + drag) above this fraction of n_full fluid neighbours; foam in between. Tested: 0.6.")]
    [Range(0f, 1f)] public float foamBubbleFrac = 0.6f;
    [Tooltip("Bubble buoyancy k_b (multiple of -gravity). Tested: 0.5.")]
    public float foamBuoyancy = 0.5f;
    [Tooltip("Bubble drag k_d towards the local fluid velocity, per step. Tested: 0.7.")]
    [Range(0f, 1f)] public float foamDrag = 0.7f;
    [Tooltip("Gravity for spray/bubbles in sim space. Tested: (0, -9.81, 0).")]
    public Vector3 foamGravity = new Vector3(0f, -9.81f, 0f);
    [Tooltip("Spray older than this (s) is culled. Tested: 3.")]
    public float foamSprayMaxAge = 3f;
    [Tooltip("Particles with no fluid within h older than this (s) are culled. Tested: 1.")]
    public float foamOrphanMaxAge = 1f;
    [Tooltip("Particles below this sim-space y are culled (the sim floor is y = 0). Tested: -0.05.")]
    public float foamFloorY = -0.05f;
    [Tooltip("Particles outside this sim-space box are culled. Tested: (-1,-0.1,-1)..(4,4,4) around the [0,3]^3 domain.")]
    public Vector3 foamDomainMin = new Vector3(-1f, -0.1f, -1f), foamDomainMax = new Vector3(4f, 4f, 4f);

    [Header("Foam — rendering")]
    [Tooltip("Whitewater colour (sRGB).")]
    public Color foamColor = new Color(0.96f, 0.98f, 1f, 1f);
    [Tooltip("Multiplier on the whitewater colour (can exceed 1 for HDR/bloom).")]
    public float foamBrightness = 1f;
    [Tooltip("0 = flat colour (as tested); 1 = colour x the main directional light's colour x intensity, so foam dims with the scene lighting.")]
    [Range(0f, 1f)] public float foamLightInfluence = 0f;
    [Tooltip("Density -> coverage: 1 - exp(-k D). Higher = denser-looking foam from the same particles. Tested: 0.8.")]
    public float foamCoverageK = 0.8f;
    [Tooltip("Max opacity of the whitewater layer.")]
    [Range(0f, 1f)] public float foamOpacity = 1f;
    [Tooltip("Multiplier on each particle's disk footprint (0 = single pixel). Tested: 1.")]
    public float foamSpriteScale = 1f;
    [Tooltip("Footprint radius as a fraction of the projected particle radius: x = spray, y = foam/bubbles. Tested: (0.33, 0.5).")]
    public Vector2 foamSpriteFrac = new Vector2(0.33f, 0.5f);
    [Tooltip("Largest footprint radius in model-window pixels. Tested: 3.")]
    [Range(0, 8)] public int foamMaxSpritePx = 3;
    [Tooltip("Density weight per type: x = spray, y = foam, z = bubble. Tested: (0.6, 1, 0.5).")]
    public Vector3 foamTypeWeights = new Vector3(0.6f, 1f, 0.5f);
    [Tooltip("Weight multiplier for particles BEHIND the predicted fluid surface (bubbles seen through the water). 0 = hide them. Tested: 0.12.")]
    [Range(0f, 1f)] public float foamHiddenWeight = 0.12f;
    [Tooltip("A particle counts as in front of the fluid if it is within this many particle radii behind the predicted surface. Tested: 4.")]
    public float foamDepthTolR = 4f;
    [Tooltip("Separable Gaussian (sigma 1 px) on the density. Off = crisper, noisier specks. Tested: on.")]
    public bool foamBlur = true;

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
    // spread inference (ActiveSpread > 0)
    IEnumerator pendingSchedule;           // ScheduleIterable in flight; null once every layer is scheduled
    Tensor<float> pendingInput;            // its input, disposed once the prediction has been read back
    bool awaitingReadback;
    int modelLayerCount = 1;
    readonly System.Diagnostics.Stopwatch pendingSw = new System.Diagnostics.Stopwatch();
    float fluidHz, lastPredTime = -1f;
    Vector3 shadeRightWS, shadeUpWS, shadeFwdWS;   // world camera basis of the splat the current prediction came from
    float kThick;
    float thickScale = 1f;                 // particle-count normalization for the thickness channel (legacy only)
    bool useV2; float v2R, v2TS, v2MinR = 1f, v2MaxR = 24f;
    bool kThickLocked;

    Material smoothMat, shadeMat, compositeMat;
    Texture2D fieldTex;
    RenderTexture smoothedRT, smoothedRT2, cRT, mRT, nRT;
    RenderTexture histA, histB;            // temporal stage ping-pong (r D, g T, b on, a M)
    Material temporalMat;
    bool hasHistory;
    Vector3 eyePrev, rightPrev, upPrev, fwdPrev; float focalPrev;
    /// <summary>Temporal-stage taps for LiveClipRecorder (valid during OnShaded): the stage's input, the history it read, its output.</summary>
    public RenderTexture TemporalIn { get; private set; }
    public RenderTexture TemporalHist { get; private set; }
    public RenderTexture TemporalOut { get; private set; }
    /// <summary>Raised every rendered frame after the shading blits.</summary>
    public event Action OnShaded;
    float bilateralSigmaR;
    GameObject quadGO;
    MeshRenderer quadMR;
    Light mainLight;

    // 059 whitewater layer (CPU): stepped when a particle frame is splatted, drawn when its prediction lands
    FoamLayer foam;
    float[] foamFluidDepth, foamDen, foamFront, foamTmp, foamStage;   // CHW, row 0 = top
    Texture2D foamTex, foamDepthTex;                                   // texture rows (row 0 = bottom)
    float lastFoamClock;

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

    // ---------- runtime controls (keyboard in HandleInput, TouchControls on phones) ----------
    public Camera TargetCamera => targetCamera;
    public bool Paused => paused;
    public bool ShowingRawInput => showRawInput;
    public void TogglePause() => paused = !paused;
    public void ToggleRawInput() => showRawInput = !showRawInput;
    public void ToggleBilateral() => bilateralSmoothing = !bilateralSmoothing;
    public void ResetSim() { simTime = 0f; provider?.ResetSim(); hasHistory = false; foam?.Reset(); lastFoamClock = 0f; }
    public bool FoamEnabled => foamEnabled;
    public void ToggleFoam() { foamEnabled = !foamEnabled; if (foamEnabled) foam?.Reset(); }
    public void ToggleFoamOnlyView() => foamOnlyView = !foamOnlyView;
    /// <summary>Spread actually in use this frame (0 = synchronous).</summary>
    public int ActiveSpread => capturing || OnInferred != null || OnShaded != null ? 0
        : Mathf.Max(0, Application.isMobilePlatform ? spreadFramesMobile : spreadFrames);

    [Serializable]
    public class ClipSettings
    {
        public string splatMode, focalMode, model, stats, backend;
        public bool fp16Active, thicknessCountNormalize;
        public float v2R, v2TS, v2MinR, v2MaxR, thickScale, refLrParticleRadius, presmoothSigma, depthBias, simScale, kThick;
        public float[] mean, std, target, simOffset;
        public int winW, winH;
        public string temporalMode; public float temporalAlphaRest, temporalV0, temporalV1, temporalRejectM, temporalDGate0, temporalDGate1; public bool temporalReproject, temporalMaskHysteresis;
        public float inferMs;      // smoothed wall-clock ms of RunModel (Schedule + blocking readback) when the clip ended
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
        winW = W, winH = H, inferMs = inferMs,
        temporalMode = temporalMode.ToString(), temporalAlphaRest = temporalAlphaRest, temporalV0 = temporalSpeedRamp.x, temporalV1 = temporalSpeedRamp.y,
        temporalRejectM = temporalRejectM, temporalDGate0 = temporalDepthGate.x, temporalDGate1 = temporalDepthGate.y, temporalReproject = temporalReproject, temporalMaskHysteresis = temporalMaskHysteresis,
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

        foam = new FoamLayer();
        foamFluidDepth = new float[HW]; foamDen = new float[HW]; foamFront = new float[HW];
        foamTmp = new float[HW]; foamStage = new float[HW];
        foamTex = new Texture2D(W, H, TextureFormat.RFloat, false) { filterMode = FilterMode.Bilinear, wrapMode = TextureWrapMode.Clamp };
        foamDepthTex = new Texture2D(W, H, TextureFormat.RFloat, false) { filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp };

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
        if (temporalMode != TemporalMode.Off)
        {
            var tsh = Shader.Find("Hidden/FluidTemporal");
            if (tsh == null) throw new Exception("Hidden/FluidTemporal shader not found");
            temporalMat = new Material(tsh);
            histA = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBFloat) { filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp };
            histB = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBFloat) { filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp };
        }

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
        quadMR = mr;
        mr.enabled = false;   // until the first prediction is shaded (a spread inference lands a few frames in)
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
        modelLayerCount = Mathf.Max(1, model.layers.Count);
        BeginInference();
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

    // Splat the current frame from the current camera and remember that camera's world basis:
    // the shading packs and the composite must use the pose the prediction was made from.
    void BeginInference()
    {
        ComputeSimCamera();
        SplatToInput(frameIdx);
        StepFoam();
        Transform c = targetCamera.transform;
        shadeRightWS = c.right; shadeUpWS = c.up; shadeFwdWS = c.forward;
    }

    // One rendered frame of a spread inference: start one if idle, schedule ~1/spread of the
    // layers, and shade the prediction once its async readback has landed.
    void StepSpreadInference(int spread)
    {
        if (pendingSchedule == null && !awaitingReadback)
        {
            BeginInference();
            pendingInput = new Tensor<float>(new TensorShape(1, 7, H, W), in7);
            pendingSchedule = worker.ScheduleIterable(pendingInput);
            pendingSw.Restart();
        }
        if (pendingSchedule != null)
        {
            int n = Mathf.CeilToInt(modelLayerCount / (float)spread);
            for (int k = 0; k < n; k++)
                if (!pendingSchedule.MoveNext())
                {
                    pendingSchedule = null;
                    worker.PeekOutput().ReadbackRequest();
                    awaitingReadback = true;
                    break;
                }
        }
        if (awaitingReadback && worker.PeekOutput().IsReadbackRequestDone())
        {
            float[] pred;
            using (var t = (worker.PeekOutput() as Tensor<float>).ReadbackAndClone())   // served from the finished async request
                pred = t.DownloadToArray();
            awaitingReadback = false;
            pendingInput.Dispose(); pendingInput = null;
            float ms = (float)pendingSw.Elapsed.TotalMilliseconds;   // splat -> prediction landed (latency, not GPU time)
            inferMs = inferMs <= 0f ? ms : Mathf.Lerp(inferMs, ms, 0.2f);
            ProcessPrediction(pred);
        }
    }

    // Finish and discard an in-flight spread inference (switching to synchronous mid-flight).
    void DrainPending()
    {
        if (pendingSchedule == null && !awaitingReadback && pendingInput == null) return;
        if (pendingSchedule != null) while (pendingSchedule.MoveNext()) { }
        using (var t = (worker.PeekOutput() as Tensor<float>).ReadbackAndClone()) { }   // wait for the GPU before freeing the input
        pendingSchedule = null; awaitingReadback = false;
        pendingInput?.Dispose(); pendingInput = null;
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

        int spread = ActiveSpread;
        if (spread > 0) StepSpreadInference(spread);
        else
        {
            DrainPending();   // no-op unless the spread was just switched off mid-inference
            BeginInference();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            float[] pred = RunModel();
            sw.Stop();
            inferMs = Mathf.Lerp(inferMs <= 0f ? (float)sw.Elapsed.TotalMilliseconds : inferMs,
                                 (float)sw.Elapsed.TotalMilliseconds, 0.1f);
            ProcessPrediction(pred);
        }

        CaptureAndStatus();
    }

    // Everything downstream of the network for one prediction: kThick, the (depth, thickness,
    // alpha) stage, smoothing, temporal, the 512² shading packs and the composite parameters.
    void ProcessPrediction(float[] pred)
    {
        lastPred = pred;
        OnInferred?.Invoke();
        float now = Time.realtimeSinceStartup;
        if (lastPredTime >= 0f && now > lastPredTime)
            fluidHz = fluidHz <= 0f ? 1f / (now - lastPredTime) : Mathf.Lerp(fluidHz, 1f / (now - lastPredTime), 0.2f);
        lastPredTime = now;

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
                float spd = 1f;
                if (temporalMat != null)
                {   // |v_cam| of the INPUT at this pixel, de-normalised from the tensor the network just read (0 where the input is empty)
                    spd = 0f;
                    if (in7[6 * HW + i] > 0f)
                    {
                        float vx = in7[2 * HW + i] * meta.std[2] + meta.mean[2], vy = in7[3 * HW + i] * meta.std[3] + meta.mean[3], vz = in7[4 * HW + i] * meta.std[4] + meta.mean[4];
                        spd = Mathf.Sqrt(vx * vx + vy * vy + vz * vz);
                    }
                }
                px[flipped] = new Color(depth, thick, a, spd);
                foamFluidDepth[i] = a > 0f ? depth : 0f;
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

        // temporal stage (067-EMA3): reprojected, motion-adaptive EMA + mask hysteresis. Off = the shading reads smoothedRT as before.
        RenderTexture shadeSrc = smoothedRT;
        if (temporalMat != null)
        {
            temporalMat.SetTexture("_FieldTex", fieldTex);
            temporalMat.SetTexture("_HistTex", histB);
            temporalMat.SetFloat("_HasHistory", hasHistory ? 1f : 0f);
            temporalMat.SetFloat("_Reproject", temporalReproject ? 1f : 0f);
            temporalMat.SetFloat("_Hyst", temporalMaskHysteresis ? 1f : 0f);
            temporalMat.SetFloat("_ARest", temporalAlphaRest);
            temporalMat.SetFloat("_V0", temporalSpeedRamp.x);
            temporalMat.SetFloat("_V1", temporalSpeedRamp.y);
            temporalMat.SetFloat("_RejectM", temporalRejectM);
            temporalMat.SetFloat("_DGate0", temporalDepthGate.x); temporalMat.SetFloat("_DGate1", temporalDepthGate.y);
            temporalMat.SetFloat("_WinAspect", winAspect);
            temporalMat.SetFloat("_FocalC", focalM); temporalMat.SetFloat("_FocalP", focalPrev);
            temporalMat.SetVector("_EyeC", eyeSim); temporalMat.SetVector("_RightC", rightSim); temporalMat.SetVector("_UpC", upSim); temporalMat.SetVector("_FwdC", fwdSim);
            temporalMat.SetVector("_EyeP", eyePrev); temporalMat.SetVector("_RightP", rightPrev); temporalMat.SetVector("_UpP", upPrev); temporalMat.SetVector("_FwdP", fwdPrev);
            Graphics.Blit(smoothedRT, histA, temporalMat);
            TemporalIn = smoothedRT; TemporalHist = histB; TemporalOut = histA;
            shadeSrc = histA;
            (histA, histB) = (histB, histA);       // this frame's output is next frame's history
            hasHistory = true;
            eyePrev = eyeSim; rightPrev = rightSim; upPrev = upSim; fwdPrev = fwdSim; focalPrev = focalM;
        }

        // scene-independent shading packs at 512² (premultiplied by coverage)
        SetShadeParams(shadeMat);
        Graphics.Blit(shadeSrc, cRT, shadeMat, 0);
        Graphics.Blit(shadeSrc, mRT, shadeMat, 1);
        Graphics.Blit(shadeSrc, nRT, shadeMat, 2);
        OnShaded?.Invoke();

        // composite quad (samples scene color/depth in-render, pass 3)
        compositeMat.SetTexture("_CTex", cRT);
        compositeMat.SetTexture("_MTex", mRT);
        compositeMat.SetTexture("_NTex", nRT);
        compositeMat.SetFloat("_FocalM", focalM);
        compositeMat.SetFloat("_WinAspect", winAspect);
        compositeMat.SetFloat("_SimScale", transform.lossyScale.x);
        compositeMat.SetFloat("_DepthBias", depthBias);
        // the SPLAT camera's basis, not the current one: between spread predictions this keeps
        // the fluid world-locked through camera rotation (identical to the current camera when synchronous)
        compositeMat.SetVector("_CamRightWS", shadeRightWS);
        compositeMat.SetVector("_CamUpWS", shadeUpWS);
        compositeMat.SetVector("_CamFwdWS", shadeFwdWS);
        DrawFoam();
        quadMR.enabled = true;
    }

    // ---------- 059 whitewater layer ----------

    // Sim time the foam advances on: the GPU solver's own clock when live (so the layer steps once per NEW
    // solver frame, whatever the render rate), else the playback clock.
    float FoamClock => provider is GpuSphProvider gpu ? gpu.SolverTime : simTime;

    float FoamRadius => foamRadiusOverride > 0f ? foamRadiusOverride
        : provider.ParticleRadius > 0f ? provider.ParticleRadius
        : meta.coarseRadius > 0f ? meta.coarseRadius : 0.0414f;

    void ApplyFoamKnobs()
    {
        foam.spawnScale = foamSpawnScale; foam.kTa = foamKTa; foam.kWc = foamKWc; foam.tauScale = foamTauScale;
        foam.tauDecay = foamTauDecay; foam.tauPercentile = foamTauPercentile; foam.tauMinFrac = foamTauMinFrac;
        foam.kineticMinSpeed = foamKineticSpeed.x; foam.kineticMaxSpeed = Mathf.Max(foamKineticSpeed.y, foamKineticSpeed.x + 1e-3f);
        foam.crestMinVelDotN = foamCrestMinVelDotN; foam.surfaceFrac = foamSurfaceFrac; foam.nFullPercentile = foamNFullPercentile;
        foam.maxDiffuse = Mathf.Max(foamMaxDiffuse, 1000);
        foam.lifeMin = foamLifetime.x; foam.lifeMax = Mathf.Max(foamLifetime.y, foamLifetime.x);
        foam.sprayFrac = foamSprayFrac; foam.bubbleFrac = foamBubbleFrac; foam.buoyancy = foamBuoyancy; foam.drag = foamDrag;
        foam.gravity = foamGravity; foam.sprayMaxAge = foamSprayMaxAge; foam.orphanMaxAge = foamOrphanMaxAge;
        foam.floorY = foamFloorY; foam.domainMin = foamDomainMin; foam.domainMax = foamDomainMax;
        foam.spriteScale = foamSpriteScale; foam.spraySprite = foamSpriteFrac.x; foam.foamSprite = foamSpriteFrac.y;
        foam.maxSpritePx = foamMaxSpritePx;
        foam.weightSpray = foamTypeWeights.x; foam.weightFoam = foamTypeWeights.y; foam.weightBubble = foamTypeWeights.z;
        foam.hiddenWeight = foamHiddenWeight; foam.depthTolR = foamDepthTolR; foam.blur = foamBlur;
    }

    // Advance the diffuse particles in the frame just splatted (GetFrame re-serves the cached readback).
    void StepFoam()
    {
        float clock = FoamClock;
        float fdt = Mathf.Clamp(clock - lastFoamClock, 0f, 0.2f);
        lastFoamClock = clock;
        if (!foamEnabled || fdt <= 0f) return;   // paused / no new solver frame: particles hold still
        ApplyFoamKnobs();
        provider.GetFrame(frameIdx, out var data, out int off, out int count);
        foam.Step(data, off, count, FoamRadius, fdt);
    }

    // Draw the particles from the SPLAT camera (the pose this prediction came from, so the layer stays
    // world-locked with the fluid in spread mode) into the model window, depth-tested against its depth.
    void DrawFoam()
    {
        compositeMat.SetFloat("_FoamOn", foamEnabled ? 1f : 0f);
        compositeMat.SetFloat("_FoamOnly", foamEnabled && foamOnlyView ? 1f : 0f);
        if (!foamEnabled) return;
        ApplyFoamKnobs();
        foam.Splat(eyeSim, rightSim, upSim, fwdSim, focalM, H, W, foamFluidDepth, FoamRadius, foamDen, foamTmp, foamFront, winAspect);
        UploadFoam(foamTex, foamDen);
        UploadFoam(foamDepthTex, foamFront);

        Color c = foamColor.linear * foamBrightness;
        if (mainLight != null && foamLightInfluence > 0f)
        {
            Color l = mainLight.color.linear * mainLight.intensity;
            c *= Color.Lerp(Color.white, l, foamLightInfluence);
        }
        compositeMat.SetTexture("_FoamTex", foamTex);
        compositeMat.SetTexture("_FoamDepthTex", foamDepthTex);
        compositeMat.SetFloat("_FoamK", foamCoverageK);
        compositeMat.SetFloat("_FoamOpacity", foamOpacity);
        compositeMat.SetVector("_FoamColor", new Vector4(c.r, c.g, c.b, 0f));   // linear, no gamma conversion
    }

    void UploadFoam(Texture2D tex, float[] chw)
    {
        for (int y = 0; y < H; y++)
            Array.Copy(chw, y * W, foamStage, (H - 1 - y) * W, W);   // tensor row 0 = top; texture row 0 = bottom
        tex.SetPixelData(foamStage, 0);
        tex.Apply(false, false);
    }

    void CaptureAndStatus()
    {
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
                 $"render {smoothedFps:F1} fps   " +
                 (ActiveSpread > 0 ? $"infer spread/{ActiveSpread} {inferMs:F0} ms latency, fluid {fluidHz:F1} Hz   "
                                   : $"infer {inferMs:F1} ms   ") +
                 $"backend={backend} fp16={fp16Active}   " +
                 $"k_thick={kThick:F3}{(showRawInput ? "   RAW INPUT" : "")}" +
                 $"{(bilateralSmoothing ? "   BILATERAL" : "")}{(paused ? "   PAUSED" : "")}" +
                 $"{(capturing ? "   CAPTURING" : "")}" +
                 (foamEnabled ? $"   FOAM{(foamOnlyView ? " ONLY" : "")} {foam.Alive} ({foam.Spray}s/{foam.Foam}f/{foam.Bubble}b) " +
                                $"+{foam.SpawnedTa}ta/{foam.SpawnedWc}wc  {foam.LastStepMs:F1}+{foam.LastSplatMs:F1} ms" : "");
    }

    void SetShadeParams(Material m)
    {
        m.SetFloat("_FocalM", focalM);
        m.SetFloat("_WinAspect", winAspect);
        m.SetVector("_CamRightWS", shadeRightWS);
        m.SetVector("_CamUpWS", shadeUpWS);
        m.SetVector("_CamFwdWS", shadeFwdWS);
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
            if (kb.pKey.wasPressedThisFrame) TogglePause();
            if (kb.bKey.wasPressedThisFrame) ToggleBilateral();
            if (kb.vKey.wasPressedThisFrame) ToggleRawInput();
            if (kb.rKey.wasPressedThisFrame) ResetSim();
            if (kb.gKey.wasPressedThisFrame) ToggleFoam();
            if (kb.hKey.wasPressedThisFrame) ToggleFoamOnlyView();
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
        if (!showStatus) return;
        float s = TouchControls.BeginScaledGUI();   // identity on desktop; dpi-scaled + safe-area on phones
        GUI.Label(new Rect(10, 8, Mathf.Max(Screen.width / s - 20, 200), 60), status);
        GUI.matrix = Matrix4x4.identity;
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
        pendingInput?.Dispose();
        worker?.Dispose();
        if (fieldTex != null) Destroy(fieldTex);
        if (foamTex != null) Destroy(foamTex);
        if (foamDepthTex != null) Destroy(foamDepthTex);
        if (smoothedRT != null) smoothedRT.Release();
        if (smoothedRT2 != null) smoothedRT2.Release();
        if (cRT != null) cRT.Release();
        if (mRT != null) mRT.Release();
        if (nRT != null) nRT.Release();
        if (histA != null) histA.Release();
        if (histB != null) histB.Release();
        if (temporalMat != null) Destroy(temporalMat);
        if (smoothMat != null) Destroy(smoothMat);
        if (shadeMat != null) Destroy(shadeMat);
        if (compositeMat != null) Destroy(compositeMat);
        if (quadGO != null) Destroy(quadGO);
    }
}
