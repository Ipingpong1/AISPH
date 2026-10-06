using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.SceneManagement;
using Unity.InferenceEngine;

/// <summary>Creates only LiquidRelay assets; never changes the research SampleScene.</summary>
public static class LiquidRelaySceneBuilder
{
    public const string Root = "Assets/LiquidRelay";
    public const string ScenePath = Root + "/Scenes/LiquidRelay.unity";
    const string ModelPath = Root + "/Models/liquid_relay_ft_v1.onnx";
    const string ModelStatsPath = Root + "/Models/liquid_relay_ft_v1_stats.json";

    [MenuItem("Tools/Liquid Relay/Create Demo Scene")]
    public static void CreateScene()
    {
        if (EditorApplication.isPlaying) throw new InvalidOperationException("Exit Play mode before building Liquid Relay.");
        EnsureFolder(Root + "/Scenes"); EnsureFolder(Root + "/Materials");
        WriteStats();
        AssetDatabase.ImportAsset(Root + "/relay_scene.json", ImportAssetOptions.ForceSynchronousImport);

        Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Additive);
        SceneManager.SetActiveScene(scene);
        var apparatus = new GameObject("LIQUID RELAY");
        var fluidGO = new GameObject("Fluid Simulation");
        fluidGO.transform.SetParent(apparatus.transform, false);
        var provider = fluidGO.AddComponent<DfsphProvider>();
        provider.sceneAsset = AssetDatabase.LoadAssetAtPath<TextAsset>(Root + "/relay_scene.json");
        provider.scenePath = Root + "/relay_scene.json";
        provider.maxParticles = 8192;
        provider.maxStepsPerFrame = 12;
        provider.liveCapPressure = 20;
        provider.liveCapDivergence = 10;
        provider.corpusTessellation = false;
        provider.propMapResolution = new Vector3Int(24, 24, 24);
        provider.domainCenterXZ = new Vector2(1.5f, 1.5f);
        provider.domainMin = Vector3.zero; provider.domainMax = Vector3.one * 3;

        var fluid = fluidGO.AddComponent<FluidSceneMVP>();
        fluid.provider = provider;
        fluid.modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelPath);
        if (fluid.modelAsset == null) throw new FileNotFoundException("Liquid Relay model missing", ModelPath);
        fluid.statsJson = AssetDatabase.LoadAssetAtPath<TextAsset>(Root + "/relay_model_meta.json");
        fluid.sceneShader = AssetDatabase.LoadAssetAtPath<Shader>("Assets/SSFR/Resources/FluidSSFRScene.shader");
        fluid.smoothShader = AssetDatabase.LoadAssetAtPath<Shader>("Assets/SSFR/Resources/FluidSSFR.shader");
        fluid.backend = BackendType.GPUCompute;
        fluid.useFp16 = false;
        fluid.splatMode = FluidSceneMVP.SplatMode.V2;
        fluid.focalMode = FluidSceneMVP.FocalMode.FitVertical;
        fluid.windowWidth = 512;
        fluid.gpuPath = true;
        fluid.spreadFrames = 0;
        fluid.flyControls = false;
        fluid.keyboardShortcuts = false;
        fluid.showStatus = false;
        fluid.speed = .42f;
        fluid.kThickOverride = .50f;
        fluid.temporalMode = FluidSceneMVP.TemporalMode.Adaptive;
        fluid.temporalAlphaRest = .35f;
        fluid.temporalDepthGate = new Vector2(.01f, .04f);
        fluid.presmoothSigma = 1.2f;
        fluid.depthBias = .015f;
        fluid.foamEnabled = false;
        fluid.sprayEnabled = false;
        fluid.defaultLook = false;
        fluid.absorbColor = new Color(.60f, .19f, .10f);
        fluid.bodyTintColor = new Color(.025f, .24f, .29f);
        fluid.refrStrength = 12;
        fluid.ks = .85f;
        fluid.shininess = 180;
        // Use the existing continuous-support classical variant for this demo.
        // It adapts sparse splats and the NR radius to this scene's particle scale.
        fluid.classicalSettings.repairSupportContinuity = true;
        fluid.classicalSettings.nrUseParticleRadius = true;
        fluid.classicalSettings.supportContinuityEndpoint = 12.5f;
        fluid.classicalSettings.sparseRadiusMult = 1.5f;

        Material steel = MaterialAsset("Anodized Graphite", Hex("273A44"), .35f, .38f);
        Material dark = MaterialAsset("Midnight Enamel", Hex("10252E"), .10f, .30f);
        Material trim = MaterialAsset("Brushed Aluminum", Hex("9AADB3"), .50f, .48f);
        Material teal = MaterialAsset("Receiver A Teal", Hex("57DEC3"), .10f, .35f);
        Material brass = MaterialAsset("Receiver B Brass", Hex("DFB56A"), .16f, .40f);
        Material coral = MaterialAsset("Waste Coral", Hex("F28A73"), .1f, .4f);
        Material ground = MaterialAsset("Studio Ground", Hex("0B1820"), .05f, .28f);

        var machine = new GameObject("Machine").transform;
        machine.SetParent(apparatus.transform, false);
        var props = new List<LiveSphProvider.Obstacle>();
        BoxBody("Reservoir floor", new Vector3(.58f, 1.62f, 1.5f), new Vector3(1.40f, .18f, 1.54f), Quaternion.Euler(0, 0, -8), steel, machine, props);
        BoxBody("Reservoir wall A", new Vector3(.58f, 2.15f, .70f), new Vector3(1.42f, 1.54f, .16f), Quaternion.identity, dark, machine, props);
        Transform frontWall = BoxBody("Reservoir wall B", new Vector3(.58f, 2.15f, 2.30f), new Vector3(1.42f, 1.54f, .16f), Quaternion.identity, dark, machine, props);
        // The near wall is a deliberate cutaway; its edge frame keeps the container readable.
        frontWall.GetComponent<MeshRenderer>().enabled = false;
        var gate = BoxBody("Release gate", LiquidRelayGame.GateCenter, new Vector3(.18f, 1.54f, 1.76f), Quaternion.identity, trim, machine, props);
        gate.GetComponent<MeshRenderer>().enabled = false;
        var shutterObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
        shutterObject.name = "Retracting shutter";
        shutterObject.transform.SetParent(gate, false);
        shutterObject.GetComponent<Renderer>().sharedMaterial = trim;
        UnityEngine.Object.DestroyImmediate(shutterObject.GetComponent<Collider>());
        var gateVisual = gate.gameObject.AddComponent<LiquidRelayGateVisual>();
        gateVisual.shutter = shutterObject.transform;
        var paddle = BoxBody("Tilting deflector", LiquidRelayGame.DeflectorCenter, new Vector3(1.42f, .16f, 1.28f), LiquidRelayGame.DeflectorRotation(0), brass, machine, props);
        BoxBody("Waste threshold", new Vector3(1.47f, .24f, 1.5f), new Vector3(.16f, .48f, 3), Quaternion.identity, steel, machine, props);
        BoxBody("Receiver divider", new Vector3(2.24f, .25f, 1.5f), new Vector3(1.52f, .50f, .16f), Quaternion.identity, steel, machine, props);
        provider.obstacles = props.ToArray();

        var decor = new GameObject("Machine Finish").transform;
        decor.SetParent(apparatus.transform, false);
        VisualBox("Plinth", new Vector3(1.5f, -.15f, 1.5f), new Vector3(3.38f, .28f, 3.38f), dark, decor);
        VisualBox("A basin floor", new Vector3(2.29f, -.012f, .70f), new Vector3(1.4f, .045f, 1.36f), teal, decor);
        VisualBox("B basin floor", new Vector3(2.29f, -.012f, 2.30f), new Vector3(1.4f, .045f, 1.36f), brass, decor);
        VisualBox("Waste basin floor", new Vector3(.70f, -.01f, 1.5f), new Vector3(1.34f, .04f, 2.95f), dark, decor);
        // Solid rear boundary and low visible rim indicate the simulation tank without hiding its contents.
        VisualBox("Tank rear wall", new Vector3(-.035f, 1.45f, 1.5f), new Vector3(.07f, 2.90f, 3.1f), steel, decor);
        VisualBox("Tank far rim", new Vector3(1.5f, .25f, -.035f), new Vector3(3.1f, .50f, .07f), steel, decor);
        VisualBox("Tank near rim", new Vector3(1.5f, .075f, 3.035f), new Vector3(3.1f, .15f, .07f), steel, decor);
        VisualBox("Tank right rim", new Vector3(3.035f, .075f, 1.5f), new Vector3(.07f, .15f, 3.1f), steel, decor);
        foreach (float z in new[] { .67f, 2.33f })
        {
            VisualBox("Reservoir corner post", new Vector3(-.005f, 2.18f, z), new Vector3(.06f, 1.57f, .06f), trim, decor);
            VisualBox("Gate guide", new Vector3(1.17f, 2.20f, z), new Vector3(.08f, 1.66f, .075f), trim, decor);
            VisualBox("Reservoir upper rail", new Vector3(.58f, 2.94f, z), new Vector3(1.30f, .055f, .06f), trim, decor);
        }
        for (int i = 0; i < 6; i++)
            VisualBox("Waste warning stripe", new Vector3(.18f + i * .20f, .018f, 2.72f), new Vector3(.095f, .008f, .25f), coral, decor);
        for (int i = 0; i < 7; i++)
        {
            float x = 1.75f + i * .16f;
            VisualBox("A measuring tick", new Vector3(x, .019f, .14f), new Vector3(.025f, .006f, .12f), trim, decor);
            VisualBox("B measuring tick", new Vector3(x, .019f, 2.86f), new Vector3(.025f, .006f, .12f), trim, decor);
        }
        VisualBox("Shutter housing", new Vector3(1.17f, 3.015f, 1.5f), new Vector3(.27f, .18f, 1.87f), steel, decor);
        VisualBox("Studio floor", new Vector3(1.5f, -.37f, 1.5f), new Vector3(200, .12f, 200), ground, decor);

        var cameraGO = new GameObject("Main Camera");
        cameraGO.tag = "MainCamera";
        var camera = cameraGO.AddComponent<Camera>();
        cameraGO.AddComponent<AudioListener>();
        camera.fieldOfView = 45;
        camera.nearClipPlane = .10f;
        camera.farClipPlane = 80;
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = Hex("0B1820");
        camera.allowHDR = true;
        camera.transform.position = LiquidRelayGame.SimToWorld(new Vector3(5.46f, 4.53f, 6.23f));
        camera.transform.rotation = Quaternion.LookRotation(LiquidRelayGame.SimToWorld(new Vector3(1.5f, 1.12f, 1.5f)) - camera.transform.position, Vector3.up);
        var cameraData = cameraGO.AddComponent<UniversalAdditionalCameraData>();
        cameraData.requiresColorTexture = true;
        cameraData.requiresDepthTexture = true;
        cameraData.renderPostProcessing = true;
        fluid.targetCamera = camera;

        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.skybox = null;
        RenderSettings.reflectionIntensity = .08f;
        RenderSettings.ambientLight = new Color(.19f, .25f, .30f);
        RenderSettings.fog = false;
        MakeLight("Key Light", new Vector3(46, -30, 0), new Color(.94f, .98f, 1), 1.15f, LightShadows.Soft);
        MakeLight("Warm Rim", new Vector3(25, 140, 0), new Color(1, .79f, .50f), .65f, LightShadows.None);
        MakeLight("Cool Fill", new Vector3(55, 65, 0), new Color(.40f, .72f, 1), .35f, LightShadows.None);

        var game = fluidGO.AddComponent<LiquidRelayGame>();
        game.provider = provider;
        game.fluidRenderer = fluid;
        game.solverScene = provider.sceneAsset;
        game.gate = gate; game.deflector = paddle;
        gateVisual.game = game;
        game.levels = LiquidRelayGame.DefaultLevels();
        // Presentation is independently authored and attached without compile-time coupling.
        Type presentationType = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("LiquidRelayPresentation")).FirstOrDefault(t => t != null);
        if (presentationType != null)
        {
            Component presentation = fluidGO.AddComponent(presentationType);
            presentationType.GetField("game")?.SetValue(presentation, game);
            presentationType.GetField("fluidRenderer")?.SetValue(presentation, fluid);
        }
        Type audioType = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType("LiquidRelayAudio")).FirstOrDefault(t => t != null);
        if (audioType != null)
        {
            Component audio = fluidGO.AddComponent(audioType);
            audioType.GetField("game")?.SetValue(audio, game);
        }
        EditorSceneManager.SaveScene(scene, ScenePath);
        AssetDatabase.SaveAssets();
        Debug.Log($"Liquid Relay: created {ScenePath}; {props.Count + 1} DFSPH bodies, fixed camera, three puzzles. Original scenes remain untouched.");
        Selection.activeGameObject = apparatus;
    }

    static void WriteStats()
    {
        var stats = JObject.Parse(File.ReadAllText(ModelStatsPath));
        stats["target"] = new JArray(1.5, 1.0, 1.5);
        stats["source"] = "Liquid Relay: selected epoch 96 fine-tune; original frozen normalization";
        stats["fps"] = 25; stats["frameCount"] = 1; stats["factor"] = 1;
        stats["height"] = 512; stats["width"] = 512;
        stats["renderer"] = "v2"; stats["coarseRadius"] = .035;
        stats["thicknessScale"] = 3.9448387491412222;
        stats["minRadiusPx"] = 1; stats["maxRadiusPx"] = 24;
        File.WriteAllText(Root + "/relay_model_meta.json", stats.ToString());
        AssetDatabase.ImportAsset(Root + "/relay_model_meta.json", ImportAssetOptions.ForceSynchronousImport);
    }

    static Transform BoxBody(string name, Vector3 center, Vector3 scale, Quaternion rotation, Material mat, Transform parent, List<LiveSphProvider.Obstacle> list)
    {
        var t = VisualBox(name, center, scale, mat, parent);
        t.rotation = rotation;
        list.Add(new LiveSphProvider.Obstacle { transform = t, shape = LiveSphProvider.Obstacle.Shape.Box, motion = LiveSphProvider.Obstacle.Motion.None });
        return t;
    }

    static Transform VisualBox(string name, Vector3 center, Vector3 scale, Material mat, Transform parent)
    {
        GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = name; go.transform.SetParent(parent, false);
        go.transform.position = LiquidRelayGame.SimToWorld(center);
        go.transform.localScale = scale;
        go.GetComponent<Renderer>().sharedMaterial = mat;
        UnityEngine.Object.DestroyImmediate(go.GetComponent<Collider>());
        return go.transform;
    }

    static void MakeLight(string name, Vector3 euler, Color color, float intensity, LightShadows shadows)
    {
        var go = new GameObject(name);
        go.transform.rotation = Quaternion.Euler(euler);
        var light = go.AddComponent<Light>();
        light.type = LightType.Directional; light.color = color; light.intensity = intensity; light.shadows = shadows;
        light.shadowBias = .025f; light.shadowNormalBias = .15f;
        go.AddComponent<UniversalAdditionalLightData>();
    }

    static Material MaterialAsset(string name, Color color, float metal, float smooth)
    {
        string path = Root + "/Materials/" + name.Replace(" ", "_") + ".mat";
        var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (mat == null)
        {
            mat = new Material(Shader.Find("Universal Render Pipeline/Lit")) { name = name };
            AssetDatabase.CreateAsset(mat, path);
        }
        mat.SetColor("_BaseColor", color); mat.SetFloat("_Metallic", metal); mat.SetFloat("_Smoothness", smooth);
        EditorUtility.SetDirty(mat);
        return mat;
    }

    static Color Hex(string value) { ColorUtility.TryParseHtmlString("#" + value, out var c); return c; }
    static void EnsureFolder(string path) { if (!AssetDatabase.IsValidFolder(path)) { string parent = Path.GetDirectoryName(path).Replace('\\', '/'); EnsureFolder(parent); AssetDatabase.CreateFolder(parent, Path.GetFileName(path)); } }
}
