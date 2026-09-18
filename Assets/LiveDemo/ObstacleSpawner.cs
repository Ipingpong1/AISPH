// ObstacleSpawner.cs — add rigid, unmoving obstacles to the live GPU-PBF scene at runtime.
// N = box, M = sphere at the mouse ray's hit on the ground plane (y = 0); Backspace removes the
// last one. Each spawn is a visible Unity primitive AND an entry in the provider's obstacle array
// (motion None), which GpuSphProvider re-uploads to the solver every step — so the fluid collides
// with exactly what the audience sees. Hard cap = GpuSphSolver.MaxObstacles (8) incl. scene props.
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class ObstacleSpawner : MonoBehaviour
{
    public GpuSphProvider provider;
    [Tooltip("Camera for the mouse ray. Empty = Camera.main.")]
    public Camera cam;
    [Tooltip("Optional material for spawned props (empty = default primitive material).")]
    public Material propMaterial;
    public float boxSize = 0.5f;
    public float sphereDiameter = 0.5f;
    public Key boxKey = Key.N, sphereKey = Key.M, removeKey = Key.Backspace;

    [Tooltip("Draw the key legend under FluidSceneMVP's status line.")]
    public bool showHelp = true;

    readonly List<GameObject> spawned = new List<GameObject>();
    public int SpawnedCount => spawned.Count;

    void Start()
    {
        if (provider == null) provider = GetComponent<GpuSphProvider>();
        if (cam == null) cam = Camera.main;
    }

    void Update()
    {
        var kb = Keyboard.current;
        if (kb == null || provider == null || cam == null) return;
        if (kb[boxKey].wasPressedThisFrame) SpawnAtMouse(LiveSphProvider.Obstacle.Shape.Box);
        if (kb[sphereKey].wasPressedThisFrame) SpawnAtMouse(LiveSphProvider.Obstacle.Shape.Sphere);
        if (kb[removeKey].wasPressedThisFrame) RemoveLast();
    }

    bool MouseOnGround(out Vector3 p)
    {
        p = Vector3.zero;
        var ms = Mouse.current;
        if (ms == null) return false;
        Ray r = cam.ScreenPointToRay(ms.position.ReadValue());
        var ground = new Plane(Vector3.up, Vector3.zero);
        if (!ground.Raycast(r, out float t)) return false;
        p = r.GetPoint(t);
        return true;
    }

    public void SpawnAtMouse(LiveSphProvider.Obstacle.Shape shape)
    {
        if (MouseOnGround(out Vector3 p)) SpawnAt(shape, p);
    }

    // Also callable from scripts / the MCP bridge for deterministic tests.
    public GameObject SpawnAt(LiveSphProvider.Obstacle.Shape shape, Vector3 groundPoint)
    {
        int n = provider.obstacles == null ? 0 : provider.obstacles.Length;
        if (n >= GpuSphSolver.MaxObstacles)
        {
            Debug.LogWarning($"ObstacleSpawner: obstacle cap {GpuSphSolver.MaxObstacles} reached (incl. scene props)");
            return null;
        }
        bool box = shape == LiveSphProvider.Obstacle.Shape.Box;
        var go = GameObject.CreatePrimitive(box ? PrimitiveType.Cube : PrimitiveType.Sphere);
        float s = box ? boxSize : sphereDiameter;
        go.name = $"Spawned{shape}{spawned.Count}";
        go.transform.localScale = Vector3.one * s;
        go.transform.position = new Vector3(groundPoint.x, s * 0.5f, groundPoint.z);   // resting on the floor
        if (propMaterial != null) go.GetComponent<Renderer>().sharedMaterial = propMaterial;

        var o = new LiveSphProvider.Obstacle();
        o.transform = go.transform;
        o.shape = shape;
        o.motion = LiveSphProvider.Obstacle.Motion.None;
        o.orbit = false;
        var arr = provider.obstacles ?? new LiveSphProvider.Obstacle[0];
        Array.Resize(ref arr, n + 1);
        arr[n] = o;
        provider.obstacles = arr;
        spawned.Add(go);
        return go;
    }

    void OnGUI()
    {
        if (!showHelp) return;
        GUI.Label(new Rect(10, 30, 1800, 24),
            "LMB drag: stir sphere   scroll: stirrer height   N / M: box / sphere at mouse   Backspace: remove last   "
            + "F: drop block   R: reset   B: bilateral   V: raw input   P: pause   WASD + RMB: fly");
    }

    public void RemoveLast()
    {
        if (spawned.Count == 0) return;
        var go = spawned[spawned.Count - 1];
        spawned.RemoveAt(spawned.Count - 1);
        if (provider.obstacles != null)
        {
            var list = new List<LiveSphProvider.Obstacle>(provider.obstacles);
            list.RemoveAll(o => o.transform == go.transform);
            provider.obstacles = list.ToArray();   // GpuSphProvider clears + re-uploads every step
        }
        Destroy(go);
    }
}
