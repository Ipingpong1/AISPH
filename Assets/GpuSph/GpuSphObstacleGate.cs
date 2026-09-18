using System.Text;
using UnityEngine;

// GpuSphObstacleGate.cs — known-answer gate for obstacle projection (GpuSph_VALIDATION.md §4).
//
// Project discipline (RESTART.md #3): a new tool gets a known-answer case BEFORE it produces
// data anyone reasons about. The torus primitive added 2026-08-20 is that new tool, and there is
// no oracle above it — LiveSphProvider IS the oracle — so the gate checks the GEOMETRIC contract
// directly and then checks that the GPU agrees with the oracle on the same points:
//
//   1. ORACLE CONTRACT — for each shape: a point inside is moved exactly onto the surface, the
//      projection is idempotent, and a point outside is returned bit-identical (no branch taken).
//   2. GPU PARITY — GpuSph.compute's ProjectOutOfObstacles agrees with
//      LiveSphProvider.ProjectOutOfPrimitive on the same point set. §3's numeric parity run was
//      done with NO obstacles, so this is the obstacle half of that check.
//
// Needs no fluid, no props and no play-mode solver. The ComputeShader is Instantiate()d so the
// gate's buffer bindings cannot disturb a live solver sharing the same asset.
//
// Run: put on any GameObject with the GpuSph.compute reference assigned, press Play (runOnStart),
// or use the "Run Obstacle Gate" context menu on the component. Results go to the Console.
public class GpuSphObstacleGate : MonoBehaviour
{
    [Tooltip("GpuSph.compute — the same asset the solver uses; it is instantiated, not bound.")]
    public ComputeShader shader;
    public bool runOnStart = true;
    [Tooltip("Points per shape. They are seeded deterministically, so a failure reproduces.")]
    public int pointsPerShape = 4096;
    [Tooltip("Surface/idempotence tolerance in LOCAL units (primitives are ~0.5 across).")]
    public float tolerance = 1e-4f;
    [Tooltip("Max allowed C#-vs-GPU disagreement in SIM units.")]
    public float parityTolerance = 1e-4f;

    // Mirrors ObstacleGpu in GpuSph.compute (and the private copy in GpuSphSolver):
    // 8 float4 + int + int + float2 pad = 144 bytes, transforms as EXPLICIT ROWS. Staged here
    // exactly as GpuSphSolver.SetObstacle stages it — if the two ever diverge, the gate is
    // testing something the solver does not do, which is worse than no gate.
    struct ObstacleGpu
    {
        public Vector4 t0, t1, t2, t3, f0, f1, f2, f3;
        public int shape, active;
        public Vector2 pad;
    }

    void Start() { if (runOnStart) Run(); }

    // Returns the report so it can also be driven headlessly (MCP execute_code) without
    // scraping the Console. Runs in EDIT mode — no Play, no fluid, no solver needed.
    [ContextMenu("Run Obstacle Gate")]
    public string Run()
    {
        var log = new StringBuilder("[GpuSphObstacleGate]\n");
        bool ok = RunOracleContract(log);
        ok &= RunGpuParity(log);
        log.Append(ok ? "RESULT: PASS" : "RESULT: FAIL");
        string report = log.ToString();
        if (ok) Debug.Log(report); else Debug.LogError(report);
        return report;
    }

    // ---------- shared helpers ----------

    // Deterministic obstacle set: one of each shape, none axis-aligned, none unit-scaled, placed
    // apart so the per-shape contract test sees exactly one primitive at a time.
    static LiveSphProvider.Obstacle MakeObstacle(LiveSphProvider.Obstacle.Shape shape)
    {
        Matrix4x4 trs;
        switch (shape)
        {
            case LiveSphProvider.Obstacle.Shape.Sphere:
                trs = Matrix4x4.TRS(new Vector3(0.7f, 0.5f, 0.6f),
                                    Quaternion.Euler(0f, 35f, 0f), new Vector3(0.6f, 0.6f, 0.6f));
                break;
            case LiveSphProvider.Obstacle.Shape.Box:
                trs = Matrix4x4.TRS(new Vector3(1.8f, 0.55f, 1.1f),
                                    Quaternion.Euler(0f, 27f, 0f), new Vector3(0.5f, 0.8f, 0.35f));
                break;
            default:
                trs = Matrix4x4.TRS(new Vector3(1.2f, 0.9f, 2.1f),
                                    Quaternion.Euler(15f, 40f, 0f), new Vector3(0.9f, 0.9f, 0.9f));
                break;
        }
        var o = new LiveSphProvider.Obstacle { shape = shape, active = true };
        o.toLocal = trs.inverse;
        o.fromLocal = trs;
        return o;
    }

    // Distance from the primitive's SURFACE, in local space. 0 on the surface, <0 inside.
    static float LocalSurfaceDistance(LiveSphProvider.Obstacle.Shape shape, Vector3 lp)
    {
        switch (shape)
        {
            case LiveSphProvider.Obstacle.Shape.Sphere:
                return lp.magnitude - 0.5f;
            case LiveSphProvider.Obstacle.Shape.Box:
                // the projection exits through the NEAREST FACE, so a projected point sits at
                // max(|component|) == 0.5
                return Mathf.Max(Mathf.Abs(lp.x), Mathf.Max(Mathf.Abs(lp.y), Mathf.Abs(lp.z))) - 0.5f;
            default:
                float rad = new Vector2(lp.x, lp.z).magnitude;
                return new Vector2(rad - LiveSphProvider.TorusMajor, lp.y).magnitude
                       - LiveSphProvider.TorusTube;
        }
    }

    // Deterministic point cloud around an obstacle: a cube of samples centred on it, so a good
    // fraction land inside. Seeded per shape — a failure is reproducible.
    Vector3[] SamplePoints(LiveSphProvider.Obstacle o, int seed, int count)
    {
        var rng = new System.Random(seed);
        Vector3 centre = o.fromLocal.MultiplyPoint3x4(Vector3.zero);
        var pts = new Vector3[count];
        for (int i = 0; i < count; i++)
            pts[i] = centre + new Vector3((float)rng.NextDouble() - 0.5f,
                                          (float)rng.NextDouble() - 0.5f,
                                          (float)rng.NextDouble() - 0.5f) * 1.6f;
        return pts;
    }

    // ---------- 1. oracle contract ----------

    bool RunOracleContract(StringBuilder log)
    {
        bool all = true;
        foreach (LiveSphProvider.Obstacle.Shape shape in
                 System.Enum.GetValues(typeof(LiveSphProvider.Obstacle.Shape)))
        {
            var o = MakeObstacle(shape);
            var pts = SamplePoints(o, 1000 + (int)shape, pointsPerShape);

            int inside = 0, badSurface = 0, badIdem = 0, badOutside = 0;
            float worstSurface = 0f, worstIdem = 0f;
            foreach (var p in pts)
            {
                Vector3 lp = o.toLocal.MultiplyPoint3x4(p);
                bool wasInside = LocalSurfaceDistance(shape, lp) < 0f;
                Vector3 q = LiveSphProvider.ProjectOutOfPrimitive(o, p);

                if (!wasInside)
                {
                    // no branch should have been taken at all
                    if (q != p) badOutside++;
                    continue;
                }
                inside++;
                float d = Mathf.Abs(LocalSurfaceDistance(shape, o.toLocal.MultiplyPoint3x4(q)));
                worstSurface = Mathf.Max(worstSurface, d);
                if (d > tolerance) badSurface++;

                float idem = Vector3.Distance(q, LiveSphProvider.ProjectOutOfPrimitive(o, q));
                worstIdem = Mathf.Max(worstIdem, idem);
                if (idem > tolerance) badIdem++;
            }

            // degenerate cases: the exact singular point of each primitive must still resolve
            Vector3 singular = shape == LiveSphProvider.Obstacle.Shape.Torus
                ? o.fromLocal.MultiplyPoint3x4(new Vector3(LiveSphProvider.TorusMajor, 0f, 0f))
                : o.fromLocal.MultiplyPoint3x4(Vector3.zero);
            Vector3 sq = LiveSphProvider.ProjectOutOfPrimitive(o, singular);
            float sd = Mathf.Abs(LocalSurfaceDistance(shape, o.toLocal.MultiplyPoint3x4(sq)));
            bool singularOk = sd <= tolerance && !float.IsNaN(sq.x);

            bool ok = badSurface == 0 && badIdem == 0 && badOutside == 0 && singularOk && inside > 100;
            all &= ok;
            log.AppendFormat(
                "  {0,-6} contract {1}: inside {2}/{3}, surface worst {4:E2} ({5} bad), " +
                "idempotence worst {6:E2} ({7} bad), outside moved {8}, singular {9}\n",
                shape, ok ? "PASS" : "FAIL", inside, pts.Length, worstSurface, badSurface,
                worstIdem, badIdem, badOutside, singularOk ? "ok" : "BAD");
        }
        return all;
    }

    // ---------- 2. GPU parity ----------

    bool RunGpuParity(StringBuilder log)
    {
        if (shader == null) { log.Append("  GPU parity SKIPPED: no shader assigned\n"); return false; }

        // all three shapes live at once — this also exercises the multi-obstacle loop order
        var obs = new[]
        {
            MakeObstacle(LiveSphProvider.Obstacle.Shape.Sphere),
            MakeObstacle(LiveSphProvider.Obstacle.Shape.Box),
            MakeObstacle(LiveSphProvider.Obstacle.Shape.Torus),
        };

        var pts = new System.Collections.Generic.List<Vector3>();
        for (int i = 0; i < obs.Length; i++) pts.AddRange(SamplePoints(obs[i], 2000 + i, pointsPerShape));
        Vector3[] input = pts.ToArray();

        var expect = new Vector3[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            Vector3 p = input[i];
            foreach (var o in obs) p = LiveSphProvider.ProjectOutOfPrimitive(o, p);
            expect[i] = p;
        }

        var cs = Instantiate(shader);          // isolate bindings from any live solver
        var stage = new ObstacleGpu[GpuSphSolver.MaxObstacles];
        for (int i = 0; i < obs.Length; i++)
            stage[i] = new ObstacleGpu
            {
                t0 = obs[i].toLocal.GetRow(0), t1 = obs[i].toLocal.GetRow(1),
                t2 = obs[i].toLocal.GetRow(2), t3 = obs[i].toLocal.GetRow(3),
                f0 = obs[i].fromLocal.GetRow(0), f1 = obs[i].fromLocal.GetRow(1),
                f2 = obs[i].fromLocal.GetRow(2), f3 = obs[i].fromLocal.GetRow(3),
                shape = (int)obs[i].shape,
                active = 1,
            };

        var bObs = new GraphicsBuffer(GraphicsBuffer.Target.Structured, GpuSphSolver.MaxObstacles, 144);
        var bIn = new GraphicsBuffer(GraphicsBuffer.Target.Structured, input.Length, 12);
        var bOut = new GraphicsBuffer(GraphicsBuffer.Target.Structured, input.Length, 12);
        var got = new Vector3[input.Length];
        float worst = 0f;
        int worstIdx = -1;
        try
        {
            bObs.SetData(stage);
            bIn.SetData(input);
            int k = cs.FindKernel("ObstacleProjectTest");
            cs.SetInt("_NumObstacles", obs.Length);
            cs.SetInt("_N", input.Length);
            cs.SetBuffer(k, "_Obstacles", bObs);
            cs.SetBuffer(k, "_TestIn", bIn);
            cs.SetBuffer(k, "_TestOut", bOut);
            cs.Dispatch(k, (input.Length + 255) / 256, 1, 1);
            bOut.GetData(got);
        }
        finally
        {
            bObs.Dispose(); bIn.Dispose(); bOut.Dispose();
            if (Application.isPlaying) Destroy(cs); else DestroyImmediate(cs);
        }

        int nan = 0;
        for (int i = 0; i < got.Length; i++)
        {
            if (float.IsNaN(got[i].x) || float.IsNaN(got[i].y) || float.IsNaN(got[i].z)) { nan++; continue; }
            float d = Vector3.Distance(got[i], expect[i]);
            if (d > worst) { worst = d; worstIdx = i; }
        }
        bool ok = nan == 0 && worst <= parityTolerance;
        log.AppendFormat("  GPU parity {0}: {1} points, worst |C#-GPU| {2:E2} (tol {3:E2}){4}{5}\n",
                         ok ? "PASS" : "FAIL", got.Length, worst, parityTolerance,
                         nan > 0 ? $", {nan} NaN" : "",
                         worstIdx >= 0 ? $", worst at input {input[worstIdx]}" : "");
        return ok;
    }
}
