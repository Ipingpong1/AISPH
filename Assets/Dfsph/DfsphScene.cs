// DfsphScene.cs — DFSPH1001: a SPlisHSPlasH scene file (the corpus format: Configuration + FluidBlocks + RigidBodies +
// Materials) parsed into a DfsphSolver configuration. Only what the DFSPH corpus uses is read; anything that would
// change the physics and is NOT reproduced (another simulation method / boundary method / non-pressure force, a mesh
// that is not one of the analytic primitives, a dynamic body, particle-file fluid models) is reported in `unsupported`
// instead of being silently ignored.
//   UnitBox.obj -> Box (edge 1), sphere.obj -> Sphere (radius 1), torus.obj -> Torus (R 1, r 0.5, axis y): the stock
//   meshes of the corpus, scaled by `scale` (uniform scale required for sphere / torus).
using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using UnityEngine;

[Serializable]
public class DfsphScene
{
    [Serializable]
    public class FluidBlock
    {
        public Vector3 start, end;
        public Vector3 scale = Vector3.one, translation;
        public int denseMode;
        public Vector3 initialVelocity;
    }

    public float particleRadius = 0.025f;
    public Vector3 gravity = new Vector3(0f, -9.81000042f, 0f);
    public float timeStepSize = 0.001f;
    public float cflFactor = 0.5f, cflMinTimeStepSize = 9.99999975e-05f, cflMaxTimeStepSize = 0.00499999989f;
    public int cflMethod = 1;
    public float maxError = 0.00999999978f, maxErrorV = 0.100000001f;
    public int minIterations = 2, maxIterations = 100, maxIterationsV = 100;
    public bool enableDivergenceSolver = true;
    public float density0 = 1000f, viscosity = 0.00999999978f, viscosityBoundary;
    public float dataExportFPS = 25f, stopAt = -1f;
    public List<FluidBlock> fluidBlocks = new List<FluidBlock>();
    public List<DfsphSolver.Body> bodies = new List<DfsphSolver.Body>();
    public List<string> unsupported = new List<string>();
    public string source;

    static Vector3 V3(JToken t, Vector3 d)
    {
        if (t == null || t.Type != JTokenType.Array || ((JArray)t).Count < 3) return d;
        return new Vector3((float)(double)t[0], (float)(double)t[1], (float)(double)t[2]);
    }
    static float F(JToken t, float d) => t == null ? d : (float)(double)t;
    static int I(JToken t, int d) => t == null ? d : (int)t;
    static bool B(JToken t, bool d) => t == null ? d : (t.Type == JTokenType.Boolean ? (bool)t : (int)t != 0);

    public static DfsphScene FromFile(string path) { var s = Parse(File.ReadAllText(path)); s.source = path; return s; }

    public static DfsphScene Parse(string json)
    {
        var root = JObject.Parse(json);
        var s = new DfsphScene();
        var c = root["Configuration"] as JObject ?? new JObject();
        s.particleRadius = F(c["particleRadius"], s.particleRadius);
        s.gravity = V3(c["gravitation"], s.gravity);
        s.timeStepSize = F(c["timeStepSize"], s.timeStepSize);
        s.cflMethod = I(c["cflMethod"], s.cflMethod);
        s.cflFactor = F(c["cflFactor"], s.cflFactor);
        s.cflMinTimeStepSize = F(c["cflMinTimeStepSize"], s.cflMinTimeStepSize);
        s.cflMaxTimeStepSize = F(c["cflMaxTimeStepSize"], s.cflMaxTimeStepSize);
        s.dataExportFPS = F(c["dataExportFPS"], s.dataExportFPS);
        s.stopAt = F(c["stopAt"], s.stopAt);
        int method = I(c["simulationMethod"], 4);
        if (method != 4) s.unsupported.Add($"simulationMethod {method} (only 4 = DFSPH)");
        int bh = I(c["boundaryHandlingMethod"], 2);
        if (bh != 2) s.unsupported.Add($"boundaryHandlingMethod {bh} (only 2 = Bender 2019 volume maps)");
        if (s.cflMethod != 1) s.unsupported.Add($"cflMethod {s.cflMethod} (only 1)");
        if (I(c["kernel"], 4) != 4 || I(c["gradKernel"], 4) != 4) s.unsupported.Add("kernel / gradKernel != 4 (precomputed cubic)");
        var d = c["DFSPH"] as JObject ?? c;   // namespaced (c5a063d) or flat (older files)
        s.maxError = F(d["maxError"], s.maxError);
        s.maxErrorV = F(d["maxErrorV"], s.maxErrorV);
        s.minIterations = I(d["minIterations"], s.minIterations);
        s.maxIterations = I(d["maxIterations"], s.maxIterations);
        s.maxIterationsV = I(d["maxIterationsV"], s.maxIterationsV);
        s.enableDivergenceSolver = B(d["enableDivergenceSolver"], s.enableDivergenceSolver);

        if (root["Materials"] is JArray mats && mats.Count > 0)
        {
            var m = (JObject)mats[0];
            s.density0 = F(m["density0"], s.density0);
            int vm = I(m["viscosityMethod"], 0);
            var sv = m["Standard viscosity"] as JObject ?? m;
            if (vm == 1) { s.viscosity = F(sv["viscosity"], s.viscosity); s.viscosityBoundary = F(sv["viscosityBoundary"], 0f); }
            else { s.viscosity = 0f; if (vm != 0) s.unsupported.Add($"viscosityMethod {vm} (only 1 = standard)"); }
            if (s.viscosityBoundary != 0f) s.unsupported.Add("viscosityBoundary != 0");
            foreach (var key in new[] { "surfaceTensionMethod", "vorticityMethod", "dragMethod", "elasticityMethod" })
                if (I(m[key], 0) != 0) s.unsupported.Add($"{key} {I(m[key], 0)}");
            var xs = m["XSPH"] as JObject ?? m;
            if (F(xs["xsph"], 0f) != 0f) s.unsupported.Add("XSPH != 0");
            if (mats.Count > 1) s.unsupported.Add("more than one material (single fluid only)");
        }
        else s.viscosity = 0f;   // no material: SPlisHSPlasH defaults to viscosityMethod 0

        if (root["FluidBlocks"] is JArray fbs)
            foreach (JObject fb in fbs)
            {
                s.fluidBlocks.Add(new FluidBlock
                {
                    start = V3(fb["start"], Vector3.zero), end = V3(fb["end"], Vector3.one),
                    scale = V3(fb["scale"], Vector3.one), translation = V3(fb["translation"], Vector3.zero),
                    denseMode = I(fb["denseMode"], 0), initialVelocity = V3(fb["initialVelocity"], Vector3.zero),
                });
                if (V3(fb["initialAngularVelocity"], Vector3.zero) != Vector3.zero) s.unsupported.Add("FluidBlock initialAngularVelocity");
            }
        if (root["FluidModels"] is JArray fms && fms.Count > 0) s.unsupported.Add("FluidModels (particle files)");
        if (root["Emitters"] is JArray ems && ems.Count > 0) s.unsupported.Add("Emitters");
        if (root["AnimationFields"] is JArray afs && afs.Count > 0) s.unsupported.Add("AnimationFields");

        if (root["RigidBodies"] is JArray rbs)
            foreach (JObject rb in rbs)
            {
                string geo = Path.GetFileName((string)rb["geometryFile"] ?? "");
                DfsphSolver.Shape shape;
                if (geo.Equals("UnitBox.obj", StringComparison.OrdinalIgnoreCase)) shape = DfsphSolver.Shape.Box;
                else if (geo.Equals("sphere.obj", StringComparison.OrdinalIgnoreCase)) shape = DfsphSolver.Shape.Sphere;
                else if (geo.Equals("torus.obj", StringComparison.OrdinalIgnoreCase)) shape = DfsphSolver.Shape.Torus;
                else { s.unsupported.Add($"RigidBody mesh '{geo}' is not an analytic primitive (skipped)"); continue; }
                var b = new DfsphSolver.Body
                {
                    shape = shape,
                    scale = V3(rb["scale"], Vector3.one),
                    translation = V3(rb["translation"], Vector3.zero),
                    rotationAxis = V3(rb["rotationAxis"], Vector3.right),
                    rotationAngle = F(rb["rotationAngle"], 0f),
                    mapInvert = B(rb["mapInvert"], false),
                    mapThickness = F(rb["mapThickness"], 0f),
                };
                var res = rb["mapResolution"] as JArray;
                b.mapResolution = res != null && res.Count >= 3 ? new Vector3Int((int)res[0], (int)res[1], (int)res[2]) : new Vector3Int(20, 20, 20);
                if (shape != DfsphSolver.Shape.Box && (Mathf.Abs(b.scale.x - b.scale.y) > 1e-6f || Mathf.Abs(b.scale.x - b.scale.z) > 1e-6f))
                    s.unsupported.Add($"non-uniform scale on {geo} (the analytic SDF uses scale.x)");
                if (B(rb["isDynamic"], false)) s.unsupported.Add($"dynamic rigid body {geo} (simulated as static)");
                s.bodies.Add(b);
            }
        return s;
    }

    /// <summary>true (corpus parity): sphere / torus bodies use the tessellated stock mesh for field 0, as SPlisHSPlasH did;
    /// false: the analytic primitive (up to ~7 mm larger than sphere.obj / torus.obj). Boxes are identical either way.</summary>
    public void SetTessellated(bool on)
    {
        foreach (var b in bodies) b.tessellated = on && b.shape != DfsphSolver.Shape.Box;
    }

    /// <summary>Copy the configuration into a solver (before Init).</summary>
    public void Configure(DfsphSolver s)
    {
        s.particleRadius = particleRadius;
        s.density0 = density0;
        s.viscosity = viscosity;
        s.gravity = gravity;
        s.maxError = maxError; s.maxErrorV = maxErrorV;
        s.minIterations = minIterations; s.maxIterations = maxIterations; s.maxIterationsV = maxIterationsV;
        s.enableDivergenceSolver = enableDivergenceSolver;
        s.cflFactor = cflFactor; s.cflMinTimeStepSize = cflMinTimeStepSize; s.cflMaxTimeStepSize = cflMaxTimeStepSize;
        s.initialTimeStep = timeStepSize;
    }

    /// <summary>Every fluid block sampled exactly as SimulatorBase::createFluidBlocks (positions + initial velocities).</summary>
    public void SampleFluid(List<Vector3> pos, List<Vector3> vel)
    {
        foreach (var fb in fluidBlocks)
        {
            var p = DfsphSolver.SampleFluidBlock(fb.start, fb.end, fb.scale, fb.translation, fb.denseMode, particleRadius);
            pos.AddRange(p);
            for (int i = 0; i < p.Count; i++) vel.Add(fb.initialVelocity);
        }
    }
}
