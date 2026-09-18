// LiveSphProvider.cs — a LIVE coarse SPH solver running inside Unity, feeding FluidSceneMVP
// through the ParticleFrameProvider seam. No baked data: the fluid is simulated in real time
// and neurally upsampled every frame.
//
// Method: Position-Based Fluids (Macklin & Müller 2013) with a cubic-spline kernel — chosen
// for unconditional stability at coarse resolution and interactive timesteps. The exported
// per-particle density is the plain SPH kernel sum with EXACTLY the training-data convention
// (cubic spline, support h = 4·particleRadius, mass = 0.8·(2r)³·ρ0), so the model's density
// input channel matches the SPlisHSPlasH training distribution by construction (an isolated
// particle reads m·W(0) ≈ 255, matching the baked reference sims).
//
// Defaults replicate the validated true-coarse scenario (sim_lowres_real / lowres_probe2
// pair_0000_coarse): r = 0.0414, an 11³ dam block dropped in a [0,3]³ domain, 25 Hz.
//
// Scene coupling: obstacles are real scene Transforms (sphere or box primitives). Each
// substep, particles are projected out of their SDFs — a MOVING obstacle therefore imparts
// velocity through the projection itself (free two-way-ish coupling, fluid reacts to props
// being animated through it). Obstacles can optionally auto-orbit for demo purposes, advanced
// with the same dt as the solver so deterministic capture stays deterministic.
//
// The solver is driven by FluidSceneMVP via Tick(dt) (fixed-step during capture, frame dt
// otherwise) and substeps internally under a CFL cap. Runtime keys: F drops a fresh dam block
// (adds to the pool, up to maxParticles).

using System;
using UnityEngine;
using UnityEngine.InputSystem;

public class LiveSphProvider : ParticleFrameProvider
{
    // ---------- inspector ----------

    [Header("Fluid (defaults = validated coarse dam-break distribution)")]
    [Tooltip("Particle radius in sim units. 0.0414 matches the true-coarse reference sims the deployed models were validated on. Support radius = 4×this.")]
    public float particleRadius = 0.0414f;
    [Tooltip("Rest density. 1000 matches training data.")]
    public float restDensity = 1000f;
    [Tooltip("Hard cap on particle count (dam block is 1331; each extra F-drop adds another block).")]
    public int maxParticles = 4096;
    [Tooltip("Solver rate the sim is stepped at, Hz. 25 matches the training sims' export rate; substeps subdivide further under a CFL cap.")]
    public float simHz = 25f;

    [Header("Domain (sim space, y-up; [0,3]³ matches training dam breaks)")]
    public Vector3 domainMin = Vector3.zero;
    public Vector3 domainMax = new Vector3(3f, 3f, 3f);
    [Tooltip("Sim-space XZ offset that FluidSceneMVP subtracts when centerXZOnTarget is on — keep equal to the statsJson 'target' xz (1.5, 1.5 for v3 dam breaks) so obstacles line up.")]
    public Vector2 domainCenterXZ = new Vector2(1.5f, 1.5f);

    [Header("Initial dam block (sim space; default = the validated 11³ drop)")]
    public Vector3 blockMin = new Vector3(0.52f, 1.33f, 0.40f);
    public Vector3Int blockCount = new Vector3Int(11, 11, 11);
    [Tooltip("Re-drop a fresh block automatically every N seconds (0 = off). Also on the F key.")]
    public float autoDropInterval = 0f;

    [Header("PBF solver")]
    [Tooltip("Constraint iterations per substep.")]
    public int solverIters = 3;
    [Tooltip("CFL number: substep dt ≤ cfl·h/|v|max.")]
    public float cfl = 0.4f;
    [Tooltip("Max substeps per Tick (safety).")]
    public int maxSubsteps = 8;
    [Tooltip("Relaxation epsilon in the lambda denominator.")]
    public float lambdaEps = 100f;
    [Tooltip("Artificial-pressure (tensile) strength k; keeps splash from clumping.")]
    public float sCorrK = 0.001f;
    [Tooltip("XSPH viscosity coefficient.")]
    public float xsph = 0.05f;

    [Header("Scene obstacles (real props the fluid collides with)")]
    public Obstacle[] obstacles;

    [Serializable]
    public class Obstacle
    {
        // Enum ORDER IS THE WIRE FORMAT: the value is uploaded verbatim as ObstacleGpu.shape
        // (0 sphere / 1 box / 2 torus in GpuSph.compute). Append only; never reorder.
        public enum Shape { Sphere, Box, Torus }

        // Prop motion, all driven by SOLVER time (never Time.time) so captures stay deterministic.
        //   Orbit  — circle around the start position (the original demo stir)
        //   Sweep  — ping-pong along motionAxis, +-motionDistance
        //   Plunge — dip down motionDistance and return, once per motionPeriod
        public enum Motion { None, Orbit, Sweep, Plunge }

        [Tooltip("Scene object the fluid collides with. Sphere/Box/Torus SDF in the object's LOCAL space (unit primitives: radius/half-extent 0.5, torus outer extent 0.5), so transform scale shapes it.")]
        public Transform transform;
        public Shape shape = Shape.Sphere;
        [Tooltip("How this prop moves. None = follow the scene Transform (hand-animated or static). Legacy 'orbit' below still works when this is None.")]
        public Motion motion = Motion.None;
        [Tooltip("LEGACY (pre-Motion scenes): equivalent to Motion.Orbit. Only consulted when motion == None.")]
        public bool orbit = false;
        [Tooltip("Orbit radius in world units.")]
        public float orbitRadius = 0.8f;
        [Tooltip("Seconds per revolution (Orbit) / per full cycle (Sweep, Plunge).")]
        public float orbitPeriod = 6f;
        [Tooltip("Sweep direction in world space (normalized; ignored by Orbit/Plunge).")]
        public Vector3 motionAxis = Vector3.right;
        [Tooltip("Sweep amplitude / plunge depth, world units.")]
        public float motionDistance = 0.8f;
        [NonSerialized] public Vector3 startPos;
        [NonSerialized] public bool started;
        [NonSerialized] public Matrix4x4 toLocal, fromLocal;   // sim <-> obstacle local, per Tick
        [NonSerialized] public bool active;
    }

    // Torus geometry, LOCAL space: axis +Y, centre circle radius TorusMajor, tube TorusTube.
    // Major+Tube = 0.5 keeps the unit-primitive rule (transform scale s = world outer diameter s),
    // and the 2:1 major:minor ratio mirrors SPlisHSPlasH's torus.obj (major 1.0 / minor 0.5), so
    // this torus is proportioned like the ones in the DFSPH obstacle corpus. Note Major > Tube,
    // which is what keeps the HOLE empty: an on-axis point is |(0-Major, y)| >= Major > Tube away
    // from the tube, i.e. correctly classified outside (a sign-flipped SDF would fill the hole).
    // MUST stay in sync with TORUS_MAJOR / TORUS_TUBE in GpuSph.compute — this side is the oracle.
    public const float TorusMajor = 1f / 3f;
    public const float TorusTube = 1f / 6f;

    // World-space offset from an obstacle's start position at a given SOLVER time. Shared by
    // LiveSphProvider and GpuSphProvider so the two paths stir identically.
    public static Vector3 MotionOffset(Obstacle o, float solverTime)
    {
        Obstacle.Motion m = o.motion;
        if (m == Obstacle.Motion.None) m = o.orbit ? Obstacle.Motion.Orbit : Obstacle.Motion.None;
        if (m == Obstacle.Motion.None) return Vector3.zero;

        float a = 2f * Mathf.PI * solverTime / Mathf.Max(o.orbitPeriod, 0.1f);
        switch (m)
        {
            case Obstacle.Motion.Orbit:
                // starts AT startPos (cos 0 - 1 == 0), circles in the XZ plane
                return new Vector3(Mathf.Cos(a) - 1f, 0f, Mathf.Sin(a)) * o.orbitRadius;
            case Obstacle.Motion.Sweep:
                return o.motionAxis.normalized * (Mathf.Sin(a) * o.motionDistance);
            case Obstacle.Motion.Plunge:
                return Vector3.up * (-(0.5f - 0.5f * Mathf.Cos(a)) * o.motionDistance);
            default:
                return Vector3.zero;
        }
    }

    // Push a point out of one obstacle's primitive, in that obstacle's LOCAL space. This is the
    // PARITY ORACLE for ProjectOutOfObstacles in GpuSph.compute — edit the two together.
    public static Vector3 ProjectOutOfPrimitive(Obstacle o, Vector3 p)
    {
        Vector3 lp = o.toLocal.MultiplyPoint3x4(p);
        if (o.shape == Obstacle.Shape.Sphere)
        {
            float r = lp.magnitude;
            if (r < 0.5f)
            {
                lp = r > 1e-5f ? lp * (0.5f / r) : new Vector3(0f, 0.5f, 0f);
                p = o.fromLocal.MultiplyPoint3x4(lp);
            }
        }
        else if (o.shape == Obstacle.Shape.Box)
        {
            if (Mathf.Abs(lp.x) < 0.5f && Mathf.Abs(lp.y) < 0.5f && Mathf.Abs(lp.z) < 0.5f)
            {
                // exit through the nearest face
                float px = 0.5f - Mathf.Abs(lp.x), py = 0.5f - Mathf.Abs(lp.y), pz = 0.5f - Mathf.Abs(lp.z);
                if (px <= py && px <= pz) lp.x = Mathf.Sign(lp.x) * 0.5f;
                else if (py <= pz) lp.y = Mathf.Sign(lp.y) * 0.5f;
                else lp.z = Mathf.Sign(lp.z) * 0.5f;
                p = o.fromLocal.MultiplyPoint3x4(lp);
            }
        }
        else
        {
            // Torus: nearest point on the tube surface. In the (radial, y) half-plane the tube is
            // a circle of radius TorusTube centred at (TorusMajor, 0).
            float rad = new Vector2(lp.x, lp.z).magnitude;
            Vector2 d = new Vector2(rad - TorusMajor, lp.y);
            float dl = d.magnitude;
            if (dl < TorusTube)
            {
                // degenerate (exactly on the centre circle): push radially outward, matching HLSL
                d = dl > 1e-5f ? d * (TorusTube / dl) : new Vector2(TorusTube, 0f);
                float nr = TorusMajor + d.x;
                Vector2 dir = rad > 1e-5f ? new Vector2(lp.x, lp.z) / rad : new Vector2(1f, 0f);
                lp = new Vector3(dir.x * nr, d.y, dir.y * nr);
                p = o.fromLocal.MultiplyPoint3x4(lp);
            }
        }
        return p;
    }

    // ---------- state ----------

    float h, h2, mass, kernelSigma, w0;         // kernel constants
    int n;                                       // active particles
    Vector3[] pos, vel, pred, dp;
    float[] dens, lambda;
    float[] records;                             // 7-float export records
    bool recordsDirty = true;

    // spatial hash (rebuilt per substep)
    int tableSize;
    int[] head, next;

    // neighbor cache (per substep)
    int[] nbr; int[] nbrCount;
    const int MaxNbr = 96;

    float simTimeAcc, timeSinceDrop, solverTime;
    float lastStepMs;

    public float LastStepMs => lastStepMs;
    public int ActiveParticles => n;

    // ---------- ParticleFrameProvider ----------

    public override int FrameCount { get { EnsureInit(); return 1; } }
    public override float NativeFps => simHz;
    public override float ParticleRadius => particleRadius;

    public override void GetFrame(int idx, out float[] data, out int offset, out int count)
    {
        EnsureInit();
        if (recordsDirty)
        {
            for (int i = 0; i < n; i++)
            {
                int b = i * 7;
                records[b] = pos[i].x; records[b + 1] = pos[i].y; records[b + 2] = pos[i].z;
                records[b + 3] = vel[i].x; records[b + 4] = vel[i].y; records[b + 5] = vel[i].z;
                records[b + 6] = dens[i];
            }
            recordsDirty = false;
        }
        data = records; offset = 0; count = n;
    }

    public override void Tick(float dt)
    {
        EnsureInit();
        if (dt <= 0f) return;
        var kb = Keyboard.current;
        if (kb != null && kb.fKey.wasPressedThisFrame) SpawnBlock();
        if (autoDropInterval > 0f)
        {
            timeSinceDrop += dt;
            if (timeSinceDrop >= autoDropInterval) { SpawnBlock(); timeSinceDrop = 0f; }
        }

        simTimeAcc = Mathf.Min(simTimeAcc + dt, 3f / simHz);   // drop time if hopelessly behind
        float step = 1f / simHz;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        while (simTimeAcc >= step)
        {
            simTimeAcc -= step;
            solverTime += step;
            AdvanceObstacles(step);
            StepFrame(step);
            recordsDirty = true;
        }
        sw.Stop();
        lastStepMs = (float)sw.Elapsed.TotalMilliseconds;
    }

    // ---------- init ----------

    bool inited;

    void EnsureInit()
    {
        if (inited) return;
        inited = true;

        h = 4f * particleRadius;
        h2 = h * h;
        float d = 2f * particleRadius;
        mass = 0.8f * d * d * d * restDensity;               // SPlisHSPlasH convention
        kernelSigma = 8f / (Mathf.PI * h * h * h);
        w0 = kernelSigma;                                     // W(0), cubic spline

        pos = new Vector3[maxParticles];
        vel = new Vector3[maxParticles];
        pred = new Vector3[maxParticles];
        dp = new Vector3[maxParticles];
        dens = new float[maxParticles];
        lambda = new float[maxParticles];
        records = new float[maxParticles * 7];
        tableSize = Mathf.NextPowerOfTwo(maxParticles * 2);
        head = new int[tableSize];
        next = new int[maxParticles];
        nbr = new int[maxParticles * MaxNbr];
        nbrCount = new int[maxParticles];

        n = 0;
        SpawnBlock();
    }

    public override void ResetSim()
    {
        EnsureInit();
        n = 0;
        solverTime = 0f;
        timeSinceDrop = 0f;
        SpawnBlock();
    }

    void SpawnBlock()
    {
        float spacing = 2f * particleRadius;
        for (int ix = 0; ix < blockCount.x; ix++)
            for (int iy = 0; iy < blockCount.y; iy++)
                for (int iz = 0; iz < blockCount.z; iz++)
            {
                if (n >= maxParticles) return;
                pos[n] = blockMin + new Vector3(ix * spacing, iy * spacing, iz * spacing);
                vel[n] = Vector3.zero;
                dens[n] = restDensity * 0.8f;
                n++;
            }
        recordsDirty = true;
    }

    // ---------- kernel (cubic spline, matches training data convention) ----------

    float W(float r)
    {
        float q = r / h;
        if (q >= 1f) return 0f;
        if (q <= 0.5f) { float q2 = q * q; return kernelSigma * (6f * (q2 * q - q2) + 1f); }
        float t = 1f - q;
        return kernelSigma * 2f * t * t * t;
    }

    // dW/dr (scalar; gradient = this * dir/r)
    float dW(float r)
    {
        float q = r / h;
        if (q >= 1f || q < 1e-6f) return 0f;
        if (q <= 0.5f) return kernelSigma * (6f / h) * (3f * q * q - 2f * q);
        float t = 1f - q;
        return -kernelSigma * (6f / h) * t * t;
    }

    // ---------- spatial hash ----------

    static readonly int[] cellOff = new int[27 * 3];
    static LiveSphProvider()
    {
        int k = 0;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++) { cellOff[k++] = dx; cellOff[k++] = dy; cellOff[k++] = dz; }
    }

    int HashCell(int cx, int cy, int cz)
    {
        unchecked
        {
            int hsh = cx * 73856093 ^ cy * 19349663 ^ cz * 83492791;
            return hsh & (tableSize - 1);
        }
    }

    void BuildGridAndNeighbors()
    {
        Array.Fill(head, -1);
        float inv = 1f / h;
        for (int i = 0; i < n; i++)
        {
            int cx = (int)MathF.Floor(pred[i].x * inv), cy = (int)MathF.Floor(pred[i].y * inv), cz = (int)MathF.Floor(pred[i].z * inv);
            int hc = HashCell(cx, cy, cz);
            next[i] = head[hc];
            head[hc] = i;
        }
        for (int i = 0; i < n; i++)
        {
            int cx = (int)MathF.Floor(pred[i].x * inv), cy = (int)MathF.Floor(pred[i].y * inv), cz = (int)MathF.Floor(pred[i].z * inv);
            int cnt = 0;
            Vector3 pi = pred[i];
            for (int c = 0; c < 27; c++)
            {
                int hc = HashCell(cx + cellOff[c * 3], cy + cellOff[c * 3 + 1], cz + cellOff[c * 3 + 2]);
                for (int j = head[hc]; j != -1; j = next[j])
                {
                    if (j == i) continue;
                    float dx = pi.x - pred[j].x, dy = pi.y - pred[j].y, dz = pi.z - pred[j].z;
                    if (dx * dx + dy * dy + dz * dz < h2 && cnt < MaxNbr)
                        nbr[i * MaxNbr + cnt++] = j;
                }
            }
            nbrCount[i] = cnt;
        }
    }

    // ---------- obstacles ----------

    void AdvanceObstacles(float dt)
    {
        if (obstacles == null) return;
        Transform anchor = transform;
        Vector3 off = new Vector3(domainCenterXZ.x, 0f, domainCenterXZ.y);
        // sim -> world: p_w = anchor.TransformPoint(FlipZ(p_sim - off))
        Matrix4x4 flip = Matrix4x4.Scale(new Vector3(1f, 1f, -1f));
        Matrix4x4 simToWorld = anchor.localToWorldMatrix * flip * Matrix4x4.Translate(-off);

        foreach (var o in obstacles)
        {
            o.active = o.transform != null && o.transform.gameObject.activeInHierarchy;
            if (!o.active) continue;
            if (!o.started) { o.startPos = o.transform.position; o.started = true; }
            Vector3 moved = MotionOffset(o, solverTime);
            if (moved != Vector3.zero) o.transform.position = o.startPos + moved;
            o.toLocal = o.transform.worldToLocalMatrix * simToWorld;
            o.fromLocal = o.toLocal.inverse;
        }
    }

    // Push a predicted position out of every obstacle SDF (in obstacle local space, where the
    // unit primitives have radius / half-extent 0.5).
    void CollideObstacles(ref Vector3 p)
    {
        if (obstacles == null) return;
        foreach (var o in obstacles)
        {
            if (!o.active) continue;
            p = ProjectOutOfPrimitive(o, p);
        }
    }

    // ---------- PBF step ----------

    void StepFrame(float frameDt)
    {
        // CFL substepping
        float vmax = 0.5f;
        for (int i = 0; i < n; i++) vmax = Mathf.Max(vmax, vel[i].sqrMagnitude);
        vmax = Mathf.Sqrt(vmax);
        float dtMax = cfl * h / vmax;
        int steps = Mathf.Clamp(Mathf.CeilToInt(frameDt / dtMax), 1, maxSubsteps);
        float dt = frameDt / steps;
        for (int s = 0; s < steps; s++) Substep(dt);
    }

    void Substep(float dt)
    {
        Vector3 g = new Vector3(0f, -9.81f, 0f);
        float wallMin = particleRadius;
        Vector3 dmin = domainMin + Vector3.one * wallMin;
        Vector3 dmax = domainMax - Vector3.one * wallMin;

        for (int i = 0; i < n; i++)
        {
            vel[i] += g * dt;
            pred[i] = pos[i] + vel[i] * dt;
        }

        BuildGridAndNeighbors();

        float sCorrDenom = W(0.2f * h);
        float invRho0 = 1f / restDensity;

        for (int it = 0; it < solverIters; it++)
        {
            // lambdas
            for (int i = 0; i < n; i++)
            {
                Vector3 pi = pred[i];
                float rho = mass * w0;
                Vector3 gradI = Vector3.zero;
                float sumGrad2 = 0f;
                int cnt = nbrCount[i], b = i * MaxNbr;
                for (int k = 0; k < cnt; k++)
                {
                    int j = nbr[b + k];
                    Vector3 d = pi - pred[j];
                    float r = d.magnitude;
                    if (r >= h) continue;
                    rho += mass * W(r);
                    Vector3 grad = (mass * invRho0 * dW(r) / Mathf.Max(r, 1e-6f)) * d;
                    gradI += grad;
                    sumGrad2 += grad.sqrMagnitude;
                }
                dens[i] = rho;
                float C = rho * invRho0 - 1f;
                lambda[i] = C > 0f ? -C / (sumGrad2 + gradI.sqrMagnitude + lambdaEps) : 0f;
            }
            // position deltas
            for (int i = 0; i < n; i++)
            {
                Vector3 pi = pred[i];
                Vector3 delta = Vector3.zero;
                int cnt = nbrCount[i], b = i * MaxNbr;
                for (int k = 0; k < cnt; k++)
                {
                    int j = nbr[b + k];
                    Vector3 d = pi - pred[j];
                    float r = d.magnitude;
                    if (r >= h) continue;
                    float sCorr = 0f;
                    if (sCorrK > 0f && sCorrDenom > 0f)
                    {
                        float w = W(r) / sCorrDenom;
                        sCorr = -sCorrK * w * w * w * w;
                    }
                    delta += ((lambda[i] + lambda[j] + sCorr) * mass * invRho0 * dW(r) / Mathf.Max(r, 1e-6f)) * d;
                }
                dp[i] = delta;
            }
            for (int i = 0; i < n; i++)
            {
                Vector3 p = pred[i] + dp[i];
                p.x = Mathf.Clamp(p.x, dmin.x, dmax.x);
                p.y = Mathf.Clamp(p.y, dmin.y, dmax.y);
                p.z = Mathf.Clamp(p.z, dmin.z, dmax.z);
                CollideObstacles(ref p);
                pred[i] = p;
            }
        }

        for (int i = 0; i < n; i++)
        {
            vel[i] = (pred[i] - pos[i]) / dt;
            pos[i] = pred[i];
        }

        // XSPH viscosity on the final velocities
        if (xsph > 0f)
        {
            for (int i = 0; i < n; i++)
            {
                Vector3 pi = pos[i], acc = Vector3.zero;
                int cnt = nbrCount[i], b = i * MaxNbr;
                for (int k = 0; k < cnt; k++)
                {
                    int j = nbr[b + k];
                    float r = (pi - pos[j]).magnitude;
                    if (r >= h) continue;
                    acc += (vel[j] - vel[i]) * (mass / Mathf.Max(dens[j], 1f) * W(r));
                }
                dp[i] = acc;   // reuse as scratch
            }
            for (int i = 0; i < n; i++) vel[i] += xsph * dp[i];
        }
    }
}
