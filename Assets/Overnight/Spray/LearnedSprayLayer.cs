// LearnedSprayLayer.cs — G2 (overnight 2026-09-30): the 4e LEARNED 3D spray emitter with the 4f "LOR" knobs, live.
//
// Python oracle (SSU_restart, sealed; imported, never copied): Experiments/4e/spray3d.py (coarse_features, Emitter,
// emitter_births, place_blob, Droplets), Helpers/foam_probe.py (potentials), Experiments/4f/spray4f.py (births: rate
// x s_R, outward launch u_o along n^) and deploy4f.py (LOR = launch 3.0 m/s + rate 2.9 on emitter L = emitter_s0.pt).
// Parity gate, timings and decisions: SSU_restart/Experiments/Overnight0930/G2_spray/NOTES.md.
//
// Step(), once per SOLVER frame (1/25 s, the emitter's own frame; hand it each new coarse state):
//   1. droplets already born: ballistic flight, `substeps` semi-implicit Euler steps under gravity, age + 1;
//   2. culls against THIS coarse state (spray3d.Droplets.cull order): domain walls (pad r_d), obstacles (pad r_d),
//      re-entry (nearest coarse particle < mergeR r_c once age >= mergeGrace frames), age * dt > maxAge;
//   3. the 124 features of every coarse particle, double precision, exactly spray3d.coarse_features on
//      foam_probe.potentials (NOT FoamLayer's variant: n_full is numpy's linear-interpolated 90th percentile, an
//      isolated particle's n^ is 0 instead of +y, pairs at |x_ij| = h count, W uses max(|x_ij|, 1e-9)):
//      own [|v_h|, v_up, height/r_c clip 60, log1p I_ta, log1p I_wc, log1p E_k, cnt/n_full, surface, v^.n^],
//      n^ in the local frame (3), then the 16 nearest neighbours within 4 r_c sorted by distance: dx/r_c (3),
//      dv/V0 (3), valid. Local frame: up = -g, e1 = horizontal velocity direction (fixed axis below 1e-6), e2 = up x e1;
//   4. the emitter MLP through Inference Engine (ONNX with the normalisation inside, export_emitter.py) -> packed
//      [N, 63] = log_rate | mixture logits (4) | mu (4x6) | log sigma (4x6) | size logits (10);
//   5. births: n ~ Poisson(min(exp(log_rate) s_R, 50 max(s_R, 1))) per particle; per event a mixture component, a
//      Gaussian (dx, dv) in the parent's local frame (units r_c, V0), + u_o n^ (LOR launch), a size bin -> k droplets
//      as the most compact cubic-lattice blob (spacing 2 r_d, jitter N(0, (0.1 r_d)^2)), velocity + N(0, sig_in^2).
//      Newborns (age 0) are not drawn until their first flight step: the oracle never shows an un-advanced droplet.
// Composite() (every rendered frame): droplets as ray-cast spheres into the SAME (depth, thickness, alpha) field the
// SSFR shading reads, depth-tested against the predicted bulk, so they get its normals, refraction, specular and the
// composite's scene occlusion. Thickness = chord length x the splat's thickness scale (SplatV2 volume units).
using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using UnityEngine;
using Unity.InferenceEngine;

public sealed class LearnedSprayLayer : IDisposable
{
    // ---- the oracle's constants (spray3d.py) ----
    public const int NFeat = 124, KNN = 16, NPack = 63, NMix = 4, NBins = 10;
    public const double HC = 4.0, V0 = 1.0;
    public static readonly int[] SizeEdges = { 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 1000 };
    const int OffLogits = 1, OffMu = 5, OffLs = 29, OffSize = 53;

    // ---- knobs (defaults = 4f LOR on emitter L) ----
    public float rateScale = 2.9f;                 // s_R: x the emitter's Poisson rate
    public float launchSpeed = 3f;                 // u_o [m/s] along the parent's colour-field normal n^
    public float sigIn = 0.08748386701656483f;     // per-droplet velocity spread [m/s] (4e consts.json sig_in)
    public float dropletRadius = 0f;               // r_d [m]; 0 = r_c / volumeRatio^(1/3)
    public float volumeRatio = 25f;                // dense/coarse particle-count ratio the emitter was trained at
    public int maxDroplets = 30000;                // global cap on live droplets (events that do not fit are dropped)
    public float mergeR = 1f;                      // x r_c re-entry radius (4e merge_calib.json)
    public int mergeGrace = 2;                     // frames of age before re-entry applies
    public float maxAge = 2f;                      // s
    public int substeps = 4;
    public double frameDt = 1.0 / 25.0;            // the emitter's frame (s); = 1 / GpuSphProvider.simHz
    public float rateCap = 50f;                    // per-particle Poisson mean cap, x max(s_R, 1) (4e / 4f)
    public Vector3 gravity = new Vector3(0f, -9.81f, 0f);
    public Vector3 domainMin = Vector3.zero, domainMax = new Vector3(3f, 3f, 3f);
    public bool cullObstacles = true, cullMerge = true;
    public bool multithreaded = true;

    /// <summary>Cull obstacle in sim space: toLocal maps sim -> the primitive's local space (sphere radius 0.5 /
    /// box half-extent 0.5 / torus major 1/3 tube 1/6, GpuSph.compute); padLocal = r_d in local units.</summary>
    public struct Obstacle { public Matrix4x4 toLocal; public int shape; public float padLocal; }
    public readonly List<Obstacle> obstacles = new List<Obstacle>();

    // ---- readouts ----
    public int Alive => nd;
    public int Visible { get; private set; }
    public int LastEvents, LastBorn, LastDropped, KillWall, KillObst, KillMerge, KillAge;
    public double LastRateSum;                     // sum of the (capped) Poisson means this step
    public double NFull => nFull;
    public double MinHeight => minH;
    public float MsFeatures, MsMlp, MsBirths, MsDroplets, MsComposite;
    /// <summary>Optional per-event log (parity gate): 10 floats per event, appended BEFORE the droplet cap:
    /// parent, k, centre xyz, velocity xyz (incl. launch), component, size bin. null = off.</summary>
    public List<float> eventLog;
    public long StepCount;
    public float RC => (float)rc;
    public float RD => (float)rd;

    // ---- coarse state (double, sim space) ----
    int n; double rc, rd, hSup;
    double[] P = new double[0], V = new double[0];
    double upX, upY, upZ, refX, refY, refZ;
    // grid (counting sort over the coarse bbox, cell >= h)
    double gx0, gy0, gz0, cell; int gnx, gny, gnz;
    int[] cellStart = new int[1], cellOf = new int[0], sorted = new int[0], cellFill = new int[0];
    // potentials / features
    int[] cnt = new int[0], cntSorted = new int[0];
    double[] ita = new double[0], iwc = new double[0], nrm = new double[0], nhat = new double[0];
    double[] e1 = new double[0], e2 = new double[0];
    bool[] surf = new bool[0];
    int[] knn = new int[0];
    double nFull, minH;
    float[] X = new float[0];
    float[] packed = new float[0];
    public float[] Features => X;                  // [n, 124] of the last Step / ComputeFeatures
    public float[] Packed => packed;               // [n, 63]
    public int[] Knn => knn;                       // [n, 16], -1 = invalid
    public double[] NHat => nhat;                  // [n, 3] world
    public int Count => n;

    // ---- droplets ----
    Vector3[] dPos = new Vector3[0], dVel = new Vector3[0];
    int[] dAge = new int[0];
    int nd;
    public Vector3[] DropletPositions => dPos;
    public Vector3[] DropletVelocities => dVel;
    public int[] DropletAges => dAge;

    // ---- model ----
    Worker worker; ModelAsset modelAsset; BackendType backend;
    public string ModelName => modelAsset != null ? modelAsset.name : "";

    // ---- RNG (xoshiro256**, seeded; births are the only consumer) ----
    ulong s0, s1, s2, s3;
    bool haveSpare; double spare;

    // lattice offsets (units of 2 r_d) sorted by |o| (+1e-6 index tie-break), prefix sums for the blob mean
    static double[] lat, latCum;

    public LearnedSprayLayer(ulong seed = 0x4E5F4C4F52UL) { Seed(seed); BuildLattice(); }

    public void Seed(ulong seed)
    {
        ulong z = seed;
        s0 = SplitMix(ref z); s1 = SplitMix(ref z); s2 = SplitMix(ref z); s3 = SplitMix(ref z);
        haveSpare = false;
    }

    public void SetModel(ModelAsset asset, BackendType be)
    {
        if (asset == null) throw new ArgumentNullException(nameof(asset));
        if (worker != null && asset == modelAsset && be == backend) return;
        worker?.Dispose();
        worker = new Worker(ModelLoader.Load(asset), be);
        modelAsset = asset; backend = be;
    }

    public void Reset() { nd = 0; Visible = 0; LastEvents = LastBorn = LastDropped = 0; KillWall = KillObst = KillMerge = KillAge = 0; }

    public void Dispose() { worker?.Dispose(); worker = null; modelAsset = null; }

    // =====================================================================================================
    // Step: one solver frame
    // =====================================================================================================
    public void Step(float[] data, int off, int count, float rCoarse)
    {
        StepCount++;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        LoadCoarse(data, off, count, rCoarse);
        BuildGrid();
        // 1-2: flight + culls against this state (droplets born last step included)
        AdvanceAndCull();
        MsDroplets = (float)sw.Elapsed.TotalMilliseconds;
        // 3: features
        sw.Restart();
        ComputeFeaturesLoaded();
        MsFeatures = (float)sw.Elapsed.TotalMilliseconds;
        // 4: MLP
        sw.Restart();
        RunMlp();
        MsMlp = (float)sw.Elapsed.TotalMilliseconds;
        // 5: births (age 0: drawn from the next step on)
        sw.Restart();
        Births();
        MsBirths = (float)sw.Elapsed.TotalMilliseconds;
    }

    /// <summary>Features only (parity gate): loads the frame, builds the grid, fills Features / Knn / NHat.</summary>
    public void ComputeFeatures(float[] data, int off, int count, float rCoarse)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        LoadCoarse(data, off, count, rCoarse);
        BuildGrid();
        ComputeFeaturesLoaded();
        MsFeatures = (float)sw.Elapsed.TotalMilliseconds;
    }

    /// <summary>MLP on the current Features (parity gate). Needs SetModel.</summary>
    public void RunMlp()
    {
        if (worker == null) throw new InvalidOperationException("LearnedSprayLayer: SetModel first");
        if (packed.Length != n * NPack) packed = new float[n * NPack];
        if (n == 0) return;
        using (var input = new Tensor<float>(new TensorShape(n, NFeat), X))
        {
            worker.Schedule(input);
            var o = worker.PeekOutput("packed") as Tensor<float>;
            using (var cpu = o.ReadbackAndClone())
                cpu.AsReadOnlyNativeArray().CopyTo(packed);
        }
    }

    // ---------------------------------------------------------------- coarse state + grid
    void LoadCoarse(float[] data, int off, int count, float rCoarse)
    {
        n = count; rc = rCoarse; hSup = HC * rc;
        rd = dropletRadius > 0f ? dropletRadius : rc / Math.Pow(Math.Max(volumeRatio, 1e-6), 1.0 / 3.0);
        if (P.Length < 3 * n)
        {
            int m = 3 * n;
            P = new double[m]; V = new double[m]; nrm = new double[m]; nhat = new double[m]; e1 = new double[m]; e2 = new double[m];
            cnt = new int[n]; cntSorted = new int[n]; ita = new double[n]; iwc = new double[n]; surf = new bool[n];
            knn = new int[n * KNN]; cellOf = new int[n]; sorted = new int[n];
        }
        if (X.Length != n * NFeat) X = new float[n * NFeat];
        for (int i = 0; i < n; i++)
        {
            int b = off + i * 7;
            P[3 * i] = data[b]; P[3 * i + 1] = data[b + 1]; P[3 * i + 2] = data[b + 2];
            V[3 * i] = data[b + 3]; V[3 * i + 1] = data[b + 4]; V[3 * i + 2] = data[b + 5];
        }
        // up = -g / |g|; the fixed horizontal axis for |v_h| < 1e-6 (spray3d.local_frames)
        double gx = gravity.x, gy = gravity.y, gz = gravity.z, gn = Math.Sqrt(gx * gx + gy * gy + gz * gz);
        if (!(gn > 0)) { gx = 0; gy = -1; gz = 0; gn = 1; }
        upX = -gx / gn; upY = -gy / gn; upZ = -gz / gn;
        double rx = Math.Abs(upX) < 0.9 ? 1.0 : 0.0, ry = 0.0, rz = Math.Abs(upX) < 0.9 ? 0.0 : 1.0;
        double ru = rx * upX + ry * upY + rz * upZ;
        rx -= ru * upX; ry -= ru * upY; rz -= ru * upZ;
        double rn = Math.Sqrt(rx * rx + ry * ry + rz * rz);
        refX = rx / rn; refY = ry / rn; refZ = rz / rn;
    }

    void BuildGrid()
    {
        double x0 = double.MaxValue, y0 = double.MaxValue, z0 = double.MaxValue, x1 = double.MinValue, y1 = double.MinValue, z1 = double.MinValue;
        for (int i = 0; i < n; i++)
        {
            double x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
            if (x < x0) x0 = x; if (x > x1) x1 = x;
            if (y < y0) y0 = y; if (y > y1) y1 = y;
            if (z < z0) z0 = z; if (z > z1) z1 = z;
        }
        if (n == 0) { x0 = y0 = z0 = 0; x1 = y1 = z1 = 0; }
        cell = hSup > 0 ? hSup : 1.0;
        long cells;
        while (true)
        {
            gnx = (int)Math.Floor((x1 - x0) / cell) + 1; gny = (int)Math.Floor((y1 - y0) / cell) + 1; gnz = (int)Math.Floor((z1 - z0) / cell) + 1;
            cells = (long)gnx * gny * gnz;
            if (cells <= 4_000_000) break;
            cell *= 2.0;                          // runaway particle: coarser cells stay correct (cell >= h)
        }
        gx0 = x0; gy0 = y0; gz0 = z0;
        if (cellStart.Length < cells + 1) { cellStart = new int[cells + 1]; cellFill = new int[cells]; }
        Array.Clear(cellStart, 0, (int)cells + 1);
        for (int i = 0; i < n; i++)
        {
            int ix = Clampi((int)((P[3 * i] - gx0) / cell), 0, gnx - 1);
            int iy = Clampi((int)((P[3 * i + 1] - gy0) / cell), 0, gny - 1);
            int iz = Clampi((int)((P[3 * i + 2] - gz0) / cell), 0, gnz - 1);
            int c = (iz * gny + iy) * gnx + ix;
            cellOf[i] = c; cellStart[c + 1]++;
        }
        for (int c = 0; c < cells; c++) cellStart[c + 1] += cellStart[c];
        Array.Copy(cellStart, cellFill, (int)cells);
        for (int i = 0; i < n; i++) sorted[cellFill[cellOf[i]]++] = i;
    }

    static int Clampi(int v, int lo, int hi) => v < lo ? lo : (v > hi ? hi : v);

    void ForRange(int count, int grain, Action<int, int> body)
    {
        if (!multithreaded || count < 2 * grain) { body(0, count); return; }
        Parallel.ForEach(Partitioner.Create(0, count, grain), r => body(r.Item1, r.Item2));
    }

    // ---------------------------------------------------------------- features (spray3d.coarse_features)
    static double Log1p(double x)
    {
        double u = 1.0 + x;
        return u == 1.0 ? x : Math.Log(u) * (x / (u - 1.0));
    }

    void ComputeFeaturesLoaded()
    {
        double h = hSup, h2 = h * h;
        // pass A: counts, trapped air, colour-field normal, KNN (strictly inside h, as cKDTree distance_upper_bound)
        ForRange(n, 128, (i0, i1) =>
        {
            var kd = new double[KNN]; var kj = new int[KNN];
            for (int i = i0; i < i1; i++)
            {
                double pxi = P[3 * i], pyi = P[3 * i + 1], pzi = P[3 * i + 2];
                double vxi = V[3 * i], vyi = V[3 * i + 1], vzi = V[3 * i + 2];
                int c0 = 0; double ta = 0, nx = 0, ny = 0, nz = 0; int kc = 0;
                int ix = Clampi((int)((pxi - gx0) / cell), 0, gnx - 1), iy = Clampi((int)((pyi - gy0) / cell), 0, gny - 1), iz = Clampi((int)((pzi - gz0) / cell), 0, gnz - 1);
                for (int dz = -1; dz <= 1; dz++)
                {
                    int cz = iz + dz; if (cz < 0 || cz >= gnz) continue;
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        int cy = iy + dy; if (cy < 0 || cy >= gny) continue;
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int cx = ix + dx; if (cx < 0 || cx >= gnx) continue;
                            int c = (cz * gny + cy) * gnx + cx;
                            for (int s = cellStart[c], e = cellStart[c + 1]; s < e; s++)
                            {
                                int j = sorted[s];
                                if (j == i) continue;
                                double xx = pxi - P[3 * j], xy = pyi - P[3 * j + 1], xz = pzi - P[3 * j + 2];
                                double d2 = xx * xx + xy * xy + xz * xz;
                                if (d2 > h2) continue;
                                // KNN: strictly inside h, sorted by (d2, index)
                                if (d2 < h2 && (kc < KNN || d2 < kd[KNN - 1] || (d2 == kd[KNN - 1] && j < kj[KNN - 1])))
                                {
                                    int p = kc < KNN ? kc++ : KNN - 1;
                                    while (p > 0 && (kd[p - 1] > d2 || (kd[p - 1] == d2 && kj[p - 1] > j))) { kd[p] = kd[p - 1]; kj[p] = kj[p - 1]; p--; }
                                    kd[p] = d2; kj[p] = j;
                                }
                                double d = Math.Max(Math.Sqrt(d2), 1e-9);
                                double w = 1.0 - d / h;
                                double hx = xx / d, hy = xy / d, hz = xz / d;
                                c0++;
                                double wx = vxi - V[3 * j], wy = vyi - V[3 * j + 1], wz = vzi - V[3 * j + 2];
                                double vn = Math.Sqrt(wx * wx + wy * wy + wz * wz);
                                double vd = Math.Max(vn, 1e-9);
                                ta += vn * (1.0 - (wx / vd * hx + wy / vd * hy + wz / vd * hz)) * w;
                                nx += hx * w; ny += hy * w; nz += hz * w;
                            }
                        }
                    }
                }
                cnt[i] = c0; ita[i] = ta;
                nrm[3 * i] = nx; nrm[3 * i + 1] = ny; nrm[3 * i + 2] = nz;
                double nm = Math.Max(Math.Sqrt(nx * nx + ny * ny + nz * nz), 1e-9);
                nhat[3 * i] = nx / nm; nhat[3 * i + 1] = ny / nm; nhat[3 * i + 2] = nz / nm;
                for (int k = 0; k < KNN; k++) knn[i * KNN + k] = k < kc ? kj[k] : -1;
            }
        });
        // n_full = max(np.percentile(cnt, 90) [linear], 4)
        nFull = 4.0;
        if (n > 0)
        {
            Array.Copy(cnt, cntSorted, n);
            Array.Sort(cntSorted, 0, n);
            double vi = (n - 1) * 0.9;
            int lo = (int)Math.Floor(vi); int hi = Math.Min(lo + 1, n - 1);
            double g = vi - lo, a = cntSorted[lo], b = cntSorted[hi], dba = b - a;
            double pct = g >= 0.5 ? b - dba * (1.0 - g) : a + dba * g;
            nFull = Math.Max(pct, 4.0);
        }
        double surfCut = 0.75 * nFull;
        minH = double.MaxValue;
        for (int i = 0; i < n; i++)
        {
            surf[i] = cnt[i] < surfCut;
            double hh = P[3 * i] * upX + P[3 * i + 1] * upY + P[3 * i + 2] * upZ;
            if (hh < minH) minH = hh;
        }
        if (n == 0) minH = 0;
        // pass B: wave crest (surface pairs, j behind i's normal), gated by v^.n^ >= 0.6; then the feature row
        ForRange(n, 128, (i0, i1) =>
        {
            for (int i = i0; i < i1; i++)
            {
                double pxi = P[3 * i], pyi = P[3 * i + 1], pzi = P[3 * i + 2];
                double vxi = V[3 * i], vyi = V[3 * i + 1], vzi = V[3 * i + 2];
                double nix = nhat[3 * i], niy = nhat[3 * i + 1], niz = nhat[3 * i + 2];
                double vm = Math.Sqrt(vxi * vxi + vyi * vyi + vzi * vzi), vmd = Math.Max(vm, 1e-9);
                double vdn = vxi / vmd * nix + vyi / vmd * niy + vzi / vmd * niz;
                double wc = 0;
                if (surf[i] && vdn >= 0.6)
                {
                    int ix = Clampi((int)((pxi - gx0) / cell), 0, gnx - 1), iy = Clampi((int)((pyi - gy0) / cell), 0, gny - 1), iz = Clampi((int)((pzi - gz0) / cell), 0, gnz - 1);
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        int cz = iz + dz; if (cz < 0 || cz >= gnz) continue;
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            int cy = iy + dy; if (cy < 0 || cy >= gny) continue;
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                int cx = ix + dx; if (cx < 0 || cx >= gnx) continue;
                                int c = (cz * gny + cy) * gnx + cx;
                                for (int s = cellStart[c], e = cellStart[c + 1]; s < e; s++)
                                {
                                    int j = sorted[s];
                                    if (j == i || !surf[j]) continue;
                                    double xx = pxi - P[3 * j], xy = pyi - P[3 * j + 1], xz = pzi - P[3 * j + 2];
                                    double d2 = xx * xx + xy * xy + xz * xz;
                                    if (d2 > h2) continue;
                                    double d = Math.Max(Math.Sqrt(d2), 1e-9);
                                    if (xx / d * nix + xy / d * niy + xz / d * niz <= 0) continue;
                                    wc += (1.0 - (nix * nhat[3 * j] + niy * nhat[3 * j + 1] + niz * nhat[3 * j + 2])) * (1.0 - d / h);
                                }
                            }
                        }
                    }
                }
                iwc[i] = wc;
                // local frame
                double vu = vxi * upX + vyi * upY + vzi * upZ;
                double hx = vxi - vu * upX, hy = vyi - vu * upY, hz = vzi - vu * upZ;
                double hn = Math.Sqrt(hx * hx + hy * hy + hz * hz);
                double ax, ay, az;
                if (hn > 1e-6) { double q = Math.Max(hn, 1e-12); ax = hx / q; ay = hy / q; az = hz / q; }
                else { ax = refX; ay = refY; az = refZ; }
                double bx = upY * az - upZ * ay, by = upZ * ax - upX * az, bz = upX * ay - upY * ax;   // e2 = up x e1
                e1[3 * i] = ax; e1[3 * i + 1] = ay; e1[3 * i + 2] = az;
                e2[3 * i] = bx; e2[3 * i + 1] = by; e2[3 * i + 2] = bz;
                int r = i * NFeat;
                double hgt = ((pxi * upX + pyi * upY + pzi * upZ) - minH) / rc;
                hgt = hgt < 0 ? 0 : (hgt > 60 ? 60 : hgt);
                double ek = 0.5 * vm * vm;
                X[r + 0] = (float)((vxi * ax + vyi * ay + vzi * az) / V0);
                X[r + 1] = (float)(vu / V0);
                X[r + 2] = (float)hgt;
                X[r + 3] = (float)Log1p(ita[i]);
                X[r + 4] = (float)Log1p(wc);
                X[r + 5] = (float)Log1p(ek);
                X[r + 6] = (float)(cnt[i] / nFull);
                X[r + 7] = surf[i] ? 1f : 0f;
                X[r + 8] = (float)((vxi * nix + vyi * niy + vzi * niz) / vmd);
                X[r + 9] = (float)(nix * ax + niy * ay + niz * az);
                X[r + 10] = (float)(nix * upX + niy * upY + niz * upZ);
                X[r + 11] = (float)(nix * bx + niy * by + niz * bz);
                for (int k = 0; k < KNN; k++)
                {
                    int o = r + 12 + 7 * k, j = knn[i * KNN + k];
                    if (j < 0) { for (int q = 0; q < 7; q++) X[o + q] = 0f; continue; }
                    double dx = P[3 * j] - pxi, dy = P[3 * j + 1] - pyi, dz = P[3 * j + 2] - pzi;
                    double wx = V[3 * j] - vxi, wy = V[3 * j + 1] - vyi, wz = V[3 * j + 2] - vzi;
                    X[o + 0] = (float)((dx * ax + dy * ay + dz * az) / rc);
                    X[o + 1] = (float)((dx * upX + dy * upY + dz * upZ) / rc);
                    X[o + 2] = (float)((dx * bx + dy * by + dz * bz) / rc);
                    X[o + 3] = (float)((wx * ax + wy * ay + wz * az) / V0);
                    X[o + 4] = (float)((wx * upX + wy * upY + wz * upZ) / V0);
                    X[o + 5] = (float)((wx * bx + wy * by + wz * bz) / V0);
                    X[o + 6] = 1f;
                }
            }
        });
    }

    // ---------------------------------------------------------------- births (spray4f.births, LOR knobs)
    /// <summary>Parity gate: one more round of births from the current Features / Packed (no flight, no culls).</summary>
    public void RunBirths() => Births();

    void Births()
    {
        LastEvents = LastBorn = LastDropped = 0; LastRateSum = 0;
        if (n == 0) return;
        double sR = rateScale, cap = rateCap * Math.Max(sR, 1.0);
        var w = new double[NMix]; var ps = new double[NBins];
        for (int i = 0; i < n; i++)
        {
            double lam = Math.Exp((double)packed[i * NPack]) * sR;
            if (lam > cap) lam = cap;
            if (!(lam > 0)) continue;
            LastRateSum += lam;
            int ne = Poisson(lam);
            for (int e = 0; e < ne; e++) BirthEvent(i, w, ps);
        }
    }

    // one event of parent i: component, (dx, dv) ~ N(mu, sigma^2) in the local frame, size bin, k, blob
    public void BirthEvent(int i, double[] w, double[] ps)
    {
        int r = i * NPack;
        Softmax(packed, r + OffLogits, NMix, w);
        double u = NextDouble(), acc = 0; int comp = 0;
        for (int m = 0; m < NMix; m++) { acc += w[m]; if (u > acc) comp++; }
        if (comp > NMix - 1) comp = NMix - 1;
        double y0, y1, y2, y3, y4, y5;
        int mo = r + OffMu + comp * 6, lo_ = r + OffLs + comp * 6;
        y0 = packed[mo + 0] + Math.Exp(packed[lo_ + 0]) * Normal();
        y1 = packed[mo + 1] + Math.Exp(packed[lo_ + 1]) * Normal();
        y2 = packed[mo + 2] + Math.Exp(packed[lo_ + 2]) * Normal();
        y3 = packed[mo + 3] + Math.Exp(packed[lo_ + 3]) * Normal();
        y4 = packed[mo + 4] + Math.Exp(packed[lo_ + 4]) * Normal();
        y5 = packed[mo + 5] + Math.Exp(packed[lo_ + 5]) * Normal();
        Softmax(packed, r + OffSize, NBins, ps);
        double u2 = NextDouble(); acc = 0; int kb = 0;
        for (int m = 0; m < NBins; m++) { acc += ps[m]; if (u2 > acc) kb++; }
        if (kb > NBins - 1) kb = NBins - 1;
        int klo = SizeEdges[kb], khi = SizeEdges[kb + 1];
        int k = klo + (int)Math.Floor(NextDouble() * (khi - klo));
        LastEvents++;
        double ax = e1[3 * i], ay = e1[3 * i + 1], az = e1[3 * i + 2], bx = e2[3 * i], by = e2[3 * i + 1], bz = e2[3 * i + 2];
        double cx = P[3 * i] + (y0 * ax + y1 * upX + y2 * bx) * rc;
        double cy = P[3 * i + 1] + (y0 * ay + y1 * upY + y2 * by) * rc;
        double cz = P[3 * i + 2] + (y0 * az + y1 * upZ + y2 * bz) * rc;
        double ux = V[3 * i] + (y3 * ax + y4 * upX + y5 * bx) * V0 + launchSpeed * nhat[3 * i];
        double uy = V[3 * i + 1] + (y3 * ay + y4 * upY + y5 * by) * V0 + launchSpeed * nhat[3 * i + 1];
        double uz = V[3 * i + 2] + (y3 * az + y4 * upZ + y5 * bz) * V0 + launchSpeed * nhat[3 * i + 2];
        if (eventLog != null)
        {
            eventLog.Add(i); eventLog.Add(k); eventLog.Add((float)cx); eventLog.Add((float)cy); eventLog.Add((float)cz);
            eventLog.Add((float)ux); eventLog.Add((float)uy); eventLog.Add((float)uz); eventLog.Add(comp); eventLog.Add(kb);
        }
        if (nd + k > maxDroplets) { LastDropped += k; return; }
        EnsureDropletCapacity(nd + k);
        double sp = 2.0 * rd;
        double mx = latCum[3 * k] / k, my = latCum[3 * k + 1] / k, mz = latCum[3 * k + 2] / k;
        double jit = 0.1 * rd, sg = sigIn;
        for (int m = 0; m < k; m++)
        {
            double px_ = cx + (lat[3 * m] - mx) * sp + jit * Normal();
            double py_ = cy + (lat[3 * m + 1] - my) * sp + jit * Normal();
            double pz_ = cz + (lat[3 * m + 2] - mz) * sp + jit * Normal();
            dPos[nd] = new Vector3((float)px_, (float)py_, (float)pz_);
            dVel[nd] = new Vector3((float)(ux + sg * Normal()), (float)(uy + sg * Normal()), (float)(uz + sg * Normal()));
            dAge[nd] = 0;
            nd++;
        }
        LastBorn += k;
    }

    static void Softmax(float[] src, int o, int m, double[] dst)
    {
        double mx = double.MinValue;
        for (int q = 0; q < m; q++) if (src[o + q] > mx) mx = src[o + q];
        double s = 0;
        for (int q = 0; q < m; q++) { dst[q] = Math.Exp(src[o + q] - mx); s += dst[q]; }
        for (int q = 0; q < m; q++) dst[q] /= s;
    }

    void EnsureDropletCapacity(int need)
    {
        if (dPos.Length >= need) return;
        int cap = Math.Max(need, Math.Max(1024, dPos.Length * 2));
        Array.Resize(ref dPos, cap); Array.Resize(ref dVel, cap); Array.Resize(ref dAge, cap);
    }

    /// <summary>Test hook: add droplets directly (age 0).</summary>
    public void AddDroplets(Vector3[] p, Vector3[] v, int count, int age = 0)
    {
        EnsureDropletCapacity(nd + count);
        for (int m = 0; m < count; m++) { dPos[nd] = p[m]; dVel[nd] = v[m]; dAge[nd] = age; nd++; }
    }

    // ---------------------------------------------------------------- flight + culls (spray3d.Droplets)
    /// <summary>Flight + culls only, against the coarse frame last loaded (Step's first half; also the self-test hook).</summary>
    public void AdvanceAndCull()
    {
        KillWall = KillObst = KillMerge = KillAge = 0;
        if (nd == 0) { Visible = 0; return; }
        float hs = (float)(frameDt / Math.Max(substeps, 1));
        Vector3 g = gravity;
        double r2m = Math.Pow(Math.Min(mergeR, 4f) * rc, 2);
        float pad = (float)rd;
        Vector3 lo = domainMin + Vector3.one * pad, hi = domainMax - Vector3.one * pad;
        int w = 0;
        for (int p = 0; p < nd; p++)
        {
            Vector3 x = dPos[p], v = dVel[p];
            for (int s = 0; s < substeps; s++) { v += g * hs; x += v * hs; }
            int age = dAge[p] + 1;
            bool wall = x.x < lo.x || x.y < lo.y || x.z < lo.z || x.x > hi.x || x.y > hi.y || x.z > hi.z;
            bool obst = !wall && cullObstacles && InsideObstacle(x);
            bool merge = !wall && !obst && cullMerge && age >= mergeGrace && NearCoarse(x, r2m);
            bool old = !wall && !obst && !merge && age * frameDt > maxAge;
            if (wall) KillWall++; else if (obst) KillObst++; else if (merge) KillMerge++; else if (old) KillAge++;
            if (wall || obst || merge || old) continue;
            dPos[w] = x; dVel[w] = v; dAge[w] = age; w++;
        }
        nd = w; Visible = w;
    }

    bool InsideObstacle(Vector3 x)
    {
        for (int o = 0; o < obstacles.Count; o++)
        {
            var ob = obstacles[o];
            Vector3 q = ob.toLocal.MultiplyPoint3x4(x);
            float pq = ob.padLocal;
            if (ob.shape == 0) { if (q.magnitude <= 0.5f + pq) return true; }
            else if (ob.shape == 1) { if (Mathf.Abs(q.x) <= 0.5f + pq && Mathf.Abs(q.y) <= 0.5f + pq && Mathf.Abs(q.z) <= 0.5f + pq) return true; }
            else
            {
                float rad = Mathf.Sqrt(q.x * q.x + q.z * q.z) - 1f / 3f;
                if (rad * rad + q.y * q.y <= (1f / 6f + pq) * (1f / 6f + pq)) return true;
            }
        }
        return false;
    }

    bool NearCoarse(Vector3 x, double r2)
    {
        if (n == 0) return false;
        int ix = (int)Math.Floor((x.x - gx0) / cell), iy = (int)Math.Floor((x.y - gy0) / cell), iz = (int)Math.Floor((x.z - gz0) / cell);
        for (int dz = -1; dz <= 1; dz++)
        {
            int cz = iz + dz; if (cz < 0 || cz >= gnz) continue;
            for (int dy = -1; dy <= 1; dy++)
            {
                int cy = iy + dy; if (cy < 0 || cy >= gny) continue;
                for (int dx = -1; dx <= 1; dx++)
                {
                    int cx = ix + dx; if (cx < 0 || cx >= gnx) continue;
                    int c = (cz * gny + cy) * gnx + cx;
                    for (int s = cellStart[c], e = cellStart[c + 1]; s < e; s++)
                    {
                        int j = sorted[s];
                        double a = x.x - P[3 * j], b = x.y - P[3 * j + 1], d = x.z - P[3 * j + 2];
                        if (a * a + b * b + d * d < r2) return true;
                    }
                }
            }
        }
        return false;
    }

    // ---------------------------------------------------------------- composite into the SSFR field
    int[] touched = new int[0]; float[] zFront = new float[0], zChord = new float[0], zSpeed = new float[0];

    /// <summary>Ray-cast every drawn droplet as a sphere into the field texture pixels (texture rows, row 0 = BOTTOM;
    /// r = depth along the camera forward [sim m], g = thickness, b = alpha, a = speed) — the layout
    /// FluidSceneMVP.ProcessPrediction uploads. Only pixels where a droplet is IN FRONT of the bulk (or where there is
    /// no bulk) change: depth = nearest droplet front, thickness += chord x thicknessScale, alpha = 1.
    /// minRadiusPx keeps far droplets at least that big (footprint only); radiusScale scales r_d for display.</summary>
    public void Composite(Color[] px, int W, int H, Vector3 eye, Vector3 right, Vector3 up, Vector3 fwd, float focal,
                          float aspect, float thicknessScale, float minRadiusPx, float radiusScale)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        int HW = W * H;
        if (zFront.Length != HW) { zFront = new float[HW]; zChord = new float[HW]; zSpeed = new float[HW]; touched = new int[HW]; for (int i = 0; i < HW; i++) zFront[i] = float.MaxValue; }
        int nt = 0;
        float halfH = 0.5f * H, pxPerUnit = focal * halfH;
        float rW = (float)rd * Mathf.Max(radiusScale, 1e-3f);
        for (int p = 0; p < nd; p++)
        {
            if (dAge[p] < 1) continue;                               // newborns: not advanced yet
            Vector3 rel = dPos[p] - eye;
            float cx = Vector3.Dot(rel, right), cy = Vector3.Dot(rel, up), depth = Vector3.Dot(rel, fwd);
            if (!(depth > 1e-2f)) continue;
            float pxf = (focal * cx / depth / aspect + 1f) * 0.5f * W;
            float pyf = (1f - focal * cy / depth) * 0.5f * H;       // row 0 = top
            float rTrue = rW * pxPerUnit / depth;
            float rPix = Mathf.Max(rTrue, minRadiusPx);
            float rEff = rPix * depth / pxPerUnit;                   // world radius actually drawn
            if (!(pxf + rPix > 0f && pxf - rPix < W && pyf + rPix > 0f && pyf - rPix < H)) continue;
            int S = (int)Math.Ceiling(rPix);
            int x0 = (int)Math.Floor(pxf), y0 = (int)Math.Floor(pyf);
            float r2 = rPix * rPix, rE2 = rEff * rEff, dScale = depth / pxPerUnit;
            float spd = dVel[p].magnitude;
            for (int oy = -S; oy <= S; oy++)
            {
                int ty = y0 + oy; if (ty < 0 || ty >= H) continue;
                float dy = ty + 0.5f - pyf;
                for (int ox = -S; ox <= S; ox++)
                {
                    int tx = x0 + ox; if (tx < 0 || tx >= W) continue;
                    float dx = tx + 0.5f - pxf;
                    float rho2 = dx * dx + dy * dy;
                    if (!(rho2 < r2)) continue;
                    float s2 = rho2 * dScale * dScale;
                    float hh = Mathf.Sqrt(Mathf.Max(rE2 - s2, 0f));
                    float front = depth - hh;
                    int ti = (H - 1 - ty) * W + tx;                  // texture row 0 = bottom
                    Color f = px[ti];
                    if (f.b >= 0.5f && front >= f.r) continue;        // behind / inside the predicted bulk
                    if (zFront[ti] == float.MaxValue) { touched[nt++] = ti; zChord[ti] = 0f; zSpeed[ti] = 0f; }
                    if (front < zFront[ti]) zFront[ti] = front;
                    zChord[ti] += 2f * hh;
                    if (spd > zSpeed[ti]) zSpeed[ti] = spd;
                }
            }
        }
        for (int t = 0; t < nt; t++)
        {
            int ti = touched[t];
            Color f = px[ti];
            float thick = (f.b >= 0.5f ? f.g : 0f) + zChord[ti] * thicknessScale;
            px[ti] = new Color(zFront[ti], thick, 1f, Mathf.Max(f.b >= 0.5f ? f.a : 0f, zSpeed[ti]));
            zFront[ti] = float.MaxValue;
        }
        DrawnPixels = nt;
        MsComposite = (float)sw.Elapsed.TotalMilliseconds;
    }
    public int DrawnPixels { get; private set; }

    // ---------------------------------------------------------------- RNG
    static ulong SplitMix(ref ulong z)
    {
        z += 0x9E3779B97F4A7C15UL;
        ulong x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
        return x ^ (x >> 31);
    }

    static ulong Rotl(ulong x, int k) => (x << k) | (x >> (64 - k));

    ulong NextU64()
    {
        ulong result = Rotl(s1 * 5, 7) * 9;
        ulong t = s1 << 17;
        s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t; s3 = Rotl(s3, 45);
        return result;
    }

    public double NextDouble() => (NextU64() >> 11) * (1.0 / 9007199254740992.0);   // [0, 1)

    public double Normal()
    {
        if (haveSpare) { haveSpare = false; return spare; }
        double u, v, s;
        do { u = 2.0 * NextDouble() - 1.0; v = 2.0 * NextDouble() - 1.0; s = u * u + v * v; } while (s >= 1.0 || s == 0.0);
        double m = Math.Sqrt(-2.0 * Math.Log(s) / s);
        spare = v * m; haveSpare = true;
        return u * m;
    }

    /// <summary>Poisson(lam) by inversion (lam <= 50 x s_R here, so the sequential search is short).</summary>
    public int Poisson(double lam)
    {
        if (!(lam > 0)) return 0;
        if (lam > 400) { double g = lam + Math.Sqrt(lam) * Normal(); return g < 0 ? 0 : (int)Math.Round(g); }
        double u = NextDouble(), p = Math.Exp(-lam), F = p; int k = 0;
        while (u > F && k < 10000) { k++; p *= lam / k; F += p; }
        return k;
    }

    static void BuildLattice()
    {
        if (lat != null) return;
        int L = 15, N = L * L * L;
        var key = new double[N]; var idx = new int[N]; var o = new double[3 * N];
        int q = 0;
        for (int a = -7; a <= 7; a++) for (int b = -7; b <= 7; b++) for (int c = -7; c <= 7; c++)   // meshgrid 'ij' order
        {
            o[3 * q] = a; o[3 * q + 1] = b; o[3 * q + 2] = c;
            key[q] = Math.Sqrt(a * a + b * b + c * c) + 1e-6 * q; idx[q] = q; q++;
        }
        Array.Sort(key, idx);
        lat = new double[3 * N]; latCum = new double[3 * (N + 1)];
        for (int m = 0; m < N; m++)
        {
            int s = idx[m];
            lat[3 * m] = o[3 * s]; lat[3 * m + 1] = o[3 * s + 1]; lat[3 * m + 2] = o[3 * s + 2];
            latCum[3 * (m + 1)] = latCum[3 * m] + lat[3 * m];
            latCum[3 * (m + 1) + 1] = latCum[3 * m + 1] + lat[3 * m + 1];
            latCum[3 * (m + 1) + 2] = latCum[3 * m + 2] + lat[3 * m + 2];
        }
    }
}
