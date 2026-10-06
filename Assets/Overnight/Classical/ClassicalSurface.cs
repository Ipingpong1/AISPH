// ClassicalSurface.cs — G3 overnight 2026-09-30: the classical C* surface reconstructor (the `t3_aniso+nr` row of the
// SSU_restart PBF tables) as a GPU pipeline that produces the SAME (depth, thickness, alpha) field the network path
// hands to FluidSSFRScene, so the two can be flipped live (FluidSceneMVP key C, see FluidSceneMVP.Classical.cs).
//
//   1. Yu & Turk 2013 anisotropic kernels (Banana AniKernelGenerator defaults, as Truong & Yuksel 2018 ship them):
//      uniform-grid neighbour search within h = 8 r (grid built on the CPU, counting sort), weighted mean, Laplacian
//      centre blend lambda 0.5, weighted covariance, Jacobi eigen-decomposition, axis-ratio 8 clamp, det-1 scaling,
//      < 25 neighbours -> 0.75 I (drawn as a sphere).
//   2. Ellipsoid depth splat at sprite radius k r (k = 3): exact paraxial ray / quadric front root per pixel,
//      nearest wins (InterlockedMin on order-preserving keys); iso kernels use splat_render_v2's sphere formula.
//      Thickness (shading only, not part of the research row): path length through the ellipsoids (farthest back
//      minus nearest front surface) x 15.851, the pysurf GT scale, i.e. the unit the network's predicted thickness is
//      in (option: the volume-matched chord sum in the v2 input unit, ~5x thinner).
//   3. Truong & Yuksel 2018 narrow-range filter, faithful port (filter_size 40, iters 2, threshold_ratio 20,
//      particle radius 0.0414, 1d2d + 4 px clean-up), then the frozen depth offset (+3.83 mm subtracted on the mask).
// Python oracle: SSU_restart Helpers/aniso_render.py (anisotropic_kernels, render_aniso) and
// Helpers/narrow_range_faithful.py (recon_narrow_range_faithful); parity + parameter sources:
// SSU_restart/Experiments/Overnight0930/G3_classical/NOTES.md.
using System;
using System.Diagnostics;
using UnityEngine;

[Serializable]
public class ClassicalSurfaceSettings
{
    [Tooltip("Sprite radius multiplier k (ellipsoids drawn at k * r). 3.0 = the PBF-suite C* selection (SSU_restart Experiments/T123_probe/coverage_probe_pbf.json selected.t3_aniso.k; T&Y's own default is 1.5).")]
    public float radiusMult = 3f;
    [Tooltip("Opt-in sparse-support experiment. Off preserves the frozen classical renderer. On interpolates sparseRadiusMult to radiusMult over the neighbor-count range, leaving dense particles unchanged.")]
    public bool repairSparseRadius = false;
    [Tooltip("Opt-in support continuity: smooth weighted-support radius, determinant-one shape, and center. Takes precedence over count-based sparse radius. Off preserves the frozen renderer; no temporal history is used.")]
    public bool repairSupportContinuity = false;
    [Tooltip("Weighted-neighbor support at the dense endpoint. 12.5 = 25 nominal neighbors times mean cubic weight 1/2 in a uniform 3D ball. Dense endpoints retain the original kernel and center.")]
    public float supportContinuityEndpoint = 12.5f;
    public float sparseRadiusMult = 1.5f;
    public int sparseNeighborsMin = 0, sparseNeighborsMax = 25;
    [Tooltip("Particle radius r [sim m] for the kernels and sprites. 0 = the live splat radius (provider.ParticleRadius, GpuSphProvider 0.0414).")]
    public float particleRadius = 0f;
    [Tooltip("Kernel support h = kernelRatio * r (Banana AniKernelGenerator: 8).")]
    public float kernelRatio = 8f;
    [Tooltip("Laplacian centre blend lambda: c = lerp(x, weighted mean, lambda). T&Y ship 0.5 (Yu & Turk used 0.9-1).")]
    public float positionBlending = 0.5f;
    [Tooltip("Max ratio of the largest to any other kernel axis (Banana: 8).")]
    public float axisRatio = 8f;
    [Tooltip("Fewer neighbours than this -> isotropic spray kernel (Banana: 25).")]
    public int neighborThreshold = 25;
    [Tooltip("Spray kernel K = spraySize * I (Banana defaultSpraySize 0.75; drawn as a sphere of the FULL sprite radius, T&Y depth-pass.vs.glsl).")]
    public float spraySize = 0.75f;
    [Tooltip("Kernels whose three column lengths agree within this are drawn as spheres (T&Y UNIT_SPHERE_ISOLATED_PARTICLE, 1e-2).")]
    public float isoTol = 1e-2f;
    [Tooltip("Footprint clamps in px: min (1), ellipsoid half-width max (96), sphere radius max = this x k (24 -> 72 px at k 3). aniso_render.render_aniso / coverage_probe.render_rows.")]
    public float minRadiusPx = 1f, maxRadiusPx = 96f, sphereMaxRadiusPxPerK = 24f;

    public enum ThicknessMode { DepthRangeGTUnit, VolumeMatchedSum }
    [Tooltip("Thickness for the shading (the research row scores depth + mask only). DepthRangeGTUnit (default): path length through the ellipsoids (farthest back minus nearest front surface, per pixel) x Thickness Scale GT = the unit the network's predicted thickness is in, so the same kThick is the same material. VolumeMatchedSum: sum of chord lengths x thicknessScale / k^3 (the v2 INPUT unit; ~5x thinner than the network's output on live frames).")]
    public ThicknessMode thicknessMode = ThicknessMode.DepthRangeGTUnit;
    [Tooltip("GT thickness per metre of path length: 15.851 = recipe.thickness_scale of the pysurf GT bakes 067a / 060a were trained on (gpupbf_v1_pysurfgt, gpupbf_poolstir_v1_pysurfgt; frozen since 030).")]
    public float thicknessScaleGT = 15.851288f;

    [Tooltip("Truong & Yuksel narrow-range filter on the ellipsoid depth (the '+nr' of t3_aniso+nr).")]
    public bool narrowRange = true;
    [Tooltip("NR filter_size (sigma_world = 0.1 * filter_size * r). 40 = 067c PBF tuning (tuned_wide.json).")]
    public int nrFilterSize = 40;
    [Tooltip("NR separable iterations (each = horizontal + vertical 1D pass). 2 = 067c PBF tuning.")]
    public int nrIters = 2;
    [Tooltip("NR threshold delta = ratio * r. 20 = 067c PBF tuning (shader default 10.5).")]
    public float nrThresholdRatio = 20f;
    [Tooltip("NR clamp mu = ratio * r (shader: 1).")]
    public float nrClampRatio = 1f;
    [Tooltip("NR particle radius [m]. 0.0414 = the recon_narrow_range_faithful default, which is what the research row used (coverage_probe.nr() never passes it).")]
    public float nrParticleRadius = 0.0414f;
    [Tooltip("Use the actual current particle radius for narrow-range filtering. Off preserves the historical fixed-radius baseline.")]
    public bool nrUseParticleRadius = false;
    [Tooltip("NR projected-radius cap in px (MAX_FILTER_SIZE 100).")]
    public int nrMaxFilterSize = 100;
    [Tooltip("NR fixed-radius 2D clean-up pass (T&Y's '+1' iteration).")]
    public bool nrCleanup = true;
    [Tooltip("Clean-up radius in px (fixedFilterRadius 4).")]
    public int nrCleanupRadiusPx = 4;
    [Tooltip("Subtracted from the final depth on the mask [sim m]. +0.003826 = the t3_aniso+nr offset of the PBF suite (coverage_probe_pbf.json offsets). The network path applies no offset.")]
    public float depthOffset = 0.0038260221f;
}

public sealed class ClassicalSurface : IDisposable
{
    const int KSTRIDE = 24, PSTRIDE = 16, RASTER_GROUPS_X = 256;
    public readonly ClassicalSurfaceSettings s;
    readonly ComputeShader cs;
    readonly int kAniso, kProject, kClear, kRaster, kResolve, kNR1D, kNR2D, kPack;

    ComputeBuffer posBuf, cellStartBuf, cellCountBuf, cellIdxBuf, cellOfBuf, kernBuf, projBuf, supportBuf;
    ComputeBuffer keyBuf, thickFixBuf, backKeyBuf, splatBuf, maskBuf, thickBuf, dA, dB, shadeDepthBuf;
    RenderTexture outRT;
    int capN, capCells, H, W, count;
    float[] posStage = new float[0];
    uint[] cellStart = new uint[0], cellCount = new uint[0], cellIdx = new uint[0];
    int[] cellOf = new int[0];
    uint[] cellOfU = new uint[0];
    ComputeBuffer finalDepth;          // the NR output buffer of the last Run (tensor layout, before the offset)

    /// <summary>(depth sim m, thickness, alpha 0/1, 0), W x H, row 0 = bottom: the network path's field layout.</summary>
    public RenderTexture Output => outRT;
    public int Count => count;
    public int Width => W;
    public int Height => H;
    public double LastCpuMs { get; private set; }     // CPU side of the last Run (grid + uploads + dispatch encoding)
    public int LastGridCells { get; private set; }

    public ClassicalSurface(ClassicalSurfaceSettings settings)
    {
        s = settings ?? new ClassicalSurfaceSettings();
        cs = Resources.Load<ComputeShader>("ClassicalSurface");
        if (cs == null) throw new Exception("ClassicalSurface: compute shader Resources/ClassicalSurface.compute not found");
        kAniso = cs.FindKernel("Aniso"); kProject = cs.FindKernel("Project"); kClear = cs.FindKernel("ClearImage");
        kRaster = cs.FindKernel("Raster"); kResolve = cs.FindKernel("Resolve"); kNR1D = cs.FindKernel("NRPass1D");
        kNR2D = cs.FindKernel("NRPass2D"); kPack = cs.FindKernel("Pack");
    }

    // ------------------------------------------------------------------ resources

    static void Grow(ref ComputeBuffer b, int n, int stride)
    {
        if (b != null && b.count >= n) return;
        b?.Release();
        b = new ComputeBuffer(Math.Max(n, 1), stride);
    }

    void EnsureParticles(int n)
    {
        if (n <= capN && posBuf != null) return;
        int cap = Math.Max(1024, Mathf.NextPowerOfTwo(n));
        posBuf?.Release(); kernBuf?.Release(); projBuf?.Release(); supportBuf?.Release(); cellIdxBuf?.Release(); cellOfBuf?.Release();
        posBuf = new ComputeBuffer(cap * 3, 4);
        cellOfBuf = new ComputeBuffer(cap, 4);
        kernBuf = new ComputeBuffer(cap * KSTRIDE, 4);
        projBuf = new ComputeBuffer(cap * PSTRIDE, 4);
        supportBuf = new ComputeBuffer(cap * 2, 4);
        cellIdxBuf = new ComputeBuffer(cap, 4);
        posStage = new float[cap * 3];
        cellIdx = new uint[cap];
        cellOf = new int[cap];
        cellOfU = new uint[cap];
        capN = cap;
    }

    void EnsureImage(int h, int w)
    {
        if (outRT != null && h == H && w == W) return;
        ReleaseImage();
        H = h; W = w;
        int n = H * W;
        keyBuf = new ComputeBuffer(n, 4); thickFixBuf = new ComputeBuffer(n, 4); backKeyBuf = new ComputeBuffer(n, 4);
        splatBuf = new ComputeBuffer(n, 4); maskBuf = new ComputeBuffer(n, 4); thickBuf = new ComputeBuffer(n, 4);
        dA = new ComputeBuffer(n, 4); dB = new ComputeBuffer(n, 4); shadeDepthBuf = new ComputeBuffer(n, 4);
        outRT = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBFloat)
        { enableRandomWrite = true, filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp, name = "ClassicalSurface.Output" };
        outRT.Create();
    }

    void ReleaseImage()
    {
        keyBuf?.Release(); thickFixBuf?.Release(); backKeyBuf?.Release(); splatBuf?.Release(); maskBuf?.Release(); thickBuf?.Release();
        dA?.Release(); dB?.Release(); shadeDepthBuf?.Release();
        keyBuf = thickFixBuf = backKeyBuf = splatBuf = maskBuf = thickBuf = dA = dB = shadeDepthBuf = null;
        if (outRT != null) { outRT.Release(); UnityEngine.Object.DestroyImmediate(outRT); outRT = null; }
    }

    public void Dispose()
    {
        ReleaseImage();
        posBuf?.Release(); kernBuf?.Release(); projBuf?.Release(); supportBuf?.Release(); cellIdxBuf?.Release(); cellOfBuf?.Release();
        cellStartBuf?.Release(); cellCountBuf?.Release();
        posBuf = kernBuf = projBuf = supportBuf = cellIdxBuf = cellOfBuf = cellStartBuf = cellCountBuf = null;
        capN = 0; capCells = 0;
    }

    // ------------------------------------------------------------------ pipeline

    /// <summary>Full C* frame: particles (stride floats per particle, xyz first, sim space) seen from the splat camera
    /// (eye / right / up / fwd in sim space, focal = the model window's NDC focal, H x W window) -> Output.
    /// r = particle radius (kernels + sprites; s.particleRadius overrides), thicknessScale = the v2 thickness unit.</summary>
    public void Run(float[] data, int off, int n, int stride, Vector3 eye, Vector3 right, Vector3 up, Vector3 fwd,
                    float focal, int h, int w, float r, float thicknessScale)
    {
        var sw = Stopwatch.StartNew();
        float rr = s.particleRadius > 0f ? s.particleRadius : r;
        EnsureImage(h, w);
        count = n;
        EnsureParticles(n);
        for (int i = 0; i < n; i++)
        {
            int b = off + i * stride;
            posStage[3 * i] = data[b]; posStage[3 * i + 1] = data[b + 1]; posStage[3 * i + 2] = data[b + 2];
        }
        if (n > 0)
        {
            posBuf.SetData(posStage, 0, 0, 3 * n);
            BuildGrid(n, (float)(s.kernelRatio * (double)rr));
            DispatchAniso(n, rr);
        }
        Render(eye, right, up, fwd, focal, rr, thicknessScale);
        LastCpuMs = sw.Elapsed.TotalMilliseconds;
    }

    /// <summary>Debug / known-answer path: kernel records given directly (per particle: centre xyz, K row-major 9),
    /// the kernel stage is skipped and the iso flag is computed here with the same rule.</summary>
    public void RunWithKernels(float[] kern12, int n, Vector3 eye, Vector3 right, Vector3 up, Vector3 fwd,
                               float focal, int h, int w, float r, float thicknessScale)
    {
        EnsureImage(h, w);
        count = n;
        EnsureParticles(n);
        var rec = new float[n * KSTRIDE];
        var support = new float[n * 2];
        for (int i = 0; i < n; i++)
        {
            int a = i * 12, b = i * KSTRIDE;
            for (int k = 0; k < 12; k++) rec[b + k] = kern12[a + k];
            var K = new Matrix4x4();
            for (int rI = 0; rI < 3; rI++) for (int c = 0; c < 3; c++) K[rI, c] = kern12[a + 3 + 3 * rI + c];
            K[3, 3] = 1f;
            Matrix4x4 Ki = K.inverse; Matrix4x4 Ki2 = Ki * Ki;
            for (int rI = 0; rI < 3; rI++) for (int c = 0; c < 3; c++) rec[b + 12 + 3 * rI + c] = Ki2[rI, c];
            float l0 = Mathf.Sqrt(K[0, 0] * K[0, 0] + K[1, 0] * K[1, 0] + K[2, 0] * K[2, 0]);
            float l1 = Mathf.Sqrt(K[0, 1] * K[0, 1] + K[1, 1] * K[1, 1] + K[2, 1] * K[2, 1]);
            float l2 = Mathf.Sqrt(K[0, 2] * K[0, 2] + K[1, 2] * K[1, 2] + K[2, 2] * K[2, 2]);
            bool iso = Mathf.Abs(l0 - l1) < s.isoTol && Mathf.Abs(l1 - l2) < s.isoTol && Mathf.Abs(l2 - l0) < s.isoTol;
            rec[b + 21] = -1f; rec[b + 22] = !s.repairSupportContinuity && iso ? 1f : 0f; rec[b + 23] = 0f;
            support[2 * i + 1] = 1f; // injected kernels retain their prescribed shape and radius
        }
        kernBuf.SetData(rec, 0, 0, rec.Length);
        supportBuf.SetData(support, 0, 0, support.Length);
        Render(eye, right, up, fwd, focal, s.particleRadius > 0f ? s.particleRadius : r, thicknessScale);
    }

    void BuildGrid(int n, float hk)
    {
        Vector3 mn = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue), mx = -mn;
        for (int i = 0; i < n; i++)
        {
            float x = posStage[3 * i], y = posStage[3 * i + 1], z = posStage[3 * i + 2];
            if (x < mn.x) mn.x = x; if (y < mn.y) mn.y = y; if (z < mn.z) mn.z = z;
            if (x > mx.x) mx.x = x; if (y > mx.y) mx.y = y; if (z > mx.z) mx.z = z;
        }
        // any cell size >= h is exact (the 27-cell search then sees every neighbour); grow it if the box is huge
        float cell = hk;
        int gx, gy, gz;
        while (true)
        {
            gx = Math.Max(1, (int)Math.Floor((mx.x - mn.x) / cell) + 1);
            gy = Math.Max(1, (int)Math.Floor((mx.y - mn.y) / cell) + 1);
            gz = Math.Max(1, (int)Math.Floor((mx.z - mn.z) / cell) + 1);
            if ((long)gx * gy * gz <= 1 << 18) break;
            cell *= 1.5f;
        }
        int cells = gx * gy * gz;
        LastGridCells = cells;
        if (cells > capCells || cellStartBuf == null)
        {
            int cap = Math.Max(4096, Mathf.NextPowerOfTwo(cells));
            cellStartBuf?.Release(); cellCountBuf?.Release();
            cellStartBuf = new ComputeBuffer(cap, 4); cellCountBuf = new ComputeBuffer(cap, 4);
            cellStart = new uint[cap]; cellCount = new uint[cap];
            capCells = cap;
        }
        Array.Clear(cellCount, 0, cells);
        float inv = 1f / cell;
        for (int i = 0; i < n; i++)
        {
            // identical cell arithmetic to the shader: floor((x - origin) / cellSize), clamped
            int cx = Mathf.Clamp((int)Math.Floor((posStage[3 * i] - mn.x) / cell), 0, gx - 1);
            int cy = Mathf.Clamp((int)Math.Floor((posStage[3 * i + 1] - mn.y) / cell), 0, gy - 1);
            int cz = Mathf.Clamp((int)Math.Floor((posStage[3 * i + 2] - mn.z) / cell), 0, gz - 1);
            int c = (cz * gy + cy) * gx + cx;
            cellOf[i] = c;
            cellOfU[i] = (uint)c;
            cellCount[c]++;
        }
        uint acc = 0;
        for (int c = 0; c < cells; c++) { cellStart[c] = acc; acc += cellCount[c]; }
        // scatter (reuse cellCount as a running cursor, then restore it)
        for (int c = 0; c < cells; c++) cellCount[c] = 0;
        for (int i = 0; i < n; i++) { int c = cellOf[i]; cellIdx[cellStart[c] + cellCount[c]++] = (uint)i; }
        cellStartBuf.SetData(cellStart, 0, 0, cells);
        cellCountBuf.SetData(cellCount, 0, 0, cells);
        cellIdxBuf.SetData(cellIdx, 0, 0, n);
        cellOfBuf.SetData(cellOfU, 0, 0, n);
        cs.SetVector("_GridOrigin", mn);
        cs.SetFloat("_CellSize", cell);
        cs.SetInts("_GridDim", gx, gy, gz);
    }

    void DispatchAniso(int n, float r)
    {
        cs.SetInt("_N", n);
        cs.SetFloat("_Hk", (float)(s.kernelRatio * (double)r));
        cs.SetFloat("_Lambda", s.positionBlending);
        cs.SetFloat("_AxisRatio", s.axisRatio);
        cs.SetInt("_NbrThreshold", s.neighborThreshold);
        cs.SetFloat("_SpraySize", s.spraySize);
        cs.SetFloat("_IsoTol", s.isoTol);
        cs.SetInt("_SupportContinuity", s.repairSupportContinuity ? 1 : 0);
        cs.SetFloat("_SupportEndpoint", Mathf.Max(s.supportContinuityEndpoint, 1e-4f));
        cs.SetBuffer(kAniso, "_Support", supportBuf);
        cs.SetBuffer(kAniso, "_Pos", posBuf);
        cs.SetBuffer(kAniso, "_CellStart", cellStartBuf);
        cs.SetBuffer(kAniso, "_CellCount", cellCountBuf);
        cs.SetBuffer(kAniso, "_CellIdx", cellIdxBuf);
        cs.SetBuffer(kAniso, "_CellOf", cellOfBuf);
        cs.SetBuffer(kAniso, "_Kern", kernBuf);
        cs.Dispatch(kAniso, (n + 63) / 64, 1, 1);
    }

    void Render(Vector3 eye, Vector3 right, Vector3 up, Vector3 fwd, float focal, float r, float thicknessScale)
    {
        int n = count;
        double rho = s.radiusMult * (double)r;
        double ppu = focal * (H / 2.0);
        cs.SetInt("_N", n);
        cs.SetInt("_Hpx", H); cs.SetInt("_Wpx", W);
        cs.SetVector("_Eye", eye); cs.SetVector("_Right", right); cs.SetVector("_Up", up); cs.SetVector("_Fwd", fwd);
        cs.SetFloat("_Focal", focal);
        cs.SetFloat("_Ppu", (float)ppu);
        cs.SetFloat("_Aspect", (float)((double)W / H));
        cs.SetInt("_RepairSparseRadius", s.repairSparseRadius ? 1 : 0);
        cs.SetInt("_SupportContinuity", s.repairSupportContinuity ? 1 : 0);
        cs.SetFloat("_SparseRadiusRatio", Mathf.Max(s.sparseRadiusMult, 1e-4f) / Mathf.Max(s.radiusMult, 1e-4f));
        cs.SetFloat("_SparseNeighborsMin", s.sparseNeighborsMin);
        cs.SetFloat("_SparseNeighborsMax", Math.Max(s.sparseNeighborsMax, s.sparseNeighborsMin + 1));
        cs.SetFloat("_Rho", (float)rho);
        cs.SetFloat("_Rho2", (float)(rho * rho));
        cs.SetFloat("_RhoPpu", (float)(rho * ppu));
        cs.SetFloat("_MinRpx", s.minRadiusPx);
        cs.SetFloat("_MaxRpx", s.maxRadiusPx);
        cs.SetFloat("_SphMaxRpx", (float)(s.sphereMaxRadiusPxPerK * (double)s.radiusMult));
        double k3 = s.radiusMult * (double)s.radiusMult * s.radiusMult;
        cs.SetFloat("_ThickPerM", (float)(thicknessScale / k3));
        cs.SetInt("_ThickMode", s.thicknessMode == ClassicalSurfaceSettings.ThicknessMode.DepthRangeGTUnit ? 0 : 1);
        cs.SetFloat("_ThickGT", s.thicknessScaleGT);

        int gx = (W + 7) / 8, gy = (H + 7) / 8;
        cs.SetBuffer(kClear, "_DepthKey", keyBuf);
        cs.SetBuffer(kClear, "_ThickFix", thickFixBuf);
        cs.SetBuffer(kClear, "_BackKey", backKeyBuf);
        cs.Dispatch(kClear, gx, gy, 1);

        if (n > 0)
        {
            cs.SetBuffer(kProject, "_Kern", kernBuf);
            cs.SetBuffer(kProject, "_Support", supportBuf);
            cs.SetBuffer(kProject, "_Proj", projBuf);
            cs.Dispatch(kProject, (n + 63) / 64, 1, 1);

            cs.SetInt("_GroupsX", RASTER_GROUPS_X);
            cs.SetBuffer(kRaster, "_Proj", projBuf);
            cs.SetBuffer(kRaster, "_DepthKey", keyBuf);
            cs.SetBuffer(kRaster, "_ThickFix", thickFixBuf);
            cs.SetBuffer(kRaster, "_BackKey", backKeyBuf);
            cs.Dispatch(kRaster, RASTER_GROUPS_X, (n + RASTER_GROUPS_X - 1) / RASTER_GROUPS_X, 1);
        }

        cs.SetBuffer(kResolve, "_DepthKey", keyBuf);
        cs.SetBuffer(kResolve, "_ThickFix", thickFixBuf);
        cs.SetBuffer(kResolve, "_BackKey", backKeyBuf);
        cs.SetBuffer(kResolve, "_SplatDepth", splatBuf);
        cs.SetBuffer(kResolve, "_DOut", dA);
        cs.SetBuffer(kResolve, "_Mask", maskBuf);
        cs.SetBuffer(kResolve, "_Thick", thickBuf);
        cs.Dispatch(kResolve, gx, gy, 1);

        ComputeBuffer cur = dA, nxt = dB;
        if (s.narrowRange && s.nrFilterSize > 0)
        {
            double fpx = focal * (H / 2.0);     // = H / (2 tan(fov/2)) for this window's vertical fov
            double nrR = s.nrUseParticleRadius ? r : s.nrParticleRadius;
            cs.SetFloat("_NRK", (float)(s.nrFilterSize * fpx * nrR * 0.1));
            cs.SetInt("_NRMaxR", s.nrMaxFilterSize);
            cs.SetFloat("_Delta", (float)(s.nrThresholdRatio * nrR));
            cs.SetFloat("_Mu", (float)(s.nrClampRatio * nrR));
            cs.SetInt("_CleanR", s.nrCleanupRadiusPx);
            for (int it = 0; it < s.nrIters; it++)
                for (int axis = 1; axis >= 0; axis--)            // horizontal (1) then vertical (0)
                {
                    cs.SetInt("_Axis", axis);
                    cs.SetBuffer(kNR1D, "_DIn", cur);
                    cs.SetBuffer(kNR1D, "_MaskIn", maskBuf);
                    cs.SetBuffer(kNR1D, "_DOut", nxt);
                    cs.Dispatch(kNR1D, gx, gy, 1);
                    (cur, nxt) = (nxt, cur);
                }
            if (s.nrCleanup)
            {
                cs.SetBuffer(kNR2D, "_DIn", cur);
                cs.SetBuffer(kNR2D, "_MaskIn", maskBuf);
                cs.SetBuffer(kNR2D, "_DOut", nxt);
                cs.Dispatch(kNR2D, gx, gy, 1);
                (cur, nxt) = (nxt, cur);
            }
        }
        finalDepth = cur;

        cs.SetFloat("_DepthOffset", s.depthOffset);
        cs.SetBuffer(kPack, "_DIn", cur);
        cs.SetBuffer(kPack, "_MaskIn", maskBuf);
        cs.SetBuffer(kPack, "_Thick", thickBuf);
        cs.SetTexture(kPack, "_OutTex", outRT);
        cs.SetBuffer(kPack, "_ShadeDepth", shadeDepthBuf);
        cs.Dispatch(kPack, gx, gy, 1);
    }

    // ------------------------------------------------------------------ readbacks (debug / parity / foam / kThick)

    public void ReadSplat(float[] depth, float[] mask, float[] thick)
    {
        if (depth != null) splatBuf.GetData(depth, 0, 0, H * W);
        if (mask != null) maskBuf.GetData(mask, 0, 0, H * W);
        if (thick != null) thickBuf.GetData(thick, 0, 0, H * W);
    }

    /// <summary>Depth after the NR filter (before the depth offset), tensor layout (row 0 = top).</summary>
    public void ReadFinalDepth(float[] depth) => finalDepth.GetData(depth, 0, 0, H * W);

    public void ReadKernels(float[] dst) => kernBuf.GetData(dst, 0, 0, count * KSTRIDE);

    /// <summary>Debug weighted support and smooth confidence, two floats per particle.</summary>
    public void ReadKernelSupport(float[] dst) => supportBuf.GetData(dst, 0, 0, count * 2);

    /// <summary>Debug projected records; slot 15 is radius divided by the bulk radius (0 when centre-culled).</summary>
    public void ReadProjected(float[] dst) => projBuf.GetData(dst, 0, 0, count * PSTRIDE);

    /// <summary>The depth handed to the shading (after NR and the offset), tensor layout, 0 = no fluid. Synchronous.</summary>
    public void ReadShadedDepth(float[] depth) => shadeDepthBuf.GetData(depth, 0, 0, H * W);

    /// <summary>Asynchronous readback of the shaded depth (tensor layout, 0 = no fluid); the callback runs on a later frame.</summary>
    public void RequestShadedDepth(Action<UnityEngine.Rendering.AsyncGPUReadbackRequest> done) =>
        UnityEngine.Rendering.AsyncGPUReadback.Request(shadeDepthBuf, H * W * 4, 0, done);

    /// <summary>Stall until the GPU work queued so far is done (for timing).</summary>
    public void Sync()
    {
        var one = new float[1];
        (finalDepth ?? dA).GetData(one, 0, 0, 1);
    }

    /// <summary>Per-stage GPU+CPU wall time (ms, mean over reps) with a sync after every stage. Order: grid (CPU),
    /// upload, aniso, project+raster+resolve (splat), nr, pack, and the whole Run without intermediate syncs.</summary>
    public string Profile(float[] data, int off, int n, int stride, Vector3 eye, Vector3 right, Vector3 up, Vector3 fwd,
                          float focal, int h, int w, float r, float thicknessScale, int reps)
    {
        float rr = s.particleRadius > 0f ? s.particleRadius : r;
        Run(data, off, n, stride, eye, right, up, fwd, focal, h, w, r, thicknessScale);   // warm-up (allocations, kernels)
        Sync();
        double tGrid = 0, tUp = 0, tAniso = 0, tSplat = 0, tNR = 0, tWhole = 0, tWholeCpu = 0;
        var swt = new Stopwatch();
        for (int rep = 0; rep < reps; rep++)
        {
            swt.Restart();
            for (int i = 0; i < n; i++)
            {
                int b = off + i * stride;
                posStage[3 * i] = data[b]; posStage[3 * i + 1] = data[b + 1]; posStage[3 * i + 2] = data[b + 2];
            }
            posBuf.SetData(posStage, 0, 0, 3 * n);
            Sync();
            tUp += swt.Elapsed.TotalMilliseconds;
            swt.Restart();
            BuildGrid(n, (float)(s.kernelRatio * (double)rr));
            tGrid += swt.Elapsed.TotalMilliseconds;
            Sync();
            swt.Restart();
            DispatchAniso(n, rr);
            Sync();
            tAniso += swt.Elapsed.TotalMilliseconds;
            // splat only (NR off, pack included; pack is ~free), then the NR increment
            bool nr = s.narrowRange;
            s.narrowRange = false;
            swt.Restart();
            Render(eye, right, up, fwd, focal, rr, thicknessScale);
            Sync();
            tSplat += swt.Elapsed.TotalMilliseconds;
            s.narrowRange = nr;
            swt.Restart();
            Render(eye, right, up, fwd, focal, rr, thicknessScale);
            Sync();
            tNR += swt.Elapsed.TotalMilliseconds;
            swt.Restart();
            Run(data, off, n, stride, eye, right, up, fwd, focal, h, w, r, thicknessScale);
            tWholeCpu += LastCpuMs;
            Sync();
            tWhole += swt.Elapsed.TotalMilliseconds;
        }
        double tSync = 0;                      // the bare round trip every synced number above contains once
        for (int rep = 0; rep < reps; rep++) { swt.Restart(); Sync(); tSync += swt.Elapsed.TotalMilliseconds; }
        double k = 1.0 / Math.Max(reps, 1);
        // the 'nr' timing ran the whole Render with NR on: subtract the splat-only time for the NR share
        return $"{{\"reps\":{reps},\"N\":{n},\"H\":{H},\"W\":{W},\"grid_cpu_ms\":{tGrid * k:F3},\"upload_ms\":{tUp * k:F3}," +
               $"\"aniso_ms\":{tAniso * k:F3},\"splat_ms\":{tSplat * k:F3},\"render_with_nr_ms\":{tNR * k:F3}," +
               $"\"nr_ms\":{(tNR - tSplat) * k:F3},\"whole_run_synced_ms\":{tWhole * k:F3},\"whole_run_cpu_ms\":{tWholeCpu * k:F3}," +
               $"\"sync_roundtrip_ms\":{tSync * k:F3},\"grid_cells\":{LastGridCells}}}";
    }
}
