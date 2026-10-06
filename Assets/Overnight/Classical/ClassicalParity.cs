// ClassicalParity.cs — G3 overnight 2026-09-30: editor-only driver for the known-answer / parity gate of
// ClassicalSurface against the Python oracle (SSU_restart Experiments/Overnight0930/G3_classical/g3_oracle.py).
// A fixture directory holds camera.json + particles.bin (N x 7 float32, the provider layout) or kernels.bin
// (N x 12 float32: centre, K row-major) for the injected-kernel raster cases. RunFixture writes u_*.bin next to them:
//   u_kern.bin (N x 24: centre, K, K^-2, n, iso, spray), u_splat_depth.bin / u_mask.bin / u_thick.bin (H x W, row 0 = top),
//   u_nr_depth.bin (after the NR filter, before the depth offset), u_timing.json.
// Called through the MCP bridge in EDIT mode (no Play mode, no scene change):
//   System.Type.GetType("ClassicalParity, Assembly-CSharp").GetMethod("RunFixture").Invoke(null, new object[]{ dir })
#if UNITY_EDITOR
using System;
using System.IO;
using UnityEngine;

public static class ClassicalParity
{
    [Serializable]
    class Cam
    {
        public string name;
        public int N, H, W;
        public float[] eye, right, up, fwd;
        public float focal, particleRadius, thicknessScale, kMult, depthOffset;
        public bool repairSparseRadius, nrUseParticleRadius, repairSupportContinuity;
        public float supportContinuityEndpoint;
        public float sparseRadiusMult;
    }

    static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, f.Length * 4);
        return f;
    }

    static void WriteF32(string path, float[] f, int n)
    {
        var bytes = new byte[n * 4];
        Buffer.BlockCopy(f, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }

    static Vector3 V(float[] a) => new Vector3(a[0], a[1], a[2]);

    public static string RunFixture(string dir) => RunFixture(dir, 0);

    public static string RunFixture(string dir, int profileReps)
    {
        var cam = JsonUtility.FromJson<Cam>(File.ReadAllText(Path.Combine(dir, "camera.json")));
        var settings = new ClassicalSurfaceSettings { radiusMult = cam.kMult > 0f ? cam.kMult : 3f };
        if (cam.depthOffset != 0f) settings.depthOffset = cam.depthOffset;
        settings.repairSparseRadius = cam.repairSparseRadius;
        settings.repairSupportContinuity = cam.repairSupportContinuity;
        if (cam.supportContinuityEndpoint > 0f) settings.supportContinuityEndpoint = cam.supportContinuityEndpoint;
        settings.nrUseParticleRadius = cam.nrUseParticleRadius;
        if (cam.sparseRadiusMult > 0f) settings.sparseRadiusMult = cam.sparseRadiusMult;
        var cs = new ClassicalSurface(settings);
        try
        {
            int H = cam.H, W = cam.W, HW = H * W;
            string kpath = Path.Combine(dir, "kernels.bin");
            var sw = System.Diagnostics.Stopwatch.StartNew();
            float[] data = null;
            int n;
            if (File.Exists(kpath))
            {
                var k12 = ReadF32(kpath);
                n = k12.Length / 12;
                cs.RunWithKernels(k12, n, V(cam.eye), V(cam.right), V(cam.up), V(cam.fwd), cam.focal, H, W,
                                  cam.particleRadius, cam.thicknessScale);
            }
            else
            {
                data = ReadF32(Path.Combine(dir, "particles.bin"));
                n = data.Length / 7;
                cs.Run(data, 0, n, 7, V(cam.eye), V(cam.right), V(cam.up), V(cam.fwd), cam.focal, H, W,
                       cam.particleRadius, cam.thicknessScale);
            }
            cs.Sync();
            double ms = sw.Elapsed.TotalMilliseconds;
            float[] d = new float[HW], m = new float[HW], t = new float[HW], nr = new float[HW];
            cs.ReadSplat(d, m, t);
            cs.ReadFinalDepth(nr);
            WriteF32(Path.Combine(dir, "u_splat_depth.bin"), d, HW);
            WriteF32(Path.Combine(dir, "u_mask.bin"), m, HW);
            WriteF32(Path.Combine(dir, "u_thick.bin"), t, HW);
            WriteF32(Path.Combine(dir, "u_nr_depth.bin"), nr, HW);
            if (data != null)
            {
                var kern = new float[n * 24];
                cs.ReadKernels(kern);
                WriteF32(Path.Combine(dir, "u_kern.bin"), kern, n * 24);
                var support = new float[n * 2];
                cs.ReadKernelSupport(support);
                WriteF32(Path.Combine(dir, "u_support.bin"), support, n * 2);
            }
            var projected = new float[n * 16];
            cs.ReadProjected(projected);
            WriteF32(Path.Combine(dir, "u_projected.bin"), projected, n * 16);
            var radii = new float[n];
            for (int i = 0; i < n; i++) radii[i] = settings.radiusMult * projected[i * 16 + 15];
            WriteF32(Path.Combine(dir, "u_radius_mult.bin"), radii, n);
            // the shading field itself (texture layout), to prove the Pack flip + offset
            var tex = new Texture2D(W, H, TextureFormat.RGBAFloat, false);
            var prev = RenderTexture.active;
            RenderTexture.active = cs.Output;
            tex.ReadPixels(new Rect(0, 0, W, H), 0, 0, false);
            tex.Apply(false);
            RenderTexture.active = prev;
            var px = tex.GetPixels();
            UnityEngine.Object.DestroyImmediate(tex);
            var field = new float[HW * 4];
            for (int i = 0; i < HW; i++) { field[4 * i] = px[i].r; field[4 * i + 1] = px[i].g; field[4 * i + 2] = px[i].b; field[4 * i + 3] = px[i].a; }
            WriteF32(Path.Combine(dir, "u_field_rgba.bin"), field, HW * 4);
            string prof = "null";
            if (profileReps > 0 && data != null)
                prof = cs.Profile(data, 0, n, 7, V(cam.eye), V(cam.right), V(cam.up), V(cam.fwd), cam.focal, H, W,
                                  cam.particleRadius, cam.thicknessScale, profileReps);
            string json = $"{{\"first_run_ms\":{ms:F3},\"cpu_ms\":{cs.LastCpuMs:F3},\"grid_cells\":{cs.LastGridCells},\"N\":{n},\"H\":{H},\"W\":{W},\"profile\":{prof}," +
                          $"\"unity\":\"{Application.unityVersion}\",\"gpu\":\"{SystemInfo.graphicsDeviceName}\",\"api\":\"{SystemInfo.graphicsDeviceType}\"}}";
            File.WriteAllText(Path.Combine(dir, "u_timing.json"), json);
            return json;
        }
        finally { cs.Dispose(); }
    }

    /// <summary>Every fixture directory under root (sorted); returns one line per fixture.</summary>
    public static string RunAll(string root, int profileReps)
    {
        var sb = new System.Text.StringBuilder();
        var dirs = Directory.GetDirectories(root);
        Array.Sort(dirs, StringComparer.Ordinal);
        foreach (var d in dirs)
        {
            if (!File.Exists(Path.Combine(d, "camera.json"))) continue;
            try { sb.AppendLine(Path.GetFileName(d) + " " + RunFixture(d, profileReps)); }
            catch (Exception e) { sb.AppendLine(Path.GetFileName(d) + " ERROR " + e.Message); }
        }
        return sb.ToString();
    }
}
#endif
