// BakedParticleProvider.cs — plays a baked binary particle sequence (FluidData/<name>_lr.bytes
// + <name>_meta.json, the bake_unity_live_sim.py format) through the ParticleFrameProvider
// seam. Same wire format as FluidLiveMVP.BakedParticleFrames.

using System;
using System.IO;
using UnityEngine;

public class BakedParticleProvider : ParticleFrameProvider
{
    [Tooltip("Baked particle sequence (FluidData/<name>_lr.bytes).")]
    public TextAsset particleData;
    [Tooltip("Sidecar metadata (FluidData/<name>_meta.json). Supplies fps here; FluidSceneMVP also falls back to it for normalization stats and domain center.")]
    public TextAsset metaJson;

    [Serializable] class MetaFps { public float fps = 25f; public float coarseRadius = 0f; }

    float[] all;
    int[] starts, counts;
    float fps = 25f;
    float coarseRadius = 0f;                 // v2 meta field (Helpers/patch_unity_meta_v2.py); 0 in legacy metas

    void EnsureLoaded()
    {
        if (all != null) return;
        if (particleData == null) throw new Exception($"{name}: BakedParticleProvider.particleData not assigned");
        if (metaJson != null) { var mf = JsonUtility.FromJson<MetaFps>(metaJson.text); fps = mf.fps; coarseRadius = mf.coarseRadius; }

        byte[] raw = particleData.bytes;
        using var br = new BinaryReader(new MemoryStream(raw));
        int magic = br.ReadInt32();
        if (magic != 0x53504C31) throw new Exception($"{particleData.name}: bad magic 0x{magic:X8}");
        int version = br.ReadInt32();
        if (version != 1) throw new Exception($"{particleData.name}: unsupported version {version}");
        int n = br.ReadInt32();
        counts = new int[n];
        starts = new int[n];
        for (int i = 0; i < n; i++) counts[i] = br.ReadInt32();
        int total = 0;
        for (int i = 0; i < n; i++) { starts[i] = total * 7; total += counts[i]; }
        all = new float[total * 7];
        Buffer.BlockCopy(raw, 12 + 4 * n, all, 0, total * 7 * 4);
    }

    public override int FrameCount { get { EnsureLoaded(); return counts.Length; } }
    public override float NativeFps { get { EnsureLoaded(); return fps; } }
    public override float ParticleRadius { get { EnsureLoaded(); return coarseRadius; } }

    public override void GetFrame(int idx, out float[] data, out int offset, out int count)
    {
        EnsureLoaded();
        data = all; offset = starts[idx]; count = counts[idx];
    }
}
