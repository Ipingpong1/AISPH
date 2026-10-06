// FluidSceneMVP.ModelBank.cs — overnight 2026-09-30 (main session, G1): runtime network switcher, key K (M / N are ObstacleSpawner's sphere / box keys).
// Cycles the network between the scene's modelAsset (entry 0) and every ModelAsset under Resources/<modelBankResources>
// (sorted by name). Tonight's bank: the pool fine-tunes of 2026-09-26 (FT-THIN, FT-TC; base 067a) and the PBF FT-RS arms
// of 2026-09-30 (pbf_rs_ft_rs = re-seeded in Unity's own GPU PBF, frozen recipe; pbf_rs_ft_rss = + quasi-static settle).
// Every bank model takes the square 512 V2 input and the union stats of 067a (pbf_live_067a_meta), i.e. the scene's own
// statsJson — so no stats swap is needed. An optional TextAsset "<model name>_stats" ({"mean": [6], "std": [6]}) next to a
// model replaces meta.mean / meta.std while that model is active (for a model trained on other stats).
// The switch runs at the start of the next LateUpdate, after any in-flight spread inference is drained; the fp16 weight
// path follows Init() (quantize, fall back to fp32 on failure). Numbers: SSU_restart/Experiments/Overnight0930/G1_pbf_rs.
using System;
using System.Linq;
using UnityEngine;
using Unity.InferenceEngine;

public partial class FluidSceneMVP
{
    [Header("Model bank (overnight 2026-09-30, key K) — cycles the network")]
    [Tooltip("Resources folder with the switchable ONNX ModelAssets (square 512, V2 splat, 067a union stats) and optional <name>_stats TextAssets. Key K cycles: scene model -> bank models (by name) -> scene model.")]
    public string modelBankResources = "SSU_ModelBank";

    ModelAsset[] bankModels;
    int bankIdx;                 // 0 = the scene's modelAsset
    bool bankPending;
    float[] bankSceneMean, bankSceneStd;
    string bankError;

    public string ActiveModelName => bankModels == null || bankIdx == 0
        ? (modelAsset != null ? modelAsset.name : "") : bankModels[bankIdx].name;

    public void RequestModelCycle() => bankPending = true;

    string ModelBankStatus() => bankModels == null ? "" :
        $"   MODEL [{bankIdx + 1}/{bankModels.Length}] {ActiveModelName}{(bankError != null ? " (" + bankError + ")" : "")}";

    void ApplyPendingModelSwitch()
    {
        if (!bankPending) return;
        bankPending = false;
        if (bankModels == null)
        {
            var extra = Resources.LoadAll<ModelAsset>(modelBankResources).OrderBy(m => m.name).ToArray();
            bankModels = new ModelAsset[extra.Length + 1];
            bankModels[0] = modelAsset;
            Array.Copy(extra, 0, bankModels, 1, extra.Length);
            bankSceneMean = (float[])meta.mean.Clone();
            bankSceneStd = (float[])meta.std.Clone();
            Debug.Log($"model bank: {extra.Length} models under Resources/{modelBankResources}: " +
                      string.Join(", ", extra.Select(m => m.name)));
        }
        if (bankModels.Length < 2) { bankError = "empty bank"; return; }
        int next = (bankIdx + 1) % bankModels.Length;
        try
        {
            DrainPending();
            var model = ModelLoader.Load(bankModels[next]);
            // fp16 only if it worked for the scene model: on Inference Engine 2.6.1 quantized weights fail at SCHEDULE
            // time ("key '126' not present"), which Init() catches and answers with fp32 (fp16Active = false).
            bool fp16 = false;
            if (useFp16 && fp16Active)
            {
                try { ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model); fp16 = true; }
                catch (Exception e) { Debug.LogWarning($"model bank: fp16 quantization failed ({e.Message}) — fp32"); model = ModelLoader.Load(bankModels[next]); }
            }
            worker?.Dispose();
            worker = new Worker(model, backend);
            modelLayerCount = Mathf.Max(1, model.layers.Count);
            if (fp16)
            {
                try { RunModel(); }                            // the same schedule-time check Init() makes
                catch (Exception e)
                {
                    Debug.LogWarning($"model bank: fp16 failed at schedule time ({e.Message}) — fp32");
                    worker.Dispose();
                    model = ModelLoader.Load(bankModels[next]);
                    worker = new Worker(model, backend);
                    fp16 = false;
                }
            }
            fp16Active = fp16;
            bankIdx = next;
            var st = next == 0 ? null : Resources.Load<TextAsset>($"{modelBankResources}/{bankModels[next].name}_stats");
            if (st != null)
            {
                var m = JsonUtility.FromJson<Meta>(st.text);
                meta.mean = m.mean; meta.std = m.std;
            }
            else { meta.mean = (float[])bankSceneMean.Clone(); meta.std = (float[])bankSceneStd.Clone(); }
            bankError = null;
            Debug.Log($"model bank: -> {ActiveModelName} (stats: {(st != null ? st.name : "scene statsJson")}, fp16 {fp16Active})");
        }
        catch (Exception e)
        {
            bankError = e.Message.Length > 60 ? e.Message.Substring(0, 60) : e.Message;
            Debug.LogException(e);
        }
    }
}
