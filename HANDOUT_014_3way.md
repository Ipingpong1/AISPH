# HANDOUT — wire up model 014, produce the 3-way render + real in-engine timings

*From the training-repo session (`~/Desktop/ScreenSpaceUpsampling`), 2026-07-02. Everything you
need is already copied into `Assets/`. This file is the task spec + report-back format.*

## Context (2 lines)
**014 is the new trunk model** — it replaced the decoder (ConvTranspose → nearest-Upsample+3×3Conv)
and KILLED the period-4px striping at the source (spectral index 116 → 1.1 = GT level, judged by a
frozen instrument). It supersedes `model_010.onnx` everywhere. Known cosmetic remnant: a faint
isotropic 2-4px "squiggle" texture (loss-side issue, being fixed by runs 015a/b tonight — decoder
and export path are final).

## What's new in `Assets/`
| File | What |
|---|---|
| `model_014.onnx` | the new trunk (36.4M params, 145.6 MB fp32, static 1×7×512×512, opset 17; ONNX↔PyTorch parity **8.8e-05** on a real held-out frame) |
| `lr_sim30_f20.bytes` / `gt_sim30_f20.bytes` / `meta_sim30_f20.json` | a SECOND baked frame: in-dist val, the "squiggle frame" we've been diagnosing — same 7×512×512 CHW layout as the sim46 pair |
| (already present) `lr/gt_sim46_f20.bytes` | held-out test_content frame — keep using it as the primary |
| (already present) `pred_sim46_f20.bytes` | STALE — that's a 010 prediction bake. Don't use it for comparisons; the worker computes 014's pred live. Re-bake from 014 later if anything in `SSFR/` depends on it. |

## Mission
1. **3-way render with 014** — `LR | NEURAL(014) | GT` on BOTH frames (sim46_f20 primary,
   sim30_f20 secondary).
2. **Proper in-engine timings** on `GPUCompute` — pipelined AND sync+readback, fp32 and fp16.
3. **Report the numbers back** in the format at the bottom.

## Steps
1. Select the FluidUpsamplerMVP GameObject → drag **`model_014.onnx`** into the Model Asset slot
   (no code change needed for the swap). Play. Record both timings from the label/Console.
   `benchmarkIters=50` is fine; do one throwaway Play first (shader compile pollutes the first run).
2. Duplicate the GameObject (or scene) → assign `lr_sim30_f20.bytes` / `gt_sim30_f20.bytes` →
   Play → screenshot. (Enable only one at a time — two workers = contended timings.) In the pred
   panel, the squiggles are a faint wavy texture in the depth interior — subtle in raw depth,
   normal-amplified once shaded; don't be surprised if it's hard to see in grayscale.
3. **fp16 pass**: quantize weights at load time, e.g.
   ```csharp
   var model = ModelLoader.Load(modelAsset);
   ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model);
   worker = new Worker(model, backend);
   ```
   (`Unity.InferenceEngine` namespace, package 2.6.1 — if the API moved, check the Inference
   Engine docs for the current quantization entry point.) Note: this quantizes WEIGHTS; the
   backend may still compute in fp32 — whatever you measure is the honest in-engine fp16 number,
   even if it beats/misses the PyTorch proxy below. Verify the pred panel still looks right after
   quantization (a visibly broken panel = quantization artifact, report it).
4. Screenshots of both 3-way renders (Game view, timings label visible) → save into the project,
   note the paths in the report.

## Reference numbers to compare against (PyTorch CUDA proxy, THIS machine, measured 2026-07-02)
| model | fp32 median @512² | fp16 median @512² | vs 5 ms budget |
|---|---|---|---|
| **014** (resize-conv, 36.4M) | **30.7 ms** | **17.5 ms** | 6.1× / 3.5× over |
| 010 (ConvTranspose, 32.7M) | 23.4 ms | 13.1 ms | 4.7× / 2.6× over |

The in-engine pipelined number should land in the same ballpark as the fp32 proxy (±2×). If you
see 100×+ (hundreds of ms), the backend silently fell back to CPU — check Player Settings →
Graphics API = **Vulkan** on Linux, and that the worker really is `GPUCompute`. Note the resize-conv
decoder costs **+31% latency** vs 010 — that's the price of the artifact-free surface; the
compression/distillation plan (later) keeps the resize-conv design but shrinks the net.

## Constraints
- **Do NOT benchmark while training is running on this GPU.** The 015 pair launches tonight and
  holds the 12GB card for ~8.5h (`systemctl --user status ssu-015` to check from a terminal, or
  `nvidia-smi`). Contended numbers are garbage numbers. Benchmark before it launches or after it
  finishes.
- Canonical copies of `FluidUpsamplerMVP.cs` + assets live in the training repo's `UnityMVP/`
  (`~/Desktop/ScreenSpaceUpsampling/UnityMVP/`). If you edit the .cs here, say so in the report
  so the canonical copy gets synced.

## Report-back format (paste this filled-in)
```
GPU: <name>            package: com.unity.ai.inference <version>   graphics API: <Vulkan/...>
model_014.onnx import: <clean / warnings: ...>

                     pipelined ms/frame    sync+readback ms/frame
fp32  GPUCompute:    ____                  ____
fp16  GPUCompute:    ____                  ____   (quantization API used: ____)

3-way screenshots: <paths>   sim46 pred panel: <looks correct? y/n>   sim30: <y/n>
Anything odd: ____
```
