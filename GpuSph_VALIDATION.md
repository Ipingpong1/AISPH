# GpuSph validation checklist — run in the next Unity session

> **2026-08-17 status (mechanical items done via MCP; eyeball items remain yours):**
> - §1 DONE: `.compute` compiles, all 10 kernels resolve, zero shader messages.
> - §3 DONE (numbers): 50-frame coarse dam block, GPU vs C# oracle — COM within 0.005 on all
>   axes, KE within 2.3%, density mean within 2%, zero NaNs. **Eyeball side-by-side still owed.**
> - §5 DONE, with a finding: at dense-GT radius (0.0089), `maxSubsteps 8` violates CFL and the
>   solve gains unphysical KE. **`maxSubsteps 32` fixes it** (262k particles, stable, ρ≈700,
>   ~1s sim in ~2s wall even while a trainer held the GPU). Coarse class is fine at 8.
>   Set maxSubsteps ≥32 on any dense-class SimExportRunner batch.
> - §6 SMOKE DONE: SimExportRunner ran a real 2-sim randomized batch (obstacles included) →
>   1LPS dumps → `unity_sim_to_bgeo.py` → `Simulations/GpuPbfSmoke/` with valid manifest;
>   frames render correctly through the training splat (`Experiments/gpupbf_smoke_render.png`).
> - §2 (your eyeball), §4 (obstacle *visual* check), idle-GPU timing, and the real data-gen
>   batch remain. The scene was never saved; the runner GameObject was removed.
>
> **2026-08-20 (obstacles for the live demo):** solver gained a **torus** primitive and three
> prop **motion modes** (Orbit/Sweep/Plunge, all solver-time driven so capture stays
> deterministic; the legacy `orbit` bool still maps to Motion.Orbit, verified bit-identical, so
> the serialized StirSphere keeps working). §4 split: **4a numeric is DONE and found a real bug
> — obstacles were a silent no-op on the GPU** (transposed transforms); **4b visual is still
> owed** and only now meaningful. Full write-up in §4.

> **2026-09-03:** §2 (live smoke) DONE — `GpuSphProvider` runs live in `FluidLiveScene` at 2.7k-8.2k particles, 0.2-0.6 ms/step idle, ~7.5 ms in splash frames, no NaN/explosion over ~60 s with drops + a fast-moving sphere. §4b partially: box + sphere spawned at runtime sit in the pool and the fluid flows around them (screenshots `Captures/livedemo_2026-09-03/`); torus, resting-on-top-face and toggle-off cases still unviewed. Scene SAVED this time.

Built 2026-08-15 while the 051 queue owned the GPU. Everything below needs the editor (or a
free GPU). The Python bridge is ALREADY validated (3/3 gates, byte-exact — see
`ScreenSpaceUpsampling/unity_sim_to_bgeo.py`); nothing here re-tests it.

Suggested first step: `git init` this project (RESTART.md flags it; backups currently go to
`/media/matias/T9/SSU/Backup/`).

## 1. Compile the compute shader (5 min)
Open the editor (or MCP `refresh_unity`), check the Console for `GpuSph.compute` errors.
It is plain SM5 HLSL, no Unity macros; the known risky bits (macro neighbor loop, interlocked
ops on structured buffers, cbuffer packing) are conventional but unproven on this toolchain.
C# side already compile-checked via the Roslyn path (0 errors, 2026-08-15).

## 2. Smoke the solver (15 min)
Add `GpuSphProvider` to a disabled duplicate of the FluidLiveScene object, assign
`GpuSph.compute`, leave defaults (they replicate LiveSphProvider's validated coarse dam
break). Assign it as `FluidSceneMVP.provider`. Play:
- fluid falls, settles, no explosion, no NaN freeze (watch `ActiveParticles`, status line)
- F drops a second block; R resets
- compare side-by-side with LiveSphProvider on the raw view (V) — should look like the same
  fluid, not the same trajectory (float ordering differs; bitwise parity is impossible)

## 3. Solver parity vs the C# oracle (30 min)
Same seed scene (default block, no obstacles), fixed-step capture path so both are
deterministic-ish. Over the first ~100 frames compare:
- center-of-mass trajectory (bounded drift, no systematic bias)
- kinetic energy curve (same settle envelope)
- density histogram at rest vs restDensity (same mode, similar spread)
- neighbor-starved edge cases: single particle, two particles at contact
Known deliberate differences (recorded in GpuSph.compute header): no 96-neighbor cap,
iteration order, gravity uniform. If drift is unbounded or the settle height differs
visibly, suspect the grid (hash constants, cell size) first.

## 4. Obstacle check

### 4a. Numeric — DONE 2026-08-20, and it found a REAL BUG (see below)
`Assets/GpuSph/GpuSphObstacleGate.cs` — run it headlessly (no Play mode, no fluid):
MCP `execute_code` → add `GpuSphObstacleGate` to a temp GameObject, assign `GpuSph.compute`,
call `Run()`, which returns the report string. Current result:

```
  Sphere contract PASS: inside 129/4096, surface worst 1.49E-007, idempotence 1.03E-007, outside moved 0, singular ok
  Box    contract PASS: inside 123/4096, surface worst 4.77E-007, idempotence 2.46E-007, outside moved 0, singular ok
  Torus  contract PASS: inside 110/4096, surface worst 2.83E-007, idempotence 2.73E-007, outside moved 0, singular ok
  GPU parity PASS: 12288 points, worst |C#-GPU| 6.66E-007 (tol 1.00E-004)
```

**THE BUG IT FOUND: obstacles were a complete no-op on the GPU.** `ObstacleGpu` stored the
transforms as `float4x4` and `SetObstacle` uploaded them `.transpose`d, on the belief (written
in both files) that HLSL reads a structured-buffer `float4x4` as consecutive rows. It does not —
matrix packing in a structured buffer defaults to column-major, which is also Unity's
`Matrix4x4` memory layout, so the transpose made every obstacle transform wrong. Measured on a
rotated/scaled sphere: the GPU moved **0** of the 267 particles the C# oracle moved. Silent — no
error, no NaN, the fluid just sailed through props. Fixed by storing EXPLICIT ROWS (`float4 t0..t3`,
`f0..f3`) and multiplying with dot products, which has one meaning on every backend; stride is
unchanged at 144 B.

Why it survived since 2026-08-15: §3 parity was run **without obstacles**, and §4 (the visual
check) was never run. The §6 smoke batch *had* obstacles but was only checked for "renders
correctly through the training splat", which a no-op obstacle passes. **No training data is
affected** — every `SimExport/GpuPbfV1` scene JSON has `"obstacles": []`, so the 053 corpus never
used them. `SimExport/GpuPbfSmoke` (2 sims) DOES have obstacles and its fluid ignored them:
treat that root as invalid for anything obstacle-related.

Also fixed at the same time: `ClearObstacles()` left `active = 1` on cleared slots while
`numObstacles` is a high-water mark, so a prop toggled off mid-scene kept colliding (ghost
collider); and `SimExportRunner` looped `o < obstacles.Count` with no `MaxObstacles` clamp.

Regression on the fluid path (the obstacle refactor touches `ApplyDelta`): 50-frame coarse dam
block, zero obstacles, 4 runs before vs 4 runs after — every delta inside the solver's own
run-to-run spread (this solver is not deterministic run-to-run; the hash grid is built with
`InterlockedExchange`). COM max |Δ| 0.0017 vs the §3 criterion 0.005; KE Δ 0.45 % against a
4.7–8.1 % natural spread; density Δ 0.3 %.

### 4b. Visual — STILL OWED (your eyeball, ~15 min)
Now worth doing for the first time, because until 2026-08-20 it would have shown nothing.
On a `GpuSphProvider` object (not `LiveSphProvider`), enable PropCube + StirSphere:
- particles flow **around** the box, the orbiting sphere **stirs**
- fluid resting ON the box top face (exercises the sign-parity case in the box branch)
- a **torus** in the flow: fluid must pass **through the hole**, not over a solid disc — the one
  failure mode no numeric gate catches as vividly
- toggle one prop off mid-play: the fluid must stop colliding with it (ghost-collider regression)
- **save the scene this time** (the 2026-08-17 session left it unsaved and deleted the runner)

## 5. Scale test (10 min)
maxParticles 262144, blockCount 64³ (needs blockMin near origin), radius 0.0089 (dense-GT
class). Watch `LastStepMs` — dispatch+readback per solver frame. If the 4-byte vmax sync
readback dominates, that's the known v1 cost (see GpuSphSolver.Step) — switch CFL to a
fixed-substep policy for data-gen runs before optimizing anything else.

## 6. First data-gen run (after the 051 queue drains — GPU-heavy)
- `SimExportRunner` on an empty GameObject: rootName `GpuPbfSmoke`, simCount 2, defaults
  (coarse class). runOnPlay, Play, wait for the Console "batch done" line.
- Bridge: `/home/matias/anaconda3/envs/SSU/bin/python unity_sim_to_bgeo.py
  --dump_dir ~/AISPH/SimExport/GpuPbfSmoke --out Simulations/GpuPbfSmoke`
- Gates on the root: eyeball a few bgeo frames via `splatRender`/`lowres_probe_render.py`;
  check density channel stats are ~in-distribution (mean ~774/std ~211 after splat is NOT
  expected to match exactly — PBF at coarse res is a different distribution; that
  measurement IS the solver-family-gap experiment).
- Then the real decision point: the bridge experiment (small PBF root -> existing chain ->
  probe train). Needs a stats decision (own-root vs frozen v3 stats) — flag for Matias.

## 7. Live-path latency note
The live seam still reads back to CPU and re-splats on CPU (unchanged from LiveSphProvider).
The GPU splat (FUTURE_PLAN.md contract; buffers already exposed on `GpuSphProvider.Solver`)
is the next infrastructure piece, and it is also what makes `FluidSceneMVP`'s
never-parity-checked splat replaceable by one that IS gated by `compare_unity_splat.py`.
