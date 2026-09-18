// ParticleFrameProvider.cs — the particle-source seam for FluidSceneMVP.
//
// A provider hands FluidSceneMVP frames of SPH particles as interleaved records of 7 floats:
//   px py pz vx vy vz density   (sim space: right-handed, y-up, solver world units)
// Any solver — baked playback, a live CPU SPH, a GPU solver with readback — plugs into the
// scene renderer by implementing this component and assigning it to FluidSceneMVP.provider.
//
// Contract notes for live solvers:
//  - Units should match the training distribution (dam-break scale: domain a few units across,
//    speeds O(1) units/s, densities O(1000)). The upsampler normalizes with fixed training
//    stats, so a solver with wildly different density/velocity units is out of distribution.
//    Scene size is still free — FluidSceneMVP's anchor transform scales sim units to world.
//  - GetFrame may be called more than once with the same idx in a frame; re-serving must be
//    cheap (hand back a cached array, don't recompute).
//  - Live sources that only expose "the current state" should return FrameCount = 1 and
//    ignore idx.

using UnityEngine;

public abstract class ParticleFrameProvider : MonoBehaviour
{
    // Number of frames available (1 for live "current state only" sources).
    public abstract int FrameCount { get; }

    // Native frames/sec of the source data (playback pacing; live solvers: your sim rate).
    public abstract float NativeFps { get; }

    // Interleaved records of 7 floats: px py pz vx vy vz density. offset is a float index
    // into data; the frame spans count records (count*7 floats) from there.
    public abstract void GetFrame(int idx, out float[] data, out int offset, out int count);

    // Called by FluidSceneMVP once per rendered frame with the elapsed fluid time (0 while
    // paused; fixed-step during deterministic capture). Live solvers advance here so sim and
    // render stay in lockstep; playback providers can ignore it.
    public virtual void Tick(float dt) { }

    // Called when the user restarts (R). Live solvers should reset their state.
    public virtual void ResetSim() { }

    // Solver particle radius in sim units, or 0 if unknown.
    //
    // Needed because the splat pass accumulates THICKNESS as a raw sum of kernel weights, which is
    // therefore proportional to particle COUNT. Measured on the 023 ratio-probe roots (same physical
    // frames rendered at subsample factor 12 / 25 / 50): foreground thickness sum scales as 1/factor
    // to within 1% (2.079 vs the ideal 2.083, and 0.500 vs 0.500). Since count for a fixed fluid
    // volume goes as 1/r^3, a solver whose radius differs from the training LR point set hands the
    // model an out-of-distribution thickness channel. Reporting the radius here lets FluidSceneMVP
    // normalize it away — see its Thickness Count Normalize option. Depth (min-reduce) and
    // velocity/density (weight-normalized means) are count-invariant by construction; thickness is
    // the only brittle channel, exactly as 023 found.
    public virtual float ParticleRadius => 0f;
}
