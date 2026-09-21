// StirPath.cs — 067: a deterministic, seeded stand-in for the player's MouseStirrer, in SIM space. A sphere plunges into
// the pool at tIn, sweeps a Lissajous figure around the tank at a bounded speed, and lifts out at tOut, leaving a settle
// tail. Pure function of time so SimExportRunner can evaluate it per CFL substep (GpuSphSolver.beforeSubstep).

using UnityEngine;

public static class StirPath
{
    public struct Params
    {
        public float diameter, yStir, yAbove, cx, cz, ax, az, fx, fz, phx, phz, tIn, tOut, ramp, maxSpeed, domain;
    }

    static float Lerp(System.Random rng, float a, float b) => a + (float)rng.NextDouble() * (b - a);

    public static Params Sample(System.Random rng, float poolDepth, float stopAt, float domain = 3f)
    {
        var p = new Params { domain = domain, ramp = 0.8f };
        p.diameter = Lerp(rng, 0.4f, 0.7f);
        // live StirSphere: diameter 0.6 with its centre at 0.24 over a 0.17 pool -> the sphere sits ON the floor
        p.yStir = Mathf.Max(p.diameter * Lerp(rng, 0.35f, 0.55f), poolDepth * 0.5f);
        p.yAbove = poolDepth + p.diameter * 0.5f + 0.15f;      // parked just clear of the surface: the plunge stays < 1 m/s
        float half = p.diameter * 0.5f + 0.06f;
        p.cx = Lerp(rng, 1.2f, 1.8f); p.cz = Lerp(rng, 1.2f, 1.8f);
        p.ax = Mathf.Min(Lerp(rng, 0.35f, 1.0f), Mathf.Min(p.cx, domain - p.cx) - half);
        p.az = Mathf.Min(Lerp(rng, 0.35f, 1.0f), Mathf.Min(p.cz, domain - p.cz) - half);
        p.fx = Lerp(rng, 0.15f, 0.45f); p.fz = Lerp(rng, 0.15f, 0.45f);
        p.phx = Lerp(rng, 0f, 6.2832f); p.phz = Lerp(rng, 0f, 6.2832f);
        p.maxSpeed = Lerp(rng, 0.4f, 2.5f);                  // mouse stirring is energetic; the live Orbit take was 0.94
        // peak speed of the Lissajous <= 2*pi*sqrt((ax fx)^2 + (az fz)^2); rescale both frequencies to hit maxSpeed
        float peak = 6.2832f * Mathf.Sqrt(p.ax * p.fx * p.ax * p.fx + p.az * p.fz * p.az * p.fz);
        float k = p.maxSpeed / Mathf.Max(peak, 1e-4f);
        p.fx *= k; p.fz *= k;
        p.tIn = Lerp(rng, 0.6f, 1.0f);                        // after the pool's spawn-lattice relaxation
        p.tOut = Mathf.Min(Lerp(rng, 3.0f, 3.5f), stopAt - 1.5f - p.ramp);
        return p;
    }

    /// <summary>Whether the prop should be an active obstacle at time t (it is parked high above the pool otherwise).</summary>
    public static bool Active(in Params p, float t) => t >= p.tIn && t <= p.tOut + p.ramp;

    public static Vector3 Eval(in Params p, float t)
    {
        float x = p.cx + p.ax * Mathf.Sin(6.2832f * p.fx * t + p.phx);
        float z = p.cz + p.az * Mathf.Sin(6.2832f * p.fz * t + p.phz);
        float down = Mathf.SmoothStep(0f, 1f, (t - p.tIn) / p.ramp) * (1f - Mathf.SmoothStep(0f, 1f, (t - p.tOut) / p.ramp));
        float y = Mathf.Lerp(p.yAbove, p.yStir, down);
        float half = p.diameter * 0.5f;
        return new Vector3(Mathf.Clamp(x, half, p.domain - half), y, Mathf.Clamp(z, half, p.domain - half));
    }
}
