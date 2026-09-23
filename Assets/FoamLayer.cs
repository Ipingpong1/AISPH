// FoamLayer.cs — whitewater (spray / foam / bubbles) from the coarse particle frame, in-engine port of the
// 059 probe (SSU_restart/Helpers/foam_probe.py, Ihmsen et al. 2012). CPU, provider-agnostic: it consumes the
// same 7-float records (px py pz vx vy vz density) the splatter consumes, so it works on baked slots and on
// the live GPU-PBF readback alike. 059 showed the COARSE solve's potentials localise the dense solve's
// whitewater (break-phase IoU 0.42, settle 0.59), which is why this is a compositor feature, not a network output.
//
// Per frame (Step):
//   1. uniform hash grid over the fluid particles, cell = h = 4 r_w (the solver's own support);
//   2. per particle: neighbour count, trapped-air potential I_ta = sum |v_ij| (1 - v^_ij . x^_ij) W,
//      outward normal n = sum x^_ij W (colour-field gradient direction), W = 1 - |x_ij|/h;
//      surface = count < 0.75 n_full (n_full = 90th percentile count);
//      wave crest I_wc = sum over surface pairs of (1 - n_i.n_j) W where j lies behind n_i, gated by v^.n >= 0.6;
//      kinetic gate Phi_k = clamp((0.5|v|^2 - 0.5) / (4.5 - 0.5)) [J/kg; 1..3 m/s], physical, same as Python;
//   3. clamp Phi = (I - tau/4) / (tau - tau/4). tau is a RUNNING calibration instead of the probe's fixed
//      percentile over calibration frames: tau = max(0.98 tau, p99.5 of this frame's potential) — the break
//      sets the scale, it decays slowly through the loop; tauScale multiplies it (lower = more foam);
//   4. spawn n = round_stochastic((kTa Phi_ta + kWc Phi_wc) Phi_k dt spawnScale) diffuse particles in the
//      velocity cylinder (radius r_w, height |v| dt), lifetime U(lifeMin, lifeMax) (0.5 + 0.5 Phi_k);
//   5. diffuse update (before spawning, as Python): fluid-neighbour count within h -> spray (< 0.15 n_full,
//      ballistic), bubble (> 0.6 n_full, buoyant k_b 0.5 + drag k_d 0.7 towards the fluid velocity), else foam
//      (carried by the W-weighted fluid velocity, lifetime ticks); culls: life, floor, domain, lost spray.
// Splat: each diffuse particle -> pixel (floor) through the SAME camera as the input splat, depth-tested
// against the panel's fluid depth (visible if in front within 4 r_w, else a hidden bubble at 0.12 weight);
// weights spray 0.6 / foam 1.0 / bubble 0.5; separable Gaussian sigma 1 px; the shader turns density into
// coverage 1 - exp(-k D). CHW row convention (row 0 = top) like every other buffer here.
// Optional frontDepth output (FluidSceneMVP): nearest VISIBLE whitewater depth per pixel (0 = none), min-dilated
// over the blur footprint, so the scene composite can occlude spray that is off the fluid surface.
// Every constant of the probe is a public field below; the defaults are the tested values.
using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using UnityEngine;

public sealed class FoamLayer
{
    // ---- tunables (mirrored from foam_probe.py; spawnScale replaces the probe's mass factor) ----
    public float kTa = 8f, kWc = 12f, spawnScale = 20f, tauScale = 1f;
    public float lifeMin = 1f, lifeMax = 4f;
    public float tauDecay = 0.98f;
    public int maxDiffuse = 30000;
    public Vector3 gravity = new Vector3(0f, -9.81f, 0f);
    public float floorY = -0.05f;
    public Vector3 domainMin = new Vector3(-1f, -0.1f, -1f), domainMax = new Vector3(4f, 4f, 4f);
    // potentials / classification
    public float surfaceFrac = 0.75f, sprayFrac = 0.15f, bubbleFrac = 0.60f;   // fractions of n_full
    public float nFullPercentile = 0.90f;
    public float crestMinVelDotN = 0.6f;              // wave crest only where v^.n >= this
    public float tauPercentile = 0.995f;              // running tau = max(decayed tau, this percentile of the frame)
    public float tauMinFrac = 0.25f;                  // Phi ramps from tauMinFrac*tau to tau
    public float kineticMinSpeed = 1f, kineticMaxSpeed = 3f;   // Phi_k ramps over 0.5 v^2 between these speeds
    // diffuse dynamics
    public float buoyancy = 0.5f, drag = 0.7f;        // bubble k_b, k_d
    public float sprayMaxAge = 3f, orphanMaxAge = 1f; // cull spray older than this / anything with no fluid nearby
    // splat
    public float spriteScale = 1f;                    // multiplier on the per-particle disk footprint (0 = single pixel)
    public float spraySprite = 0.33f, foamSprite = 0.5f;   // footprint radius as a fraction of the projected r_w
    public int maxSpritePx = 3;
    public float weightSpray = 0.6f, weightFoam = 1f, weightBubble = 0.5f;
    public float hiddenWeight = 0.12f;                // weight multiplier for particles behind the fluid surface
    public float depthTolR = 4f;                      // "in front" = within depthTolR * r_w behind the fluid depth
    public bool blur = true;                          // separable Gaussian sigma 1 px on the density
    // Neighbour passes, the diffuse update and the blur run across cores. Every parallel loop writes only
    // its own index/row, and spawning (the only RNG user) stays sequential, so results match single-threaded.
    public bool multithreaded = true;

    // ---- diffuse state (struct-of-arrays, compacted in place) ----
    Vector3[] dPos, dVel; float[] dLife, dAge; byte[] dTyp; int n;
    bool[] dAlive = new bool[0];
    // ---- fluid scratch ----
    // open-addressing hash grid: hKeys[slot] = cell key (or Empty), hHead[slot] = first particle in that cell
    long[] hKeys = new long[0]; int[] hHead = new int[0]; int hMask;
    const long Empty = long.MinValue;
    int[] next = new int[0];
    float[] ITa = new float[0], IWc = new float[0], scratch = new float[0];
    int[] cnt = new int[0];
    Vector3[] nrm = new Vector3[0], fPos = new Vector3[0], fVel = new Vector3[0];
    bool[] surface = new bool[0];
    float tauTa = 0f, tauWc = 0f;
    readonly System.Random rng = new System.Random(0);

    // ---- readouts ----
    public int Alive => n;
    public int Spray, Foam, Bubble, SpawnedTa, SpawnedWc;
    public float NFull;
    public float TauTa => tauTa;
    public float TauWc => tauWc;
    public float LastStepMs, LastSplatMs;

    public FoamLayer() { Allocate(); }

    void Allocate()
    {
        dPos = new Vector3[maxDiffuse]; dVel = new Vector3[maxDiffuse];
        dLife = new float[maxDiffuse]; dAge = new float[maxDiffuse]; dTyp = new byte[maxDiffuse];
        n = 0;
    }

    // cap changed at runtime: keep the live particles (the newest ones if shrinking below n)
    void Resize()
    {
        if (n > maxDiffuse) DropOldest(n - maxDiffuse);
        Array.Resize(ref dPos, maxDiffuse); Array.Resize(ref dVel, maxDiffuse);
        Array.Resize(ref dLife, maxDiffuse); Array.Resize(ref dAge, maxDiffuse); Array.Resize(ref dTyp, maxDiffuse);
    }

    public void Reset() { n = 0; tauTa = 0f; tauWc = 0f; Spray = Foam = Bubble = SpawnedTa = SpawnedWc = 0; }

    // ---------- grid ----------
    static long Key(int x, int y, int z) => ((long)(x + 100000) << 42) ^ ((long)(y + 100000) << 21) ^ (long)(z + 100000);

    static int Slot(long k, int mask) { ulong x = (ulong)k * 0x9E3779B97F4A7C15UL; return (int)(x >> 40) & mask; }

    int Find(long k)
    {
        int s = Slot(k, hMask);
        while (true)
        {
            long kk = hKeys[s];
            if (kk == k) return hHead[s];
            if (kk == Empty) return -1;
            s = (s + 1) & hMask;
        }
    }

    void BuildGrid(int count, float h)
    {
        int size = 1024; while (size < 2 * count) size <<= 1;
        if (hKeys.Length != size) { hKeys = new long[size]; hHead = new int[size]; }
        hMask = size - 1;
        for (int s = 0; s < size; s++) hKeys[s] = Empty;
        if (next.Length < count) next = new int[count];
        float inv = 1f / h;
        for (int i = 0; i < count; i++)
        {
            var p = fPos[i];
            long k = Key((int)Math.Floor(p.x * inv), (int)Math.Floor(p.y * inv), (int)Math.Floor(p.z * inv));
            int s = Slot(k, hMask);
            while (hKeys[s] != Empty && hKeys[s] != k) s = (s + 1) & hMask;
            if (hKeys[s] == Empty) { hKeys[s] = k; hHead[s] = -1; }
            next[i] = hHead[s];
            hHead[s] = i;
        }
    }

    static float Clamp01(float v) => v < 0f ? 0f : (v > 1f ? 1f : v);

    // body(i0, i1) over [0, n) in chunks of `grain`, in parallel when multithreaded
    void ForRange(int n, int grain, Action<int, int> body)
    {
        if (!multithreaded || n < 2 * grain) { body(0, n); return; }
        Parallel.ForEach(Partitioner.Create(0, n, grain), r => body(r.Item1, r.Item2));
    }

    // ---------- fluid potentials + diffuse update + spawn ----------
    public void Step(float[] data, int off, int count, float rW, float dt)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        if (dPos.Length != maxDiffuse) Resize();
        float h = 4f * rW, h2 = h * h, inv = 1f / h;
        if (fPos.Length < count)
        {
            fPos = new Vector3[count]; fVel = new Vector3[count]; ITa = new float[count]; IWc = new float[count];
            cnt = new int[count]; nrm = new Vector3[count]; surface = new bool[count]; scratch = new float[count];
        }
        for (int i = 0; i < count; i++)
        {
            int b = off + i * 7;
            fPos[i] = new Vector3(data[b], data[b + 1], data[b + 2]);
            fVel[i] = new Vector3(data[b + 3], data[b + 4], data[b + 5]);
        }
        BuildGrid(count, h);

        // pass A: counts, trapped air, normals
        ForRange(count, 256, (i0, i1) => {
        for (int i = i0; i < i1; i++)
        {
            Vector3 pi = fPos[i], vi = fVel[i], nn = Vector3.zero;
            int c = 0; float ta = 0f;
            int cx = (int)Math.Floor(pi.x * inv), cy = (int)Math.Floor(pi.y * inv), cz = (int)Math.Floor(pi.z * inv);
            for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
            {
                int j = Find(Key(cx + dx, cy + dy, cz + dz));
                for (; j != -1; j = next[j])
                {
                    if (j == i) continue;
                    Vector3 xij = pi - fPos[j];
                    float d2 = xij.sqrMagnitude;
                    if (d2 >= h2) continue;
                    float d = Mathf.Sqrt(Mathf.Max(d2, 1e-18f));
                    float w = 1f - d * inv;
                    Vector3 xh = xij / d;
                    c++;
                    nn += xh * w;
                    Vector3 vij = vi - fVel[j];
                    float vn = vij.magnitude;
                    if (vn > 1e-9f) ta += vn * (1f - Vector3.Dot(vij / vn, xh)) * w;
                }
            }
            cnt[i] = c; ITa[i] = ta;
            float nm = nn.magnitude; nrm[i] = nm > 1e-9f ? nn / nm : Vector3.up;
        }
        });
        // n_full = 90th percentile of counts
        for (int i = 0; i < count; i++) scratch[i] = cnt[i];
        NFull = Mathf.Max(Percentile(scratch, count, nFullPercentile), 4f);
        for (int i = 0; i < count; i++) surface[i] = cnt[i] < surfaceFrac * NFull;

        // pass B: wave crest on surface particles moving along their normal
        ForRange(count, 256, (i0, i1) => {
        for (int i = i0; i < i1; i++)
        {
            IWc[i] = 0f;
            if (!surface[i]) continue;
            Vector3 vi = fVel[i]; float vm = vi.magnitude;
            if (vm < 1e-6f || Vector3.Dot(vi / vm, nrm[i]) < crestMinVelDotN) continue;
            Vector3 pi = fPos[i], ni = nrm[i]; float wc = 0f;
            int cx = (int)Math.Floor(pi.x * inv), cy = (int)Math.Floor(pi.y * inv), cz = (int)Math.Floor(pi.z * inv);
            for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
            {
                int j = Find(Key(cx + dx, cy + dy, cz + dz));
                for (; j != -1; j = next[j])
                {
                    if (j == i || !surface[j]) continue;
                    Vector3 xij = pi - fPos[j];
                    float d2 = xij.sqrMagnitude;
                    if (d2 >= h2) continue;
                    float d = Mathf.Sqrt(Mathf.Max(d2, 1e-18f));
                    if (Vector3.Dot(xij / d, ni) <= 0f) continue;            // j must lie behind i's normal (convex)
                    wc += (1f - Vector3.Dot(ni, nrm[j])) * (1f - d * inv);
                }
            }
            IWc[i] = wc;
        }
        });

        // running calibration of tau (99.5th percentile, slow decay)
        for (int i = 0; i < count; i++) scratch[i] = ITa[i];
        float pTa = Percentile(scratch, count, tauPercentile);
        tauTa = Mathf.Max(tauTa * tauDecay, pTa);
        int m = 0; for (int i = 0; i < count; i++) if (IWc[i] > 0f) scratch[m++] = IWc[i];
        if (m >= 20) tauWc = Mathf.Max(tauWc * tauDecay, Percentile(scratch, m, tauPercentile));
        else tauWc *= tauDecay;

        // diffuse update in this frame's field (before spawning, as Python)
        if (dt > 0f) UpdateDiffuse(count, h, h2, inv, dt);

        // spawn
        SpawnedTa = SpawnedWc = 0;
        if (dt > 0f && tauTa > 1e-9f)
        {
            float tTaMax = tauTa * tauScale, tTaMin = tauMinFrac * tTaMax;
            float tWcMax = tauWc * tauScale, tWcMin = tauMinFrac * tWcMax;
            float kMin = 0.5f * kineticMinSpeed * kineticMinSpeed, kMax = 0.5f * kineticMaxSpeed * kineticMaxSpeed;
            float kInv = 1f / Mathf.Max(kMax - kMin, 1e-6f);
            for (int i = 0; i < count; i++)
            {
                float ek = 0.5f * fVel[i].sqrMagnitude;
                float phiK = Clamp01((ek - kMin) * kInv);
                if (phiK <= 0f) continue;
                float phiTa = Clamp01((ITa[i] - tTaMin) / Mathf.Max(tTaMax - tTaMin, 1e-12f));
                float phiWc = tWcMax > 1e-9f ? Clamp01((IWc[i] - tWcMin) / Mathf.Max(tWcMax - tWcMin, 1e-12f)) : 0f;
                float xTa = kTa * phiTa * phiK * dt * spawnScale, xWc = kWc * phiWc * phiK * dt * spawnScale;
                int nTa = StochasticRound(xTa), nWc = StochasticRound(xWc);
                if (nTa + nWc == 0) continue;
                SpawnAt(fPos[i], fVel[i], nTa + nWc, rW, phiK, dt);
                SpawnedTa += nTa; SpawnedWc += nWc;
            }
        }
        Spray = Foam = Bubble = 0;
        for (int p = 0; p < n; p++) { if (dTyp[p] == 0) Spray++; else if (dTyp[p] == 1) Foam++; else Bubble++; }
        sw.Stop(); LastStepMs = (float)sw.Elapsed.TotalMilliseconds;
    }

    int StochasticRound(float x)
    {
        int f = (int)Math.Floor(x);
        return f + (rng.NextDouble() < x - f ? 1 : 0);
    }

    void SpawnAt(Vector3 p0, Vector3 v, int m, float r, float phiK, float dt)
    {
        float vn = v.magnitude;
        Vector3 vh = vn > 1e-9f ? v / vn : Vector3.up;
        Vector3 a = Mathf.Abs(vh.x) < 0.9f ? Vector3.right : Vector3.up;
        Vector3 e1 = Vector3.Cross(vh, a); e1 /= Mathf.Max(e1.magnitude, 1e-9f);
        Vector3 e2 = Vector3.Cross(vh, e1);
        for (int k = 0; k < m; k++)
        {
            if (n >= maxDiffuse) DropOldest(maxDiffuse / 10);
            float rr = r * Mathf.Sqrt((float)rng.NextDouble()), th = 2f * Mathf.PI * (float)rng.NextDouble();
            float hh = (float)rng.NextDouble() * dt;
            dPos[n] = p0 + rr * (Mathf.Cos(th) * e1 + Mathf.Sin(th) * e2) + hh * v;
            dVel[n] = v;
            dLife[n] = lifeMin + (lifeMax - lifeMin) * (float)rng.NextDouble() * (0.5f + 0.5f * phiK);
            dAge[n] = 0f; dTyp[n] = 1;
            n++;
        }
    }

    void DropOldest(int k)
    {
        // remove the k oldest particles (age threshold via a sorted copy; rare event)
        if (k <= 0 || n == 0) return;
        if (scratch.Length < n) scratch = new float[n];
        Array.Copy(dAge, scratch, n);
        Array.Sort(scratch, 0, n);
        float thr = scratch[Math.Max(n - k, 0)];
        int w = 0;
        for (int p = 0; p < n; p++)
        {
            if (dAge[p] >= thr) continue;
            if (w != p) { dPos[w] = dPos[p]; dVel[w] = dVel[p]; dLife[w] = dLife[p]; dAge[w] = dAge[p]; dTyp[w] = dTyp[p]; }
            w++;
        }
        n = w;
    }

    void UpdateDiffuse(int count, float h, float h2, float inv, float dt)
    {
        // advance every particle in place (parallel: each writes only its own slot), then compact the survivors
        if (dAlive.Length < n) dAlive = new bool[dPos.Length];
        ForRange(n, 512, (p0, p1) => {
        for (int p = p0; p < p1; p++)
        {
            Vector3 x = dPos[p];
            int cx = (int)Math.Floor(x.x * inv), cy = (int)Math.Floor(x.y * inv), cz = (int)Math.Floor(x.z * inv);
            int c = 0; float wsum = 0f; Vector3 vf = Vector3.zero;
            for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
            {
                int j = Find(Key(cx + dx, cy + dy, cz + dz));
                for (; j != -1; j = next[j])
                {
                    float d2 = (x - fPos[j]).sqrMagnitude;
                    if (d2 >= h2) continue;
                    c++;
                    float ww = 1f - Mathf.Sqrt(d2) * inv;
                    wsum += ww; vf += fVel[j] * ww;
                }
            }
            bool hasF = wsum > 0f;
            if (hasF) vf /= wsum;
            byte typ = c < sprayFrac * NFull ? (byte)0 : (c > bubbleFrac * NFull ? (byte)2 : (byte)1);
            if (!hasF) typ = 0;
            Vector3 v = dVel[p];
            if (typ == 0) v += gravity * dt;
            else if (typ == 1) v = vf;
            else v += (-buoyancy * gravity) * dt + drag * (vf - v);
            x += v * dt;
            float life = dLife[p] - (typ == 1 ? dt : 0f);
            float age = dAge[p] + dt;
            dAlive[p] = life > 0f && x.y > floorY
                        && x.x > domainMin.x && x.y > domainMin.y && x.z > domainMin.z
                        && x.x < domainMax.x && x.y < domainMax.y && x.z < domainMax.z
                        && !(typ == 0 && age > sprayMaxAge) && !(!hasF && age > orphanMaxAge);
            dPos[p] = x; dVel[p] = v; dLife[p] = life; dAge[p] = age; dTyp[p] = typ;
        }
        });
        int w = 0;
        for (int p = 0; p < n; p++)
        {
            if (!dAlive[p]) continue;
            if (w != p) { dPos[w] = dPos[p]; dVel[w] = dVel[p]; dLife[w] = dLife[p]; dAge[w] = dAge[p]; dTyp[w] = dTyp[p]; }
            w++;
        }
        n = w;
    }

    static float Percentile(float[] a, int count, float q)
    {
        if (count == 0) return 0f;
        Array.Sort(a, 0, count);
        int k = Mathf.Clamp((int)Math.Round(q * (count - 1)), 0, count - 1);
        return a[k];
    }

    // ---------- screen-space density (CHW, row 0 = top) ----------
    // fluidDepth: per-pixel front depth in world units (0 = background) of the panel this layer is drawn over.
    // frontDepth (optional, same layout): receives the nearest visible whitewater depth, 0 = none.
    // aspect: window W/H for a non-square window (SplatV2's projection); 1 = the square window.
    public void Splat(Vector3 eye, Vector3 right, Vector3 up, Vector3 fwd, float focal, int H, int W,
                      float[] fluidDepth, float rW, float[] density, float[] tmp,
                      float[] frontDepth = null, float aspect = 1f)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        Array.Clear(density, 0, H * W);
        if (frontDepth != null) Array.Clear(frontDepth, 0, H * W);
        float tol = depthTolR * rW;
        int maxR = Math.Max(maxSpritePx, 0);
        for (int p = 0; p < n; p++)
        {
            Vector3 r = dPos[p] - eye;
            float cx = Vector3.Dot(r, right), cy = Vector3.Dot(r, up), cz = -Vector3.Dot(r, fwd);
            float depth = -cz;
            if (!(depth > 1e-3f)) continue;
            int px = (int)Math.Floor((focal * cx / depth / aspect + 1f) * 0.5f * W);
            int py = (int)Math.Floor((1f - focal * cy / depth) * 0.5f * H);
            if (px < 0 || px >= W || py < 0 || py >= H) continue;
            int i = py * W + px;
            float fd = fluidDepth[i];
            bool front = fd <= 0f || depth <= fd + tol;
            float wt = dTyp[p] == 0 ? weightSpray : (dTyp[p] == 1 ? weightFoam : weightBubble);
            wt = front ? wt : hiddenWeight * wt;
            bool writeDepth = front && frontDepth != null;
            // footprint: a disk of half the projected coarse radius (spray a third), 0..maxSpritePx; weight spread over it
            float rp = (dTyp[p] == 0 ? spraySprite : foamSprite) * rW * focal * (H * 0.5f) / depth * spriteScale;
            int R = rp < 0.75f ? 0 : (rp > maxR ? maxR : (int)Math.Round(rp));
            if (R == 0)
            {
                density[i] += wt;
                if (writeDepth && (frontDepth[i] <= 0f || depth < frontDepth[i])) frontDepth[i] = depth;
                continue;
            }
            int R2 = R * R, area = 0;
            for (int oy = -R; oy <= R; oy++) for (int ox = -R; ox <= R; ox++) if (ox * ox + oy * oy <= R2) area++;
            float wa = wt / area;
            for (int oy = -R; oy <= R; oy++)
            {
                int ty = py + oy; if (ty < 0 || ty >= H) continue;
                for (int ox = -R; ox <= R; ox++)
                {
                    int tx = px + ox; if (tx < 0 || tx >= W || ox * ox + oy * oy > R2) continue;
                    int t = ty * W + tx;
                    density[t] += wa;
                    if (writeDepth && (frontDepth[t] <= 0f || depth < frontDepth[t])) frontDepth[t] = depth;
                }
            }
        }
        if (!blur)
        {
            sw.Stop(); LastSplatMs = (float)sw.Elapsed.TotalMilliseconds;
            return;
        }
        // separable Gaussian sigma 1 px (5 taps)
        float k0 = 0.40262f, k1 = 0.24420f, k2 = 0.05449f;
        ForRange(H, 16, (y0, y1) => {
        for (int y = y0; y < y1; y++)
        {
            int row = y * W;
            for (int x = 0; x < W; x++)
            {
                float s = density[row + x] * k0;
                if (x >= 1) s += density[row + x - 1] * k1;
                if (x + 1 < W) s += density[row + x + 1] * k1;
                if (x >= 2) s += density[row + x - 2] * k2;
                if (x + 2 < W) s += density[row + x + 2] * k2;
                tmp[row + x] = s;
            }
        }
        });
        ForRange(H, 16, (y0, y1) => {
        for (int y = y0; y < y1; y++)
            for (int x = 0; x < W; x++)
            {
                float s = tmp[y * W + x] * k0;
                if (y >= 1) s += tmp[(y - 1) * W + x] * k1;
                if (y + 1 < H) s += tmp[(y + 1) * W + x] * k1;
                if (y >= 2) s += tmp[(y - 2) * W + x] * k2;
                if (y + 2 < H) s += tmp[(y + 2) * W + x] * k2;
                density[y * W + x] = s;
            }
        });
        // the blur spreads density 2 px past the footprint: give that halo a depth too (min over 5x5, 0 = none)
        if (frontDepth != null) DilateMin(frontDepth, tmp, H, W, 2);
        sw.Stop(); LastSplatMs = (float)sw.Elapsed.TotalMilliseconds;
    }

    void DilateMin(float[] d, float[] tmp, int H, int W, int R)
    {
        ForRange(H, 16, (y0, y1) => {
        for (int y = y0; y < y1; y++)
        {
            int row = y * W;
            for (int x = 0; x < W; x++)
            {
                float m = 0f;
                for (int o = Math.Max(x - R, 0), e = Math.Min(x + R, W - 1); o <= e; o++)
                {
                    float v = d[row + o];
                    if (v > 0f && (m <= 0f || v < m)) m = v;
                }
                tmp[row + x] = m;
            }
        }
        });
        ForRange(H, 16, (y0, y1) => {
        for (int y = y0; y < y1; y++)
            for (int x = 0; x < W; x++)
            {
                float m = 0f;
                for (int o = Math.Max(y - R, 0), e = Math.Min(y + R, H - 1); o <= e; o++)
                {
                    float v = tmp[o * W + x];
                    if (v > 0f && (m <= 0f || v < m)) m = v;
                }
                d[y * W + x] = m;
            }
        });
    }
}
