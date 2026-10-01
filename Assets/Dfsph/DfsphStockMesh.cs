// DfsphStockMesh.cs — DFSPH1001: the corpus' stock meshes (UnitBox.obj, sphere.obj, torus.obj from the pySPlisHSPlasH data
// dir, copied to Resources/StockMeshes/*.txt) and Discregrid's TriangleMeshDistance on them, in double, so a volume map can
// be built on EXACTLY the tessellated geometry SPlisHSPlasH used (field 0), instead of the analytic primitive.
//   import: OBJLoader (stof(text) * scale in float), faces 1-based -> 0-based;
//   sign: angle-weighted vertex / edge (sum of the two face normals) / face pseudonormal of the nearest entity,
//   distance: Eberly point-triangle regions (Discregrid point_triangle_sq_unsigned), brute force over all triangles.
using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;

public class DfsphStockMesh
{
    public readonly double[] vx, vy, vz;                 // scaled vertices (float product, then double)
    public readonly int[] tri;                             // 3 per triangle
    readonly double[] fn;                                  // face pseudonormals, 3 per triangle
    readonly double[] en;                                  // edge pseudonormals, 9 per triangle (E01, E12, E02)
    readonly double[] vn;                                  // vertex pseudonormals, 3 per vertex
    public readonly double[] aabbMin = new double[3], aabbMax = new double[3];

    static readonly Dictionary<string, (List<float[]> v, List<int> f)> cache = new Dictionary<string, (List<float[]>, List<int>)>();

    public static string ResourceName(DfsphSolver.Shape s) => s == DfsphSolver.Shape.Box ? "UnitBox" : s == DfsphSolver.Shape.Sphere ? "sphere" : "torus";

    static (List<float[]> v, List<int> f) LoadObj(string name)
    {
        if (cache.TryGetValue(name, out var c)) return c;
        var ta = Resources.Load<TextAsset>("StockMeshes/" + name);
        if (ta == null) throw new Exception($"stock mesh Resources/StockMeshes/{name}.txt missing");
        var v = new List<float[]>(); var f = new List<int>();
        foreach (var raw in ta.text.Split('\n'))
        {
            var line = raw.Trim();
            if (line.StartsWith("v "))
            {
                var p = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                v.Add(new[] { float.Parse(p[1], CultureInfo.InvariantCulture), float.Parse(p[2], CultureInfo.InvariantCulture), float.Parse(p[3], CultureInfo.InvariantCulture) });
            }
            else if (line.StartsWith("f "))
            {
                var p = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (p.Length != 4) throw new Exception($"{name}: only triangles are supported");
                for (int k = 1; k <= 3; k++) f.Add(int.Parse(p[k].Split('/')[0], CultureInfo.InvariantCulture) - 1);
            }
        }
        c = (v, f);
        cache[name] = c;
        return c;
    }

    public DfsphStockMesh(DfsphSolver.Shape shape, Vector3 scale)
    {
        var (v, f) = LoadObj(ResourceName(shape));
        int nv = v.Count, nt = f.Count / 3;
        vx = new double[nv]; vy = new double[nv]; vz = new double[nv];
        for (int i = 0; i < nv; i++)
        {
            vx[i] = (double)(v[i][0] * scale.x); vy[i] = (double)(v[i][1] * scale.y); vz[i] = (double)(v[i][2] * scale.z);
            if (i == 0) { aabbMin[0] = aabbMax[0] = vx[i]; aabbMin[1] = aabbMax[1] = vy[i]; aabbMin[2] = aabbMax[2] = vz[i]; }
            aabbMin[0] = Math.Min(aabbMin[0], vx[i]); aabbMax[0] = Math.Max(aabbMax[0], vx[i]);
            aabbMin[1] = Math.Min(aabbMin[1], vy[i]); aabbMax[1] = Math.Max(aabbMax[1], vy[i]);
            aabbMin[2] = Math.Min(aabbMin[2], vz[i]); aabbMax[2] = Math.Max(aabbMax[2], vz[i]);
        }
        tri = f.ToArray();
        fn = new double[3 * nt]; en = new double[9 * nt]; vn = new double[3 * nv];
        var edge = new Dictionary<long, double[]>();
        void AddEdge(int a, int b, double nx, double ny, double nz)
        {
            long key = (long)Math.Min(a, b) * nv + Math.Max(a, b);
            if (!edge.TryGetValue(key, out var e)) edge[key] = new[] { nx, ny, nz };
            else { e[0] += nx; e[1] += ny; e[2] += nz; }
        }
        for (int t = 0; t < nt; t++)
        {
            int a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];
            double abx = vx[b] - vx[a], aby = vy[b] - vy[a], abz = vz[b] - vz[a];
            double acx = vx[c] - vx[a], acy = vy[c] - vy[a], acz = vz[c] - vz[a];
            double nx = aby * acz - abz * acy, ny = abz * acx - abx * acz, nz = abx * acy - aby * acx;
            double nl = Math.Sqrt(nx * nx + ny * ny + nz * nz); nx /= nl; ny /= nl; nz /= nl;
            fn[3 * t] = nx; fn[3 * t + 1] = ny; fn[3 * t + 2] = nz;
            double a0 = Angle(vx[b] - vx[a], vy[b] - vy[a], vz[b] - vz[a], vx[c] - vx[a], vy[c] - vy[a], vz[c] - vz[a]);
            double a1 = Angle(vx[a] - vx[b], vy[a] - vy[b], vz[a] - vz[b], vx[c] - vx[b], vy[c] - vy[b], vz[c] - vz[b]);
            double a2 = Angle(vx[b] - vx[c], vy[b] - vy[c], vz[b] - vz[c], vx[a] - vx[c], vy[a] - vy[c], vz[a] - vz[c]);
            vn[3 * a] += a0 * nx; vn[3 * a + 1] += a0 * ny; vn[3 * a + 2] += a0 * nz;
            vn[3 * b] += a1 * nx; vn[3 * b + 1] += a1 * ny; vn[3 * b + 2] += a1 * nz;
            vn[3 * c] += a2 * nx; vn[3 * c + 1] += a2 * ny; vn[3 * c + 2] += a2 * nz;
            AddEdge(a, b, nx, ny, nz); AddEdge(b, c, nx, ny, nz); AddEdge(a, c, nx, ny, nz);
        }
        for (int i = 0; i < nv; i++) Normalize(vn, 3 * i);
        for (int t = 0; t < nt; t++)
        {
            int a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];
            int[,] pr = { { a, b }, { b, c }, { a, c } };
            for (int k = 0; k < 3; k++)
            {
                var e = edge[(long)Math.Min(pr[k, 0], pr[k, 1]) * nv + Math.Max(pr[k, 0], pr[k, 1])];
                double l = Math.Sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
                en[9 * t + 3 * k] = e[0] / l; en[9 * t + 3 * k + 1] = e[1] / l; en[9 * t + 3 * k + 2] = e[2] / l;
            }
        }
    }

    static double Angle(double ax, double ay, double az, double bx, double by, double bz)
    {
        double la = Math.Sqrt(ax * ax + ay * ay + az * az), lb = Math.Sqrt(bx * bx + by * by + bz * bz);
        return Math.Acos(Math.Abs((ax / la) * (bx / lb) + (ay / la) * (by / lb) + (az / la) * (bz / lb)));
    }
    static void Normalize(double[] a, int o)
    {
        double l = Math.Sqrt(a[o] * a[o] + a[o + 1] * a[o + 1] + a[o + 2] * a[o + 2]);
        if (l > 0) { a[o] /= l; a[o + 1] /= l; a[o + 2] /= l; }
    }

    // nearest entity: 0 V0, 1 V1, 2 V2, 3 E01, 4 E12, 5 E02, 6 F (Discregrid order)
    static double PointTriangleSq(out int ent, out double qx, out double qy, out double qz,
                                  double px, double py, double pz, double v0x, double v0y, double v0z,
                                  double v1x, double v1y, double v1z, double v2x, double v2y, double v2z)
    {
        double dfx = v0x - px, dfy = v0y - py, dfz = v0z - pz;
        double e0x = v1x - v0x, e0y = v1y - v0y, e0z = v1z - v0z;
        double e1x = v2x - v0x, e1y = v2y - v0y, e1z = v2z - v0z;
        double a00 = e0x * e0x + e0y * e0y + e0z * e0z;
        double a01 = e0x * e1x + e0y * e1y + e0z * e1z;
        double a11 = e1x * e1x + e1y * e1y + e1z * e1z;
        double b0 = dfx * e0x + dfy * e0y + dfz * e0z;
        double b1 = dfx * e1x + dfy * e1y + dfz * e1z;
        double c = dfx * dfx + dfy * dfy + dfz * dfz;
        double det = Math.Abs(a00 * a11 - a01 * a01);
        double s = a01 * b1 - a11 * b0;
        double t = a01 * b0 - a00 * b1;
        double d2;
        if (s + t <= det)
        {
            if (s < 0)
            {
                if (t < 0)   // region 4
                {
                    if (b0 < 0)
                    {
                        t = 0;
                        if (-b0 >= a00) { ent = 1; s = 1; d2 = a00 + 2 * b0 + c; }
                        else { ent = 3; s = -b0 / a00; d2 = b0 * s + c; }
                    }
                    else
                    {
                        s = 0;
                        if (b1 >= 0) { ent = 0; t = 0; d2 = c; }
                        else if (-b1 >= a11) { ent = 2; t = 1; d2 = a11 + 2 * b1 + c; }
                        else { ent = 5; t = -b1 / a11; d2 = b1 * t + c; }
                    }
                }
                else         // region 3
                {
                    s = 0;
                    if (b1 >= 0) { ent = 0; t = 0; d2 = c; }
                    else if (-b1 >= a11) { ent = 2; t = 1; d2 = a11 + 2 * b1 + c; }
                    else { ent = 5; t = -b1 / a11; d2 = b1 * t + c; }
                }
            }
            else if (t < 0)  // region 5
            {
                t = 0;
                if (b0 >= 0) { ent = 0; s = 0; d2 = c; }
                else if (-b0 >= a00) { ent = 1; s = 1; d2 = a00 + 2 * b0 + c; }
                else { ent = 3; s = -b0 / a00; d2 = b0 * s + c; }
            }
            else             // region 0
            {
                ent = 6;
                double invDet = 1 / det;
                s *= invDet; t *= invDet;
                d2 = s * (a00 * s + a01 * t + 2 * b0) + t * (a01 * s + a11 * t + 2 * b1) + c;
            }
        }
        else
        {
            double tmp0, tmp1, numer, denom;
            if (s < 0)       // region 2
            {
                tmp0 = a01 + b0; tmp1 = a11 + b1;
                if (tmp1 > tmp0)
                {
                    numer = tmp1 - tmp0; denom = a00 - 2 * a01 + a11;
                    if (numer >= denom) { ent = 1; s = 1; t = 0; d2 = a00 + 2 * b0 + c; }
                    else { ent = 4; s = numer / denom; t = 1 - s; d2 = s * (a00 * s + a01 * t + 2 * b0) + t * (a01 * s + a11 * t + 2 * b1) + c; }
                }
                else
                {
                    s = 0;
                    if (tmp1 <= 0) { ent = 2; t = 1; d2 = a11 + 2 * b1 + c; }
                    else if (b1 >= 0) { ent = 0; t = 0; d2 = c; }
                    else { ent = 5; t = -b1 / a11; d2 = b1 * t + c; }
                }
            }
            else if (t < 0)  // region 6
            {
                tmp0 = a01 + b1; tmp1 = a00 + b0;
                if (tmp1 > tmp0)
                {
                    numer = tmp1 - tmp0; denom = a00 - 2 * a01 + a11;
                    if (numer >= denom) { ent = 2; t = 1; s = 0; d2 = a11 + 2 * b1 + c; }
                    else { ent = 4; t = numer / denom; s = 1 - t; d2 = s * (a00 * s + a01 * t + 2 * b0) + t * (a01 * s + a11 * t + 2 * b1) + c; }
                }
                else
                {
                    t = 0;
                    if (tmp1 <= 0) { ent = 1; s = 1; d2 = a00 + 2 * b0 + c; }
                    else if (b0 >= 0) { ent = 0; s = 0; d2 = c; }
                    else { ent = 3; s = -b0 / a00; d2 = b0 * s + c; }
                }
            }
            else             // region 1
            {
                numer = a11 + b1 - a01 - b0;
                if (numer <= 0) { ent = 2; s = 0; t = 1; d2 = a11 + 2 * b1 + c; }
                else
                {
                    denom = a00 - 2 * a01 + a11;
                    if (numer >= denom) { ent = 1; s = 1; t = 0; d2 = a00 + 2 * b0 + c; }
                    else { ent = 4; s = numer / denom; t = 1 - s; d2 = s * (a00 * s + a01 * t + 2 * b0) + t * (a01 * s + a11 * t + 2 * b1) + c; }
                }
            }
        }
        if (d2 < 0) d2 = 0;
        qx = v0x + s * e0x + t * e1x; qy = v0y + s * e0y + t * e1y; qz = v0z + s * e0z + t * e1z;
        return d2;
    }

    /// <summary>TriangleMeshDistance::signed_distance(p).distance.</summary>
    public double SignedDistance(double px, double py, double pz)
    {
        int nt = tri.Length / 3;
        double best = double.MaxValue; int bt = -1, be = 0; double bqx = 0, bqy = 0, bqz = 0;
        for (int t = 0; t < nt; t++)
        {
            int a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];
            double d2 = PointTriangleSq(out int ent, out double qx, out double qy, out double qz, px, py, pz,
                                        vx[a], vy[a], vz[a], vx[b], vy[b], vz[b], vx[c], vy[c], vz[c]);
            if (d2 < best) { best = d2; bt = t; be = ent; bqx = qx; bqy = qy; bqz = qz; }
        }
        double nx, ny, nz;
        int ta = tri[3 * bt], tb = tri[3 * bt + 1], tc = tri[3 * bt + 2];
        switch (be)
        {
            case 0: nx = vn[3 * ta]; ny = vn[3 * ta + 1]; nz = vn[3 * ta + 2]; break;
            case 1: nx = vn[3 * tb]; ny = vn[3 * tb + 1]; nz = vn[3 * tb + 2]; break;
            case 2: nx = vn[3 * tc]; ny = vn[3 * tc + 1]; nz = vn[3 * tc + 2]; break;
            case 3: nx = en[9 * bt]; ny = en[9 * bt + 1]; nz = en[9 * bt + 2]; break;
            case 4: nx = en[9 * bt + 3]; ny = en[9 * bt + 4]; nz = en[9 * bt + 5]; break;
            case 5: nx = en[9 * bt + 6]; ny = en[9 * bt + 7]; nz = en[9 * bt + 8]; break;
            default: nx = fn[3 * bt]; ny = fn[3 * bt + 1]; nz = fn[3 * bt + 2]; break;
        }
        double ux = px - bqx, uy = py - bqy, uz = pz - bqz;
        double dist = Math.Sqrt(best);
        return (ux * nx + uy * ny + uz * nz >= 0.0) ? dist : -dist;
    }
}
