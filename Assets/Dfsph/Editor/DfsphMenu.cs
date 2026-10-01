// DfsphMenu.cs — DFSPH1001: switch the open live scene between the GPU PBF (GpuSphProvider) and the GPU DFSPH
// (DfsphProvider) behind FluidSceneMVP.provider. Undo-able; the scene is marked dirty, never saved.
using UnityEditor;
using UnityEngine;

public static class DfsphMenu
{
    [MenuItem("Tools/DFSPH1001/Use GPU DFSPH (SPlisHSPlasH) in the live scene")]
    static void UseDfsph()
    {
        var f = Object.FindAnyObjectByType<FluidSceneMVP>();
        if (f == null) { Debug.LogError("DFSPH1001: no FluidSceneMVP in the open scene"); return; }
        var go = f.gameObject;
        var d = go.GetComponent<DfsphProvider>();
        if (d == null) d = Undo.AddComponent<DfsphProvider>(go);
        var g = go.GetComponent<GpuSphProvider>();
        if (g != null)
        {
            Undo.RecordObject(d, "DFSPH1001 props");
            d.obstacles = g.obstacles;              // the same props collide with the DFSPH fluid (kinematic volume maps)
            d.domainCenterXZ = g.domainCenterXZ;
        }
        Undo.RecordObject(f, "DFSPH1001 provider");
        f.provider = d;
        Debug.Log($"DFSPH1001: FluidSceneMVP.provider = DfsphProvider (scene {d.scenePath}); enter Play mode");
    }

    [MenuItem("Tools/DFSPH1001/Use GPU PBF in the live scene")]
    static void UsePbf()
    {
        var f = Object.FindAnyObjectByType<FluidSceneMVP>();
        if (f == null) return;
        var g = f.GetComponent<GpuSphProvider>();
        if (g == null) { Debug.LogError("DFSPH1001: no GpuSphProvider on the FluidSceneMVP object"); return; }
        Undo.RecordObject(f, "DFSPH1001 provider");
        f.provider = g;
        var d = f.GetComponent<DfsphProvider>();
        if (d != null) Undo.DestroyObjectImmediate(d);
        Debug.Log("DFSPH1001: FluidSceneMVP.provider = GpuSphProvider");
    }
}
