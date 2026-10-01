// FluidSceneMVP.GpuSpray.cs — GPU1001: the learned spray inside the GPU chain.
// Until the GPU spray layer is in, the CPU layer (LearnedSprayLayer) still works on the GPU path through ONE field
// readback per rendered frame (the CPU spray needs the bulk field for its depth test / Field-mode merge).
using UnityEngine;
using UnityEngine.Rendering;

public partial class FluidSceneMVP
{
    const string GpuSprayLayerSample = "GPU1001.Spray";

    void RecordGpuSprayComposite(CommandBuffer cb)
    {
        if (!sprayEnabled) { SetSprayOverlayVisible(false); return; }
        // CPU-layer fallback: run what is recorded so far, read the field back, let the CPU layer composite, re-upload
        Graphics.ExecuteCommandBuffer(cb);
        cb.Clear();
        gpuSplat.Field.GetData(px);
        SprayComposite();
        if (sprayRenderMode == SprayRenderMode.Field) gpuSplat.Field.SetData(px);
    }

    void AfterGpuExecute() { }

    void DisposeGpuSpray() { }
}
