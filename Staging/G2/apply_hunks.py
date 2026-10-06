#!/usr/bin/env python3
"""G2: apply the learned-spray hooks to FluidSceneMVP.cs (idempotent; every anchor must match exactly once).
  python3 apply_hunks.py <in FluidSceneMVP.cs> <out path>
All hooks are additive one-liners; the logic lives in Assets/Overnight/Spray/FluidSceneMVP.Spray.cs (partial class)."""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
MARK = "G2 learned spray"

HUNKS = [
    # (anchor, replacement, already-applied probe)
    ("public class FluidSceneMVP : MonoBehaviour", "public partial class FluidSceneMVP : MonoBehaviour", "public partial class FluidSceneMVP"),
    ("// B bilateral smoothing, [ / ] playback fps −/+5, G whitewater layer (FoamLayer), H whitewater only.\n",
     "// B bilateral smoothing, [ / ] playback fps −/+5, G whitewater layer (FoamLayer), H whitewater only.\n"
     "// J learned spray droplets (G2 learned spray, Overnight/Spray/FluidSceneMVP.Spray.cs; off by default).\n",
     "// J learned spray droplets"),
    ("        fieldTex.SetPixels(px);\n",
     "        SprayComposite();   // G2 learned spray: droplets into the field before upload (no-op while off)\n"
     "        fieldTex.SetPixels(px);\n",
     "SprayComposite();"),
    ("            if (kb.hKey.wasPressedThisFrame) ToggleFoamOnlyView();\n",
     "            if (kb.hKey.wasPressedThisFrame) ToggleFoamOnlyView();\n"
     "            if (kb.jKey.wasPressedThisFrame) ToggleSpray();   // G2 learned spray\n",
     "ToggleSpray();"),
    ("{foam.LastStepMs:F1}+{foam.LastSplatMs:F1} ms\" : \"\");\n",
     "{foam.LastStepMs:F1}+{foam.LastSplatMs:F1} ms\" : \"\") + SprayStatus();   // G2 learned spray\n",
     "SprayStatus();"),
    ("    void OnDestroy()\n    {\n",
     "    void OnDestroy()\n    {\n        SprayShutdown();   // G2 learned spray\n",
     "SprayShutdown();"),
]
for anchor, rep, probe in HUNKS:
    if probe in s:
        continue
    c = s.count(anchor)
    assert c == 1, f"anchor found {c} times: {anchor!r}"
    s = s.replace(anchor, rep)
open(dst, "w").write(s)
print("ok", dst)
