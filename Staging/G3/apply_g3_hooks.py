#!/usr/bin/env python3
"""apply_g3_hooks.py IN [OUT] — add G3's minimal, additive hooks to FluidSceneMVP.cs (idempotent).
Hooks (all behaviour lives in Assets/Overnight/Classical/FluidSceneMVP.Classical.cs, OFF by default):
  1. `partial` on the class (G2 may already have added it)
  2. LateUpdate: classical frame instead of splat + inference when surfaceSource == Classical
  3. HandleInput: key C toggles the surface source
  4. OnDestroy: release the classical GPU resources
  5. header comment: document key C
Fails loudly if an anchor is missing (the file changed shape) instead of guessing."""
import sys

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else src
s = open(src).read()
MARK = "G3 classical"


def once(s, anchor, new, where="after"):
    if new.strip() in s:
        return s
    assert s.count(anchor) == 1, f"anchor not unique/missing: {anchor!r} (count {s.count(anchor)})"
    return s.replace(anchor, anchor + new if where == "after" else new + anchor)


if "public partial class FluidSceneMVP : MonoBehaviour" not in s:
    assert s.count("public class FluidSceneMVP : MonoBehaviour") == 1
    s = s.replace("public class FluidSceneMVP : MonoBehaviour", "public partial class FluidSceneMVP : MonoBehaviour")

s = once(s, "        frameIdx = (int)(simTime * playbackFps) % provider.FrameCount;\n",
         "\n        if (ClassicalActive) { ClassicalLateUpdate(); return; }   // G3 classical C* surface instead of splat + network (key C; Overnight/Classical)\n")
s = once(s, "            if (kb.hKey.wasPressedThisFrame) ToggleFoamOnlyView();\n",
         "            if (kb.cKey.wasPressedThisFrame) ToggleSurfaceSource();   // G3 classical C* surface\n")
s = once(s, "    void OnDestroy()\n    {\n",
         "        DisposeClassical();   // G3 classical C* surface\n")
s = once(s, "// B bilateral smoothing, [ / ] playback fps −/+5, G whitewater layer (FoamLayer), H whitewater only.\n",
         "// C surface source: network / classical C* (G3 classical, Overnight/Classical/FluidSceneMVP.Classical.cs; network by default).\n")
open(dst, "w").write(s)
print("hooks:", s.count("G3 classical"), "->", dst)
