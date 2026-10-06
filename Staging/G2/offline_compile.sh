#!/usr/bin/env bash
# G2: offline type-check of Assembly-CSharp with the staged spray files, using Unity's own Roslyn + the editor's last
# response file (references/defines). Touches nothing under Assets/ or Library/ (output goes to $OUT).
set -e
E=~/Unity/Hub/Editor/6000.5.2f1/Editor/Data
OUT=${OUT:-/tmp/claude-1000/-home-matias-Desktop-SSU-restart/36e4d35c-312a-4b27-905d-79b17233ccfd/scratchpad/cc}
mkdir -p "$OUT"
cd ~/AISPH
RSP=Library/Bee/artifacts/2400b0aE.dag/Assembly-CSharp.rsp
python3 - "$RSP" "$OUT/g2.rsp" <<'PY'
import sys, glob, os
rsp, out = sys.argv[1], sys.argv[2]
lines = open(rsp).read().splitlines()
keep = []
for l in lines:
    if l.startswith("-out:") or l.startswith("-refout:") or l.startswith("-analyzer:") or l.startswith("/additionalfile:"): continue
    if l.startswith('"Assets/'): continue
    keep.append(l)
keep.append(f'-out:"{os.path.dirname(out)}/Assembly-CSharp.g2.dll"')
# current non-editor Assets sources (+ staged replacements)
srcs = sorted(p for p in glob.glob("Assets/**/*.cs", recursive=True) if "/Editor/" not in p)
srcs = [p for p in srcs if p != "Assets/FluidSceneMVP.cs" and not p.startswith("Assets/Overnight/Spray/")]
srcs += ["Staging/G2/FluidSceneMVP.staged.cs"] + sorted(glob.glob("Staging/G2/Overnight/Spray/*.cs"))
keep += [f'"{p}"' for p in srcs]
open(out, "w").write("\n".join(keep) + "\n")
PY
$E/NetCoreRuntime/dotnet $E/DotNetSdk/sdk/8.0.318/Roslyn/bincore/csc.dll @"$OUT/g2.rsp" 2>&1 | grep -v "^warning CS0\(618\|414\)" | grep -E "error|warning CS" | head -60
echo "exit ${PIPESTATUS[0]}"
