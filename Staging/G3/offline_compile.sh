#!/usr/bin/env bash
# G3: offline type-check of Assembly-CSharp with the staged classical files (Unity's own Roslyn + the editor's last
# response file for references/defines; adapted from Staging/G2/offline_compile.sh). Touches nothing under Assets/ or Library/.
# Usage: offline_compile.sh [staged FluidSceneMVP.cs]   (default: the working-tree Assets/FluidSceneMVP.cs)
set -e
E=~/Unity/Hub/Editor/6000.5.2f1/Editor/Data
OUT=/tmp/claude-1000/-home-matias-Desktop-SSU-restart/36e4d35c-312a-4b27-905d-79b17233ccfd/scratchpad/g3cc
mkdir -p "$OUT"
cd ~/AISPH
RSP=$(ls -t Library/Bee/artifacts/*.dag/Assembly-CSharp.rsp | head -1)
MVP=${1:-Assets/FluidSceneMVP.cs}
python3 - "$RSP" "$OUT/g3.rsp" "$MVP" <<'PY'
import sys, glob, os
rsp, out, mvp = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(rsp).read().splitlines()
keep = []
for l in lines:
    if l.startswith("-out:") or l.startswith("-refout:") or l.startswith("-analyzer:") or l.startswith("/additionalfile:"): continue
    if l.startswith('"Assets/'): continue
    keep.append(l)
keep.append(f'-out:"{os.path.dirname(out)}/Assembly-CSharp.g3.dll"')
srcs = sorted(p for p in glob.glob("Assets/**/*.cs", recursive=True) if "/Editor/" not in p)
srcs = [p for p in srcs if p != "Assets/FluidSceneMVP.cs" and not p.startswith("Assets/Overnight/Classical/")]
srcs += [mvp] + sorted(glob.glob("Staging/G3/*.cs"))
keep += [f'"{p}"' for p in srcs]
open(out, "w").write("\n".join(keep) + "\n")
print("rsp:", rsp, "sources:", len(srcs))
PY
$E/NetCoreRuntime/dotnet $E/DotNetSdk/sdk/8.0.318/Roslyn/bincore/csc.dll @"$OUT/g3.rsp" 2>&1 | grep -v "^warning CS0\(618\|414\)" | grep -E "error|warning CS" | head -60
echo "exit ${PIPESTATUS[0]}"
