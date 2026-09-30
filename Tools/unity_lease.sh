#!/usr/bin/env bash
# unity_lease.sh — one-editor mutex for the 2026-09-30 overnight run (several Claude agents share ONE Unity editor + MCP bridge).
# Hold the lease for ANY of: copying files into Assets/, refresh_unity / AssetDatabase.Refresh, compiling, entering Play mode,
# execute_code runs, scene loads. Release as soon as the editor is back to a clean state (compiles, not playing).
#   Tools/unity_lease.sh acquire <owner> [max_wait_s=5400]   -> exit 0 when held (prints owner), 1 on timeout
#   Tools/unity_lease.sh release <owner>                      -> releases only if <owner> holds it
#   Tools/unity_lease.sh status                               -> prints holder + age, or FREE
# A lease older than 50 min is considered stale and may be broken (logged) — renew by re-acquiring as the same owner.
# PRIORITY: while owner "main" is waiting it drops a want-file; every other owner waits for it to clear (the critical-path
# solver/data runs go next). A want-file older than 30 min is ignored.
L=/home/matias/AISPH/.unity_lease.d
LOG=/home/matias/AISPH/Tools/unity_lease.log
cmd=${1:-status}; owner=${2:-}; wait=${3:-5400}
now() { date +%s; }
case "$cmd" in
  acquire)
    [ -z "$owner" ] && { echo "owner required"; exit 2; }
    t0=$(now)
    W=/home/matias/AISPH/.unity_lease.want_main
    [ "$owner" = "main" ] && echo "$(now)" > "$W"
    while true; do
      if [ "$owner" != "main" ] && [ -f "$W" ] && [ $(( $(now) - $(cat "$W" 2>/dev/null || echo 0) )) -lt 1800 ]; then
        [ $(( $(now) - t0 )) -gt "$wait" ] && { echo "timeout waiting for lease (main has priority)"; exit 1; }
        sleep 10; continue
      fi
      if mkdir "$L" 2>/dev/null; then
        [ "$owner" = "main" ] && rm -f "$W"
        echo "$owner $(now)" > "$L/owner"; echo "[$(date '+%F %T')] ACQ $owner" >> "$LOG"; echo "held by $owner"; exit 0; fi
      read -r o ts 2>/dev/null < "$L/owner" || { sleep 2; continue; }
      if [ "$o" = "$owner" ]; then echo "$owner $(now)" > "$L/owner"; [ "$owner" = "main" ] && rm -f "$W"; echo "renewed by $owner"; exit 0; fi
      if [ $(( $(now) - ${ts:-0} )) -gt 3000 ]; then echo "[$(date '+%F %T')] STALE $o broken by $owner" >> "$LOG"; rm -rf "$L"; continue; fi
      [ $(( $(now) - t0 )) -gt "$wait" ] && { echo "timeout waiting for lease (held by $o)"; exit 1; }
      sleep 15
    done ;;
  release)
    read -r o ts 2>/dev/null < "$L/owner"
    if [ "$o" = "$owner" ]; then rm -rf "$L"; echo "[$(date '+%F %T')] REL $owner" >> "$LOG"; echo released; else echo "not held by $owner (holder: ${o:-none})"; exit 1; fi ;;
  status)
    if [ -f "$L/owner" ] && read -r o ts < "$L/owner"; then echo "held by $o for $(( $(now) - ts )) s"; else echo FREE; fi ;;
esac
