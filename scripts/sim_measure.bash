#!/usr/bin/env bash
# Manual full-pipeline sim run with PID-based cleanup (avoids pkill self-match).
# Usage: scripts/sim_measure.bash [duration] [tag]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"
DUR="${1:-45}"
TAG="${2:-run}"
OUT="$ROOT/measurement/sim_profile"
mkdir -p "$OUT"
cd "$BUILD" || exit 1

cp "$ROOT/config/BS_Sim.yaml"   BS.yaml
cp "$ROOT/config/UE_Sim.yaml" UE.yaml
# Enable demod timing reports via hierarchical logging (replaces profiling_modules).
python3 - <<'PY'
from pathlib import Path
import re
p = Path("UE.yaml")
text = p.read_text()
if "logging:" not in text:
    text += "\nlogging:\n  default_level: warn\n  force_error: true\n  modules:\n    demod_profiling: info\n"
elif re.search(r"(?m)^    demod_profiling:\s*", text):
    text = re.sub(
        r"(?m)^(    demod_profiling:)\s*.*$",
        r"\1 info",
        text,
        count=1,
    )
else:
    modules_line = r"(?m)^  modules:\s*(?:\{\})?\s*(?:#.*)?$"
    if re.search(modules_line, text):
        text = re.sub(modules_line, "  modules:\n    demod_profiling: info", text, count=1)
    else:
        text = re.sub(r"(?m)^(logging:\s*\n)", r"\1  modules:\n    demod_profiling: info\n", text, count=1)
p.write_text(text)
PY
sed -i 's/^control_port:.*/control_port: 10044/' UE.yaml

./ChannelSimulator BS.yaml >"$OUT/sim_${TAG}.log" 2>&1 &
SIM=$!; sleep 2
./UE >"$OUT/demod_${TAG}.log" 2>&1 &
DEMOD=$!; sleep 1
./BS >"$OUT/mod_${TAG}.log" 2>&1 &
MOD=$!
echo "[sim_measure $TAG] sim=$SIM demod=$DEMOD mod=$MOD running ${DUR}s"
sleep "$DUR"
kill "$DEMOD" "$MOD" "$SIM" 2>/dev/null; sleep 1
kill -9 "$DEMOD" "$MOD" "$SIM" 2>/dev/null

echo "=== reports: $(grep -c 'CPU demod worker profiling' "$OUT/demod_${TAG}.log") ==="
grep -A16 "CPU demod worker profiling" "$OUT/demod_${TAG}.log" | tail -18
