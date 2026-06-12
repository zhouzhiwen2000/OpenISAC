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

cp "$ROOT/config/Modulator_Sim.yaml"   Modulator.yaml
cp "$ROOT/config/Demodulator_Sim.yaml" Demodulator.yaml
sed -i 's/^profiling_modules:.*/profiling_modules: "demodulation"/' Demodulator.yaml
grep -q '^profiling_modules' Demodulator.yaml || echo 'profiling_modules: "demodulation"' >> Demodulator.yaml
sed -i 's/^control_port:.*/control_port: 10044/' Demodulator.yaml

./ChannelSimulator Modulator.yaml >"$OUT/sim_${TAG}.log" 2>&1 &
SIM=$!; sleep 2
./OFDMDemodulator >"$OUT/demod_${TAG}.log" 2>&1 &
DEMOD=$!; sleep 1
./OFDMModulator >"$OUT/mod_${TAG}.log" 2>&1 &
MOD=$!
echo "[sim_measure $TAG] sim=$SIM demod=$DEMOD mod=$MOD running ${DUR}s"
sleep "$DUR"
kill "$DEMOD" "$MOD" "$SIM" 2>/dev/null; sleep 1
kill -9 "$DEMOD" "$MOD" "$SIM" 2>/dev/null

echo "=== reports: $(grep -c 'process_ofdm_frame Profiling' "$OUT/demod_${TAG}.log") ==="
grep -A16 "process_ofdm_frame Profiling" "$OUT/demod_${TAG}.log" | tail -18
