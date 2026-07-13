#!/usr/bin/env bash
# Run the full BS<->UE pipeline in simulation mode (no USRP)
# and capture the UE per-stage profiling report.
#
# Usage: scripts/sim_profile_run.bash [duration_seconds] [out_tag]
# Requires: build/ already compiled (ChannelSimulator, BS, UE).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"
DUR="${1:-25}"
TAG="${2:-run}"
OUTDIR="$ROOT/measurement/sim_profile"
mkdir -p "$OUTDIR"

cd "$BUILD" || exit 1

# Seed sim configs into the CWD the binaries read from.
cp "$ROOT/config/BS_Sim.yaml"   "$BUILD/BS.yaml"
cp "$ROOT/config/UE_Sim.yaml" "$BUILD/UE.yaml"

# Enable demodulation profiling in the demod config.
if grep -q '^profiling_modules:' "$BUILD/UE.yaml"; then
    sed -i 's/^profiling_modules:.*/profiling_modules: "demodulation"/' "$BUILD/UE.yaml"
else
    echo 'profiling_modules: "demodulation"' >> "$BUILD/UE.yaml"
fi

# Port 10000 is held by a stale/invisible process in this WSL2 env; move the
# demod control ROUTER aside so the bind succeeds for measurement runs.
sed -i 's/^control_port:.*/control_port: 10044/' "$BUILD/UE.yaml"

DEMOD_LOG="$OUTDIR/demod_${TAG}.log"
MOD_LOG="$OUTDIR/mod_${TAG}.log"
SIM_LOG="$OUTDIR/sim_${TAG}.log"

cleanup() {
    kill "${DEMOD_PID:-}" "${MOD_PID:-}" "${SIM_PID:-}" 2>/dev/null
    sleep 1
    kill -9 "${DEMOD_PID:-}" "${MOD_PID:-}" "${SIM_PID:-}" 2>/dev/null
}
trap cleanup EXIT INT TERM

# 1) Channel simulator owns the SHM control block; start it first.
./ChannelSimulator BS.yaml >"$SIM_LOG" 2>&1 &
SIM_PID=$!
sleep 2

# 2) UE (consumer) then BS (producer).
./UE >"$DEMOD_LOG" 2>&1 &
DEMOD_PID=$!
sleep 1
./BS   >"$MOD_LOG" 2>&1 &
MOD_PID=$!

echo "[sim] running ${DUR}s  (sim=$SIM_PID mod=$MOD_PID demod=$DEMOD_PID)"
sleep "$DUR"

cleanup
trap - EXIT INT TERM

echo "===== UE profiling report(s) [$TAG] ====="
grep -A 16 "process_ofdm_frame Profiling" "$DEMOD_LOG" | tail -32
echo "===== (demod log: $DEMOD_LOG) ====="
