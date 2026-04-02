#!/usr/bin/env bash

set -euo pipefail

HOUSEKEEPING_CPUS="${HOUSEKEEPING_CPUS:-0,1,10-15}"
TURBOSTAT_CPUS="${TURBOSTAT_CPUS:-0}"
INTERVAL="${INTERVAL:-0.5}"
LOG_FILE="${1:-/tmp/openisac_turbostat.log}"

mkdir -p "$(dirname "$LOG_FILE")"

exec taskset -c "$HOUSEKEEPING_CPUS" \
    turbostat --quiet --Summary -c "$TURBOSTAT_CPUS" \
    --show PkgTmp,PkgWatt,CorWatt,RAMWatt,Busy%,Bzy_MHz,TSC_MHz \
    -i "$INTERVAL" -o "$LOG_FILE"
