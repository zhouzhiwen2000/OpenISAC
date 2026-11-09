#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
backend="${1:-cpu}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$backend" != cpu ]]; then
    echo "Usage: $0 [cpu] [docker build options...]" >&2
    exit 2
fi
exec docker build -f "$root/Dockerfile" -t "openisac:$backend" \
    --build-arg "BUILD_IMAGE=${BUILD_IMAGE:-ubuntu:24.04}" \
    --build-arg "RUNTIME_IMAGE=${RUNTIME_IMAGE:-ubuntu:24.04}" \
    --build-arg "BUILD_JOBS=${BUILD_JOBS:-4}" \
    --build-arg "UHD_REF=${UHD_REF:-v4.11.0.0}" \
    --build-arg "CPU_ARCH=${CPU_ARCH-x86-64-v3}" \
    "$@" "$root"
