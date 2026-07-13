#!/bin/bash
# Thin wrapper for backward compatibility. Implementation lives in isolate_cpus.py.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/isolate_cpus.py" "$@"
