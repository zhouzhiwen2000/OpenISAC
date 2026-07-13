#!/usr/bin/env bash
set -euo pipefail

app_home="${OPENISAC_HOME:-/opt/openisac}"
run_dir="${OPENISAC_RUN_DIR:-/work/build}"

usage() {
    cat <<'EOF'
Usage:
  openisac-run BS [x310|b210] [extra args...]
  openisac-run UE [x310|b210] [extra args...]
  openisac-run bash

Environment:
  OPENISAC_RUN_DIR=/work/build       Directory used as the runtime cwd.
  OPENISAC_CONFIG=/path/to/file.yaml Copy this config instead of a preset.
  OPENISAC_REFRESH_CONFIG=1          Overwrite existing BS.yaml/UE.yaml.
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

target="$1"
shift || true

case "$target" in
    bs|BS)
        binary="${app_home}/bin/BS"
        runtime_yaml="BS.yaml"
        preset_prefix="BS"
        ;;
    ue|UE)
        binary="${app_home}/bin/UE"
        runtime_yaml="UE.yaml"
        preset_prefix="UE"
        ;;
    *)
        exec "$target" "$@"
        ;;
esac

profile="${1:-${OPENISAC_PROFILE:-x310}}"
if [[ "$profile" == "x310" || "$profile" == "X310" || "$profile" == "b210" || "$profile" == "B210" ]]; then
    shift || true
else
    profile="${OPENISAC_PROFILE:-x310}"
fi

profile_upper="$(printf '%s' "$profile" | tr '[:lower:]' '[:upper:]')"
default_config="${app_home}/config/${preset_prefix}_${profile_upper}.yaml"
source_config="${OPENISAC_CONFIG:-$default_config}"

if [[ ! -x "$binary" ]]; then
    echo "OpenISAC binary not found or not executable: $binary" >&2
    exit 127
fi

if [[ ! -f "$source_config" ]]; then
    echo "Config file not found: $source_config" >&2
    exit 2
fi

mkdir -p "$run_dir"
cd "$run_dir"

if [[ ! -f "$runtime_yaml" || "${OPENISAC_REFRESH_CONFIG:-0}" == "1" ]]; then
    cp "$source_config" "$runtime_yaml"
fi

echo "OpenISAC: running $(basename "$binary") with $PWD/$runtime_yaml"
exec "$binary" "$@"
