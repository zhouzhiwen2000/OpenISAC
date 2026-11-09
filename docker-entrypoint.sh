#!/usr/bin/env bash
set -euo pipefail
app_home="${OPENISAC_HOME:-/opt/openisac}"
run_dir="${OPENISAC_RUN_DIR:-/work/build}"
usage() {
    cat <<'HELP'
Usage: openisac-run {BS|UE} [x310|b210|sim|sim-duplex]
       openisac-run ChannelSimulator [BS.yaml]
       openisac-run doctor [config.yaml]
       openisac-run usrp-check [config.yaml | --args addr=IP]
       openisac-run <command> [args...]

OPENISAC_CONFIG=/config/BS.yaml   External YAML; loaded on every start.
OPENISAC_RUN_DIR=/work/build      Writable cwd for logs and FFTW wisdom.
OPENISAC_REQUIRE_RT=1            Fail if FIFO priority/locked memory is unavailable.
OPENISAC_REFRESH_CONFIG=1        Replace an existing config with the selected preset.
CPU numbers in YAML are host CPU IDs. Use scripts/docker_run.sh for host networking,
real-time limits and USB access. Mount external configuration read-only.
HELP
}
if [[ $# -eq 0 || "$1" == --help || "$1" == -h ]]; then usage; exit 0; fi
target="$1"
shift
case "$target" in
    bs|BS) target=BS; role=BS ;;
    ue|UE) target=UE; role=UE ;;
    usrp-check) exec python3 "$app_home/docker_uhd_prepare.py" "$@" ;;
    doctor) exec python3 "$app_home/docker_preflight.py" "$@" ;;
    *) exec "$target" "$@" ;;
esac
binary="$app_home/bin/$target"
if [[ ! -x "$binary" ]]; then
    echo "$target is not installed in this image; build the CPU image with scripts/docker_build.sh cpu." >&2
    exit 127
fi
profile="${OPENISAC_PROFILE:-x310}"
case "${1:-}" in x310|b210|sim|sim-duplex) profile="$1"; shift ;; esac
case "$profile" in
    x310|X310) suffix=X310 ;; b210|B210) suffix=B210 ;;
    sim) suffix=Sim ;; sim-duplex) suffix=Sim_Duplex ;;
    *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac
mkdir -p "$run_dir"
cd "$run_dir"
runtime_yaml="$role.yaml"
source_config="${OPENISAC_CONFIG:-$app_home/config/${role}_${suffix}.yaml}"
if [[ -n "${OPENISAC_CONFIG:-}" || ! -f "$runtime_yaml" || "${OPENISAC_REFRESH_CONFIG:-0}" == 1 ]]; then
    if [[ ! -f "$source_config" ]]; then echo "Config file not found: $source_config" >&2; exit 2; fi
    if [[ ! "$source_config" -ef "$runtime_yaml" ]]; then
        config_tmp=$(mktemp "$run_dir/.${role}.yaml.XXXXXX")
        trap 'rm -f "$config_tmp"' EXIT
        cp -- "$source_config" "$config_tmp"
        mv -f -- "$config_tmp" "$runtime_yaml"
        trap - EXIT
    fi
fi
python3 "$app_home/docker_preflight.py" "$runtime_yaml"
# Bash defers these traps while the foreground updater finishes its flash transaction.
trap 'exit 130' INT
trap 'exit 143' TERM
python3 "$app_home/docker_uhd_prepare.py" --role "$role" "$runtime_yaml"
trap - INT TERM
echo "OpenISAC: $target config=$PWD/$runtime_yaml"
exec "$binary" "$@"
