#!/usr/bin/env bash
# Keep host NIC access and host CPU IDs; only the software environment is isolated.
set -euo pipefail
usage() {
    cat <<'HELP'
Usage: scripts/docker_run.sh [options] {BS|UE|ChannelSimulator|doctor|usrp-check} [args...]
  --image IMAGE       Override openisac:cpu
  --config FILE       Read-only external YAML (loaded on every startup)
  --work-dir DIR      Writable logs/wisdom directory (default ./docker-run/TARGET)
  --cpuset-cpus LIST  Optional host CPU IDs; YAML binding still selects each thread
  --memory SIZE      Optional memory ceiling; disables container swap when set
  --isolation-plan FILE  Use the process CPU set saved by isolate_cpus.py
  --cgroup-parent NAME  Override dedicated rt-workload.slice (systemd hosts)
  --name NAME         Container name
  --usb               Pass through /dev/bus/usb for B210 (including re-enumeration)
  --firmware-state DIR  Persistent verified backups/journals (shared across containers)
  --uhd-images DIR    Host UHD firmware/FPGA image directory
  --dry-run           Print the exact docker command
Environment: OPENISAC_REQUIRE_RT=0 permits a non-real-time development run.
Native Linux host network and IPC are used; no bridge, NAT or CPU quota is configured.
HELP
}
image= config= work= cpus= name= images= isolation_plan= cgroup_parent=auto memory= firmware_state=
usb=0; dry=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image|--config|--work-dir|--cpuset-cpus|--name|--uhd-images|--isolation-plan|--cgroup-parent|--memory|--firmware-state)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --image) image="$2" ;; --config) config="$2" ;; --work-dir) work="$2" ;;
                --cpuset-cpus) cpus="$2" ;; --name) name="$2" ;; --uhd-images) images="$2" ;;
                --isolation-plan) isolation_plan="$2" ;; --cgroup-parent) cgroup_parent="$2" ;;
                --memory) memory="$2" ;; --firmware-state) firmware_state="$2" ;;
            esac
            shift 2 ;;
        --usb) usb=1; shift ;; --dry-run) dry=1; shift ;;
        --help|-h) usage; exit 0 ;; --) shift; break ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
[[ $# -gt 0 ]] || { usage; exit 2; }
target="$1"
image="${image:-openisac:cpu}"
if [[ -n "$isolation_plan" ]]; then
    [[ -z "$cpus" ]] || { echo "Use either --cpuset-cpus or --isolation-plan" >&2; exit 2; }
    cpus=$(python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))["process_cpus"]; assert p and all(type(c) is int and c >= 0 for c in p); print(",".join(map(str,p)))' "$isolation_plan")
fi
if [[ "$cgroup_parent" == auto ]]; then
    cgroup_parent=
    if [[ "$(docker info --format '{{.CgroupDriver}}')" == systemd ]]; then
        cgroup_parent=rt-workload.slice
    fi
fi
work="${work:-$PWD/docker-run/$target}"
if [[ ! -d "$work" ]]; then
    mkdir -p "$work"
    if [[ -n "${SUDO_UID:-}" ]]; then
        chown "${SUDO_UID}:${SUDO_GID}" "$work"
    fi
fi
work=$(realpath "$work")
caller_home=$(getent passwd "${SUDO_UID:-$(id -u)}" | cut -d: -f6)
firmware_state="${firmware_state:-$caller_home/.local/state/openisac/firmware}"
if [[ ! -d "$firmware_state" ]]; then
    mkdir -p "$firmware_state"
    chmod 700 "$firmware_state"
    if [[ -n "${SUDO_UID:-}" ]]; then chown "${SUDO_UID}:${SUDO_GID}" "$firmware_state"; fi
fi
firmware_state=$(realpath "$firmware_state")
args=(run --rm --init --network host --ipc host --stop-timeout 1200
      --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit rtprio=99 --ulimit memlock=-1
      --user "${SUDO_UID:-$(id -u)}:${SUDO_GID:-$(id -g)}"
      --mount "type=bind,src=$work,dst=/work/build"
      --mount "type=bind,src=$firmware_state,dst=/work/firmware"
      -e OPENISAC_FIRMWARE_STATE=/work/firmware
      -e "OPENISAC_AUTO_UPDATE_FIRMWARE=${OPENISAC_AUTO_UPDATE_FIRMWARE:-1}"
      -e "OPENISAC_REQUIRE_RT=${OPENISAC_REQUIRE_RT:-1}")
# Preserve host supplementary groups for udev-managed USB and other device nodes.
for group_id in $(id -G "${SUDO_UID:-$(id -u)}"); do
    args+=(--group-add "$group_id")
done
[[ -z "$cpus" ]] || args+=(--cpuset-cpus "$cpus")
[[ -z "$memory" ]] || args+=(--memory "$memory" --memory-swap "$memory")
[[ -z "$cgroup_parent" ]] || args+=(--cgroup-parent "$cgroup_parent")
[[ -z "$name" ]] || args+=(--name "$name")
if [[ -n "$config" ]]; then
    [[ -f "$config" ]] || { echo "Config not found: $config" >&2; exit 2; }
    config=$(realpath "$config")
    # A directory mount also observes atomic saves of the host YAML on restart.
    args+=(--mount "type=bind,src=$(dirname "$config"),dst=/config,readonly"
           -e "OPENISAC_CONFIG=/config/$(basename "$config")")
fi
if [[ -n "$images" ]]; then
    args+=(--mount "type=bind,src=$(realpath "$images"),dst=/usr/share/uhd/images,readonly"
           -e UHD_IMAGES_DIR=/usr/share/uhd/images)
fi
if [[ "$usb" == 1 ]]; then
    args+=(--mount type=bind,src=/dev/bus/usb,dst=/dev/bus/usb --device-cgroup-rule 'c 189:* rmw')
fi
args+=("$image" "$@")
if [[ "$dry" == 1 ]]; then printf '%q ' docker "${args[@]}"; printf '\n'; else exec docker "${args[@]}"; fi
