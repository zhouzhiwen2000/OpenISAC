#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: set_performance.bash [--dry-run]

Tune the host for OpenISAC real-time workloads.

Environment overrides:
  OPENISAC_MIN_NIC_SPEED_MBPS   Minimum NIC speed to tune (default: 10000)
  OPENISAC_NIC_RING_SIZE        Ring size for RX/TX (default: 4096)
  OPENISAC_NIC_MTU              MTU for tuned NICs (default: 9000)
  OPENISAC_TARGET_IFACES        Optional comma-separated NIC allowlist
                                Example: enp5s0f0,enp5s0f1

NIC queue selection and IRQ affinity are configured by isolate_cpus.py.

Examples:
  ./scripts/set_performance.bash
  OPENISAC_TARGET_IFACES=enp5s0f0 ./scripts/set_performance.bash
EOF
}

DRY_RUN=0
while (($# > 0)); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

MIN_NIC_SPEED_MBPS="${OPENISAC_MIN_NIC_SPEED_MBPS:-10000}"
NIC_RING_SIZE="${OPENISAC_NIC_RING_SIZE:-4096}"
NIC_MTU="${OPENISAC_NIC_MTU:-9000}"
TARGET_IFACES="${OPENISAC_TARGET_IFACES:-}"

if [[ ! "$MIN_NIC_SPEED_MBPS" =~ ^[0-9]+$ ]] || [[ ! "$NIC_RING_SIZE" =~ ^[0-9]+$ ]] || \
   [[ ! "$NIC_MTU" =~ ^[0-9]+$ ]]; then
    echo "Numeric environment overrides are invalid." >&2
    exit 1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] no privileged writes will be performed"
fi

if [[ $EUID -eq 0 ]]; then
    SUDO=()
else
    SUDO=(sudo)
fi

run_priv() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run]" "${SUDO[@]}" "$@"
    else
        "${SUDO[@]}" "$@"
    fi
}

get_high_speed_ifaces() {
    local iface speed
    local -A target_map=()
    local -a selected=()

    if [[ -n "$TARGET_IFACES" ]]; then
        IFS=',' read -ra requested <<< "$TARGET_IFACES"
        for iface in "${requested[@]}"; do
            [[ -n "$iface" ]] && target_map["$iface"]=1
        done
    fi

    # Use glob instead of ls for more reliable enumeration
    for iface_path in /sys/class/net/*; do
        [[ -e "$iface_path" ]] || continue
        iface="${iface_path##*/}"
        [[ "$iface" == "lo" ]] && continue
        [[ -z "$iface" ]] && continue
        [[ -n "$TARGET_IFACES" && -z "${target_map[$iface]:-}" ]] && continue
        [[ ! -r "/sys/class/net/$iface/speed" ]] && continue
        speed="$(<"/sys/class/net/$iface/speed")"
        if [[ "$speed" =~ ^[0-9]+$ ]] && (( speed >= MIN_NIC_SPEED_MBPS )); then
            selected+=("$iface")
        fi
    done

    printf '%s\n' "${selected[@]}"
}

configure_cpufreq() {
    if ! command -v cpufreq-set >/dev/null 2>&1; then
        echo "cpufreq-set not found, skipping CPU governor tuning."
        return 0
    fi

    echo "Setting CPU frequency governor to performance..."
    local i
    for ((i=0; i<$(nproc --all); ++i)); do
        run_priv cpufreq-set -c "$i" -r -g performance
    done
}

configure_iface() {
    local iface="$1"
    local speed="$2"

    echo "  Found high-speed interface: $iface (${speed} Mbps)"

    echo "    Setting ring buffers to ${NIC_RING_SIZE}..."
    run_priv ethtool -G "$iface" tx "$NIC_RING_SIZE" rx "$NIC_RING_SIZE" 2>/dev/null || echo "      (ring buffer setting failed, continuing anyway)"

    echo "    Setting MTU to ${NIC_MTU}..."
    run_priv ip link set dev "$iface" mtu "$NIC_MTU" 2>/dev/null || echo "      (MTU setting failed, continuing anyway)"
}

echo "Increasing kernel socket buffer maximum sizes..."
run_priv sysctl -w net.core.wmem_max=250000000
run_priv sysctl -w net.core.rmem_max=250000000
run_priv sysctl -w net.core.wmem_default=250000000
run_priv sysctl -w net.core.rmem_default=250000000

configure_cpufreq

echo "Configuring network interfaces..."
mapfile -t HIGH_SPEED_IFACES < <(get_high_speed_ifaces)
if ((${#HIGH_SPEED_IFACES[@]} == 0)); then
    echo "  No high-speed interfaces detected (threshold: ${MIN_NIC_SPEED_MBPS} Mbps)."
else
    for iface in "${HIGH_SPEED_IFACES[@]}"; do
        speed="$(<"/sys/class/net/$iface/speed")"
        configure_iface "$iface" "$speed"
    done
fi

echo "Disabling real-time throttling..."
run_priv bash -lc 'cd /sys/kernel/debug/sched/ && echo RT_RUNTIME_SHARE > features'

echo "Done."
