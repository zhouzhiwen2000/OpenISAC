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
  OPENISAC_NIC_COMBINED         Combined queue count for tuned NICs when IRQ pinning is active (default: 1)
  OPENISAC_PIN_NIC_IRQS         Set to 0 to skip IRQ pinning (default: 1)
  OPENISAC_IRQ_CORES_PER_IFACE  Dedicated cores per tuned NIC (default: NIC_COMBINED)
  OPENISAC_IRQ_CORE_LIST        Optional CPU list/range used as the dedicated IRQ core pool
                                Example: 14-15 or 10,11,12,13
                                If unset, the script uses the first isolated CPUs from
                                /sys/devices/system/cpu/isolated. If no isolated CPUs are
                                configured, IRQ-specific tuning is skipped.
  OPENISAC_TARGET_IFACES        Optional comma-separated NIC allowlist
                                Example: enp5s0f0,enp5s0f1
  OPENISAC_STOP_IRQBALANCE      Set to 1 to stop irqbalance before pinning IRQs (default: 0)

Examples:
  ./scripts/set_performance.bash
  OPENISAC_TARGET_IFACES=enp5s0f0 OPENISAC_IRQ_CORE_LIST=14-15 ./scripts/set_performance.bash
  OPENISAC_NIC_COMBINED=1 OPENISAC_IRQ_CORES_PER_IFACE=1 ./scripts/set_performance.bash
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
NIC_COMBINED="${OPENISAC_NIC_COMBINED:-1}"
PIN_NIC_IRQS="${OPENISAC_PIN_NIC_IRQS:-1}"
IRQ_CORES_PER_IFACE="${OPENISAC_IRQ_CORES_PER_IFACE:-$NIC_COMBINED}"
IRQ_CORE_LIST="${OPENISAC_IRQ_CORE_LIST:-}"
TARGET_IFACES="${OPENISAC_TARGET_IFACES:-}"
STOP_IRQBALANCE="${OPENISAC_STOP_IRQBALANCE:-0}"

if [[ ! "$MIN_NIC_SPEED_MBPS" =~ ^[0-9]+$ ]] || [[ ! "$NIC_RING_SIZE" =~ ^[0-9]+$ ]] || \
   [[ ! "$NIC_MTU" =~ ^[0-9]+$ ]] || [[ ! "$NIC_COMBINED" =~ ^[0-9]+$ ]] || \
   [[ ! "$IRQ_CORES_PER_IFACE" =~ ^[0-9]+$ ]]; then
    echo "Numeric environment overrides are invalid." >&2
    exit 1
fi

if [[ "$PIN_NIC_IRQS" != "0" && "$PIN_NIC_IRQS" != "1" ]]; then
    echo "OPENISAC_PIN_NIC_IRQS must be 0 or 1." >&2
    exit 1
fi

if [[ "$STOP_IRQBALANCE" != "0" && "$STOP_IRQBALANCE" != "1" ]]; then
    echo "OPENISAC_STOP_IRQBALANCE must be 0 or 1." >&2
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

expand_cpu_list() {
    local spec="$1"
    local item start end cpu
    [[ -z "$spec" ]] && return 0
    IFS=',' read -ra parts <<< "$spec"
    for item in "${parts[@]}"; do
        [[ -z "$item" ]] && continue
        if [[ "$item" == *-* ]]; then
            start="${item%-*}"
            end="${item#*-}"
            if (( start <= end )); then
                for ((cpu=start; cpu<=end; ++cpu)); do
                    echo "$cpu"
                done
            else
                for ((cpu=start; cpu>=end; --cpu)); do
                    echo "$cpu"
                done
            fi
        else
            echo "$item"
        fi
    done
}

cpu_in_list() {
    local needle="$1"
    shift
    local cpu
    for cpu in "$@"; do
        if [[ "$cpu" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

get_allowed_cpu_pool() {
    local source_list cpu
    local -a expanded=()
    local -a deduped=()
    HAVE_DEDICATED_IRQ_CPU_POOL=1
    if [[ -n "$IRQ_CORE_LIST" ]]; then
        source_list="$IRQ_CORE_LIST"
        PREFER_HIGH_CORES=0
    elif [[ -r /sys/devices/system/cpu/isolated ]] && [[ -n "$(tr -d ' \t\n' </sys/devices/system/cpu/isolated)" ]]; then
        source_list="$(tr -d ' \t\n' </sys/devices/system/cpu/isolated)"
        PREFER_HIGH_CORES=0
    else
        HAVE_DEDICATED_IRQ_CPU_POOL=0
        ALLOWED_CPU_POOL=()
        return 0
    fi

    mapfile -t expanded < <(expand_cpu_list "$source_list")
    if [[ "$PREFER_HIGH_CORES" -eq 1 ]]; then
        mapfile -t expanded < <(printf '%s\n' "${expanded[@]}" | sort -n | uniq)
    fi

    for cpu in "${expanded[@]}"; do
        if ! cpu_in_list "$cpu" "${deduped[@]:-}"; then
            deduped+=("$cpu")
        fi
    done
    ALLOWED_CPU_POOL=("${deduped[@]}")
    if ((${#ALLOWED_CPU_POOL[@]} == 0)); then
        echo "Failed to derive an IRQ CPU pool." >&2
        exit 1
    fi
}

allocate_irq_cores() {
    local count="$1"
    local -a picked=()
    local idx cpu

    ALLOCATED_IRQ_CORES=""

    if (( count == 0 )); then
        ALLOCATED_IRQ_CORES=""
        return 0
    fi

    if [[ "${PREFER_HIGH_CORES:-1}" -eq 1 ]]; then
        for ((idx=${#ALLOWED_CPU_POOL[@]}-1; idx>=0 && ${#picked[@]}<count; --idx)); do
            cpu="${ALLOWED_CPU_POOL[$idx]}"
            if cpu_in_list "$cpu" "${USED_IRQ_CPUS[@]:-}"; then
                continue
            fi
            picked+=("$cpu")
            USED_IRQ_CPUS+=("$cpu")
        done
    else
        for ((idx=0; idx<${#ALLOWED_CPU_POOL[@]} && ${#picked[@]}<count; ++idx)); do
            cpu="${ALLOWED_CPU_POOL[$idx]}"
            if cpu_in_list "$cpu" "${USED_IRQ_CPUS[@]:-}"; then
                continue
            fi
            picked+=("$cpu")
            USED_IRQ_CPUS+=("$cpu")
        done
    fi

    if ((${#picked[@]} == 0)); then
        echo "Not enough CPUs available to reserve IRQ cores." >&2
        exit 1
    fi

    ALLOCATED_IRQ_CORES="$(printf '%s\n' "${picked[@]}" | paste -sd, -)"
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

    while IFS= read -r iface; do
        [[ "$iface" == "lo" ]] && continue
        [[ -n "$TARGET_IFACES" && -z "${target_map[$iface]:-}" ]] && continue
        [[ ! -r "/sys/class/net/$iface/speed" ]] && continue
        speed="$(<"/sys/class/net/$iface/speed")"
        if [[ "$speed" =~ ^[0-9]+$ ]] && (( speed >= MIN_NIC_SPEED_MBPS )); then
            selected+=("$iface")
        fi
    done < <(ls /sys/class/net)

    printf '%s\n' "${selected[@]}"
}

get_iface_irqs() {
    local iface="$1"
    grep -i "$iface" /proc/interrupts | cut -d: -f1 | tr -d ' '
}

pin_iface_irqs() {
    local iface="$1"
    local cpu_csv="$2"
    local -a cpus
    local -a irqs
    local idx irq cpu

    IFS=',' read -ra cpus <<< "$cpu_csv"
    mapfile -t irqs < <(get_iface_irqs "$iface")
    if ((${#irqs[@]} == 0)); then
        echo "    No IRQs found for $iface, skipping IRQ pinning."
        return 0
    fi

    echo "    Pinning IRQs for $iface to CPUs: $cpu_csv"
    for ((idx=0; idx<${#irqs[@]}; ++idx)); do
        irq="${irqs[$idx]}"
        cpu="${cpus[$((idx % ${#cpus[@]}))]}"
        run_priv bash -lc "echo $cpu > /proc/irq/$irq/smp_affinity_list"
    done
}

stop_irqbalance_if_requested() {
    if [[ "$STOP_IRQBALANCE" != "1" ]]; then
        return 0
    fi
    if command -v systemctl >/dev/null 2>&1; then
        local state
        state="$(systemctl is-active irqbalance 2>/dev/null || true)"
        if [[ "$state" == "active" ]]; then
            echo "Stopping irqbalance so manual IRQ affinity will persist..."
            run_priv systemctl stop irqbalance
        fi
    fi
}

maybe_warn_irqbalance() {
    if ! command -v systemctl >/dev/null 2>&1; then
        return 0
    fi
    local state
    state="$(systemctl is-active irqbalance 2>/dev/null || true)"
    if [[ "$state" == "active" ]]; then
        echo "Warning: irqbalance is active and may override manual IRQ pinning."
        echo "         Set OPENISAC_STOP_IRQBALANCE=1 to stop it from this script."
    fi
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
    local combined_target="$NIC_COMBINED"
    local combined_current=""
    local combined_max=""
    local irq_cpu_csv=""
    local can_pin_irqs=0

    echo "  Found high-speed interface: $iface (${speed} Mbps)"

    echo "    Setting ring buffers to ${NIC_RING_SIZE}..."
    run_priv ethtool -G "$iface" tx "$NIC_RING_SIZE" rx "$NIC_RING_SIZE"

    echo "    Setting MTU to ${NIC_MTU}..."
    run_priv ip link set dev "$iface" mtu "$NIC_MTU"

    if [[ "$PIN_NIC_IRQS" == "1" && "${HAVE_DEDICATED_IRQ_CPU_POOL:-0}" == "1" ]]; then
        can_pin_irqs=1
    fi

    if command -v ethtool >/dev/null 2>&1 && [[ "$can_pin_irqs" -eq 1 ]]; then
        local channels_output
        channels_output="$(ethtool -l "$iface" 2>/dev/null || true)"
        combined_max="$(awk '/Pre-set maximums:/ {flag=1; next} flag && /Combined:/ {print $2; exit}' <<< "$channels_output")"
        combined_current="$(awk '/Current hardware settings:/ {flag=1; next} flag && /Combined:/ {print $2; exit}' <<< "$channels_output")"
        if [[ "$combined_current" =~ ^[0-9]+$ ]] && (( combined_current > 0 )); then
            if [[ "$combined_max" =~ ^[0-9]+$ ]] && (( combined_max > 0 )) && (( combined_target > combined_max )); then
                echo "    Requested combined=${combined_target}, interface maximum is ${combined_max}; using ${combined_max}."
                combined_target="$combined_max"
            fi
            echo "    Setting combined queues to ${combined_target}..."
            run_priv ethtool -L "$iface" combined "$combined_target"
        else
            echo "    Interface does not expose configurable combined channels, skipping ethtool -L."
        fi
    fi

    if [[ "$PIN_NIC_IRQS" == "1" && "${HAVE_DEDICATED_IRQ_CPU_POOL:-0}" != "1" ]]; then
        echo "    No isolated CPU cores detected; skipping IRQ-specific tuning for $iface."
        echo "    Hint: define OPENISAC_IRQ_CORE_LIST explicitly if you want manual NIC IRQ pinning."
    fi

    if [[ "$can_pin_irqs" -eq 1 ]]; then
        allocate_irq_cores "$IRQ_CORES_PER_IFACE"
        irq_cpu_csv="$ALLOCATED_IRQ_CORES"
        pin_iface_irqs "$iface" "$irq_cpu_csv"
        echo "    Reserved IRQ CPU(s) for $iface: $irq_cpu_csv"
        echo "    Reminder: do not place real-time application threads on these CPU(s)."
    fi
}

echo "Increasing kernel socket buffer maximum sizes..."
run_priv sysctl -w net.core.wmem_max=250000000
run_priv sysctl -w net.core.rmem_max=250000000
run_priv sysctl -w net.core.wmem_default=250000000
run_priv sysctl -w net.core.rmem_default=250000000

configure_cpufreq
get_allowed_cpu_pool
USED_IRQ_CPUS=()
maybe_warn_irqbalance
stop_irqbalance_if_requested

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
