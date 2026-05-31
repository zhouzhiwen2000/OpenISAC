#!/bin/bash
#
# DPDK Enable/Disable Script
# Usage: ./dpdk_control.sh start|stop|status
#

set -e

# ============== Auto-detection Functions ==============
# DPDK-compatible high-speed drivers (10Gbps+)
# ixgbe: Intel 10GbE, i40e: Intel 40GbE, ice: Intel 100GbE
# mlx5_core: Mellanox ConnectX-4/5/6, bnxt_en: Broadcom NetXtreme
DPDK_COMPATIBLE_DRIVERS="ixgbe|i40e|ice|mlx5_core|mlx4_core|bnxt_en|qede|ena"

# Get all DPDK-compatible PCI addresses (space-separated)
get_all_dpdk_compatible_pci() {
    local devbind="${DPDK_DEVBIND:-dpdk-devbind.py}"
    
    if ! command -v "$devbind" &>/dev/null; then
        echo ""
        return 0
    fi
    
    local status_output
    status_output=$("$devbind" --status 2>/dev/null) || { echo ""; return 0; }
    
    # Find all PCI addresses with DPDK-compatible drivers (both kernel and DPDK-bound)
    local pci_list
    pci_list=$(echo "$status_output" | grep -E "($DPDK_COMPATIBLE_DRIVERS|drv=vfio-pci|drv=igb_uio|drv=uio_pci_generic)" | \
               grep -oE "^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]" | sort -u | tr '\n' ' ')
    
    echo "$pci_list"
    return 0
}

# ============== Configuration ==============
# DPDK tools path (must be set first for detection)
DPDK_DEVBIND="${DPDK_DEVBIND:-dpdk-devbind.py}"
# DPDK driver to use: vfio-pci (recommended), uio_pci_generic, or igb_uio
DPDK_DRIVER="${DPDK_DRIVER:-vfio-pci}"
# Number of 2MB hugepages (e.g., 1024 = 2GB)
HUGEPAGES_2M="${HUGEPAGES_2M:-1024}"

# Auto-detect all DPDK-compatible NICs (can be overridden by DPDK_PCI_LIST)
if [[ -z "${DPDK_PCI_LIST:-}" ]]; then
    DPDK_PCI_LIST=$(get_all_dpdk_compatible_pci)
fi
# ===========================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (sudo)"
        exit 1
    fi
}



setup_hugepages() {
    log_info "Setting up hugepages (${HUGEPAGES_2M} x 2MB)..."
    
    # Mount hugetlbfs if not already mounted
    if ! mount | grep -q hugetlbfs; then
        mkdir -p /mnt/huge
        mount -t hugetlbfs nodev /mnt/huge
        log_info "Mounted hugetlbfs at /mnt/huge"
    fi
    
    # Allocate hugepages
    echo "$HUGEPAGES_2M" > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
    
    # Verify allocation
    local allocated=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages)
    log_info "Hugepages allocated: $allocated"
}

teardown_hugepages() {
    log_info "Releasing hugepages..."
    echo 0 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
    
    if mount | grep -q "/mnt/huge"; then
        umount /mnt/huge 2>/dev/null || true
        log_info "Unmounted hugetlbfs"
    fi
}

load_dpdk_driver() {
    log_info "Loading DPDK driver: $DPDK_DRIVER..."
    
    case "$DPDK_DRIVER" in
        vfio-pci)
            modprobe vfio-pci
            # Enable IOMMU if using vfio-pci
            if [[ -d /sys/kernel/iommu_groups ]]; then
                log_info "IOMMU detected"
            else
                log_warn "IOMMU not detected. vfio-pci may require 'enable_unsafe_noiommu_mode'"
                echo 1 > /sys/module/vfio/parameters/enable_unsafe_noiommu_mode 2>/dev/null || true
            fi
            ;;
        uio_pci_generic)
            modprobe uio_pci_generic
            ;;
        igb_uio)
            # igb_uio is typically compiled with DPDK
            modprobe igb_uio 2>/dev/null || {
                log_warn "igb_uio module not found. Make sure DPDK is compiled with igb_uio support."
            }
            ;;
        *)
            log_error "Unknown DPDK driver: $DPDK_DRIVER"
            exit 1
            ;;
    esac
}

# Bind a single interface to DPDK by PCI address
bind_pci() {
    local pci="$1"
    
    # Try to get interface name for logging
    local iface_info
    iface_info=$($DPDK_DEVBIND --status 2>/dev/null | grep "^$pci" | grep -oE "if=[a-zA-Z0-9]+" | cut -d= -f2 || echo "")
    
    if [[ -n "$iface_info" ]]; then
        log_info "Binding $iface_info (PCI: $pci) to $DPDK_DRIVER..."
        # Bring interface down first
        ip link set "$iface_info" down 2>/dev/null || true
    else
        log_info "Binding PCI $pci to $DPDK_DRIVER..."
    fi
    
    # Bind to DPDK driver
    $DPDK_DEVBIND --bind="$DPDK_DRIVER" "$pci"
    log_info "PCI $pci bound to DPDK driver $DPDK_DRIVER"
}

# Unbind a single interface from DPDK and restore kernel driver
unbind_pci() {
    local pci="$1"
    log_info "Unbinding PCI $pci from DPDK driver..."
    
    # Unbind from DPDK driver
    $DPDK_DEVBIND --unbind "$pci" 2>/dev/null || true
    
    # Try to rebind to original kernel driver
    # High-speed drivers first: ixgbe (10GbE), i40e (40GbE), ice (100GbE), mlx5_core
    # Lower-speed drivers last: igb (1GbE), e1000e (1GbE)
    for driver in ixgbe i40e ice mlx5_core mlx4_core bnxt_en igb e1000e; do
        if $DPDK_DEVBIND --bind="$driver" "$pci" 2>/dev/null; then
            log_info "PCI $pci rebound to kernel driver: $driver"
            return 0
        fi
    done
    
    log_warn "Could not rebind PCI $pci to kernel driver. Manual intervention may be required."
    log_info "Try: dpdk-devbind.py --bind=<original_driver> $pci"
    return 0
}

# Bind all detected interfaces to DPDK
bind_all_interfaces() {
    if [[ -z "$DPDK_PCI_LIST" ]]; then
        log_error "No DPDK-compatible NICs detected. Set DPDK_PCI_LIST manually."
        exit 1
    fi
    
    local count=0
    for pci in $DPDK_PCI_LIST; do
        bind_pci "$pci"
        ((count++)) || true
    done
    log_info "Total $count interface(s) bound to DPDK"
}

# Unbind all detected interfaces from DPDK
unbind_all_interfaces() {
    if [[ -z "$DPDK_PCI_LIST" ]]; then
        log_warn "No DPDK-compatible NICs detected in DPDK_PCI_LIST"
        return 0
    fi
    
    local count=0
    for pci in $DPDK_PCI_LIST; do
        unbind_pci "$pci"
        ((count++)) || true
    done
    log_info "Total $count interface(s) unbound from DPDK"
}

start_dpdk() {
    log_info "========== Starting DPDK =========="
    log_info "Detected NICs: $DPDK_PCI_LIST"
    check_root
    
    setup_hugepages
    load_dpdk_driver
    bind_all_interfaces
    
    log_info "========== DPDK Started =========="
    show_status
}

stop_dpdk() {
    log_info "========== Stopping DPDK =========="
    log_info "Detected NICs: $DPDK_PCI_LIST"
    check_root
    
    unbind_all_interfaces
    teardown_hugepages
    
    log_info "========== DPDK Stopped =========="
}

show_status() {
    log_info "DPDK Status:"
    echo ""
    
    # Show hugepage status
    echo "=== Hugepages ==="
    local hp_total=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages 2>/dev/null || echo 0)
    local hp_free=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages 2>/dev/null || echo 0)
    echo "  2MB Hugepages: $hp_free free / $hp_total total"
    echo ""
    
    # Show loaded DPDK drivers
    echo "=== DPDK Drivers ==="
    for mod in vfio-pci uio_pci_generic igb_uio; do
        if lsmod | grep -q "^${mod//-/_}"; then
            echo "  $mod: loaded"
        else
            echo "  $mod: not loaded"
        fi
    done
    echo ""
    
    # Show device bindings
    echo "=== Network Device Bindings ==="
    $DPDK_DEVBIND --status 2>/dev/null || {
        log_warn "dpdk-devbind.py not found in PATH"
    }
}

print_usage() {
    echo "Usage: $0 {start|stop|status}"
    echo ""
    echo "Commands:"
    echo "  start   - Setup hugepages, load DPDK driver, bind all DPDK-compatible interfaces"
    echo "  stop    - Unbind all interfaces, release hugepages"
    echo "  status  - Show current DPDK status"
    echo ""
    echo "Environment Variables:"
    echo "  DPDK_PCI_LIST  - Space-separated list of PCI addresses (auto-detects all 10Gbps+ NICs)"
    echo "  DPDK_DRIVER    - DPDK driver: vfio-pci, uio_pci_generic, igb_uio (default: vfio-pci)"
    echo "  HUGEPAGES_2M   - Number of 2MB hugepages to allocate (default: 1024)"
    echo "  DPDK_DEVBIND   - Path to dpdk-devbind.py (default: dpdk-devbind.py)"
    echo ""
    echo "Supported NICs: Intel 10/40/100GbE (ixgbe/i40e/ice), Mellanox ConnectX (mlx5_core)"
    echo ""
    echo "Example:"
    echo "  sudo $0 start                                    # Auto-detect all NICs"
    echo "  sudo DPDK_PCI_LIST='0000:01:00.0 0000:01:00.1' $0 start  # Specific NICs"
}

# Main
case "${1:-}" in
    start)
        start_dpdk
        ;;
    stop)
        stop_dpdk
        ;;
    status)
        show_status
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
