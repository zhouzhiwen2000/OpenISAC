#!/bin/bash
# Increase kernel socket buffer maximum sizes to ~32MB
sudo sysctl -w net.core.wmem_max=33554432
sudo sysctl -w net.core.rmem_max=33554432
# Set default socket buffer sizes to ~32MB
sudo sysctl -w net.core.wmem_default=33554432
sudo sysctl -w net.core.rmem_default=33554432

# Set CPU frequency governor to performance for all cores
for ((i=0;i<$(nproc --all);i++)); do sudo cpufreq-set -c $i -r -g performance; done

# Detect and configure high-speed interfaces (>= 10Gbps)
echo "Configuring network interfaces..."
for iface in $(ls /sys/class/net/); do
    # Skip loopback and check for speed file
    if [ "$iface" != "lo" ] && [ -r "/sys/class/net/$iface/speed" ]; then
        speed=$(cat /sys/class/net/$iface/speed 2>/dev/null)
        # Check if speed is a number and >= 10000 (10Gbps)
        if [[ "$speed" =~ ^[0-9]+$ ]] && [ "$speed" -ge 10000 ]; then
            echo "  Found high-speed interface: $iface ($speed Mbps)"
            
            # Increase NIC ring buffer sizes to 4096
            echo "    Setting ring buffers to 4096..."
            sudo ethtool -G $iface tx 4096 rx 4096
            
            # Set MTU to 9000
            echo "    Setting MTU to 9000..."
            sudo ifconfig $iface mtu 9000
        fi
    fi
done

# Disable Real-Time Throttling
sudo bash -c "cd /sys/kernel/debug/sched/ && echo RT_RUNTIME_SHARE > features"

# Optional: Bring down interfaces
# sudo ifconfig enp1s0f0 down && sudo ifconfig enp1s0f1 down

# Load VFIO-PCI driver
# sudo modprobe vfio-pci

# Bind network devices to VFIO-PCI for DPDK usage (Example)
# sudo dpdk-devbind.py --bind=vfio-pci 01:00.0 01:00.1