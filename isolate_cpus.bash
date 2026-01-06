#!/bin/bash

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Error: Please run this script with sudo"
  exit 1
fi

# Define paths
ATOM_CPUS_PATH="/sys/devices/cpu_atom/cpus"
CORE_CPUS_PATH="/sys/devices/cpu_core/cpus"

# Get max CPU index (use --all to ignore current affinity limits)
MAX_CPU=$(($(nproc --all) - 1))
TOTAL_CPUS=$(($(nproc --all)))

# Default config: First 8 cores (0-7) reserved for app, rest for system
DEFAULT_RESERVED_COUNT=8

if [ "$TOTAL_CPUS" -gt "$DEFAULT_RESERVED_COUNT" ]; then
    DEFAULT_APP_CPUS="0-$(($DEFAULT_RESERVED_COUNT - 1))"
    DEFAULT_ISOLATION="$DEFAULT_RESERVED_COUNT-$MAX_CPU"
else
    # If not enough cores, cannot isolate effectively, let system use all cores
    DEFAULT_APP_CPUS="0-$MAX_CPU"
    DEFAULT_ISOLATION="0-$MAX_CPU"
fi

# Auto-detect Hybrid Architecture (Intel Hybrid Architecture)
if [ -f "$ATOM_CPUS_PATH" ] && [ -f "$CORE_CPUS_PATH" ]; then
    ATOM_CPUS=$(cat "$ATOM_CPUS_PATH")
    CORE_CPUS=$(cat "$CORE_CPUS_PATH")
else
    ATOM_CPUS=""
    CORE_CPUS=""
fi

# --- Handle subcommands ---

# 1. Reset
if [ "$1" == "reset" ]; then
    echo "Resetting CPU affinity settings..."
    systemctl set-property --now system.slice AllowedCPUs=
    systemctl set-property --now user.slice   AllowedCPUs=
    systemctl set-property --now init.scope   AllowedCPUs=
    echo "Reset complete. System can now use all CPUs."
    exit 0
fi

# 2. Run (Run command on reserved cores)
if [ "$1" == "run" ]; then
    shift # Remove 'run'
    RUN_CPUS="$DEFAULT_APP_CPUS"
    
    # If default app cores is empty (e.g. not enough cores), try using detected P-Cores
    if [ -z "$RUN_CPUS" ] && [ -n "$CORE_CPUS" ]; then
        RUN_CPUS="$CORE_CPUS"
    fi

    if [ -z "$RUN_CPUS" ]; then
        echo "Error: Cannot determine core range for running application."
        exit 1
    fi
    echo "Starting process on reserved cores ($RUN_CPUS)..."
    # Use systemd-run to create a new slice to bypass user.slice limits
    # -p WorkingDirectory=$(pwd) ensures working directory matches current
    systemd-run --slice=rt-workload.slice --pty -p AllowedCPUs=$RUN_CPUS -p WorkingDirectory=$(pwd) "$@"
    exit $?
fi

# --- Main logic: Set isolation ---

ISOLATION_TARGET=""
FREE_CPUS=""

# Check for manual arguments (and not reset/run)
if [ -n "$1" ]; then
    # If argument is a number, treat as 'reserved core count'
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        RESERVED_COUNT="$1"
        echo "Manually specified reserved core count: $RESERVED_COUNT"
        
        # Boundary check
        if [ "$RESERVED_COUNT" -ge "$TOTAL_CPUS" ]; then
             RESERVED_COUNT=$(($TOTAL_CPUS - 1))
             echo "Warning: Reserved core count must be less than total cores. Adjusted to: $RESERVED_COUNT"
        fi
        
        if [ "$RESERVED_COUNT" -le 0 ]; then
             ISOLATION_TARGET="0-$MAX_CPU"
             FREE_CPUS="None"
        else
             ISOLATION_TARGET="$RESERVED_COUNT-$MAX_CPU"
             FREE_CPUS="0-$(($RESERVED_COUNT - 1))"
        fi
    else
        # Otherwise treat as direct system core range specification
        ISOLATION_TARGET="$1"
        echo "Manually specified system CPU range: $ISOLATION_TARGET"
        FREE_CPUS="[CPUs not in $ISOLATION_TARGET]"
    fi
else
    # Use default policy
    ISOLATION_TARGET="$DEFAULT_ISOLATION"
    FREE_CPUS="$DEFAULT_APP_CPUS"
    
    if [ -n "$ATOM_CPUS" ]; then
        echo "Detected Hybrid Architecture CPU (Reference):"
        echo "  P-Cores: $CORE_CPUS"
        echo "  E-Cores: $ATOM_CPUS"
    fi
    echo "Applied policy: System uses cores ($ISOLATION_TARGET), reserves first $DEFAULT_RESERVED_COUNT cores ($FREE_CPUS) for application."
fi

echo "------------------------------------------------"
echo "Configuring system isolation..."
echo "Restricting system services, user sessions, and Init processes to CPU: $ISOLATION_TARGET"
echo "Reserving CPU $FREE_CPUS for high-performance applications"
echo "------------------------------------------------"

# Apply settings
systemctl set-property --now system.slice AllowedCPUs=$ISOLATION_TARGET
systemctl set-property --now user.slice   AllowedCPUs=$ISOLATION_TARGET
systemctl set-property --now init.scope   AllowedCPUs=$ISOLATION_TARGET

echo "Setup successful!"
echo "Since the current Shell is also restricted to system cores, you cannot use taskset directly."
echo "Please use the following command to run your application on reserved cores:"
echo ""
echo "  sudo ./isolate_cpus.bash run ./your_application"
echo ""