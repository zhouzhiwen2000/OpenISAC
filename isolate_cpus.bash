#!/bin/bash

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Error: Please run this script with sudo"
  exit 1
fi

# Define paths
ATOM_CPUS_PATH="/sys/devices/cpu_atom/cpus"
CORE_CPUS_PATH="/sys/devices/cpu_core/cpus"
APP_CPUS_CONFIG="/tmp/isolate_cpus_app.conf"

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

# 0. Help
show_help() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "A script to isolate CPUs for high-performance applications by restricting"
    echo "system services to a subset of CPUs."
    echo ""
    echo "Commands:"
    echo "  reset              Reset CPU affinity, allowing system to use all CPUs"
    echo "  run <command>      Run a command on the reserved (app) CPUs"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Options (for isolation setup):"
    echo "  <number>           Reserve the first N cores for app (e.g., 8)"
    echo "  <range>            Specify app CPU range (e.g., 8-15 means app uses 8-15)"
    echo "  <comma-list>       Specify app CPUs as comma-separated list (e.g., 0,1,2,3)"
    echo ""
    echo "Examples:"
    echo "  sudo $0                    # Use default (first $DEFAULT_RESERVED_COUNT cores for app)"
    echo "  sudo $0 4                  # Reserve first 4 cores (0-3) for app"
    echo "  sudo $0 8-15               # Reserve cores 8-15 for app"
    echo "  sudo $0 0,2,4,6            # Reserve cores 0,2,4,6 for app"
    echo "  sudo $0 reset              # Reset to allow all CPUs for system"
    echo "  sudo $0 run ./my_app       # Run my_app on reserved cores"
    echo ""
    echo "System Info:"
    echo "  Total CPUs: $TOTAL_CPUS (0-$MAX_CPU)"
    if [ -n "$ATOM_CPUS" ]; then
        echo "  P-Cores: $CORE_CPUS"
        echo "  E-Cores: $ATOM_CPUS"
    fi
    exit 0
}

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_help
fi

# 1. Reset
if [ "$1" == "reset" ]; then
    echo "Resetting CPU affinity settings..."
    
    ALL_CPUS="0-$MAX_CPU"
    
    # Method 1: Set to all CPUs explicitly (more reliable than empty value)
    systemctl set-property --now system.slice AllowedCPUs="$ALL_CPUS" 2>/dev/null
    systemctl set-property --now user.slice   AllowedCPUs="$ALL_CPUS" 2>/dev/null
    systemctl set-property --now init.scope   AllowedCPUs="$ALL_CPUS" 2>/dev/null
    
    # Method 2: Remove persistent override files (if any)
    rm -f /etc/systemd/system/system.slice.d/50-AllowedCPUs.conf 2>/dev/null
    rm -f /etc/systemd/system/user.slice.d/50-AllowedCPUs.conf 2>/dev/null
    rm -f /etc/systemd/system/init.scope.d/50-AllowedCPUs.conf 2>/dev/null
    
    # Remove entire override directories if empty
    rmdir /etc/systemd/system/system.slice.d 2>/dev/null
    rmdir /etc/systemd/system/user.slice.d 2>/dev/null
    rmdir /etc/systemd/system/init.scope.d 2>/dev/null
    
    # Reload systemd to pick up the changes
    systemctl daemon-reload
    
    # Clear saved app CPU config
    rm -f "$APP_CPUS_CONFIG" 2>/dev/null
    
    echo "Reset complete. System can now use all CPUs ($ALL_CPUS)."
    echo ""
    echo "Note: You may need to restart your shell or re-login for changes to fully apply."
    exit 0
fi

# 2. Run (Run command on reserved cores)
if [ "$1" == "run" ]; then
    shift # Remove 'run'
    
    # Priority 1: Read from saved config file
    if [ -f "$APP_CPUS_CONFIG" ]; then
        RUN_CPUS=$(cat "$APP_CPUS_CONFIG")
        echo "Using saved app CPU config: $RUN_CPUS"
    else
        # Priority 2: Use default
        RUN_CPUS="$DEFAULT_APP_CPUS"
        echo "No saved config found, using default: $RUN_CPUS"
    fi
    
    # Priority 3: If still empty, try using detected P-Cores
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
    elif [[ "$1" =~ ^[0-9]+(,[0-9]+)+$ ]]; then
        # Comma-separated list of app CPUs (e.g., "0,1,2,3" or "0,2,4,6")
        APP_CPU_LIST="$1"
        echo "Manually specified app CPU list: $APP_CPU_LIST"
        
        # Parse the comma-separated list into an array
        IFS=',' read -ra APP_CPUS_ARRAY <<< "$APP_CPU_LIST"
        
        # Build array of all CPUs
        declare -A APP_CPU_SET
        for cpu in "${APP_CPUS_ARRAY[@]}"; do
            if [ "$cpu" -gt "$MAX_CPU" ]; then
                echo "Warning: CPU $cpu exceeds max CPU $MAX_CPU, ignoring."
                continue
            fi
            APP_CPU_SET[$cpu]=1
        done
        
        # Calculate complement (system CPUs)
        SYSTEM_CPUS=()
        for ((i=0; i<=MAX_CPU; i++)); do
            if [ -z "${APP_CPU_SET[$i]}" ]; then
                SYSTEM_CPUS+=($i)
            fi
        done
        
        if [ ${#SYSTEM_CPUS[@]} -eq 0 ]; then
            echo "Error: No CPUs left for system after reserving $APP_CPU_LIST"
            exit 1
        fi
        
        # Convert system CPUs array to comma-separated string
        ISOLATION_TARGET=$(IFS=','; echo "${SYSTEM_CPUS[*]}")
        FREE_CPUS="$APP_CPU_LIST"
    elif [[ "$1" =~ ^[0-9]+-[0-9]+$ ]]; then
        # Range specification for app CPUs (e.g., "8-15" means app uses 8-15)
        APP_CPU_RANGE="$1"
        echo "Manually specified app CPU range: $APP_CPU_RANGE"
        
        # Parse range
        RANGE_START=$(echo "$APP_CPU_RANGE" | cut -d'-' -f1)
        RANGE_END=$(echo "$APP_CPU_RANGE" | cut -d'-' -f2)
        
        # Boundary check
        if [ "$RANGE_END" -gt "$MAX_CPU" ]; then
            RANGE_END=$MAX_CPU
            echo "Warning: Range end exceeds max CPU, adjusted to $RANGE_END"
        fi
        
        # Build set of app CPUs
        declare -A APP_CPU_SET
        for ((i=RANGE_START; i<=RANGE_END; i++)); do
            APP_CPU_SET[$i]=1
        done
        
        # Calculate complement (system CPUs)
        SYSTEM_CPUS=()
        for ((i=0; i<=MAX_CPU; i++)); do
            if [ -z "${APP_CPU_SET[$i]}" ]; then
                SYSTEM_CPUS+=($i)
            fi
        done
        
        if [ ${#SYSTEM_CPUS[@]} -eq 0 ]; then
            echo "Error: No CPUs left for system after reserving $APP_CPU_RANGE"
            exit 1
        fi
        
        # Convert system CPUs array to comma-separated string
        ISOLATION_TARGET=$(IFS=','; echo "${SYSTEM_CPUS[*]}")
        FREE_CPUS="$APP_CPU_RANGE"
    else
        # Unknown format
        echo "Error: Unrecognized format '$1'"
        echo "Use: <number>, <start>-<end>, or <cpu1>,<cpu2>,..."
        exit 1
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

# Save app CPU config for 'run' command
if [ "$FREE_CPUS" != "None" ] && [[ ! "$FREE_CPUS" =~ ^\[.*\]$ ]]; then
    echo "$FREE_CPUS" > "$APP_CPUS_CONFIG"
    echo "Saved app CPU config to $APP_CPUS_CONFIG"
fi

echo "Setup successful!"
echo "Since the current Shell is also restricted to system cores, you cannot use taskset directly."
echo "Please use the following command to run your application on reserved cores:"
echo ""
echo "  sudo ./isolate_cpus.bash run ./your_application"
echo ""