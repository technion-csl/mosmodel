#!/bin/bash

# Check and set CPU governor to performance
echo "Checking CPU governor settings..."
current_governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
if [ "$current_governor" != "performance" ]; then
    echo "Setting CPU frequency governor to performance..."
    sudo cpupower frequency-set --governor performance
    
    # Get maximum frequency
    max_freq=$(cpupower frequency-info -l | grep -oP '\d+\.?\d*' | sort -n | tail -1)
    current_max=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)
    if [ "$current_max" != "$max_freq" ]; then
        echo "Setting CPU frequency to maximum: $max_freq"
        sudo cpupower frequency-set --max $max_freq
    else
        echo "Maximum frequency already set to $max_freq"
    fi
    
    # Set performance governor for all CPUs
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        if [ -f "$cpu/cpufreq/scaling_governor" ]; then
            cpu_governor=$(cat "$cpu/cpufreq/scaling_governor")
            if [ "$cpu_governor" != "performance" ]; then
                echo "Setting $(basename $cpu) to performance governor"
                echo performance | sudo tee "$cpu/cpufreq/scaling_governor"
            fi
        fi
    done
else
    echo "CPU governor already set to performance"
fi

# Check and disable Transparent Huge Pages
echo "Checking Transparent Huge Pages (THP) status..."
thp_enabled=$(cat /sys/kernel/mm/transparent_hugepage/enabled)
if [[ $thp_enabled != *"[never]"* ]]; then
    echo "Disabling Transparent Huge Pages (THP)..."
    echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
else
    echo "THP already disabled"
fi

thp_defrag=$(cat /sys/kernel/mm/transparent_hugepage/defrag)
if [[ $thp_defrag != *"[never]"* ]]; then
    echo "Disabling THP defrag..."
    echo never | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
else
    echo "THP defrag already disabled"
fi

# Check and disable NUMA balancing
echo "Checking NUMA balancing status..."
numa_balancing=$(cat /proc/sys/kernel/numa_balancing)
if [ "$numa_balancing" != "0" ]; then
    echo "Disabling automatic NUMA balancing..."
    echo 0 | sudo tee /proc/sys/kernel/numa_balancing
else
    echo "NUMA balancing already disabled"
fi

# Check and disable hyper-threading
# echo "Checking hyper-threading status..."
# if [ -f /sys/devices/system/cpu/smt/control ]; then
#     smt_status=$(cat /sys/devices/system/cpu/smt/control)
#     if [ "$smt_status" != "off" ]; then
#         echo "Disabling hyper-threading..."
#         echo off | sudo tee /sys/devices/system/cpu/smt/control
#     else
#         echo "Hyper-threading already disabled"
#     fi
# else
#     echo "SMT control not available on this system"
# fi

# Check and disable C-states
echo "Checking C-states..."
c_states_changed=0
for cpu in /sys/devices/system/cpu/cpu*/cpuidle/state[1-3]/disable; do
    if [ -f "$cpu" ]; then
        state_status=$(cat "$cpu")
        if [ "$state_status" != "1" ]; then
            if [ $c_states_changed -eq 0 ]; then
                echo "Disabling C-states..."
                c_states_changed=1
            fi
            echo 1 | sudo tee "$cpu"
        fi
    fi
done
if [ $c_states_changed -eq 0 ]; then
    echo "C-states already disabled"
fi

# Check and disable Intel CPU Turbo
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    turbo_status=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
    if [ "$turbo_status" != "1" ]; then
        echo "Disabling Intel CPU Turbo..."
        echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
    else
        echo "Intel CPU Turbo already disabled"
    fi
else
    echo "Intel CPU Turbo control not available on this system"
fi

# Check and disable watchdog
if [ -f /proc/sys/kernel/watchdog ]; then
    watchdog_status=$(cat /proc/sys/kernel/watchdog)
    if [ "$watchdog_status" != "0" ]; then
        echo "Turning off watchdog..."
        echo 0 | sudo tee /proc/sys/kernel/watchdog
    else
        echo "Watchdog already disabled"
    fi
else
    echo "Watchdog control not available on this system"
fi

# Check and set vmstat interval
if [ -f /proc/sys/vm/stat_interval ]; then
    current_interval=$(cat /proc/sys/vm/stat_interval)
    if [ "$current_interval" != "1000000" ]; then
        echo "Extending vmstat_update interval to 1M seconds..."
        echo 1000000 | sudo tee /proc/sys/vm/stat_interval
    else
        echo "vmstat interval already set to 1M seconds"
    fi
else
    echo "vmstat interval control not available on this system"
fi

# Check and disable ASLR
aslr_status=$(cat /proc/sys/kernel/randomize_va_space)
if [ "$aslr_status" != "0" ]; then
    echo "Disabling ASLR..."
    echo 0 | sudo tee /proc/sys/kernel/randomize_va_space
else
    echo "ASLR already disabled"
fi

# Check and disable Kernel Samepage Merging
if [ -f /sys/kernel/mm/ksm/run ]; then
    ksm_status=$(cat /sys/kernel/mm/ksm/run)
    if [ "$ksm_status" != "0" ]; then
        echo "Disabling Kernel Samepage Merging (KSM)..."
        echo 0 | sudo tee /sys/kernel/mm/ksm/run
    else
        echo "KSM already disabled"
    fi
else
    echo "KSM control not available on this system"
fi

echo "Performance optimization settings check completed."