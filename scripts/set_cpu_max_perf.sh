#!/bin/bash
set -euo pipefail

MODE=""
STATE=""

usage() {
  echo "Usage: $0 --mode st|smt --state <state_file>" >&2
  exit 2
}

while (( $# )); do
  case "$1" in
    --mode)  MODE="${2:-}"; shift 2 ;;
    --state) STATE="${2:-}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

[[ "$MODE" == "st" || "$MODE" == "smt" ]] || usage
[[ -n "$STATE" ]] || usage

mkdir -p "$(dirname "$STATE")"
: > "$STATE"

sudo_write() { echo "$2" | sudo tee "$1" >/dev/null; }

# Parse selected THP option from: "always madvise [never]"
thp_selected() { sed -n 's/.*\[\([^]]\+\)\].*/\1/p' "$1" | head -n1; }

save_kv() { echo "$1=$2" >> "$STATE"; }
save_path() {
  local p="$1"
  [[ -f "$p" ]] || return 0
  save_kv "PATH::$p" "$(cat "$p")"
}

echo "[cpu_max_perf] Mode: $MODE"
echo "[cpu_max_perf] Saving state to: $STATE"
save_kv "MODE" "$MODE"

# --------------------------------------------------------------------
# CPU governor + max frequency
# --------------------------------------------------------------------
echo "Checking CPU governor settings..."
if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
  save_path /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
fi
if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq ]]; then
  save_path /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
fi

current_governor="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "")"
if [[ "$current_governor" != "performance" ]]; then
  echo "Setting CPU frequency governor to performance..."
  if command -v cpupower >/dev/null 2>&1; then
    sudo cpupower frequency-set --governor performance || true

    # Get maximum frequency from cpupower
    max_freq="$(cpupower frequency-info -l | grep -oP '\d+\.?\d*' | sort -n | tail -1 || true)"
    current_max="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo "")"
    if [[ -n "$max_freq" && "$current_max" != "$max_freq" ]]; then
      echo "Setting CPU frequency to maximum: $max_freq"
      sudo cpupower frequency-set --max "$max_freq" || true
    else
      echo "Maximum frequency already set to $max_freq"
    fi
  else
    echo "cpupower is not available; only setting per-cpu governor via sysfs where possible"
  fi

  # Set performance governor for all CPUs (save + set)
  for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    if [[ -f "$cpu/cpufreq/scaling_governor" ]]; then
      save_path "$cpu/cpufreq/scaling_governor"
      cpu_governor="$(cat "$cpu/cpufreq/scaling_governor")"
      if [[ "$cpu_governor" != "performance" ]]; then
        echo "Setting $(basename "$cpu") to performance governor"
        sudo_write "$cpu/cpufreq/scaling_governor" "performance"
      fi
    fi
    if [[ -f "$cpu/cpufreq/scaling_max_freq" ]]; then
      save_path "$cpu/cpufreq/scaling_max_freq"
    fi
  done
else
  echo "CPU governor already set to performance"
  # still save per-cpu governors for restore
  for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    [[ -f "$cpu/cpufreq/scaling_governor" ]] && save_path "$cpu/cpufreq/scaling_governor"
    [[ -f "$cpu/cpufreq/scaling_max_freq" ]] && save_path "$cpu/cpufreq/scaling_max_freq"
  done
fi

# --------------------------------------------------------------------
# THP
# --------------------------------------------------------------------
echo "Checking Transparent Huge Pages (THP) status..."
if [[ -f /sys/kernel/mm/transparent_hugepage/enabled ]]; then
  save_kv "THP_ENABLED" "$(thp_selected /sys/kernel/mm/transparent_hugepage/enabled)"
  thp_enabled="$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  if [[ $thp_enabled != *"[never]"* ]]; then
    echo "Disabling Transparent Huge Pages (THP)..."
    sudo_write /sys/kernel/mm/transparent_hugepage/enabled "never"
  else
    echo "THP already disabled"
  fi
fi

if [[ -f /sys/kernel/mm/transparent_hugepage/defrag ]]; then
  save_kv "THP_DEFRAG" "$(thp_selected /sys/kernel/mm/transparent_hugepage/defrag)"
  thp_defrag="$(cat /sys/kernel/mm/transparent_hugepage/defrag)"
  if [[ $thp_defrag != *"[never]"* ]]; then
    echo "Disabling THP defrag..."
    sudo_write /sys/kernel/mm/transparent_hugepage/defrag "never"
  else
    echo "THP defrag already disabled"
  fi
fi

# --------------------------------------------------------------------
# NUMA balancing
# --------------------------------------------------------------------
echo "Checking NUMA balancing status..."
if [[ -f /proc/sys/kernel/numa_balancing ]]; then
  save_path /proc/sys/kernel/numa_balancing
  numa_balancing="$(cat /proc/sys/kernel/numa_balancing)"
  if [[ "$numa_balancing" != "0" ]]; then
    echo "Disabling automatic NUMA balancing..."
    sudo_write /proc/sys/kernel/numa_balancing "0"
  else
    echo "NUMA balancing already disabled"
  fi
fi

# --------------------------------------------------------------------
# Hyper-threading (SMT) control: mode-dependent
# --------------------------------------------------------------------
echo "Checking hyper-threading (SMT) status..."
if [[ -f /sys/devices/system/cpu/smt/control ]]; then
  save_path /sys/devices/system/cpu/smt/control
  smt_status="$(cat /sys/devices/system/cpu/smt/control)"
  echo "Current SMT control: $smt_status"
  if [[ "$MODE" == "st" ]]; then
    if [[ "$smt_status" != "off" ]]; then
      echo "Disabling hyper-threading..."
      sudo_write /sys/devices/system/cpu/smt/control "off" || true
    else
      echo "Hyper-threading already disabled"
    fi
  else
    if [[ "$smt_status" != "on" ]]; then
      echo "Enabling hyper-threading..."
      sudo_write /sys/devices/system/cpu/smt/control "on" || true
    else
      echo "Hyper-threading already enabled"
    fi
  fi
else
  echo "SMT control not available on this system"
fi

# --------------------------------------------------------------------
# C-states
# --------------------------------------------------------------------
echo "Checking C-states..."
c_states_changed=0
for p in /sys/devices/system/cpu/cpu*/cpuidle/state[1-3]/disable; do
  [[ -f "$p" ]] || continue
  save_path "$p"
  state_status="$(cat "$p")"
  if [[ "$state_status" != "1" ]]; then
    if [[ $c_states_changed -eq 0 ]]; then
      echo "Disabling C-states..."
      c_states_changed=1
    fi
    sudo_write "$p" "1"
  fi
done
if [[ $c_states_changed -eq 0 ]]; then
  echo "C-states already disabled"
fi

# --------------------------------------------------------------------
# Turbo
# --------------------------------------------------------------------
if [[ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
  save_path /sys/devices/system/cpu/intel_pstate/no_turbo
  turbo_status="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
  if [[ "$turbo_status" != "1" ]]; then
    echo "Disabling Intel CPU Turbo..."
    sudo_write /sys/devices/system/cpu/intel_pstate/no_turbo "1"
  else
    echo "Intel CPU Turbo already disabled"
  fi
else
  echo "Intel CPU Turbo control not available on this system"
fi

# --------------------------------------------------------------------
# Watchdog
# --------------------------------------------------------------------
if [[ -f /proc/sys/kernel/watchdog ]]; then
  save_path /proc/sys/kernel/watchdog
  watchdog_status="$(cat /proc/sys/kernel/watchdog)"
  if [[ "$watchdog_status" != "0" ]]; then
    echo "Turning off watchdog..."
    sudo_write /proc/sys/kernel/watchdog "0"
  else
    echo "Watchdog already disabled"
  fi
else
  echo "Watchdog control not available on this system"
fi

# --------------------------------------------------------------------
# vmstat interval
# --------------------------------------------------------------------
if [[ -f /proc/sys/vm/stat_interval ]]; then
  save_path /proc/sys/vm/stat_interval
  current_interval="$(cat /proc/sys/vm/stat_interval)"
  if [[ "$current_interval" != "1000000" ]]; then
    echo "Extending vmstat_update interval to 1M seconds..."
    sudo_write /proc/sys/vm/stat_interval "1000000"
  else
    echo "vmstat interval already set to 1M seconds"
  fi
else
  echo "vmstat interval control not available on this system"
fi

# --------------------------------------------------------------------
# ASLR
# --------------------------------------------------------------------
if [[ -f /proc/sys/kernel/randomize_va_space ]]; then
  save_path /proc/sys/kernel/randomize_va_space
  aslr_status="$(cat /proc/sys/kernel/randomize_va_space)"
  if [[ "$aslr_status" != "0" ]]; then
    echo "Disabling ASLR..."
    sudo_write /proc/sys/kernel/randomize_va_space "0"
  else
    echo "ASLR already disabled"
  fi
fi

# --------------------------------------------------------------------
# KSM
# --------------------------------------------------------------------
if [[ -f /sys/kernel/mm/ksm/run ]]; then
  save_path /sys/kernel/mm/ksm/run
  ksm_status="$(cat /sys/kernel/mm/ksm/run)"
  if [[ "$ksm_status" != "0" ]]; then
    echo "Disabling Kernel Samepage Merging (KSM)..."
    sudo_write /sys/kernel/mm/ksm/run "0"
  else
    echo "KSM already disabled"
  fi
else
  echo "KSM control not available on this system"
fi

echo "Performance optimization settings check completed."
echo "[cpu_max_perf] State saved to: $STATE"
