#!/usr/bin/env bash
#
# Usage:
#   export ISOLATED_CPUS="2-3,5"
#   export ISOLATED_MEMORY_NODE="0"
#   ./check_isolated_setup.sh
#
# This script checks:
#   1) ISOLATED_CPUS is set and matches /sys/devices/system/cpu/isolated
#   2) ISOLATED_MEMORY_NODE is set if ISOLATED_CPUS is set
#   3) All CPUs in ISOLATED_CPUS actually belong to the node in ISOLATED_MEMORY_NODE

set -euo pipefail

#######################################
# Function to expand CPU ranges, e.g.:
#   "2-3,5" -> "2 3 5"
#######################################
expand_cpu_ranges() {
  local cpurange="$1"
  python3 -c "
import sys
s = sys.argv[1]
expanded = []
for part in s.split(','):
    if '-' in part:
        start, end = part.split('-')
        for c in range(int(start), int(end)+1):
            expanded.append(str(c))
    else:
        expanded.append(part)
print(' '.join(expanded))
" "$cpurange"
}

#########################################
# 1) Check if ISOLATED_CPUS is set
#########################################
if [ -z "${ISOLATED_CPUS-}" ]; then
  echo "WARNING: ISOLATED_CPUS is not set — skipping isolation checks." >&2
  exit 0
fi

#########################################
# Compare ISOLATED_CPUS to the system's
#  /sys/devices/system/cpu/isolated
#########################################
SYSTEM_ISOLATED_CPUS="$(cat /sys/devices/system/cpu/isolated 2>/dev/null || true)"

if [ "$ISOLATED_CPUS" != "$SYSTEM_ISOLATED_CPUS" ]; then
  echo "ERROR: Mismatch in isolated CPUs." >&2
  echo "       System indicates: '$SYSTEM_ISOLATED_CPUS'" >&2
  echo "       You set:         '$ISOLATED_CPUS'" >&2
  exit 1
fi

#########################################
# 2) Ensure ISOLATED_MEMORY_NODE is set
#########################################
if [ -z "${ISOLATED_MEMORY_NODE-}" ]; then
  echo "ERROR: ISOLATED_MEMORY_NODE is not set, but ISOLATED_CPUS is set!" >&2
  exit 1
fi

#########################################
# 3) Expand ISOLATED_CPUS and check nodes
#########################################
EXPANDED_ISOLATED="$(expand_cpu_ranges "$ISOLATED_CPUS")"

mismatch=0

for core in $EXPANDED_ISOLATED; do
  # readlink -f /sys/devices/system/cpu/cpuX/nodeY -> /sys/devices/system/node/nodeY
  node_path="$(readlink -f /sys/devices/system/cpu/cpu"$core"/node* 2>/dev/null || true)"
  node_id="${node_path##*node}"

  # If node is empty or mismatch
  if [ -z "$node_id" ] || [ "$node_id" != "$ISOLATED_MEMORY_NODE" ]; then
    echo "ERROR: CPU $core is on node '$node_id', not node '$ISOLATED_MEMORY_NODE'!" >&2
    mismatch=1
  fi
done

if [ "$mismatch" -eq 1 ]; then
  echo "ERROR: One or more isolated CPUs are not on node $ISOLATED_MEMORY_NODE." >&2
  exit 1
fi

#########################################
# 4) Success if we reach here
#########################################
echo "All isolated CPUs ($ISOLATED_CPUS) match node $ISOLATED_MEMORY_NODE."
exit 0

