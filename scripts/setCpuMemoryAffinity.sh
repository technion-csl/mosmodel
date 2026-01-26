#! /bin/bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  $0 <node_number> <command...>                     # ST: bind to node cpulist"
  echo "  $0 <node_number> --smt <1|2> <command...>         # SMT: bind to a sibling CPU within the node"
  echo "  $0 <node_number> --cpu <cpu_list> <command...>    # Explicit CPU list"
  exit 1
}

if (( $# < 2 )); then
  usage
fi

node_number="$1"
shift

mode="st"
which_smt=""
explicit_cpu_list=""

if [[ "${1:-}" == "--smt" ]]; then
  if (( $# < 3 )); then usage; fi
  mode="smt"
  which_smt="$2"   # "1" or "2"
  shift 2
elif [[ "${1:-}" == "--cpu" ]]; then
  if (( $# < 3 )); then usage; fi
  mode="cpu"
  explicit_cpu_list="$2"
  shift 2
fi

command=("$@")

node_cpulist_path="/sys/devices/system/node/node${node_number}/cpulist"
if [[ ! -r "$node_cpulist_path" ]]; then
  echo "Error: cannot read $node_cpulist_path"
  exit 2
fi

node_cpulist="$(cat "$node_cpulist_path")"

# Return the first CPU id from a cpulist string like "0-15,32-47"
first_cpu_from_cpulist() {
  local s="$1"
  local first_chunk="${s%%,*}"     # up to first comma
  local first_cpu="${first_chunk%%-*}"  # up to dash (or the whole chunk)
  echo "$first_cpu"
}

# Return a concrete CPU id from a siblings list like "0,32" (choose the one != base)
second_sibling_cpu() {
  local base="$1"
  local sib_path="/sys/devices/system/cpu/cpu${base}/topology/thread_siblings_list"
  if [[ ! -r "$sib_path" ]]; then
    echo "Error: cannot read $sib_path"
    exit 3
  fi
  local sibs="$(cat "$sib_path")"
  # Normalize: take first two entries before any '-' ranges
  IFS=',' read -r a b _ <<< "$sibs"
  a="${a%%-*}"
  b="${b%%-*}"

  if [[ -z "${b:-}" ]]; then
    echo "Error: CPU${base} has no second sibling in $sibs (SMT may be disabled)"
    exit 4
  fi
  if [[ "$a" == "$base" ]]; then
    echo "$b"
  else
    echo "$a"
  fi
}

cpu_list="$node_cpulist"
if [[ "$mode" == "cpu" ]]; then
  cpu_list="$explicit_cpu_list"
elif [[ "$mode" == "smt" ]]; then
  base="$(first_cpu_from_cpulist "$node_cpulist")"
  sib="$(second_sibling_cpu "$base")"
  if [[ "$which_smt" == "1" ]]; then
    cpu_list="$base"
  elif [[ "$which_smt" == "2" ]]; then
    cpu_list="$sib"
  else
    echo "Error: --smt must be 1 or 2"
    exit 5
  fi
fi

command="$@"

echo "Binding the process to memory node: $node_number"
echo "and CPU(s): $cpu_list"

taskset_command="taskset --cpu-list $cpu_list"
numactl_command="numactl --membind $node_number"

submit_command="$taskset_command $numactl_command"

echo submit command: $submit_command
echo running command: $command
$submit_command $command


