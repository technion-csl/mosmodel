#!/bin/bash
set -euo pipefail

STATE=""

usage() { echo "Usage: $0 --state <state_file>" >&2; exit 2; }

while (( $# )); do
  case "$1" in
    --state) STATE="${2:-}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

[[ -n "$STATE" ]] || usage
[[ -f "$STATE" ]] || { echo "State file not found: $STATE" >&2; exit 2; }

sudo_write() { echo "$2" | sudo tee "$1" >/dev/null; }

# Restore PATH::<path>=<value>
while IFS= read -r line; do
  [[ "$line" == PATH::*=* ]] || continue
  key="${line%%=*}"
  val="${line#*=}"
  path="${key#PATH::}"
  [[ -f "$path" ]] || continue
  sudo_write "$path" "$val" || true
done < "$STATE"

# Restore THP selections (expects "always|madvise|never")
thp_en="$(grep -E '^THP_ENABLED=' "$STATE" | tail -n1 | cut -d= -f2- || true)"
thp_df="$(grep -E '^THP_DEFRAG='  "$STATE" | tail -n1 | cut -d= -f2- || true)"
[[ -n "$thp_en" && -f /sys/kernel/mm/transparent_hugepage/enabled ]] && sudo_write /sys/kernel/mm/transparent_hugepage/enabled "$thp_en" || true
[[ -n "$thp_df" && -f /sys/kernel/mm/transparent_hugepage/defrag  ]] && sudo_write /sys/kernel/mm/transparent_hugepage/defrag  "$thp_df" || true

echo "Restored CPU tuning from: $STATE"

