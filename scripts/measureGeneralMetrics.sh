#! /bin/bash

if (( $# < 1 )); then
    echo "Usage: $0 \"command_to_execute\""
    exit -1
fi

if [ -z "$MOSMODEL_RUN_OUT_DIR" ]; then
  MOSMODEL_RUN_OUT_DIR="./"
  echo "$0: [WARNING] - MOSMODEL_RUN_OUT_DIR variable is not set, initialize with current dir"
fi

command="$@"

# Resolve the directory of the currently executing script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Full path to perf_command.txt
PERF_COMMAND_FILE="$SCRIPT_DIR/perf_command.txt"
# Check if perf_command.txt exists
if [ -f "$PERF_COMMAND_FILE" ]; then
    perf_command=`eval echo $(cat ${PERF_COMMAND_FILE})`
else
    echo "❌ ${SCRIPT_DIR}/perf_command.txt does NOT exist."
    exit 1  # Exit the script with a non-zero status to indicate failure
fi

echo "Running the following command:"
echo "$perf_command $command"
$perf_command $command

