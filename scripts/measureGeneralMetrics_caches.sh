#! /bin/bash

if (( $# < 1 )); then
    echo "Usage: $0 \"command_to_execute\""
    exit -1
fi

command="$@"

perf_events="cpu-cycles,instructions,dtlb_load_misses.miss_causes_a_walk,dtlb_load_misses.stlb_hit,dtlb_load_misses.walk_duration,dtlb_load_misses.walk_completed,dtlb_store_misses.miss_causes_a_walk,dtlb_store_misses.stlb_hit,dtlb_store_misses.walk_duration,dtlb_store_misses.walk_completed,offcore_response.all_data_rd.llc_miss.any_response,llc_misses.mem_read,llc_misses.mem_write"
prefix_perf_command="perf stat --field-separator=, --output=perf.out"
perf_command="$prefix_perf_command --event $perf_events -- "

time_format="seconds-elapsed,%e\nuser-time-seconds,%U\n"
time_format+="kernel-time-seconds,%S\nmax-resident-memory-kb,%M"
time_command="time --format=$time_format --output=time.out"

submit_command="$perf_command $time_command"
echo "Running the following command:"
echo "$submit_command $command"
$submit_command $command

