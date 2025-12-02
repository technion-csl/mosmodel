#! /bin/bash

if (( $# < 1 )); then
    echo "Usage: $0 \"perf_command_output_file\""
    exit -1
fi

perf_cmd_file=$1

general_events="context-switches,page-faults,minor-faults,major-faults,ref-cycles,cpu-cycles,instructions,"

# We no longer measure the cache events because we want to reduce sampling and improve the measuring accuracy.
#general_events+=",L1-dcache-loads,L1-dcache-stores,L1-dcache-load-misses,L1-dcache-store-misses"
#general_events+=",LLC-loads,LLC-stores,LLC-load-misses,LLC-store-misses"

prefix_perf_command="perf stat --field-separator=, --output=\${MOSMODEL_RUN_OUT_DIR}/perf.out"
# extract architecture specific dtlb events from 'perf list'
dtlb_events=`perf list | \grep -o "dtlb_.*_misses\.\w*" | \grep -v "[124][gmk]" | sort -u | tr '\n' ','`
dtlb_events=${dtlb_events%?} # remove the trailing , charachter
#dtlb_events=dtlb_load_misses.miss_causes_a_walk,dtlb_load_misses.walk_duration,dtlb_store_misses.miss_causes_a_walk,dtlb_store_misses.walk_duration

perf_command="$prefix_perf_command --event $general_events$dtlb_events -- "

# We also measure energy if the system allows it.
# energy_events=`perf list | \grep -o "\w*\/energy.*\/" | sort -u | tr '\n' ',i'`
# energy_events=${energy_events%?} # remove the trailing , charachter
# if [[ -z "$energy_events" ]]; then
#     echo "this CPU does not support energy events"
# else
#     echo "this CPU supports energy events"
#     perf_command+="$prefix_perf_command --event $energy_events --cpu=0 -- "
# fi

time_format="\"seconds-elapsed,%e\nuser-time-seconds,%U\nkernel-time-seconds,%S\nmax-resident-memory-kb,%M\""
time_command="time --format=$time_format --output=\${MOSMODEL_RUN_OUT_DIR}/time.out"

final_command="$perf_command $time_command"
echo $final_command > ${perf_cmd_file}


