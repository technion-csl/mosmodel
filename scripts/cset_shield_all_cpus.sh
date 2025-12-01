#!/bin/bash
echo " ======> Start configuring cset-shield <======"

# Get the number of nodes dynamically
NUMBER_OF_NODES=$(ls -d /sys/devices/system/node/node*/ | wc -l)

# Initialize flags
ALL_CSETS_EXIST=true
PARTIAL_CSETS_EXIST=false

# Check if nodeX_cset exists for each node
for ((node=0; node<$NUMBER_OF_NODES; node++)); do
    if [ ! -d "/sys/fs/cgroup/cpuset/node${node}_cset" ]; then
        ALL_CSETS_EXIST=false
    else
        PARTIAL_CSETS_EXIST=true
    fi
done

# Handle the different scenarios
if [ "$ALL_CSETS_EXIST" = true ]; then
    echo "All node csets (node0_cset to node$((NUMBER_OF_NODES-1))_cset) already exist. No further configuration needed."
    exit 0
elif [ "$PARTIAL_CSETS_EXIST" = true ]; then
    echo "Error: Only some node csets exist. Clearing all existing configurations and reconfiguring them."
    
    # Clear all existing configurations
    sudo cset shield --reset --userset=node0_cset
    for ((node=1; node<$NUMBER_OF_NODES; node++)); do
        sudo cset set --destroy --set=node${node}_cset
    done
else
    echo "No node csets found. Proceeding to configure all csets."
fi

# Initialize a variable to store CPUs for isolation
ALL_CPUS=""

# Iterate over each node and gather CPU lists dynamically
for ((node=0; node<$NUMBER_OF_NODES; node++)); do
    # Assign for each node all of its CPUs except the first one to keep for system use
    NODE_CPUS=$(cat /sys/devices/system/node/node$node/cpulist | cut -d , -f 2-)
    
    # Collect all CPUs for cset shield isolation
    if [ -z "$ALL_CPUS" ]; then
        ALL_CPUS="$NODE_CPUS"
    else
        ALL_CPUS="$ALL_CPUS,$NODE_CPUS"
    fi

    # Store CPUs for each node in variables (e.g., NODE0_CPUS, NODE1_CPUS, etc.)
    eval NODE${node}_CPUS=$NODE_CPUS
done

# Isolate all CPUs except the first core in each socket (to keep for the system)
sudo cset shield --cpu=$ALL_CPUS --userset=node0_cset

# Remove exclusiveness from node0_cset to allow creating other node sets
echo 0 | sudo tee /sys/fs/cgroup/cpuset/node0_cset/cpuset.cpu_exclusive

# Shrink node0_cset to keep node0 cores only to allow creating other node csets
echo ${NODE0_CPUS} | sudo tee /sys/fs/cgroup/cpuset/node0_cset/cpuset.cpus

# Loop to create csets for the remaining nodes
for ((node=1; node<$NUMBER_OF_NODES; node++)); do
    # Dynamically get the CPUs for each node
    NODE_CPUS=$(eval echo \${NODE${node}_CPUS})
    
    # Create a cset for each node and set it as exclusive
    sudo cset set --cpu=${NODE_CPUS} --set=node${node}_cset --cpu_exclusive
    
    # Bind each set to its local memory node
    echo $node | sudo tee /sys/fs/cgroup/cpuset/node${node}_cset/cpuset.mems
done

# Set exclusiveness again for all nodes
echo 1 | sudo tee /sys/fs/cgroup/cpuset/node*/cpuset.cpu_exclusive

echo " ======> Cset-shield configuration completed <======"
