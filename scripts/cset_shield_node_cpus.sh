#!/bin/bash

NODE=$1

echo " ======> Start configuring cset-shield for memory node ${NODE} <======"

# NODE_CPUS=$(cat /sys/devices/system/node/node$NODE/cpulist | cut -d , -f 2-)
NODE_CPUS=$(cat /sys/devices/system/node/node$NODE/cpulist)

# cset-shield node CPUs
sudo cset shield --reset
sudo cset shield -c ${NODE_CPUS}
# bind the cset set to local memory node
echo ${NODE} | sudo tee /sys/fs/cgroup/cpuset/user/cpuset.mems
# kernel-threads
sudo cset shield -k off
# enable shielding
sudo cset shield

echo " ======> Cset-shield configuration completed <======"
