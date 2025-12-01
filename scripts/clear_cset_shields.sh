#!/bin/bash

sudo cset shield --reset
# Clear all existing configurations
sudo cset shield --reset --userset=node0_cset;

nodes=$(cd /sys/fs/cgroup/cpuset/ && ls node*_cset -d)
for node in ${nodes}; do
    sudo cset set --destroy --set=${node};
done
