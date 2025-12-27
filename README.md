# Introduction
Mosmodel is primarily useful for architectural virtual memory studies. It allows researchers to construct mathematical models that predict the runtime of applications from their virtual memory performance (e.g., the L1/L2 TLB miss rate and the latency of page table walks). Such models are a key component in the partial simulation methodology, which architects use to predict the performance of a newly proposed virtual memory design. Mosmodel is built on top of Mosalloc, a new memory allocator for hugepages. This repo is fully automated and contains all required tools (including Mosalloc) to produce the Mosmodel for any workload on any x86-64 Linux system.

More details about Mosmodel and Mosalloc can be found in the [MICRO'20](https://www.microarch.org/micro53/) paper:
["Predicting execution times with partial simulations in virtual memory research: why and how"](https://www.cs.technion.ac.il/~dan/papers/mosalloc-micro-2020.pdf)
by Mohammad Agbarya, Idan Yaniv, Jayneel Gandhi, Dan Tsafrir 

# Quick Start
Simply clone this repo, enter the repo directory, and run `make`.
This will produce Mosmodel for the toy benchmark (random access over a 1GB array, takes ~15 seconds) provided in the repo.

# Software Prerequisites
- **Sudo permissions**: multiple steps require sudo priviliges, most notably, reserving hugepages and installing apt packages (perf, numactl, ...). We recommend to [configure sudo permissions without password](https://www.cyberciti.biz/faq/linux-unix-running-sudo-command-without-a-password/) to fully automate the workflow. Sudo with password may stop the workflow at these steps prompting for your password.
- **Linux distro**: The code was tested on Ubuntu 20 LTS. Please note that all necessary apt packages are downloaded automatically. If you are using a different Linux distribution, you should probably modify the makefile to use the proper package management software and package names.
- **Python**: Our scripts are written in Python3 and rely on python packages like numpy, pandas, etc. We highly recommend installing the [Anaconda python distribution](https://www.anaconda.com/products/individual), which comes with the required modules built in. We successfully tested our code on Anaconda Individual Edition version 2019.07.

# Hardware Prerequisites
- **Intel CPUs**: Mosmodel collects and analyzes the hardware performance counters of Intel CPUs. Additionally, our code uses Intel Precise Event Based Sampling (PEBS) to find interesting Mosalloc layouts. We successfully tested our code on Intel Broadwell processors or newer.

# Setup and Configuration
Before you start building and running Mosmodel, you need to set and configure the following:
- Update the following variables in `benchmark.mk`:
    - `BENCHMARK_PATH`: the full path to a directory containing the benchmark. This directory must contain the following files:
        - `pre_run.sh` - a script running before the "actual" benchmark is measured.
        - `run.sh` - the main script of the benchmark which will be measured with perf.
        - `post_run.sh` - a script running after the "actual" benchmark is measured.

# Mosmodel Directory Structure
- `mosalloc` - a git submodule pointing to the Mosalloc memory allocator.
- `scripts` - python scripts to run the experiments, collect the results, build Mosmodel and everything in between.
- `experiments` - every experiment (== a single run of the benchmark) is stored under this directory.
- `analysis` - CSV files with raw data and the model coefficients.
- `toy_benchmark` - a small-memory benchmark supplied with this repo to quickly demonstrate Mosmodel and how it is built. It allocates a 1GB array and reads it randomly.
- `client_server_example` - a demo of how to create a benchmark infrastructure (`pre_run.sh`, `run.sh`, `post_run.sh`) for a client-server workloads, e.g., memcached.

# Limitations (Future Work)
- Currently, Mosmodel scans only Mosalloc layouts on the `brk()` pool because it assumes that the benchmark allocates memory through `malloc()`. In case the benchmark uses different allocators (than glibc `malloc()`), then this assumption may not hold. We need to customize the python scripts and makefile infrastructure that create the Mosalloc layouts. The first step toward this goal is measuring the relative performance impact of hugepages in the `mmap()` and `brk()` pools, respectively.

