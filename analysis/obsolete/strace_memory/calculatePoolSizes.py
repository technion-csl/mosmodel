#! /usr/bin/env python3

from memory_address_space import MemoryAddressSpace

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', required=True,
                    help='the directory of input files (i.e., strace outputs)')
parser.add_argument('-o', '--output_dir', required=True,
                    help='the directory for all output files')
args = parser.parse_args()

import glob
strace_files = glob.glob(args.input_dir + '/repeat0/strace.out.*')
print('Processing the following strace files:', strace_files)

for strace_file in strace_files:
    print(strace_file)
    mas = MemoryAddressSpace()
    with open(strace_file, 'r') as strace_fid:
        mas.followStraceFile(strace_fid)
    brk_pool_size, anon_pool_size, file_pool_size = \
    mas.max_brk_pool_size, mas.max_anon_pool_size, mas.max_file_pool_size
    print(brk_pool_size, anon_pool_size, file_pool_size)
