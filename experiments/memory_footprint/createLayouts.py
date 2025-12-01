#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(sys.argv[0])+"/..")
from Utils.utils import *
from Utils.ConfigurationFile import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--mem_max_size_kb',
        type=int, help='The upper bound of the brk pool size in kB')
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()

import math
def round_up(x, base):
    return int(base * math.ceil(x/base))

def round_down(x, base):
    return (int(x / base) * base)

mem_max_size = args.mem_max_size_kb * 1024
brk_pool_size = round_down(mem_max_size, 4096)
mmap_pool_size = brk_pool_size


configuration = Configuration()
configuration.setPoolsSize(brk_size=brk_pool_size,
                           file_size=1*gb,
                           mmap_size=mmap_pool_size)
configuration.exportToCSV(args.output, "layout4kb")

