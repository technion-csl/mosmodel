#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(sys.argv[0])+"/..")
from Utils.utils import *
from Utils.ConfigurationFile import *
import argparse
import math
import os
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--memory_footprint', default='memory_footprint.txt')
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()

import pandas as pd
footprint_df = pd.read_csv(args.memory_footprint)
mmap_footprint = footprint_df['anon-mmap-max'][0]
brk_footprint = footprint_df['brk-max'][0]

brk_pool_size = round_up(brk_footprint, 4*kb)
mmap_pool_size = round_up(mmap_footprint, 4*kb)
file_pool_size = 1 * gb

# building configuration for layout_4kb_pages
configuration = Configuration()
configuration.setPoolsSize(brk_size=brk_pool_size,
                           file_size=file_pool_size,
                           mmap_size=mmap_pool_size)
configuration.exportToCSV(args.output, "layout4kb")

# building configuration for layout_2mb_pages
configuration = Configuration()
configuration.setPoolsSize(brk_size=round_up(brk_pool_size, 2*mb),
                           file_size=file_pool_size,
                           mmap_size=round_up(mmap_pool_size, 2*mb))

configuration.addWindow(type=configuration.TYPE_MMAP,
                        page_size=2*mb,
                        start_offset=0,
                        end_offset=round_up(mmap_pool_size, 2*mb))

configuration.addWindow(type=configuration.TYPE_BRK,
                        page_size=2*mb,
                        start_offset=0,
                        end_offset=round_up(brk_pool_size, 2*mb))

configuration.exportToCSV(args.output, "layout2mb")

# building configuration dor layout_1gb_pages
configuration = Configuration()
configuration.setPoolsSize(brk_size=round_up(brk_pool_size, 1*gb),
                           file_size=file_pool_size,
                           mmap_size=round_up(mmap_pool_size, 1*gb))
configuration.addWindow(type=configuration.TYPE_MMAP,
                        page_size=1*gb,
                        start_offset=0,
                        end_offset=round_up(mmap_pool_size, 1*gb))

configuration.addWindow(type=configuration.TYPE_BRK,
                        page_size=1*gb,
                        start_offset=0,
                        end_offset=round_up(brk_pool_size, 1*gb))

configuration.exportToCSV(args.output, "layout1gb")

