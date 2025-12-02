#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(sys.argv[0])+"/..")
from Utils.utils import *
from Utils.ConfigurationFile import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--memory_footprint', default='memory_footprint.txt')
parser.add_argument('-i', '--input_file', default='mem_bins_2mb.csv')
parser.add_argument('-n', '--num_layouts', type=int, default=9)
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()

import pandas as pd
# read memory-footprints
footprint_df = pd.read_csv(args.memory_footprint)
mmap_footprint = footprint_df['anon-mmap-max'][0]
brk_footprint = footprint_df['brk-max'][0]

import math
page_size = 1 << 21
total_pages = math.floor(brk_footprint / page_size)
step_size = 100 / args.num_layouts

def writeLayout(num_layout, windows, output):
    configuration = Configuration()
    configuration.setPoolsSize(
            brk_size=brk_footprint,
            file_size=1*gb,
            mmap_size=mmap_footprint)
    for w in windows:
        configuration.addWindow(
                type=configuration.TYPE_BRK,
                page_size=page_size,
                start_offset=w * page_size,
                end_offset=(w+1) * page_size)
    configuration.exportToCSV(output, 'layout'+str(num_layout))

def findTlbCoverageWindows(df, tlb_coverage_percentage):
    epsilon = 0.5
    windows = []
    total_weight = 0
    for index, row in df.iterrows():
        weight = row['NUM_ACCESSES']
        page_number = row['PAGE_NUMBER']
        if (total_weight + weight) <= (tlb_coverage_percentage + epsilon):
            #print('page: {page} - weight: {weight}'.format(page=page_number, weight=weight))
            total_weight += weight
            windows.append(page_number)
    return windows

# read mem-bins
df = pd.read_csv(args.input_file, delimiter=',')

df = df[df['PAGE_TYPE'].str.contains('brk')]
if df.empty:
    sys.exit('Input file does not contain page accesses information about the brk pool!')
df = df[['PAGE_NUMBER', 'NUM_ACCESSES']]
df = df.reset_index()

# transform NUM_ACCESSES from absolute number to percentage
total_access = df['NUM_ACCESSES'].sum()
df['NUM_ACCESSES'] = df['NUM_ACCESSES'].mul(100).divide(total_access)
df = df.sort_values('NUM_ACCESSES', ascending=False)

# build layouts
for num_layout in range(args.num_layouts):
    tlb_coverage_percentage = step_size * (num_layout+1)
    windows = findTlbCoverageWindows(df, tlb_coverage_percentage)
    print('TLB-coverage = {coverage} - Paegs = {pages}'.format(coverage=tlb_coverage_percentage, pages=windows))
    writeLayout(num_layout+1, windows, args.output)


