#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(sys.argv[0])+"/..")
from Utils.utils import *
from Utils.ConfigurationFile import *
sys.path.append(os.path.dirname(sys.argv[0])+"/../../analysis")
from performance_statistics import PerformanceStatistics

import re
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def loadDataframe(mean_file):
    mean_ps = PerformanceStatistics(mean_file)
    mean_df = mean_ps.getDataFrame()
    mean_df['cpu-cycles'] = mean_ps.getRuntime()
    mean_df['walk_cycles'] = mean_ps.getWalkDuration()
    mean_df['stlb_hits'] = mean_ps.getStlbHits()
    mean_df['stlb_misses'] = mean_ps.getStlbMisses()
    df = mean_df[['layout', 'walk_cycles', 'stlb_hits', 'stlb_misses', 'cpu-cycles']]
    # drop duplicated rows
    important_columns = list(df.columns)
    important_columns.remove('layout')
    #df.drop_duplicates(inplace=True, subset=important_columns)
    df = df.drop_duplicates(subset=important_columns)
    return df

import random
def writeLayout(layout_name, windows, output):
    page_size = 1 << 21
    random.seed(len(windows))
    start_deviation = random.randint(0, 511)
    offset_deviation = start_deviation * 4096

    configuration = Configuration()
    configuration.setPoolsSize(
            brk_size=brk_footprint,
            file_size=1*gb,
            mmap_size=mmap_footprint)
    for w in windows:
        configuration.addWindow(
                type=configuration.TYPE_BRK,
                page_size=page_size,
                start_offset=w * page_size + offset_deviation,
                end_offset=(w+1) * page_size + offset_deviation)
    configuration.exportToCSV(output, layout_name)
    #configuration.exportToCSV(output, layout)

def isPagesListUnique(pages_list, all_windows):
    pages_set = set(pages_list)
    for w in all_windows:
        if set(w) == pages_set:
            return False
    return True

def calculateTlbCoverage(pebs_df, pages):
    total_weight = pebs_df.query(
            'PAGE_NUMBER in {pages}'.format(pages=pages))\
                    ['NUM_ACCESSES'].sum()
    return total_weight

def findTlbCoverageWindows(df, tlb_coverage_percentage, base_pages, exclude_pages=None):
    epsilon = 0.5
    windows = None
    while windows == None:
        windows = _findTlbCoverageWindows(df, tlb_coverage_percentage, base_pages, epsilon, exclude_pages)
        epsilon += 0.5
    return windows

def _findTlbCoverageWindows(df, tlb_coverage_percentage, base_pages, epsilon, exclude_pages):
    # based on the fact that selected pages in base_pages are ordered
    # from heaviest to the lightest
    for i in range(len(base_pages)+1):
        windows = _findTlbCoverageWindowsBasedOnSubset(df, tlb_coverage_percentage, base_pages[i:], epsilon, exclude_pages)
        if windows:
            return windows

def _findTlbCoverageWindowsBasedOnSubset(df, tlb_coverage_percentage, base_pages, epsilon, exclude_pages):
    total_weight = calculateTlbCoverage(df, base_pages)
    # use a new list instead of using the existing base_pages list to
    # keep it sorted according to page weights
    windows = []
    for index, row in df.iterrows():
        weight = row['NUM_ACCESSES']
        page_number = row['PAGE_NUMBER']
        if exclude_pages and page_number in exclude_pages:
            continue
        if base_pages and page_number in base_pages:
            # pages from base_pages already included in the total weight
            # just add them without increasing the total weight
            windows.append(page_number)
            continue
        if (total_weight + weight) <= (tlb_coverage_percentage + epsilon):
            #print('page: {page} - weight: {weight}'.format(page=page_number, weight=weight))
            total_weight += weight
            windows.append(page_number)
        if total_weight >= (tlb_coverage_percentage - epsilon):
            break

    if total_weight > (tlb_coverage_percentage + epsilon) \
            or total_weight < (tlb_coverage_percentage - epsilon):
        return []
    # add tailed pages from base_pages that were not selected (because
    # we are already reached the goal weight)
    windows += list(set(base_pages) - set(windows))
    return windows

import itertools
def findLayouts(pebs_df, tlb_coverage_percentage, base_pages, exclude_pages, num_layouts, layouts):
    if len(layouts) == num_layouts:
        return
    windows = findTlbCoverageWindows(pebs_df, tlb_coverage_percentage, base_pages, exclude_pages)
    if not windows:
        return
    if isPagesListUnique(windows, layouts):
        layouts.append(windows)
        print(len(layouts))
    else:
        return
    for subset_size in range(len(windows)):
        for subset in list(itertools.combinations(windows, subset_size+1)):
            cosubset = set(windows) - set(subset)
            findLayouts(pebs_df, tlb_coverage_percentage, subset, cosubset, num_layouts, layouts)
            if len(layouts) == num_layouts:
                return

def findLayoutsRandomly(pebs_df, tlb_coverage_percentage, num_layouts, layouts):
    random.seed(tlb_coverage_percentage)
    randomness = 100
    while len(layouts) < num_layouts:
        query = pebs_df.query(
                'NUM_ACCESSES < {weight}'.format(weight=tlb_coverage_percentage))
        pages = query['PAGE_NUMBER']
        weights = query['NUM_ACCESSES']
        total_weight = 0
        base_pages = []
        exclude_pages = []
        for i in range(min(randomness, len(pages))):
            random_page = random.randint(0, len(pages)-1)
            select = random.randint(0, 1)
            if select:
                if (total_weight + weights[random_page]) < tlb_misses_coverage_ratio:
                    base_pages.append(random_page)
                    total_weight += weights[random_page]
            else:
                exclude_pages.append(random_page)
        # find windows based on the random selected pages
        windows = findTlbCoverageWindows(pebs_df, tlb_coverage_percentage, base_pages, exclude_pages)
        if windows and isPagesListUnique(windows, layouts):
            layouts.append(windows)
            print(windows)
        if not windows:
            randomness -= 1


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--memory_footprint', default='memory_footprint.txt')
parser.add_argument('-s', '--single_page_size_mean', default='results/single_page_size/mean.csv')
parser.add_argument('-p', '--pebs_mem_bins', default='mem_bins_2mb.csv')
parser.add_argument('-w', '--walk_cycles', type=float, required=True)
parser.add_argument('-n', '--num_layouts', type=int, default=55)
parser.add_argument('-d', '--layouts_dir', required=True)

args = parser.parse_args()

import pandas as pd

# read memory-footprints
footprint_df = pd.read_csv(args.memory_footprint)
mmap_footprint = footprint_df['anon-mmap-max'][0]
brk_footprint = footprint_df['brk-max'][0]

# read single_page_size mean file
single_page_size_df = loadDataframe(args.single_page_size_mean)
max_walk_cycles = single_page_size_df[single_page_size_df['layout'] == 'layout4kb'].iloc[0]['walk_cycles']
min_walk_cycles = single_page_size_df[single_page_size_df['layout'] == 'layout2mb'].iloc[0]['walk_cycles']
walk_cycles_span = max_walk_cycles - min_walk_cycles

# read mem-bins
pebs_df = pd.read_csv(args.pebs_mem_bins, delimiter=',')

pebs_df = pebs_df[pebs_df['PAGE_TYPE'].str.contains('brk')]
if pebs_df.empty:
    sys.exit('Input file does not contain page accesses information about the brk pool!')
pebs_df = pebs_df[['PAGE_NUMBER', 'NUM_ACCESSES']]
pebs_df = pebs_df.reset_index()

# transform NUM_ACCESSES from absolute number to percentage
total_access = pebs_df['NUM_ACCESSES'].sum()
pebs_df['NUM_ACCESSES'] = pebs_df['NUM_ACCESSES'].mul(100).divide(total_access)

# build layouts
#tlb_misses_coverage_ratio = (max_walk_cycles - args.walk_cycles) / max_walk_cycles
tlb_misses_coverage_ratio = (max_walk_cycles - args.walk_cycles) / walk_cycles_span
tlb_misses_coverage_ratio *= 100
print('=====================================')
print('generating layouts for table-walk cycles = ' + str(args.walk_cycles))
print('generating layouts to cover ~'+str(int(tlb_misses_coverage_ratio))+'% TLB-misses')
print('=====================================')

import math
layouts = []
pebs_df = pebs_df.sort_values('NUM_ACCESSES', ascending=True)
findLayouts(pebs_df, tlb_misses_coverage_ratio, [], [], int(args.num_layouts/2), layouts)

pebs_df = pebs_df.sort_values('NUM_ACCESSES', ascending=False)
findLayouts(pebs_df, tlb_misses_coverage_ratio, [], [], args.num_layouts, layouts)

#findLayoutsRandomly(pebs_df, tlb_misses_coverage_ratio, args.num_layouts, layouts)

layout_num = 1
for l in layouts:
    writeLayout('layout'+str(layout_num), l, args.layouts_dir)
    print('layout'+str(layout_num)+' --> '+str(l))
    layout_num += 1

