#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(sys.argv[0])+"/..")
#sys.path.append('../')
#sys.path.append('../../')
from Utils.utils import *
from Utils.ConfigurationFile import *
import pandas as pd
sys.path.append(os.path.dirname(sys.argv[0])+"/../../analysis")
from performance_statistics import PerformanceStatistics
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

page_size = 1 << 21
def writeLayout(layout, windows, start_offset, output):
    configuration = Configuration(output, layout)
    configuration.setPoolsSize(
            brk_size=brk_footprint,
            file_size=1*gb,
            mmap_size=mmap_footprint)
    for w in windows:
        configuration.addWindow(
                type=configuration.TYPE_BRK,
                page_size=page_size,
                start_offset=w * page_size + start_offset,
                end_offset=(w+1) * page_size + start_offset)
    configuration.exportToCSV()

import random
def combineGenes(father_allele, mother_allele, gene_length):
    gene = []
    father_set = set(father_allele)
    mother_set = set(mother_allele)
    only_in_father = list(father_set - mother_set)
    only_in_mother = list(mother_set - father_set)
    in_both = list(father_set & mother_set)
    gene += in_both
    gene += only_in_father[0::2]
    gene += only_in_mother[1::2]

    random.seed(len(gene))
    gene_deviation = random.randint(0, 511)

    return gene, gene_deviation

def getLayoutHugepages(layout, layouts_dir):
    #FIXME don't hardcode the path
    if layout.find('4kb') > 0 or layout.find('2mb') > 0:
        layouts_dir = 'experiments/single_page_size'
    df = pd.read_csv('{dir}/layouts/{file}.csv'.format(dir=layouts_dir, file=layout))
    df = df[df['type'] == 'brk']
    df = df[df['pageSize'] == 2097152]
    pages = []
    for index, row in df.iterrows():
        start_page = int(row['startOffset'] / 2097152)
        end_page = int(row['endOffset'] / 2097152)
        for i in range(start_page, end_page, 1):
            pages.append(i)
    return pages

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--memory_footprint', default='memory_footprint.txt')
parser.add_argument('-s', '--single_page_size_results', default='results/single_page_size/mean.csv')
parser.add_argument('-g', '--genetic_results', default='results/genetic_scan/mean.csv')
parser.add_argument('-l', '--layout', required=True)
parser.add_argument('-d', '--layouts_dir', required=True)
args = parser.parse_args()

# read memory-footprints
footprint_df = pd.read_csv(args.memory_footprint)
mmap_footprint = footprint_df['anon-mmap-max'][0]
brk_footprint = round_up(footprint_df['brk-max'][0], page_size)

# read single-page-siez mean file
sp_df = loadDataframe(args.single_page_size_results)
assert(sp_df.layout.str.findall('4kb').sum())
assert(sp_df.layout.str.findall('2mb').sum())

from os import path
if path.exists(args.genetic_results):
    # read genetic_window mean file
    gen_df = loadDataframe(args.genetic_results)
    # append mean files
    gen_df = sp_df.append(gen_df)
else:
    gen_df= sp_df
df = gen_df.sort_values('walk_cycles').reset_index()

# find the maximum gap of walk-cycles between measurements
max_gap_idx = df['walk_cycles'].diff().abs().idxmax()

print(df)
print('max diff between: [',
        max_gap_idx, ' - ', max_gap_idx-1, '] - [',
        df.iloc[max_gap_idx]['layout'], ' - ', df.iloc[max_gap_idx-1]['layout'], ']')

#high_measurement = df.iloc[max_gap_idx]
high_pages = getLayoutHugepages(df.iloc[max_gap_idx]['layout'], args.layouts_dir)
high_pages.sort()

#low_measurement = df.iloc[max_gap_idx-1]
low_pages = getLayoutHugepages(df.iloc[max_gap_idx-1]['layout'], args.layouts_dir)
low_pages.sort()

num_of_huge_pages = int(brk_footprint / page_size)
windows, hugepage_start_location = combineGenes(high_pages, low_pages, num_of_huge_pages)
#print('Low measurement pages: ',low_pages)
#print('Next measurement pages: ',windows)
#print('High measurement pages: ',high_pages)
writeLayout(args.layout, windows, hugepage_start_location*4096, args.layouts_dir)


