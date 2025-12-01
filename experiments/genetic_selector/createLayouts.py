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
    max_page = max(windows)
    brk_size = max((max_page+2)*page_size, brk_footprint) # +2 are: one for the end of the last page, and additional one for rounding up the last_page+offset
    configuration = Configuration()
    configuration.setPoolsSize(
            brk_size=brk_size,
            file_size=1*gb,
            mmap_size=mmap_footprint)
    for w in windows:
        configuration.addWindow(
                type=configuration.TYPE_BRK,
                page_size=page_size,
                start_offset=w * page_size + start_offset,
                end_offset=(w+1) * page_size + start_offset)
    configuration.exportToCSV(output, layout)

import math
import random
def combineGenes(father_allele, mother_allele, gene_length, gene_weights_df, father_gene_weight, mother_gene_weight):
    gene = []
    father_set = set(father_allele)
    mother_set = set(mother_allele)
    only_in_father = list(father_set - mother_set)
    only_in_mother = list(mother_set - father_set)
    in_both = list(father_set & mother_set)
    gene += in_both

    only_in_father.sort()
    only_in_mother.sort()
    gene += only_in_father[0:math.ceil(len(only_in_father)/2)] + only_in_mother[0:math.ceil(len(only_in_mother)/2)]

    random.seed(len(gene))
    gene_deviation = random.randint(0, 511)

    return gene, gene_deviation
'''
    weights_df = pd.DataFrame(columns=['origin', 'id', 'weight'])
    # calculate the gene variance-weight in father
    only_in_father_weight = 0
    for l in only_in_father:
        freq = gene_weights_df[gene_weights_df['PAGE_NUMBER'] == l]['NUM_ACCESSES']
        if freq.empty:
            continue
        weight = freq.iloc[0] * father_gene_weight
        only_in_father_weight += weight
        weights_df = weights_df.append({
            'origin':'father',
            'id':l,
            'weight':weight}, ignore_index=True)
    # calculate the gene variance-weight in mother
    only_in_mother_weight = 0
    for l in only_in_mother:
        freq = gene_weights_df[gene_weights_df['PAGE_NUMBER'] == l]['NUM_ACCESSES']
        if freq.empty:
            continue
        weight = freq.iloc[0] * mother_gene_weight
        only_in_mother_weight += weight
        weights_df = weights_df.append({
            'origin':'mother',
            'id':l,
            'weight':weight}, ignore_index=True)
    # find smallest set that covers the expected weight
    if not weights_df.empty:
        weights_df = weights_df.sort_values('weight', ascending=False).reset_index()
        gene_expected_weight = (only_in_father_weight + only_in_mother_weight) / 2
        total_weight = 0
        epsilon = 0.5
        for index, row in weights_df.iterrows():
            if (total_weight + row['weight']) <= (gene_expected_weight + epsilon):
                total_weight += row['weight']
                gene.append(row['id'])
            if total_weight > gene_expected_weight:
                break
'''


import numpy as np
def calculatePageWalkLatency(df, pages, pebs_df):
    total_misses = 0
    for p in pages:
        row = pebs_df[pebs_df['PAGE_NUMBER'] == p]
        if row['NUM_ACCESSES'].any():
                total_misses += row.iloc[0]['NUM_ACCESSES']
    if total_misses == 0:
        return np.inf
    walk_cycles = df['walk_cycles']
    return walk_cycles / total_misses

def getLayoutHugepages(result_df, experiments_root_dir):
    layout_file = str.format('{exp_root}/{exp_dir}/layouts/{layout}.csv',
            exp_root=experiments_root_dir,
            exp_dir=result_df['experiment'],
            layout=result_df['layout'])
    df = pd.read_csv(layout_file)
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
parser.add_argument('-r', '--results_root_dir', default='results')
parser.add_argument('-e', '--experiments_root_dir', default='experiments')
parser.add_argument('-f', '--mean_file_name', default='mean.csv')
parser.add_argument('-b', '--pebs_mem_bins', required=True)
parser.add_argument('-l', '--layout', required=True)
parser.add_argument('-d', '--layouts_dir', required=True)
args = parser.parse_args()

# read memory-footprints
footprint_df = pd.read_csv(args.memory_footprint)
mmap_footprint = footprint_df['anon-mmap-max'][0]
brk_footprint = round_up(footprint_df['brk-max'][0], page_size)

# read mem-bins
mem_bins_df = pd.read_csv(args.pebs_mem_bins, delimiter=',')
mem_bins_df = mem_bins_df[mem_bins_df['PAGE_TYPE'].str.contains('brk')]
if mem_bins_df.empty:
    sys.exit('The mem_bins input file does not contain page accesses information about the brk pool!')
mem_bins_df = mem_bins_df[['PAGE_NUMBER', 'NUM_ACCESSES']]
mem_bins_df.reset_index(inplace=True)
'''
# transform NUM_ACCESSES from absolute number to percentage
total_access = mem_bins_df['NUM_ACCESSES'].sum()
mem_bins_df['MISS_RATIO'] = mem_bins_df['NUM_ACCESSES']
mem_bins_df['MISS_RATIO'] = mem_bins_df['MISS_RATIO'].divide(total_access)
mem_bins_df = mem_bins_df.sort_values('MISS_RATIO', ascending=False)
mem_bins_df['REVISED_MISS_RATIO'] = mem_bins_df['MISS_RATIO']
'''

df = None
for root, dirs, files in os.walk(args.results_root_dir):
    # read all mean.csv files under the results root directory
    if args.mean_file_name in files:
        # load a dataframe rfom this mean.csv file
        exp_df = loadDataframe(root + '/' + args.mean_file_name)

        # add a new column that contains the experiment name (without the results dir suffix)
        if root.startswith(args.results_root_dir + '/'):
            experiment = root.replace(args.results_root_dir + '/', '', 1)
        else:
            experiment = root.replace(args.results_root_dir, '', 1)
        exp_df['experiment'] = experiment

        # merge all mean.csv files into one dataframe
        if df is None:
            df = exp_df
        else:
            df = pd.concat([df, exp_df], ignore_index=True)

df = df.sort_values('walk_cycles').reset_index()

# find the maximum gap of walk-cycles between measurements
max_gap_idx = df['walk_cycles'].diff().abs().argmax()

father = df.iloc[max_gap_idx]
mother = df.iloc[max_gap_idx-1]

print(str.format('max diff between: [{exp1}:{layout1}] - [{exp2}:{layout2}]',
    exp1=father['experiment'], layout1=father['layout'],
    exp2=mother['experiment'], layout2=mother['layout']))

#high_measurement = df.iloc[max_gap_idx]
father_pages = getLayoutHugepages(father, args.experiments_root_dir)
father_pages.sort()

#low_measurement = df.iloc[max_gap_idx-1]
mother_pages = getLayoutHugepages(mother, args.experiments_root_dir)
mother_pages.sort()

father_page_walk_latency = calculatePageWalkLatency(father, father_pages, mem_bins_df)
mother_page_walk_latency = calculatePageWalkLatency(mother, mother_pages, mem_bins_df)

num_of_huge_pages = int(brk_footprint / page_size)
windows, hugepage_start_location = combineGenes(father_pages, mother_pages, num_of_huge_pages, mem_bins_df, father_page_walk_latency, mother_page_walk_latency)
#print('Low measurement pages: ',low_pages)
#print('Next measurement pages: ',windows)
#print('High measurement pages: ',high_pages)
writeLayout(args.layout, windows, hugepage_start_location*4096, args.layouts_dir)


