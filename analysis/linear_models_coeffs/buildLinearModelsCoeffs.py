#! /usr/bin/env python3

import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(sys.argv[0])+"/..")
from performance_statistics import PerformanceStatistics

def loadDataframe(mean_file):
    mean_ps = PerformanceStatistics(mean_file, 'layout')
    mean_df = mean_ps.getDataFrame()
    mean_df['walk_cycles'] = mean_ps.getWalkDuration()
    mean_df['cpu-cycles'] = mean_ps.getRuntime()
    mean_df['stlb_hits'] = mean_ps.getStlbHits()
    mean_df['stlb_misses'] = mean_ps.getStlbMisses()
    df = mean_df[['walk_cycles', 'stlb_hits', 'stlb_misses', 'cpu-cycles', 'instructions']]

    return df

def calculateBasuCoeffs(df):
    '''
    Basu_runtime = A * TLB_MISSES + B
    A = PAGE_WALK_LATENCY / TLB_MISSES
    B = 4kb_runtime - PAGE_WALK_LATENCY
    '''
    runtime = df['cpu-cycles']
    walk_latency = df['walk_cycles']
    stlb_misses = df['stlb_misses']
    B = runtime - walk_latency
    A = walk_latency / stlb_misses
    assert(abs(runtime - (A*stlb_misses + B)) < 10)
    return A, B

def calculateAlamCoeffs(df):
    '''
    Alam_runtime = PAGE_WALK_LATENCY + B
    B = 2mb_runtime - PAGE_WALK_LATENCY
    '''
    runtime = df['cpu-cycles']
    walk_latency = df['walk_cycles']
    B = runtime - walk_latency
    return B

def calculatePhamCoeffs(df):
    '''
    Pham_runtime = 1xDTLB_HITS + 7xSTLB_HITS + PAGE_WALK_LATENCY + B
    DTLB_HITS can be ignored, therefore:
    B = 4kb_runtime - (7xSTLB_HITS + PAGE_WALK_LATENCY)
    '''
    runtime = df['cpu-cycles']
    walk_latency = df['walk_cycles']
    stlb_hits = df['stlb_hits']
    B = runtime - walk_latency - 7*stlb_hits
    return B

def calculateGandhiCoeffs(df_2mb, df_4kb):
    '''
    runtime = B + A*stlb_misses
    Where:
    B = runtime[2mb] - walk_cycles[2mb]
    A = walk_cycles / stlb_misses
    '''
    B = df_2mb['cpu-cycles'] - df_2mb['walk_cycles']
    A = df_4kb['walk_cycles'] / df_4kb['stlb_misses']
    return A, B

# def calculateYanivCoeffs(df_4kb, df_2mb):
#     delta_y = (df_4kb['cpu-cycles'] / df_4kb['instructions']) - (df_2mb['cpu-cycles'] / df_2mb['instructions'])
#     delta_x = 1000 * ((df_4kb['stlb_misses'] / df_4kb['instructions']) - (df_2mb['stlb_misses'] / df_2mb['instructions']))
#     A = delta_y / delta_x
#     B = (df_4kb['cpu-cycles'] / df_4kb['instructions']) - A * 1000 * (df_4kb['stlb_misses'] / df_4kb['instructions'])
#     return A,B

def calculateYanivCoeffs(df_4kb, df_2mb):
    delta_y = df_4kb['cpu-cycles'] - df_2mb['cpu-cycles']
    delta_x = df_4kb['walk_cycles'] - df_2mb['walk_cycles']
    A = delta_y / delta_x
    B = df_4kb['cpu-cycles'] - A * df_4kb['walk_cycles']
    return A,B

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--mean_file', default='mean.csv')
parser.add_argument('-o', '--output',  default='linear_models_coeffs.csv')
args = parser.parse_args()

res_df = pd.DataFrame(columns=[\
        'basu_A', 'basu_B', \
        'alam_B', \
        'pham_B', \
        'gandhi_A', 'gandhi_B', \
        'yaniv_A', 'yaniv_B'])
df = loadDataframe(args.mean_file)
df_4kb = df.loc['layout4kb']
df_2mb = df.loc['layout2mb']
basu_A, basu_B = calculateBasuCoeffs(df_4kb)
alam_B = calculateAlamCoeffs(df_2mb)
pham_B = calculatePhamCoeffs(df_4kb)
gandhi_A, gandhi_B = calculateGandhiCoeffs(df_2mb, df_4kb)
yaniv_A, yaniv_B = calculateYanivCoeffs(df_4kb, df_2mb)
res_df.loc[0] = {
        'basu_A': basu_A, 'basu_B':basu_B,
        'alam_B': alam_B,
        'pham_B':pham_B,
        'gandhi_A':gandhi_A, 'gandhi_B':gandhi_B,
        'yaniv_A':yaniv_A, 'yaniv_B':yaniv_B
        }

res_df.to_csv(args.output)



