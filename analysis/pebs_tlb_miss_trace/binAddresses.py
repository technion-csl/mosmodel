#! /usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import datetime

def writeDataframeToCsv(df, file_name):
    df.to_csv(file_name, na_rep='NaN', index=False)

def getAddress(strAddr):
    try:
        return int(strAddr, 16)
    except Exception:
        return 0

def applyBins(df, pid, tid, bin_width,
        anon_start, anon_end, brk_start, brk_end, file_start, file_end):
    anon_mask = (df['PID'] == pid) & (df['TID'] == tid) & (df['ADDR'] >= anon_start) & (df['ADDR'] < anon_end)
    brk_mask = (df['PID'] == pid) & (df['TID'] == tid) & (df['ADDR'] >= brk_start) & (df['ADDR'] < brk_end)
    file_mask = (df['PID'] == pid) & (df['TID'] == tid) & (df['ADDR'] >= file_start) & (df['ADDR'] < file_end)
    df.loc[anon_mask, 'ADDR'] -= anon_start
    df.loc[anon_mask, 'ADDR'] /= bin_width
    df.loc[anon_mask, 'PAGE_TYPE'] = 'anon'
    df.loc[brk_mask, 'ADDR'] -= brk_start
    df.loc[brk_mask, 'ADDR'] /= bin_width
    df.loc[brk_mask, 'PAGE_TYPE'] = 'brk'
    df.loc[file_mask, 'ADDR'] -= file_start
    df.loc[file_mask, 'ADDR'] /= bin_width
    df.loc[file_mask, 'PAGE_TYPE'] = 'file'

def normalizePebsAccesses(pebs_df):
    # filter and eep only brk pool accesses
    pebs_df = pebs_df[pebs_df['PAGE_TYPE'].str.contains('brk')]
    if pebs_df.empty:
        sys.exit('Input file does not contain page accesses information about the brk pool!')
    pebs_df = pebs_df[['PAGE_NUMBER', 'NUM_ACCESSES']]
    pebs_df = pebs_df.reset_index()

    # transform NUM_ACCESSES from absolute number to percentage
    total_access = pebs_df['NUM_ACCESSES'].sum()
    pebs_df['TLB_COVERAGE'] = pebs_df['NUM_ACCESSES'].mul(100).divide(total_access)
    pebs_df = pebs_df.sort_values('TLB_COVERAGE', ascending=False)
    return pebs_df

import argparse
import sys
import math
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--width', type=int, default=1<<21,
                    help='the bin width')
parser.add_argument('-p', '--pools_range_file', required=True,
                    help='file that contains the pools addresses range')
parser.add_argument('-i', '--input_file', type=argparse.FileType('r'),
                    default=sys.stdin,
                    help='input file that contains "perf mem -D report" results')
parser.add_argument('-o', '--output_file', required=True,
                    help='the output file to write memory accesses according to input file')
args = parser.parse_args()

df = pd.read_csv(args.input_file, delimiter=';',
                 converters={' ADDR': lambda x: getAddress(x)})
df.columns = ['PID', 'TID', 'IP', 'ADDR', 'LOCAL WEIGHT', 'DSRC', 'SYMBOL']
df['NUM_ACCESSES'] = 1
df['PAGE_TYPE'] = 'unknown'
df = df[['PID', 'TID', 'ADDR','NUM_ACCESSES', 'PAGE_TYPE']]

pools_df=pd.read_csv(args.pools_range_file)
for index, row in pools_df.iterrows():
    applyBins(df, row['pid'], row['tid'], args.width,
            getAddress(row['anon-mmap-start']), getAddress(row['anon-mmap-end']),
            getAddress(row['brk-start']), getAddress(row['brk-end']),
            getAddress(row['file-mmap-start']), getAddress(row['file-mmap-end']))

df['ADDR'] = df['ADDR'].astype(int)
df.columns = ['PID', 'TID', 'PAGE_NUMBER','NUM_ACCESSES', 'PAGE_TYPE']
df_g = df.groupby(['PID', 'TID', 'PAGE_NUMBER', 'PAGE_TYPE'], sort=False).size().rename('NUM_ACCESSES').reset_index()
df_g.sort_values('NUM_ACCESSES', ascending=False, inplace=True)

#Write each process results to separate file and plot it
for index, row in pools_df.iterrows():
    pid = row['pid']
    tid = row['tid']
    df_g_pid = df_g[df_g['PID']==pid]
    df_g_pid = df_g_pid[df_g_pid['TID']==tid]
    output_file = args.output_file + '.pid-' + str(pid) + '.tid-' + str(tid)
    writeDataframeToCsv(df_g_pid, output_file)

df_g = df_g[['PAGE_NUMBER','NUM_ACCESSES', 'PAGE_TYPE']]
df_g=df_g.groupby(['PAGE_NUMBER', 'PAGE_TYPE'], sort=False).sum().reset_index()
df_g.sort_values('NUM_ACCESSES', ascending=False, inplace=True)
#Write all processes data to single file
writeDataframeToCsv(df_g, args.output_file)

df=normalizePebsAccesses(df_g)
writeDataframeToCsv(df, args.output_file+'.normalized')

