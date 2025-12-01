#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--responses', default='responses.csv')
parser.add_argument('-o', '--output', default='effects.csv')
args = parser.parse_args()

import pandas as pd
df = pd.read_csv(args.responses, index_col=False)
df_max = df.max(axis=1, numeric_only=True)
res_frame = pd.DataFrame(columns=['benchmark', 'q_const', 'q_mmap', 'q_brk', 'q_cross'])
res_frame['benchmark'] = df['benchmark']
# normalize responses according to maximum reponse for each benchmark
df = df.select_dtypes(include=['number']).div(df_max, axis=0)

import numpy as np
q0_vec = (1, 1, 1, 1)
qA_vec = (-1, 1, -1, 1)
qB_vec = (-1, -1, 1, 1)
qAB_vec = (1, -1, -1, 1)
# reorder df columns to force dot operation working with correct fields
df = df[['mmap_4k_brk_4k', 'mmap_1g_brk_4k', 'mmap_4k_brk_1g', 'mmap_1g_brk_1g']]
res_frame['q_const'] = pd.DataFrame(np.dot(df, q0_vec)/4)
res_frame['q_mmap'] = pd.DataFrame(np.dot(df, qA_vec)/4)
res_frame['q_brk'] = pd.DataFrame(np.dot(df, qB_vec)/4)
res_frame['q_cross'] = pd.DataFrame(np.dot(df, qAB_vec)/4)

res_frame.to_csv(args.output, index=False, float_format='%.3f')

#res_frame['q0'] = (df['mmap_4k_brk_4k'] + df['mmap_1g_brk_4k'] + df['mmap_4k_brk_1g'] + df['mmap_1g_brk_1g'])/4
#res_frame['qA'] = (-df['mmap_4k_brk_4k'] + df['mmap_1g_brk_4k'] - df['mmap_4k_brk_1g'] + df['mmap_1g_brk_1g'])/4
#res_frame['qB'] = (-df['mmap_4k_brk_4k'] - df['mmap_1g_brk_4k'] + df['mmap_4k_brk_1g'] + df['mmap_1g_brk_1g'])/4
#res_frame['qAB'] = (df['mmap_4k_brk_4k'] - df['mmap_1g_brk_4k'] - df['mmap_4k_brk_1g'] + df['mmap_1g_brk_1g'])/4


