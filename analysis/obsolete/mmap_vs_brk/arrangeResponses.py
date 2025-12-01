#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-r00', '--response00', default='mmap_4k_brk_4k.csv')
parser.add_argument('-r11', '--response11', default='mmap_1g_brk_1g.csv')
parser.add_argument('-r10', '--response10', default='mmap_1g_brk_4k.csv')
parser.add_argument('-r01', '--response01', default='mmap_4k_brk_1g.csv')
parser.add_argument('-o', '--output', default='responses.csv')
args = parser.parse_args()

input_files = {}
input_files['mmap_4k_brk_4k'] = args.response00
input_files['mmap_1g_brk_1g'] = args.response11
input_files['mmap_1g_brk_4k'] = args.response10
input_files['mmap_4k_brk_1g'] = args.response01

import pandas as pd
data_frames = {}
columns = {}
for metric, input_file in input_files.items():
    data_frame = pd.read_csv(input_file, index_col=False)
    columns['benchmark'] = data_frame['benchmark']
    columns[metric] = data_frame['cpu-cycles']

data_frame = pd.DataFrame(columns)
reordered_columns = ['benchmark', 'mmap_4k_brk_4k', 'mmap_1g_brk_1g', 'mmap_1g_brk_4k', 'mmap_4k_brk_1g']
data_frame = data_frame[reordered_columns]
data_frame.to_csv(args.output, index=False)
