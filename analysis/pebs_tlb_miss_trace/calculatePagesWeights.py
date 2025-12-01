#!/usr/bin/env python3
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='brk')
parser.add_argument('-i', '--input_file', default='mem_bins_2mb.csv')
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()

import pandas as pd

# read mem-bins
df = pd.read_csv(args.input_file, delimiter=',')

df = df[df['PAGE_TYPE'].str.contains(args.type)]
if df.empty:
    sys.exit('Input file does not contain page accesses information about the ' + args.type)
df = df[['PAGE_NUMBER', 'NUM_ACCESSES']]

# transform NUM_ACCESSES from absolute number to percentage
total_access = df['NUM_ACCESSES'].sum()
df['NUM_ACCESSES'] = df['NUM_ACCESSES'].mul(100).divide(total_access)
df = df.sort_values('NUM_ACCESSES', ascending=False)

df.to_csv(args.output, float_format='%.2f', index=False)
