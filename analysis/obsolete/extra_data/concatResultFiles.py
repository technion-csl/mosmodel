#! /usr/bin/env python3

import sys
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--files', required=True,
                    help='a comma-separated list of result files to be concated.')
parser.add_argument('-o', '--output', default='mean.csv',
                    help='output file to save the concated result files.')
args = parser.parse_args()

result_files = args.files.split(',')

res_df = pd.DataFrame()
for f in result_files:
    df = pd.read_csv(f)
    res_df = pd.concat([res_df, df])
res_df.to_csv(args.output)



