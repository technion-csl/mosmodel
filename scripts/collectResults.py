#! /usr/bin/env python3

import argparse
import datetime
import os
import sys

import pandas as pd

from experiment_list import ExperimentList


def writeDataframeToCsv(df, file_name):
    df.to_csv(file_name, na_rep='NaN')


def parse_instruction_spec(arg):
    """Parse --instruction_count value.

    Accepted forms:
      - None => full-run aggregation
      - Numeric literal => interval [0, value]
      - Path to a file whose first non-empty line is either:
          * one number           => interval [0, value]
          * two numbers          => interval [I_start, I_end]
        Separators supported for two numbers: comma or whitespace.
    """
    if arg is None:
        return ('full_run', None, None)

    arg = str(arg).strip()
    if arg == '':
        return ('full_run', None, None)

    def _parse_line(text):
        text = text.strip()
        if text == '':
            raise ValueError('empty instruction spec')
        if ',' in text:
            parts = [p.strip() for p in text.split(',') if p.strip() != '']
        else:
            parts = text.split()
        if len(parts) == 1:
            return 0.0, float(parts[0])
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
        raise ValueError(f'expected one or two numbers, got: {text}')

    if os.path.exists(arg):
        with open(arg, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() == '':
                    continue
                I_start, I_end = _parse_line(line)
                return ('interval', I_start, I_end)
        raise ValueError(f'no instruction spec found in file: {arg}')

    I_start, I_end = _parse_line(arg)
    return ('interval', I_start, I_end)


parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--experiments_root', default='experiments/',
    help='the directory containing results in a tree structure of layout/repeat'
)
parser.add_argument(
    '-l', '--layouts', required=False, default=None,
    help='a comma-separated list of layouts'
)
parser.add_argument(
    '-r', '--repeats', default=1, type=int,
    help='repeats number of each experiment layout'
)
parser.add_argument(
    '-d', '--remove_outliers', action='store_true',
    help='if specified, then layouts with outliers will be removed'
)
parser.add_argument(
    '-s', '--skip_outliers', action='store_true',
    help='if specified, then will skip validating outliers existance'
)
parser.add_argument(
    '-o', '--output_dir', required=True,
    help='the directory for all output files'
)
parser.add_argument(
    '-i', '--instruction_count', default=None,
    help='optional instruction interval: number => [0, I_end], start,end => [I_start, I_end], or a file containing one of those formats'
)
args = parser.parse_args()

if args.remove_outliers and args.skip_outliers:
    sys.exit('Error: either --skip_outliers or --remove_outliers should be used')

try:
    instruction_mode, I_start, I_end = parse_instruction_spec(args.instruction_count)
except Exception as e:
    sys.exit(f'Error: could not parse --instruction_count: {e}')

layout_list = []
if args.layouts is None:
    for f in os.scandir(args.experiments_root):
        if f.is_dir() and f.name.startswith('layout') and f.name != 'layouts' and 'outlier' not in f.name:
            layout_list.append(f.name)
    if layout_list == []:
        print('layouts argument is empty, skipping...')
        sys.exit(0)
else:
    if args.layouts.replace(' ', '') == '':
        print('layouts argument is empty, skipping...')
        sys.exit(0)

    try:
        layout_list = args.layouts.strip().split(',')
    except KeyError:
        sys.exit('Error: could not parse the --layouts argument')

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
output_dir = args.output_dir + '/'


def collect_for_repeat(experiments_root, layout_list, repeat, instruction_mode, I_start, I_end):
    experiment_list = ExperimentList(layout_list, experiments_root)

    if instruction_mode == 'full_run':
        return experiment_list.collect(repeat)

    if instruction_mode == 'interval':
        return experiment_list.collect(repeat, I_start=I_start, I_end=I_end)

    raise ValueError(f'Unknown instruction mode: {instruction_mode}')


dataframe_list = []
for repeat in range(1, args.repeats + 1):
    df = collect_for_repeat(
        args.experiments_root,
        layout_list,
        repeat,
        instruction_mode,
        I_start,
        I_end,
    )
    csv_file_name = 'repeat' + str(repeat) + '.csv'
    if len(layout_list) > 1:
        writeDataframeToCsv(df, output_dir + csv_file_name)
    df['repeat'] = repeat
    dataframe_list.append(df)

df_with_repeats = pd.concat(dataframe_list)
df = df_with_repeats.drop(columns=['repeat'])
mean_df = df.groupby(df.index).mean()
median_df = df.groupby(df.index).median()
std_df = df.groupby(df.index).std()

interesting_metrics = ['seconds-elapsed', 'ref-cycles', 'cpu-cycles']
interesting_metrics = [metric for metric in interesting_metrics if metric in mean_df.columns]
variation = std_df[interesting_metrics] / mean_df[interesting_metrics]
outlier_threshold = 0.02
outliers = variation > outlier_threshold

if not args.skip_outliers:
    if outliers.any().any():
        print('Error: the results in', args.experiments_root, 'showed considerable variation')
        print(outliers)
        if args.remove_outliers:
            now = str(datetime.datetime.now())[:19]
            now = now.replace(' ', '_').replace(':', '-')
            for layout, outlier in outliers.iterrows():
                if not outlier['seconds-elapsed'] and not outlier['cpu-cycles']:
                    continue
                l_old_path = args.experiments_root + '/' + layout
                l_new_path = l_old_path + '.outlier.' + now
                print('remove outlier: ', l_old_path, ' --> ', l_new_path)
                os.rename(l_old_path, l_new_path)
            print('The results with outliers have been removed, please try to run them again')
        else:
            sys.exit('Cells marked with True are the outliers.')

writeDataframeToCsv(mean_df, output_dir + 'mean.csv')
writeDataframeToCsv(median_df, output_dir + 'median.csv')
writeDataframeToCsv(df_with_repeats, output_dir + 'all_repeats.csv')
writeDataframeToCsv(std_df, output_dir + 'std.csv')
