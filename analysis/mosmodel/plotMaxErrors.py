#! /usr/bin/env python3

import math
def round_up(num, to_nearest_num):
    return math.ceil((num / to_nearest_num)) * to_nearest_num

import matplotlib
matplotlib.use('Agg') #use a non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def plotModels(df, models, output):
    error_suffix = '_error'
    max_errors_df = pd.DataFrame(columns=['model', 'max-error'])
    for m in models:
        max_error = df[m + error_suffix].abs().max() * 100
        new_row = pd.Series({'model': m, 'max-error': max_error})
        max_errors_df = pd.concat([max_errors_df, new_row.to_frame().T], ignore_index=True)

    csv_output = output.replace('.pdf', '.csv')
    if csv_output == output:
        csv_output += '.csv'
    max_errors_df.to_csv(csv_output, float_format='%.3f')

    fig, ax = plt.subplots(figsize=(4,3))
    ind = np.arange(len(models))  # the x locations for the groups
    ax.bar(ind, max_errors_df['max-error'])

    bar_top = round_up(max_errors_df['max-error'].max(), 10)
    bar_step = bar_top / 5

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('max absolute errors [%]')
    ax.set_xticks(ind)
    #ax.set_xlim(left=-0.5, right=len(models)+0.5)
    ax.set_xticklabels(models, rotation=-30, ha='left')
    ax.grid(axis='y')
    '''
    ax.set_yscale('symlog', basey=2)
    ax.set_ylim(bottom=1)
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.get_yaxis().get_major_formatter().labelOnlyBase = False
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    '''
    ax.set_yticks(np.arange(0, bar_top+1, min(10, bar_step)))
    ax.set_ylim(0, bar_top)
    fig.tight_layout()
    #plt.show()
    plt.savefig(output)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-', '--errors', required=True,
        help='the model errors file path to plot the maximum error per each model')
parser.add_argument('-t', '--plot_title', default="Undefined", type=str,
        help='the plot title: the machine architecture which used to gain the model errors')
parser.add_argument('-o', '--output_dir',  default='./')
args = parser.parse_args()

import pandas as pd
df = pd.read_csv(args.errors)

plotModels(df, ['basu', 'pham', 'gandhi', 'yaniv'], args.output_dir + '/linear_models.pdf')
plotModels(df, ['poly1', 'poly2', 'poly3', 'mosmodel'], args.output_dir + '/mosalloc_models.pdf')
