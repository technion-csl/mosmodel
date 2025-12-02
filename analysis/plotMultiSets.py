#! /usr/bin/env python3

import pandas as pd
from performance_statistics import PerformanceStatistics
def loadDataframe(mean_file, output):
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
    df.to_csv(output)
    return df

import matplotlib.pyplot as plt
def plot(mean_files, labels, x_axis, x_label, output_dir, output, equal_axis=True):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4,3))

    colors = ['b', 'y', 'g', 'r', 'c', 'm', 'k']
    styles = ['o', 'v', '+', '^', 's', '*', '.', 'p', 'x']
    dataframes= []
    max_runtime = 0
    # read mean files and calculate max runtime
    for i in range(len(mean_files)):
        df = loadDataframe(mean_files[i], output_dir + '/' + labels[i]+'_scatter.csv')
        dataframes.append(df)
        max_runtime = max(max_runtime, df['cpu-cycles'].max())

    # plot the mean files
    for i in range(len(mean_files)):
        fmt = colors[-i] + styles[-i]
        df = dataframes[i]
        df['cpu-cycles-ratio'] = df['cpu-cycles'] / max_runtime
        ax.plot(df[x_axis], df[['cpu-cycles']], fmt, label=labels[i])

    ax2 = ax.twinx()
    mn, mx = ax.get_ylim()
    ax2.set_ylim(mn / max_runtime, mx / max_runtime)
    ax2.set_ylabel('runtime [%]')
    ax.set_ylabel('runtime [cycles]')

    plt.gca().yaxis.grid(True)
    #if equal_axis:
    #    ax.axis('equal')
    ax.margins(x=0)
    ax.margins(y=0)

    # set x, y labels
    plt.xlabel(x_label)

    handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles, labels, loc='best')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                      ncol=2, fancybox=True, shadow=False)
    # save to pdf
    fig.savefig(output, bbox_inches='tight')
    #plt.show()
    plt.close('all')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--mean_files', required=True)
parser.add_argument('-l', '--labels', required=True)
parser.add_argument('-o', '--output',  default='all_points', required=True)
args = parser.parse_args()

mean_files = args.mean_files.strip().split(',')
labels = args.labels.strip().split(',')

plot(mean_files, labels, 'walk_cycles', 'table walk cycles', args.output, args.output + 'table_walks.pdf', True)
plot(mean_files, labels, 'stlb_misses', 'tlb misses', args.output, args.output + 'tlb_misses.pdf', False)
plot(mean_files, labels, 'stlb_hits', 'tlb hits', args.output, args.output + 'tlb_hits.pdf', False)

