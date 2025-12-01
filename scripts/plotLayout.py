#! /usr/bin/env python3

import pandas as pd
def getLayoutHugepages(layout_file):
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

import matplotlib.pyplot as plt
def plot(layout_pages, output, start_page, end_page):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4,3))

    x = list(range(start_page, end_page+1))
    y = [1 if p in layout_pages else -1 for p in x]

    plt.yticks([-1,1], ['4KB', '2MB'])
    ax.bar(x, y)
    # set x, y labels
    plt.xlabel('2MB bin')
    plt.ylabel('page size')

    # save to pdf
    fig.savefig(output, bbox_inches='tight')
    #plt.show()
    plt.close('all')

import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--layout_file', required=True)
parser.add_argument('-s', '--start_page', type=int, default=0)
parser.add_argument('-e', '--end_page', type=int, default=sys.maxsize)
parser.add_argument('-o', '--output',  default='layout_hugepages.pdf', required=True)
args = parser.parse_args()

layout_pages = getLayoutHugepages(args.layout_file)

plot(layout_pages, args.output, args.start_page, args.end_page)


