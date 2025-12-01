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

def sumTlbCoverage(df, pages):
    return df.query(f'PAGE_NUMBER in {pages}')['TLB_COVERAGE'].sum()

def printLayoutDetails(layout, layout_pages, pebs_df):
    head_size = 10
    print(f'===> {layout} <===')
    query = pebs_df.query(f'PAGE_NUMBER in {layout_pages}')
    pebs_coverage = query['TLB_COVERAGE'].sum()
    pages_in_pebs = query['PAGE_NUMBER'].count()
    all_pages = len(layout_pages)
    query = query.sort_values('TLB_COVERAGE', ascending=False)
    head_pages_pebs_coverage = query.head(head_size)['TLB_COVERAGE'].sum()
    head_pages = query.head(head_size)[['PAGE_NUMBER', 'TLB_COVERAGE']]
    query = pebs_df.query(f'PAGE_NUMBER not in {layout_pages}')
    rest_pages = query['PAGE_NUMBER'].count()
    rest_coverage = query['TLB_COVERAGE'].sum()
    print(f'pebs coverage: {pebs_coverage}')
    print(f'num pages reported by pebs: {pages_in_pebs}')
    print(f'num all pages in layout: {all_pages}')
    print(f'num rest pages not in layout (but in PEBS): {rest_pages}')
    print(f'pebs coverage of rest pages not in layout (but in PEBS): {rest_coverage}')
    print(f'pebs coverage for first {head_size} pages: {head_pages_pebs_coverage}')
    print('head pages:')
    print(head_pages)
    print('------------------------------------------------------')

def printLayoutsComparison(layout1, layout1_pages, layout2, layout2_pages, pebs_df):
    print(f'===> {layout1} vs. {layout2} <===')
    P = set(pebs_df['PAGE_NUMBER'].to_list())
    L1 = set(layout1_pages)
    L2 = set(layout2_pages)
    L1_pages = list(L1)
    L2_pages = list(L2)
    common_pages = list(L1 & L2)
    only_in_L1 = list(L1 - L2)
    only_in_L2 = list(L2 - L1)
    union = L1 | L2
    union_pages = list(union)
    L1_equals_L2 = (L1 == L2)
    L1_included_in_L2 = (L1 == (L1 & L2))
    L2_included_in_L1 = (L2 == (L1 & L2))
    not_in_L1_pages = list(P - L1)
    not_in_L2_pages = list(P - L2)
    not_in_union_pages = list(P - union)

    print(f'All pages:')
    print(f'\t{layout1} - num pages: {len(L1_pages)} \t\t pebs coverage: {sumTlbCoverage(pebs_df, L1_pages)}')
    print(f'\t{layout2} - num pages: {len(L2_pages)} \t\t pebs coverage: {sumTlbCoverage(pebs_df, L2_pages)}')

    print('Distinct pages (that are only in one of the two layouts):')
    print(f'\t{layout1} - num pages: {len(only_in_L1)} \t\t pebs coverage: {sumTlbCoverage(pebs_df, only_in_L1)}')
    print(f'\t{layout2} - num pages: {len(only_in_L2)} \t\t pebs coverage: {sumTlbCoverage(pebs_df, only_in_L2)}')

    print('Union pages - {layout1} ∪ {layout2}:')
    print(f'\tnum pages: {len(union_pages)} \t\t pebs coverage: {sumTlbCoverage(pebs_df, union_pages)}')

    print(f'Common pages - {layout1} ∩ {layout2}:')
    print(f'\tnum pages: {len(common_pages)} \t\t pebs coverage: {sumTlbCoverage(pebs_df, common_pages)}')

    print(f'Complement pages:')
    print(f'\tnot in {layout1} - All-PEBS-pages - {layout1}:')
    print(f'\t\tnum pages: {len(not_in_L1_pages)} \t\t pebs coverage: {sumTlbCoverage(pebs_df, not_in_L1_pages)}')
    print(f'\tnot in {layout2} - All-PEBS-pages - {layout2}:')
    print(f'\t\tnum pages: {len(not_in_L2_pages)} \t\t pebs coverage: {sumTlbCoverage(pebs_df, not_in_L2_pages)}')
    print(f'\tnot in both {layout1} and {layout2} - All-PEBS-pages - ({layout1} ∪ {layout2}):')
    print(f'\t\tnum pages: {len(not_in_union_pages)} \t\t pebs coverage: {sumTlbCoverage(pebs_df, not_in_union_pages)}')

    print(f'{layout1} and {layout2} relation:')
    if L1_equals_L2:
        print(f'\t{layout1} = {layout2}')
    elif L1_included_in_L2:
        print(f'\t{layout1} ⫋ {layout2}')
    elif L2_included_in_L1:
        print(f'\t{layout1} ⫌ {layout2}')
    else:
        print(f'\t{layout1} ≠ {layout2}')
    print('------------------------------------------------------')

def compareLayouts(layout1, layout2, layouts_dir, pebs_df):
    layout1_pages = getLayoutHugepages(f'{layouts_dir}/{layout1}.csv')
    layout2_pages = getLayoutHugepages(f'{layouts_dir}/{layout2}.csv')

    printLayoutsComparison(layout1, layout1_pages, layout2, layout2_pages, pebs_df)
    printLayoutDetails(layout1, layout1_pages, pebs_df)
    printLayoutDetails(layout2, layout2_pages, pebs_df)

    all_pages = pebs_df['PAGE_NUMBER'].to_list()
    not_in_union_pages = list(set(all_pages) - (set(layout1_pages) | set(layout2_pages)))
    printLayoutDetails(f'Not in both layouts [ALL-PAGES - ({layout1} U {layout2}])', not_in_union_pages, pebs_df)

import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-l1', '--layout1', required=True)
parser.add_argument('-l2', '--layout2', required=True)
parser.add_argument('-d', '--layouts_dir', required=True)
parser.add_argument('-b', '--normalized_mem_bins_file', required=True)
args = parser.parse_args()

pebs_df = pd.read_csv(args.normalized_mem_bins_file)

compareLayouts(args.layout1, args.layout2, args.layouts_dir, pebs_df)

