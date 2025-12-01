#! /usr/bin/env python3

import sys
import pandas as pd
import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', type=argparse.FileType('r'),
                    default=sys.stdin,
                    help='input file with memory pages access summary to calculate the top half accessed pages')
parser.add_argument('-o', '--output_file', required=True,
                    help='the output file to save the top half pages')
parser.add_argument('-m', '--memory_footprint', required=True,
                    help='the benchmark memory footprint of its pool which is specified in --pool option')
parser.add_argument('-p', '--pool', choices=['mmap', 'brk'], default='brk',
                    help='the pool name which should find its top half accessed pages')
parser.add_argument('-s', '--page_size', choices=['4KB', '2MB', '1GB'], default='4KB',
                    help='the page-size to use as the bin size')
args = parser.parse_args()

df = pd.read_csv(args.input_file, delimiter=',')

df = df[df['PAGE_TYPE'].str.contains(args.pool)]
if df.empty:
    sys.exit('Input file does not contain page accesses information about the specified pool!')
df = df[['PAGE_NUMBER', 'NUM_ACCESSES']]
df = df.sort_values('PAGE_NUMBER', ascending=True)
df = df.reset_index()

total_access = df['NUM_ACCESSES'].sum()

import math
page_size = 4096
if args.page_size == '2MB':
    page_size = 1 << 21
elif args.page_size == '1GB':
    page_size = 1 << 30
actual_total_pages = math.floor(int(args.memory_footprint) / page_size)

cummulative_arr = [0] * actual_total_pages
for index, row in df.iterrows():
    i = row['PAGE_NUMBER']
    if i < actual_total_pages:
        cummulative_arr[row['PAGE_NUMBER']] = row['NUM_ACCESSES']
for i in range(len(cummulative_arr)):
    if i > 0:
        cummulative_arr[i] += cummulative_arr[i-1]

def sumAccesses(from_page, to_page):
    if from_page == 0:
        return cummulative_arr[to_page]
    return cummulative_arr[to_page] - cummulative_arr[from_page-1]

# do a binary search for the shortest window with more than X% of total tlb-misses
def findWeightedWindow(weight):
    print('Find the window that is responsible for ', str(weight), ' of tlb misses')
    l = 0
    r = actual_total_pages - 1
    max_access_start_page = -1
    max_access_end_page = 1
    weighted_accesses = math.floor(total_access * weight)
    max_sum = 0
    while l < r:
        mid = math.floor((l+r)/2)
        l_sum = sumAccesses(l, mid)
        r_sum = sumAccesses(mid+1, r)
        #print(str.format('[DEBUG] [1]: weight = {0} , left_sum = {1} , right_sum = {2} , weighted-accesses = {3} , left={4}/right={5}/middle={6}',
            #weight, l_sum, r_sum, weighted_accesses, l,r,mid))
        if l_sum <  weighted_accesses and r_sum < weighted_accesses:
            break
        if l_sum >= weighted_accesses and l_sum > r_sum:
            max_access_start_page = l
            max_access_end_page = mid
            r = mid
            max_sum = l_sum
        elif r_sum >= weighted_accesses:
            max_access_start_page = mid+1
            max_access_end_page = r
            l = mid+1
            max_sum = r_sum
        else:
            sys.exit('Unexpected number of accesses is found: right-sum=' + \
                    str(r_sum) + ' , left-sum=' + str(l_sum) + \
                    ' , weigted-accesses=' + str(weighted_accesses))

    if max_access_start_page == -1:
        return -1, -1, -1, -1

    # try to shrink the window from two sides by removing pages one by one
    l = max_access_start_page
    r = max_access_end_page
    #print(str.format('[DEBUG] [2]: weight = {0} , left = {1} , right = {2}', weight, l, r))
    while l < r:
        l_sum = sumAccesses(l+1, r)
        r_sum = sumAccesses(l, r-1)
        if l_sum > r_sum and l_sum > weighted_accesses:
            l += 1
        elif r_sum > weighted_accesses:
            r -= 1
        else:
            break

    #print(str.format('[DEBUG] [3]: weight = {0} , left = {1} , right = {2}', weight, l, r))
    max_access_start_page = l
    max_access_end_page = r
    left_weight = sumAccesses(0, l-1) / total_access
    right_weight = sumAccesses(r+1, actual_total_pages-1) / total_access
    left_weight = int(left_weight * 100)
    right_weight = int(right_weight * 100)
    max_access_length = max_access_end_page - max_access_start_page + 1
    return max_access_start_page, max_access_length, left_weight, right_weight

header ='#[page_number],[num_pages],[window_accesses_weight],[pool-start-byte],[pool-length-bytes]\n'
header += str.format('window-start,window-length,window-weight,left-side-weight,right-side-weight,{0}-start,{0}-length', args.pool)
window_info = ''

weights = [0.8, 0.6, 0.5, 0.4, 0.3, 0.2]
for w in weights:
    max_access_start_page, max_access_length, left_weight, right_weight = findWeightedWindow(w)
    window_info += str.format('{0},{1},{2},{3},{4},{5},{6}',
            max_access_start_page, max_access_length, str(int(w*100)),
            left_weight, right_weight, 0, args.memory_footprint)
    window_info += '\n'

with open(args.output_file, 'w') as output_fid:
    print(header, file=output_fid)
    print(window_info , file=output_fid)


