#!/usr/bin/env python3
import sys
import os
import pandas as pd
import itertools
import os.path

sys.path.append(os.path.dirname(sys.argv[0])+"/..")
from Utils.utils import Utils
from Utils.ConfigurationFile import Configuration

sys.path.append(os.path.dirname(sys.argv[0])+"/../../analysis")
from performance_statistics import PerformanceStatistics

class LayoutGenerator():
    def __init__(self, pebs_df, exp_dir):
        self.pebs_df = pebs_df
        self.exp_dir = exp_dir

    def buildGroupsSequentially(self, desired_weights, all_pages):
        pebs_df = self.pebs_df[['PAGE_NUMBER', 'TLB_COVERAGE']]
        pebs_df = pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        groups = []
        i = 0
        for index, row in pebs_df.iterrows():
            current_total_weight = 0
            current_group = []

            page_number = int(row['PAGE_NUMBER'])
            if page_number in all_pages:
                continue

            weight = row['TLB_COVERAGE']

            epsilon = desired_weights[i] * 0.1
            if weight > (desired_weights[i] + epsilon):
                continue

            group_max_pages = 50
            max_page = page_number + int(group_max_pages/2)
            min_page = page_number - int(group_max_pages/2)
            query = pebs_df.query(
                'PAGE_NUMBER <= {max_page} and PAGE_NUMBER >= {min_page} and PAGE_NUMBER not in {all_pages}'.format(
                    max_page=max_page,
                    min_page=min_page,
                    all_pages=all_pages))
            if query['TLB_COVERAGE'].sum() < (desired_weights[i] - epsilon):
                continue
            query_df = query.sort_values('PAGE_NUMBER', ascending=True).reset_index()
            query_df = query_df[['PAGE_NUMBER', 'TLB_COVERAGE']]

            page_index = query_df[query_df['PAGE_NUMBER'] == page_number].index.to_list()[0]
            left_index = page_index-1 if page_index > 0 else 0
            right_index = page_index+1 if page_index < len(query_df)-1 else page_index

            current_total_weight += weight
            current_group.append(page_number)

            found = True
            while right_index < len(query_df) or left_index >= 0:
                # if we already achieved our goal then stop
                if current_total_weight > (desired_weights[i] + epsilon):
                    found = False
                    break
                elif current_total_weight > (desired_weights[i] - epsilon):
                    found = True
                    break
                # if there is a right side then get its weight
                if right_index != page_index and right_index < len(query_df):
                    right_page = query_df.iat[right_index, 0]
                    right_weight = query_df.iat[right_index, 1]
                else:
                    right_page = -1
                    right_weight = -1
                # if there is a left side then get its weight
                if left_index != page_index and left_index >= 0:
                    left_page = query_df.iat[left_index, 0]
                    left_weight = query_df.iat[left_index, 1]
                else:
                    left_page = -1
                    left_weight = -1
                # if we are adding too much then stop
                if (current_total_weight + right_weight) > (desired_weights[i] + epsilon) \
                or (current_total_weight + left_weight) > (desired_weights[i] + epsilon):
                    found = False
                    break
                # take the larger side
                if right_weight > left_weight:
                    current_total_weight += right_weight
                    current_group.append(right_page)
                    right_index += 1
                else:
                    current_total_weight += left_weight
                    current_group.append(left_page)
                    left_index -= 1
            if found:
                groups.append(list(set(current_group)))
                all_pages += current_group
                i += 1
            if i == len(desired_weights):
                break
        return groups

    def findGroup(self, desired_weight, exclude_pages):
        pebs_df = self.pebs_df[['PAGE_NUMBER', 'TLB_COVERAGE']]
        pebs_df = pebs_df.query('PAGE_NUMBER not in {pages}'.format(pages=exclude_pages))
        pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        total_weight = 0
        pages = []
        for index, row in pebs_df.iterrows():
            weight = row['TLB_COVERAGE']
            page = row['PAGE_NUMBER']
            total_weight += weight
            pages.append(page)
            if total_weight > desired_weight:
                break
        return pages


    def buildGroupsSparsely(self, desired_weights):
        groups = []
        all_pages = []
        for w in desired_weights:
            g = self.findGroup(w, all_pages)
            groups.append(g)
            all_pages += g
        return groups

    def createGroups(self, write_layouts=True):
        i = 1
        desired_weights = [56, 28, 14]
        '''
        groups = []
        groups_pages = []
        # 1.1.1. create three groups of pages that are responsible for (50%, 20%, 10%)
        while len(groups) != 3:
            g = self.buildGroupsSequentially(desired_weights, groups_pages)
            groups += g
            desired_weights = desired_weights[len(g):len(desired_weights)]
            # if we could not find the required groups with current weights
            # then try to lower bound the desired weights
            desired_weights = [0.9*w for w in desired_weights]
        '''
        groups = self.buildGroupsSparsely(desired_weights)
        # 1.1.2. create eight layouts as all sub-groups of these three groups
        for subset_size in range(len(groups)+1):
            for subset in itertools.combinations(groups, subset_size):
                windows = []
                for l in subset:
                    windows += l
                layout_name = 'layout' + str(i)
                i += 1
                pebs_coverage = LayoutGeneratorUtils.calculateTlbCoverage(pebs_df, windows)
                print(layout_name)
                print('#hugepages: '+ str(len(windows)))
                print('weight: ' + str(pebs_coverage))
                print('hugepages: ' + str(windows))
                print('---------------')
                LayoutGeneratorUtils.writeLayout(layout_name, windows, self.exp_dir)
        # 1.1.3. create additional layout in which all pages are backed with 2MB
        layout_name = 'layout' + str(i)
        print(layout_name)
        print('weight: 100%')
        print('hugepages: all pages')
        LayoutGeneratorUtils.writeLayoutAll2mb(layout_name, self.exp_dir)

class LayoutGeneratorUtils():
    HUGE_PAGE_2MB_SIZE = 2097152
    BASE_PAGE_4KB_SIZE = 4096

    def __init__(self):
        pass

    def loadDataframe(mean_file):
        if not os.path.isfile(mean_file):
            return None
        mean_ps = PerformanceStatistics(mean_file)
        results_df = mean_ps.getDataFrame()
        results_df['cpu-cycles'] = mean_ps.getRuntime()
        results_df['walk_cycles'] = mean_ps.getWalkDuration()
        results_df['stlb_hits'] = mean_ps.getStlbHits()
        results_df['stlb_misses'] = mean_ps.getStlbMisses()
        df = results_df[['layout', 'walk_cycles', 'stlb_hits', 'stlb_misses', 'cpu-cycles']]
        # drop duplicated rows
        important_columns = list(df.columns)
        important_columns.remove('layout')
        #df.drop_duplicates(inplace=True, subset=important_columns)
        df = df.drop_duplicates(subset=important_columns)
        return df

    def calculateTlbCoverage(pebs_df, pages):
        selected_pages = pebs_df.query(
                'PAGE_NUMBER in {pages}'.format(pages=pages))
        return selected_pages['TLB_COVERAGE'].sum()

    def writeLayoutAll2mb(layout, output):
        brk_pool_size = Utils.round_up(
            brk_footprint,
            LayoutGeneratorUtils.HUGE_PAGE_2MB_SIZE)
        configuration = Configuration()
        configuration.setPoolsSize(
                brk_size=brk_pool_size,
                file_size=1*Utils.GB,
                mmap_size=mmap_footprint)
        configuration.addWindow(
                type=Configuration.TYPE_BRK,
                page_size=LayoutGeneratorUtils.HUGE_PAGE_2MB_SIZE,
                start_offset=0,
                end_offset=brk_pool_size)
        configuration.exportToCSV(output, layout)

    def writeLayout(layout, windows, output, sliding_index=0):
        page_size= LayoutGeneratorUtils.HUGE_PAGE_2MB_SIZE
        hugepages_start_offset = sliding_index * LayoutGeneratorUtils.BASE_PAGE_4KB_SIZE
        brk_pool_size = Utils.round_up(brk_footprint, page_size) + hugepages_start_offset
        configuration = Configuration()
        configuration.setPoolsSize(
                brk_size=brk_pool_size,
                file_size=1*Utils.GB,
                mmap_size=mmap_footprint)
        for w in windows:
            configuration.addWindow(
                    type=configuration.TYPE_BRK,
                    page_size=page_size,
                    start_offset=(w * page_size) + hugepages_start_offset,
                    end_offset=((w+1) * page_size) + hugepages_start_offset)
        configuration.exportToCSV(output, layout)

    def normalizePebsAccesses(pebs_mem_bins):
        # read mem-bins
        pebs_df = pd.read_csv(pebs_mem_bins, delimiter=',')

        # filter and eep only brk pool accesses
        pebs_df = pebs_df[pebs_df['PAGE_TYPE'].str.contains('brk')]
        if pebs_df.empty:
            sys.exit('Input file does not contain page accesses information about the brk pool!')
        pebs_df = pebs_df[['PAGE_NUMBER', 'NUM_ACCESSES']]
        pebs_df = pebs_df.reset_index()

        # transform NUM_ACCESSES from absolute number to percentage
        total_access = pebs_df['NUM_ACCESSES'].sum()
        pebs_df['TLB_COVERAGE'] = pebs_df['NUM_ACCESSES'].mul(100).divide(total_access)
        pebs_df = pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        return pebs_df


import argparse
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--memory_footprint', default='memory_footprint.txt')
    parser.add_argument('-p', '--pebs_mem_bins', default='mem_bins_2mb.csv')
    parser.add_argument('-d', '--exp_dir', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArguments()

    # read memory-footprints
    footprint_df = pd.read_csv(args.memory_footprint)
    mmap_footprint = footprint_df['anon-mmap-max'][0]
    brk_footprint = footprint_df['brk-max'][0]

    pebs_df = LayoutGeneratorUtils.normalizePebsAccesses(args.pebs_mem_bins)

    layout_generator = LayoutGenerator(pebs_df, args.exp_dir)
    layout_generator.createGroups()
