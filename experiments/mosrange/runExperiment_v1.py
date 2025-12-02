#!/usr/bin/env python3
# import cProfile
import pandas as pd
import itertools
import numpy as np
import subprocess
import math
import os, sys
import logging
import random

curr_file_dir = os.path.dirname(os.path.abspath(__file__))
experiments_root_dir = os.path.join(curr_file_dir, '..')
sys.path.append(experiments_root_dir)
from Utils.utils import Utils

class MosrangeExperiment:
    DEFAULT_HUGEPAGE_SIZE = 1 << 21 # 2MB 0

    def __init__(self,
                 memory_footprint_file, pebs_mem_bins_file,
                 collect_reults_cmd, results_file,
                 run_experiment_cmd, exp_root_dir,
                 num_layouts, metric_name,
                 metric_val, metric_coverage) -> None:
        self.last_layout_num = 0
        self.collect_reults_cmd = collect_reults_cmd
        self.results_file = results_file
        self.memory_footprint_file = memory_footprint_file
        self.pebs_mem_bins_file = pebs_mem_bins_file
        self.run_experiment_cmd = run_experiment_cmd
        self.exp_root_dir = exp_root_dir
        self.num_layouts = num_layouts
        self.metric_val = metric_val
        self.metric_coverage = metric_coverage
        self.metric_name = metric_name
        self.layouts = []
        self.layout_names = []
        self.search_pebs_threshold = 0.5
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.load()

    def load(self):
        # read memory-footprints
        self.footprint_df = pd.read_csv(self.memory_footprint_file)
        self.mmap_footprint = self.footprint_df['anon-mmap-max'][0]
        self.brk_footprint = self.footprint_df['brk-max'][0]

        self.hugepage_size = MosrangeExperiment.DEFAULT_HUGEPAGE_SIZE
        self.num_hugepages = math.ceil(self.brk_footprint / self.hugepage_size) # bit vector length

        # round up the memory footprint to match the new boundaries of the new hugepage-size
        self.memory_footprint = (self.num_hugepages + 1) * self.hugepage_size
        self.brk_footprint = self.memory_footprint

        self.all_pages = [i for i in range(self.num_hugepages)]
        if self.pebs_mem_bins_file is None:
            logging.error('pebs_mem_bins_file argument is missing, skipping loading PEBS results...')
            self.pebs_df = None
        else:
            self.pebs_df = Utils.load_pebs(self.pebs_mem_bins_file, True)
            self.pebs_pages = list(set(self.pebs_df['PAGE_NUMBER'].to_list()))
            self.pages_not_in_pebs = list(set(self.all_pages) - set(self.pebs_pages))
            self.total_misses = self.pebs_df['NUM_ACCESSES'].sum()
        # load results file
        self.results_df = self.get_runs_measurements()

        self.all_4kb_layout = []
        self.all_2mb_layout = [i for i in range(self.num_hugepages)]
        self.all_pebs_pages_layout = self.pebs_pages

    def run_command(command, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Run the command
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        # Get the output and error messages
        output = output.decode('utf-8')
        error = error.decode('utf-8')

        # Check the return code
        return_code = process.returncode

        output_log = f'{out_dir}/benchmark.log'
        error_log = f'{out_dir}/benchmark.log'
        with open(output_log, 'w+') as out:
            out.write(output)
            out.write('============================================')
            out.write(f'the process exited with status: {return_code}')
            out.write('============================================')
        with open(error_log, 'w+') as err:
            err.write(error)
            err.write('============================================')
            err.write(f'the process exited with status: {return_code}')
            err.write('============================================')
        if return_code != 0:
            # log the output and error
            logging.error('============================================')
            logging.error(f'Failed to run the following command with exit code: {return_code}')
            logging.error(f'Command line: {command}')
            logging.error('Output:', output)
            logging.error('Error:', error)
            logging.error('Return code:', return_code)
            logging.error('============================================')

        return return_code

    def collect_results(collect_reults_cmd, results_file):
        # Extract the directory path
        results_dir = os.path.dirname(results_file)
        # Create the directory if it doesn't exist
        if not os.path.exists(results_dir):
            logging.debug(f'creating new directory: {results_dir}')
            os.makedirs(results_dir)
        else:
            logging.debug(f'collecting results to the existing directory: {results_dir}')

        logging.debug(f'running collect-results script: {collect_reults_cmd}')
        ret_code = MosrangeExperiment.run_command(collect_reults_cmd, results_dir)
        if ret_code != 0:
            raise RuntimeError(f'Error: collecting experiment results failed with error code: {ret_code}')
        if os.path.exists(results_file):
            results_df = Utils.load_dataframe(results_file)
        else:
            logging.warning(f'could not find results file: {results_file}')
            results_df = pd.DataFrame()
        logging.info(f'** results of {len(results_df)} layouts were collected **')

        return results_df

    def predictTlbMisses(self, mem_layout):
        assert self.pebs_df is not None
        expected_tlb_coverage = self.pebs_df.query(f'PAGE_NUMBER in {mem_layout}')['NUM_ACCESSES'].sum()
        expected_tlb_misses = self.total_misses - expected_tlb_coverage
        logging.debug(f'mem_layout of size {len(mem_layout)} has an expected-tlb-coverage={expected_tlb_coverage} and expected-tlb-misses={expected_tlb_misses}')
        return expected_tlb_misses

    def pebsTlbCoverage(self, mem_layout):
        assert self.pebs_df is not None
        df = self.pebs_df.query(f'PAGE_NUMBER in {mem_layout}')
        expected_tlb_coverage = df['TLB_COVERAGE'].sum()
        return expected_tlb_coverage

    def realMetricCoverage(self, layout_results, metric_name=None):
        if metric_name is None:
            metric_name = self.metric_name
        layout_metric_val = layout_results[metric_name]
        all_2mb_metric_val = self.all_2mb_r[metric_name]
        all_4kb_metric_val = self.all_4kb_r[metric_name]
        min_val = min(all_2mb_metric_val, all_4kb_metric_val)
        max_val = max(all_2mb_metric_val, all_4kb_metric_val)

        # either metric_val or metric_coverage will be provided
        delta = max_val - min_val
        coverage = ((max_val - layout_metric_val) / delta) * 100

        return coverage

    def generate_layout_from_pebs(self, pebs_coverage, pebs_df):
        mem_layout = []
        total_weight = 0
        for index, row in pebs_df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            if (total_weight + weight) < (pebs_coverage + self.search_pebs_threshold):
                mem_layout.append(page)
                total_weight += weight
            if total_weight >= pebs_coverage:
                break
        # could not find subset of pages to add that leads to the required coverage
        if total_weight < (pebs_coverage - self.search_pebs_threshold):
            logging.debug(f'generate_layout_from_pebs(): total_weight < (pebs_coverage - self.search_pebs_threshold): {total_weight} < ({pebs_coverage} - {self.search_pebs_threshold})')
            return []
        logging.debug(f'generate_layout_from_pebs(): found layout of length: {len(mem_layout)}')
        return mem_layout

    def split_pages_to_working_sets(self, upper_pages, lower_pages):
        pebs_set = set(self.pebs_df['PAGE_NUMBER'].to_list())
        upper_set = set(upper_pages)
        lower_set = set(lower_pages)
        all_set = lower_set | upper_set | pebs_set
        all = list(all_set)

        union_set = lower_set | upper_set
        union = list(union_set)
        intersection = list(lower_set & upper_set)
        only_in_lower = list(lower_set - upper_set)
        only_in_upper = list(upper_set - lower_set)
        not_in_upper = list(all_set - upper_set)

        not_in_pebs = list(all_set - pebs_set)
        out_union_based_on_pebs = list(pebs_set - union_set)
        out_union = list(all_set - union_set)

        return only_in_upper, only_in_lower, out_union, all

    def get_layout_results(self, layout_name):
        self.get_runs_measurements()
        layout_results = self.results_df[self.results_df['layout'] == layout_name].iloc[0]
        return layout_results

    def fill_buckets(self, buckets_weights, start_from_tail=False, fill_min_buckets_first=True):
        assert self.pebs_df is not None

        group_size = len(buckets_weights)
        group = [ [] for _ in range(group_size) ]
        df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=start_from_tail)

        threshold = 2
        i = 0
        for index, row in df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            selected_weight = None
            selected_index = None
            completed_buckets = 0
            # count completed buckets and find bucket with minimal remaining
            # space to fill, i.e., we prefer to place current page in the
            # bicket that has the lowest remaining weight/space
            for i in range(group_size):
                if buckets_weights[i] <= 0:
                    completed_buckets += 1
                elif buckets_weights[i] >= weight - threshold:
                    if selected_index is None:
                        selected_index = i
                        selected_weight = buckets_weights[i]
                    elif fill_min_buckets_first and buckets_weights[i] < selected_weight:
                        selected_index = i
                        selected_weight = buckets_weights[i]
                    elif not fill_min_buckets_first and buckets_weights[i] > selected_weight:
                        selected_index = i
                        selected_weight = buckets_weights[i]
            if completed_buckets == group_size:
                break
            # if there is a bucket that has a capacity of current page, add it
            if selected_index is not None:
                group[selected_index].append(page)
                buckets_weights[selected_index] -= weight
        return group

    def moselect_initial_samples(self):
        # desired weights for each group layout
        buckets_weights = [56, 28, 14]
        group = self.fill_buckets(buckets_weights)
        mem_layouts = []
        # create eight layouts as all subgroups of these three group layouts
        for subset_size in range(len(group)+1):
            for subset in itertools.combinations(group, subset_size):
                subset_pages = list(itertools.chain(*subset))
                mem_layouts.append(subset_pages)
        mem_layouts.append(self.all_2mb_layout)
        return mem_layouts

    def generate_random_layout(self):
        mem_layout = []
        random_mem_layout = np.random.randint(2, size=self.num_hugepages)
        for i in range(len(random_mem_layout)):
            if random_mem_layout[i] == 1:
                mem_layout.append(i)
        return mem_layout

    def get_runs_measurements(self):
        res_df = MosrangeExperiment.collect_results(self.collect_reults_cmd, self.results_file)
        if res_df.empty:
            return res_df
        res_df['hugepages'] = None
        for index, row in res_df.iterrows():
            layout_name = row['layout']
            mem_layout_pages = Utils.load_layout_hugepages(layout_name, self.exp_root_dir)
            res_df.at[index, 'hugepages'] = mem_layout_pages
        res_df = res_df.query(f'layout in {self.layout_names}').reset_index(drop=True)
        logging.info(f'** kept results of {len(res_df)} collected layouts **')
        self.results_df = res_df

        # print results of previous runs
        for index, row in res_df.iterrows():
            logging.debug(f'{row["layout"]}: coverage={self.pebsTlbCoverage(list(row["hugepages"]))} ({len(row["hugepages"])} x hugepages), runtime={row["cpu_cycles"]} , tlb-misses={row["stlb_misses"]}')
        return res_df

    def get_surrounding_layouts(self, res_df):
        # sort pebs by stlb-misses
        df = res_df.sort_values(self.metric_name, ascending=True).reset_index(drop=True)
        lower_layout = None
        upper_layout = None
        for index, row in df.iterrows():
            row_metric_val = row[self.metric_name]
            if row_metric_val <= self.metric_val:
                lower_layout = row
            else:
                upper_layout = row
                break
        return lower_layout, upper_layout

    def add_missing_pages_to_pebs(self):
        pebs_pages = self.pebs_df['PAGE_NUMBER'].tolist()
        missing_pages = list(set(self.all_2mb_layout) - set(pebs_pages))
        #self.total_misses
        all_pebs_real_coverage = self.realMetricCoverage(self.all_pebs_r, 'stlb_misses')
        # normalize pages recorded by pebs based on their real coverage
        ratio = all_pebs_real_coverage / 100
        self.pebs_df['TLB_COVERAGE'] *= ratio
        self.pebs_df['NUM_ACCESSES'] = (self.pebs_df['NUM_ACCESSES'] * ratio).astype(int)
        # add missing pages with a unified coverage ratio
        missing_pages_total_coverage = 100 - all_pebs_real_coverage
        total_missing_pages = len(missing_pages)
        missing_pages_coverage_ratio = missing_pages_total_coverage / total_missing_pages
        # update total_misses acording to the new ratio
        old_total_misses = self.total_misses
        self.total_misses *= ratio
        missing_pages_total_misses = self.total_misses - old_total_misses
        missing_pages_misses_ratio = missing_pages_total_misses / total_missing_pages
        # update pebs_df dataframe
        missing_pages_df = pd.DataFrame(
            {'PAGE_NUMBER': missing_pages,
             'NUM_ACCESSES': missing_pages_misses_ratio,
             'TLB_COVERAGE': missing_pages_coverage_ratio})
        self.pebs_df = pd.concat([self.pebs_df, missing_pages_df], ignore_index=True)


    def update_metric_values(self, res_df):
        all_4kb_set = set(self.all_4kb_layout)
        all_2mb_set = set(self.all_2mb_layout)
        all_pebs_set= set(self.all_pebs_pages_layout)
        for index, row in res_df.iterrows():
            hugepages_set = set(row['hugepages'])
            if hugepages_set == all_4kb_set:
                self.all_4kb_r = row
            elif hugepages_set == all_2mb_set:
                self.all_2mb_r = row
            elif hugepages_set == all_pebs_set:
                self.all_pebs_r = row

        all_2mb_metric_val = self.all_2mb_r[self.metric_name]
        all_4kb_metric_val = self.all_4kb_r[self.metric_name]
        min_val = min(all_2mb_metric_val, all_4kb_metric_val)
        max_val = max(all_2mb_metric_val, all_4kb_metric_val)

        # either metric_val or metric_coverage will be provided
        delta = max_val - min_val
        if self.metric_val is None:
            self.metric_val = max_val - delta * (self.metric_coverage / 100)
        else:
            self.metric_coverage = ((max_val - self.metric_val) / delta) * 100

        # add missing pages to pebs
        self.add_missing_pages_to_pebs()

    def get_head_pages(self):
        coverage_threshold = 2
        head_pages_df = self.pebs_df.query(f'TLB_COVERAGE >= {coverage_threshold}')
        head_pages_list = head_pages_df['PAGE_NUMBER'].tolist()
        return head_pages_list

    def get_closest_moselect_layout(self):
        moselect_layouts = self.moselect_initial_samples()
        min_coverage_delta = 100
        closest_layout = None
        for l in moselect_layouts:
            coverage = self.pebsTlbCoverage(l)
            coverage_delta = abs(coverage - self.metric_coverage)
            if coverage_delta < min_coverage_delta:
                min_coverage_delta = coverage_delta
                closest_layout = l
        assert l is not None
        return l

    def generate_initial_samples(self):
        res_df = self.get_runs_measurements()
        num_prev_samples = len(res_df)
        # mem_layouts = self.moselect_initial_samples()
        mem_layouts = [self.all_4kb_layout, self.all_2mb_layout, self.all_pebs_pages_layout]

        for i, mem_layout in enumerate(mem_layouts):
            logging.info(f'** Producing initial sample #{i} using a memory layout with {len(mem_layout)} (x2MB) hugepages')
            self.run_next_layout(mem_layout)
        res_df = self.get_runs_measurements()
        self.update_metric_values(res_df)
        return res_df

    def generate_layout_from_base(self, base_pages, search_space, coverage, sort=True):
        logging.debug(f'generate_layout_from_base(): len(base_pages)={len(base_pages)} , len(search_space)={len(search_space)} , coverage={coverage}')
        expected_coverage = coverage - self.pebsTlbCoverage(base_pages)
        df = self.pebs_df.query(f'PAGE_NUMBER in {search_space} and PAGE_NUMBER not in {base_pages}')
        if sort:
            df = df.sort_values('TLB_COVERAGE', ascending=False)
        logging.debug(f'generate_layout_from_base() after filtering pages: len(df)={len(df)}')
        layout = self.generate_layout_from_pebs(expected_coverage, df)
        if layout:
            return layout+base_pages
        else:
            return []

    def shuffle_and_generate_layout_from_base(self, base_pages, exclude_pages, coverage):
        logging.debug(f'shuffle_and_generate_layout_from_base(): len(base_pages)={len(base_pages)} , len(exclude_pages)={len(exclude_pages)} , coverage={coverage}')
        expected_coverage = coverage - self.pebsTlbCoverage(base_pages)
        df = self.pebs_df.query(f'PAGE_NUMBER not in {exclude_pages+base_pages}')

        iterations_len = min(100, len(df))
        for i in range(iterations_len):
            df = df.sample(frac=1).reset_index(drop=True)
            layout = self.generate_layout_from_pebs(expected_coverage, df)
            if layout and self.isPagesListUnique(layout, self.layouts):
                return layout
        return []

    def add_hugepages_to_base(self, next_coverage, base_pages, other_pages, all_pages):
        other_even_pages = [p for p in other_pages if p%2==0]
        all_even_pages = [p for p in all_pages if p%2==0]
        search_space_options = [other_pages, all_pages, other_even_pages, all_even_pages]
        for s in search_space_options:
            layout = self.generate_layout_from_base(base_pages, s, next_coverage)
            if layout and self.isPagesListUnique(layout, self.layouts):
                return layout
        return []

    def remove_hugepages_from_base(self, pebs_coverage, base_pages, pages_to_remove):
        mem_layout = []
        df = self.pebs_df.query(f'PAGE_NUMBER in {base_pages}')
        df = df.sort_values('TLB_COVERAGE', ascending=False)
        total_weight = df['TLB_COVERAGE'].sum()
        # if the coverage of the base_pages less than expected,
        # then we can not remove pages from it
        if total_weight < (pebs_coverage - self.search_pebs_threshold):
            return []
        for index, row in df.iterrows():
            page = row['PAGE_NUMBER']
            if page not in pages_to_remove:
                continue
            weight = row['TLB_COVERAGE']
            if (total_weight - weight) > (pebs_coverage - self.search_pebs_threshold):
                mem_layout.append(page)
                total_weight -= weight
            if total_weight <= pebs_coverage:
                break
        # could not find subset to remove that leads to the required coverage
        if total_weight > (pebs_coverage + self.search_pebs_threshold):
            return []
        if mem_layout and self.isPagesListUnique(mem_layout, self.layouts):
            return mem_layout
        return []

    def write_layout(self, layout_name, mem_layout):
        logging.info(f'writing {layout_name} with {len(mem_layout)} hugepages')
        Utils.write_layout(layout_name, mem_layout, self.exp_root_dir, self.brk_footprint, self.mmap_footprint)
        self.layouts.append(mem_layout)
        self.layout_names.append(layout_name)

    def layout_was_run(self, layout_name, mem_layout):
        prev_layout_res = None
        if not self.results_df.empty:
            prev_layout_res = self.results_df.query(f'layout == "{layout_name}"')
        # prev_layout_res = self.results_df[self.results_df['layout'] == layout_name]
        if prev_layout_res is None or prev_layout_res.empty:
            # the layout does not exist in the results file
            return False, None
        prev_layout_res = prev_layout_res.iloc[0]
        prev_layout_hugepages = prev_layout_res['hugepages']
        if set(prev_layout_hugepages) != set(mem_layout):
            # the existing layout has different hugepages set than the new one
            return False, None

        # the layout exists and has the same hugepages set
        return True, prev_layout_res

    def run_workload(self, mem_layout, layout_name):
        found, prev_res = self.layout_was_run(layout_name, mem_layout)
        if found:
            logging.info(f'+++ {layout_name} already exists, skipping running it +++')
            self.layouts.append(mem_layout)
            self.layout_names.append(layout_name)
            return prev_res

        self.write_layout(layout_name, mem_layout)
        out_dir = f'{self.exp_root_dir}/{layout_name}'
        run_cmd = f'{self.run_experiment_cmd} {layout_name}'

        logging.info('-------------------------------------------')
        logging.info(f'*** start running {layout_name} with {len(mem_layout)} hugepages ***')
        logging.info(f'start running workload')
        logging.info(f'\texperiment: {out_dir}')
        logging.debug(f'\tscript: {run_cmd}')
        logging.info(f'\tlayout: {layout_name}')
        logging.info(f'\t#hugepages: {len(mem_layout)}')

        ret_code = MosrangeExperiment.run_command(run_cmd, out_dir)
        if ret_code != 0:
            raise RuntimeError(f'Error: running {layout_name} failed with error code: {ret_code}')

        layout_res = self.get_layout_results(layout_name)
        tlb_misses = layout_res['stlb_misses']
        tlb_hits = layout_res['stlb_hits']
        walk_cycles = layout_res['walk_cycles']
        runtime = layout_res['cpu_cycles']

        logging.info('-------------------------------------------')
        logging.info(f'Results:')
        logging.info(f'\tstlb-misses={tlb_misses/1e9:.2f} Billions')
        logging.info(f'\tstlb-hits={tlb_hits/1e9:.2f} Billions')
        logging.info(f'\twalk-cycles={walk_cycles/1e9:.2f} Billion cycles')
        logging.info(f'\truntime={runtime/1e9:.2f} Billion cycles')
        logging.info('===========================================')
        return layout_res

    def isPagesListUnique(self, pages_list, all_layouts):
        pages_set = set(pages_list)
        for l in all_layouts:
            if set(l) == pages_set:
                return False
        return True

    def find_general_layout(self, pebs_coverage, include_pages, exclude_pages, sort_ascending=False):
        df = self.pebs_df.query(f'PAGE_NUMBER not in {exclude_pages}')
        df = df.sort_values('TLB_COVERAGE', ascending=sort_ascending)
        search_space = df['PAGE_NUMBER'].to_list()
        layout = self.generate_layout_from_base(include_pages, search_space, pebs_coverage)
        return layout

    def select_layout_from_subsets(self, pebs_coverage, include_pages, exclude_pages, layouts, sort_ascending=False):
        layout = self.find_general_layout(pebs_coverage, include_pages, exclude_pages, sort_ascending)
        if not layout:
            return
        if self.isPagesListUnique(layout, self.layouts + layouts):
            layouts.append(layout)
            return

        # for i in range(len(layout)):
        #     self.select_layout_from_subsets(pebs_coverage, layout[i:], layout[:i], layouts, sort_ascending)
        #     if self.isPagesListUnique(layout, self.layouts + layouts):
        #         layouts.append(layout)
        #         return
        for subset_size in range(len(layout)):
            for subset in list(itertools.combinations(layout, subset_size+1)):
                cosubset = set(layout) - set(subset)
                self.select_layout_from_subsets(pebs_coverage, list(subset), list(cosubset), layouts, sort_ascending)
                if self.isPagesListUnique(layout, self.layouts + layouts):
                    layouts.append(layout)
                    return

    def select_layout_generally(self, pebs_coverage):
        logging.debug(f'select_layout_generally() -->: pebs_coverage={pebs_coverage}')
        mem_layouts = []
        self.select_layout_from_subsets(pebs_coverage, [], [], mem_layouts, False)
        for l in mem_layouts:
            if l and self.isPagesListUnique(l, self.layouts):
                return l
        mem_layouts = []
        self.select_layout_from_subsets(pebs_coverage, [], [], mem_layouts, True)
        for l in mem_layouts:
            if l and self.isPagesListUnique(l, self.layouts):
                return l
        return []

    def remove_hugepages_blindly(self, pebs_coverage, upper_mem_layout, lower_mem_layout, ratio=None):
        if ratio is None:
            upper_layout_pebs = self.pebsTlbCoverage(upper_mem_layout)
            lower_layout_pebs = self.pebsTlbCoverage(lower_mem_layout)
            ratio = abs(lower_layout_pebs - pebs_coverage) / abs(lower_layout_pebs - upper_layout_pebs)
            if ratio >= 1:
                return ratio, []

        hugepages_only_in_lower = list(set(lower_mem_layout) - set(upper_mem_layout))
        remove_set_size = int(len(hugepages_only_in_lower) * ratio)
        remove_set = hugepages_only_in_lower[:remove_set_size]
        layout = list(set(lower_mem_layout) - set(remove_set))

        if self.isPagesListUnique(layout, self.layouts):
            return ratio, layout
        return ratio, []

    def add_hugepages_blindly(self, pebs_coverage, upper_mem_layout, lower_mem_layout, ratio=None):
        if ratio is None:
            upper_layout_pebs = self.pebsTlbCoverage(upper_mem_layout)
            lower_layout_pebs = self.pebsTlbCoverage(lower_mem_layout)
            ratio = abs(pebs_coverage - upper_layout_pebs) / abs(lower_layout_pebs - upper_layout_pebs)
            if ratio >= 1:
                return ratio, []

        hugepages_only_in_lower = list(set(lower_mem_layout) - set(upper_mem_layout))
        add_set_size = int(len(hugepages_only_in_lower) * ratio)
        add_set = hugepages_only_in_lower[:add_set_size]
        layout = list(set(upper_mem_layout) | set(add_set))

        if self.isPagesListUnique(layout, self.layouts):
            return ratio, layout
        return ratio, []

    def select_layout_blindly(self, upper_mem_layout, num_hugepages_to_add=None):
        layout = []
        add_set = list(set(self.all_pages) - set(upper_mem_layout))
        if num_hugepages_to_add is None:
            num_hugepages_to_add = math.ceil(len(add_set) / 2)
        num_hugepages_to_add = max(1, num_hugepages_to_add)
        num_hugepages_to_add = min(num_hugepages_to_add, len(add_set))
        while not layout or not self.isPagesListUnique(layout, self.layouts):
            random_subset = random.sample(add_set, num_hugepages_to_add)
            layout = list(set(upper_mem_layout) | set(random_subset))
        return layout

    def run_next_layout(self, mem_layout):
        self.last_layout_num += 1
        layout_name = f'layout{self.last_layout_num}'
        logging.info(f'run workload under {layout_name} with {len(mem_layout)} hugepages')
        last_result = self.run_workload(mem_layout, layout_name)
        return last_result

    def findTlbCoverageLayout(self, df, tlb_coverage_percentage, base_pages, exclude_pages=None):
        epsilon = 0.5
        layout = None
        while not layout:
            layout = self._findTlbCoverageLayout(df, tlb_coverage_percentage, base_pages, epsilon, exclude_pages)
            epsilon += 0.5
        return layout

    def _findTlbCoverageLayout(self, df, tlb_coverage_percentage, base_pages, epsilon, exclude_pages):
        # based on the fact that selected pages in base_pages are ordered
        # from heaviest to the lightest
        for i in range(len(base_pages)+1):
            layout = self._findTlbCoverageLayoutBasedOnSubset(df, tlb_coverage_percentage, base_pages[i:], epsilon, exclude_pages)
            if layout and self.isPagesListUnique(layout, self.layouts):
                return layout
            return None

    def _findTlbCoverageLayoutBasedOnSubset(self, df, tlb_coverage_percentage, base_pages, epsilon, exclude_pages):
        total_weight = self.pebsTlbCoverage(base_pages)
        # use a new list instead of using the existing base_pages list to
        # keep it sorted according to page weights
        layout = []
        for index, row in df.iterrows():
            weight = row['TLB_COVERAGE']
            page_number = row['PAGE_NUMBER']
            if exclude_pages and page_number in exclude_pages:
                continue
            if base_pages and page_number in base_pages:
                # pages from base_pages already included in the total weight
                # just add them without increasing the total weight
                layout.append(page_number)
                continue
            if (total_weight + weight) <= (tlb_coverage_percentage + epsilon):
                #print('page: {page} - weight: {weight}'.format(page=page_number, weight=weight))
                total_weight += weight
                layout.append(page_number)
            if total_weight >= (tlb_coverage_percentage - epsilon):
                break

        if total_weight > (tlb_coverage_percentage + epsilon) \
                or total_weight < (tlb_coverage_percentage - epsilon):
            return []
        # add tailed pages from base_pages that were not selected (because
        # we are already reached the goal weight)
        layout += list(set(base_pages) - set(layout))
        return layout

    def _findLayouts(self, pebs_df, tlb_coverage_percentage, base_pages, exclude_pages, num_layouts, layouts):
        if len(layouts) == num_layouts:
            return
        layout = self.findTlbCoverageLayout(pebs_df, tlb_coverage_percentage, base_pages, exclude_pages)
        if not layout:
            return
        if self.isPagesListUnique(layout, self.layouts):
            layouts.append(layout)
        else:
            return
        for subset_size in range(len(layout)):
            for subset in list(itertools.combinations(layout, subset_size+1)):
                cosubset = set(layout) - set(subset)
                self._findLayouts(pebs_df, tlb_coverage_percentage, subset, cosubset, num_layouts, layouts)
                if len(layouts) == num_layouts:
                    return

    def findLayouts(self, tlb_coverage_percentage, base_pages, exclude_pages, num_layouts, layouts, sort_ascending):
        pebs_df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=sort_ascending)
        return self._findLayouts(pebs_df, tlb_coverage_percentage, base_pages, exclude_pages, num_layouts, layouts)


    def select_layout_with_sampled_headpages(self, pebs_coverage, include_pages, exclude_pages):
        head_pages_coverage_threshold = 2
        all_tail_pages_df = self.pebs_df.query(f'TLB_COVERAGE < {head_pages_coverage_threshold} and PAGE_NUMBER not in {include_pages} and PAGE_NUMBER not in {exclude_pages}')
        all_tail_pages = all_tail_pages_df['PAGE_NUMBER'].tolist()
        all_head_pages_df = self.pebs_df.query(f'TLB_COVERAGE >= {head_pages_coverage_threshold} and PAGE_NUMBER not in {include_pages} and PAGE_NUMBER not in {exclude_pages}')

        if self.pebsTlbCoverage(include_pages) > pebs_coverage:
            return []
        if self.pebsTlbCoverage(exclude_pages) > (100 - pebs_coverage):
            return []

        for i in range(len(all_head_pages_df)):
            sampled_headpages = all_head_pages_df.sample(frac=0.5)['PAGE_NUMBER'].tolist()
            search_pages = list(set(all_tail_pages + sampled_headpages) - set(exclude_pages))
            layout = self.generate_layout_from_base(include_pages, search_pages, pebs_coverage, True)
            if layout and self.isPagesListUnique(layout, self.layouts):
                return layout
        return []

    def select_all_subsets(self, pebs_coverage, base_pages, exclude_pages, num_layouts, layouts):
        if len(layouts) == num_layouts:
            return
        layout = self.find_general_layout(pebs_coverage, base_pages, exclude_pages)
        if not layout:
            return
        if self.isPagesListUnique(layout, self.layouts + layouts):
            layouts.append(layout)
        else:
            return
        for subset_size in range(len(layout)):
            for subset in list(itertools.combinations(layout, subset_size+1)):
                cosubset = set(layout) - set(subset)
                self.select_all_subsets(pebs_coverage, subset, cosubset, num_layouts, layouts)
                if len(layouts) == num_layouts:
                    return

    def generate_and_run_layouts(self):
        pebs_df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        selected_head_pages_df = pebs_df.head(3)
        selected_head_pages = selected_head_pages_df['PAGE_NUMBER'].tolist()

        head_pages_coverage_threshold = 2
        all_head_pages_df = self.pebs_df.query(f'TLB_COVERAGE >= {head_pages_coverage_threshold}')
        all_head_pages = all_head_pages_df['PAGE_NUMBER'].tolist()

        last_result = None
        excluded_headpages = []
        included_headpages = []
        deviation_threshold = 0.05

        for headpages_subset_size in range(len(selected_head_pages)):
            for headpages_subset in list(itertools.combinations(selected_head_pages, headpages_subset_size+1)):
                include_pages = list(headpages_subset)
                exclude_pages = list(set(selected_head_pages) - set(include_pages))
                layout = self.select_layout_with_sampled_headpages(self.metric_coverage, include_pages, exclude_pages)
                if layout:
                    last_result = self.run_next_layout(layout)
                    deviation = abs(last_result[self.metric_name] - self.metric_val) / self.metric_val
                    if deviation <= deviation_threshold:
                        excluded_headpages.append(exclude_pages)
                        included_headpages.append(include_pages)
        for i in range(len(included_headpages)):
            if self.last_layout_num >= self.num_layouts:
                break
            print(i)
            exclude_pages = excluded_headpages[i]
            include_pages = included_headpages[i]

            mem_layouts = []
            for i in range(10):
                layout = self.shuffle_and_generate_layout_from_base(include_pages, exclude_pages, self.metric_coverage)
                if layout:
                    mem_layouts.append(layout)
            for layout in mem_layouts:
                if self.last_layout_num >= self.num_layouts:
                    break
                last_result = self.run_next_layout(layout)
                deviation = abs(last_result[self.metric_name] - self.metric_val) / self.metric_val
                if deviation > deviation_threshold:
                    continue

        while self.last_layout_num < self.num_layouts:
            logging.info(f'findLayouts: pebs_coverage={self.metric_coverage}')
            ascending_layouts = []
            descending_layouts = []
            rem_layouts = self.num_layouts - self.last_layout_num
            self.findLayouts(self.metric_coverage, [], [], (rem_layouts // 2) + 1, ascending_layouts, True)
            self.findLayouts(self.metric_coverage, [], [], rem_layouts // 2, descending_layouts, False)
            general_mem_layouts = ascending_layouts + descending_layouts
            logging.info(f'findLayouts returned #{len(general_mem_layouts)}')

            for general_layout in general_mem_layouts:
                if self.last_layout_num >= self.num_layouts:
                    break

                if general_layout and self.isPagesListUnique(general_layout, self.layouts):
                    last_result = self.run_next_layout(general_layout)
                    deviation = abs(last_result[self.metric_name] - self.metric_val) / self.metric_val
                    if deviation > deviation_threshold:
                        continue

                include_pages = list(set(all_head_pages) & set(general_layout))
                exclude_pages = list(set(all_head_pages) - set(general_layout))

                for i in range(10):
                    layout = self.shuffle_and_generate_layout_from_base(include_pages, exclude_pages, self.metric_coverage)
                    if not layout:
                        break
                    last_result = self.run_next_layout(layout)
                    deviation = abs(last_result[self.metric_name] - self.metric_val) / self.metric_val
                    if deviation > deviation_threshold:
                        break

                # logging.info(f'findLayouts: pebs_coverage={self.metric_coverage} , #include_pages={len(include_pages)} , #exclude_pages={len(exclude_pages)}')
                # ascending_layouts = []
                # descending_layouts = []
                # self.findLayouts(self.metric_coverage, include_pages, exclude_pages, 3, ascending_layouts, True)
                # self.findLayouts(self.metric_coverage, include_pages, exclude_pages, 3, descending_layouts, False)
                # mem_layouts = ascending_layouts + descending_layouts
                # logging.info(f'findLayouts returned #{len(mem_layouts)}')

                # for layout in mem_layouts:
                #     if self.last_layout_num >= self.num_layouts:
                #         break
                #     last_result = self.run_next_layout(layout)
                #     deviation = abs(last_result[self.metric_name] - self.metric_val) / self.metric_val
                    # if deviation > deviation_threshold:
                    #     break

    def run(self):
        # Define the initial data samples
        res_df = self.generate_initial_samples()

        num_layouts = max(0, (self.num_layouts - len(res_df)))
        if num_layouts == 0:
            logging.info('================================================')
            logging.info(f'No more layouts to run for the experiment:\n{self.exp_root_dir}')
            logging.info('================================================')
            return

        self.generate_and_run_layouts()

        logging.info('================================================')
        logging.info(f'Finished running MosRange process for:\n{self.exp_root_dir}')
        logging.info('================================================')

import argparse
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mem', '--memory_footprint', default='memory_footprint.txt')
    parser.add_argument('-pebs', '--pebs_mem_bins', default=None)
    parser.add_argument('-exp', '--exp_root_dir', required=True)
    parser.add_argument('-res', '--results_file', required=True)
    parser.add_argument('-c', '--collect_reults_cmd', required=True)
    parser.add_argument('-x', '--run_experiment_cmd', required=True)
    parser.add_argument('-n', '--num_layouts', required=True, type=int)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-m', '--metric', choices=['stlb_misses', 'stlb_hits', 'walk_cycles'], default='stlb_misses')
    parser.add_argument('-v', '--metric_value', type=float, default=None)
    parser.add_argument('-p', '--metric_coverage', type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArguments()

    # profiler = cProfile.Profile()
    # profiler.enable()
    if args.metric_value is None and args.metric_coverage is None:
        raise ValueError('Should provide either metric_value or metric_coverage arguments: None was provided!')
    if args.metric_value is not None and args.metric_coverage is not None:
        raise ValueError('Should provide either metric_value or metric_coverage arguments: Both were provided!')

    exp = MosrangeExperiment(args.memory_footprint, args.pebs_mem_bins,
                             args.collect_reults_cmd, args.results_file,
                             args.run_experiment_cmd, args.exp_root_dir,
                             args.num_layouts, args.metric,
                             args.metric_value, args.metric_coverage)
    exp.run()
    # profiler.disable()
    # profiler.dump_stats('profile_results.prof')
    # profiler.print_stats()
