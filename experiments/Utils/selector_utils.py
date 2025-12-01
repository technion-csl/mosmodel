#!/usr/bin/env python3
# import cProfile
import pandas as pd
import subprocess
import math
import os
import logging
from Utils.collect_results import CollectResults
from Utils.layout_utils import LayoutUtils
from Utils.utils import Utils

class Selector:
    DEFAULT_HUGEPAGE_SIZE = 1 << 21 # 2MB 0

    def __init__(self,
                 memory_footprint_file, pebs_mem_bins_file,
                 exp_root_dir, results_dir,
                 run_experiment_cmd,
                 num_layouts, num_repeats,
                 metric_name='stlb_misses',
                 rebuild_pebs=True,
                 skip_outliers=False,
                 generate_endpoints=True,
                 rerun_modified_layouts=False) -> None:
        self.memory_footprint_file = memory_footprint_file
        self.pebs_mem_bins_file = pebs_mem_bins_file
        self.exp_root_dir = exp_root_dir
        self.num_layouts = num_layouts
        self.num_repeats = num_repeats
        self.collectResults = CollectResults(exp_root_dir, results_dir, num_repeats, outlier_method='std')
        self.run_experiment_cmd = run_experiment_cmd
        self.metric_name = metric_name
        self.rebuild_pebs = rebuild_pebs
        self.skip_outliers = skip_outliers
        self.generate_endpoints = generate_endpoints
        self.rerun_modified_layouts = rerun_modified_layouts
        self.last_layout_num = 0
        self.num_generated_layouts = 0
        self.layouts = []
        self.phase_layout_names = []
        self.layout_names = []
        self.budget = None
        self.logger = logging.getLogger(__name__)
        self.all_2mb_r = None
        self.all_4kb_r = None
        self.load_completed = False
        self.last_run_layout_counter = 0
        self.__load()

    def __load(self):
        # read memory-footprints
        self.footprint_df = pd.read_csv(self.memory_footprint_file)
        self.mmap_footprint = self.footprint_df['anon-mmap-max'][0]
        self.brk_footprint = self.footprint_df['brk-max'][0]

        self.hugepage_size = Selector.DEFAULT_HUGEPAGE_SIZE
        self.num_hugepages = math.ceil(self.brk_footprint / self.hugepage_size) # bit vector length

        # round up the memory footprint to match the new boundaries of the new hugepage-size
        self.memory_footprint = (self.num_hugepages + 1) * self.hugepage_size
        self.brk_footprint = self.memory_footprint

        self.all_pages = [i for i in range(self.num_hugepages)]
        if self.pebs_mem_bins_file is None:
            self.logger.warning('pebs_mem_bins_file argument is missing, skipping loading PEBS results...')
            self.pebs_df = None
        else:
            self.pebs_df = Selector.load_pebs(self.pebs_mem_bins_file, True)
            self.pebs_pages = list(set(self.pebs_df['PAGE_NUMBER'].to_list()))
            self.pages_not_in_pebs = list(set(self.all_pages) - set(self.pebs_pages))
            self.total_misses = self.pebs_df['NUM_ACCESSES'].sum()

         # load results file
        self.results_df, _ = self.collect_results(filter_results=False)

        self.all_4kb_layout = []
        self.all_2mb_layout = [i for i in range(self.num_hugepages)]
        self.all_pebs_pages_layout = self.pebs_pages

        if self.generate_endpoints:
            self.run_endpoint_layouts()

        self.load_completed = True

        # update results_df after endpoint layouts were added
        for index, row in self.results_df.iterrows():
            mem_layout_pages = self.results_df.at[index, 'hugepages']
            self.results_df.at[index, 'pebs_coverage'] = self.pebsTlbCoverage(mem_layout_pages)
            self.results_df.at[index, 'real_coverage'] = self.realMetricCoverage(self.results_df.loc[index])

    def run_command(self, command, out_dir):
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
        with open(output_log, 'a+') as out:
            out.write(output)
            out.write('============================================')
            out.write(f'the process exited with status: {return_code}')
            out.write('============================================')
        with open(error_log, 'a+') as err:
            err.write(error)
            err.write('============================================')
            err.write(f'the process exited with status: {return_code}')
            err.write('============================================')
        if return_code != 0:
            # log the output and error
            self.logger.error('============================================')
            self.logger.error(f'Failed to run the following command with exit code: {return_code}')
            self.logger.error(f'Command line: {command}')
            self.logger.error('Output:', output)
            self.logger.error('Error:', error)
            self.logger.error('Return code:', return_code)
            self.logger.error('============================================')

        return return_code

    def collect_results(self, filter_results=True):
        results_df, found_outliers = self.collectResults.collectResults(True, True, True)
        if found_outliers:
            self.logger.info(f'-- outliers were found and removed --')
        if results_df is None or results_df.empty:
            return results_df, found_outliers

        results_df['hugepages'] = None
        results_df['pebs_coverage'] = None
        results_df['real_coverage'] = None
        for index, row in results_df.iterrows():
            layout_name = row['layout']
            mem_layout_pages = LayoutUtils.load_layout_hugepages(layout_name, self.exp_root_dir)
            results_df.at[index, 'hugepages'] = mem_layout_pages
            if self.load_completed:
                results_df.at[index, 'pebs_coverage'] = self.pebsTlbCoverage(mem_layout_pages)
                results_df.at[index, 'real_coverage'] = self.realMetricCoverage(results_df.loc[index])

        self._full_results_df = results_df.copy()
        if filter_results:
            results_df = results_df.query(f'layout in {self.layout_names}').reset_index(drop=True)
            self.logger.info(f'collect results and keep the following {len(self.layout_names)} layouts: {self.layout_names[0]}--{self.layout_names[-1]}')
            self.logger.info(f'** kept results of {len(results_df)} collected layouts **')

        # print results of previous runs
        for index, row in results_df.iterrows():
            self.logger.debug(f'{row["layout"]}: coverage={self.pebsTlbCoverage(list(row["hugepages"]))} ({len(row["hugepages"])} x hugepages), runtime={row["cpu_cycles"]} , tlb-misses={row["stlb_misses"]}')

        return results_df, found_outliers

    def load_pebs(pebs_out_file, normalize=True):
        # read mem-bins
        pebs_df = pd.read_csv(pebs_out_file, delimiter=',')

        if not normalize:
            pebs_df = pebs_df[['PAGE_NUMBER', 'NUM_ACCESSES']]
            pebs_df = pebs_df.reset_index()
            pebs_df = pebs_df.sort_values('NUM_ACCESSES', ascending=False)
            return pebs_df

        # filter and eep only brk pool accesses
        pebs_df = pebs_df[pebs_df['PAGE_TYPE'].str.contains('brk')]
        if pebs_df.empty:
            raise SyntaxError('Input file does not contain page accesses information about the brk pool!')
        pebs_df = pebs_df[['PAGE_NUMBER', 'NUM_ACCESSES']]

        # transform NUM_ACCESSES from absolute number to percentage
        total_access = pebs_df['NUM_ACCESSES'].sum()
        pebs_df['TLB_COVERAGE'] = pebs_df['NUM_ACCESSES'].mul(100).divide(total_access)
        pebs_df = pebs_df.sort_values('TLB_COVERAGE', ascending=False)

        pebs_df = pebs_df.reset_index()

        return pebs_df

    def predictTlbMisses(self, mem_layout, pebs_df=None):
        if pebs_df is None:
            pebs_df = self.pebs_df

        assert pebs_df is not None
        expected_tlb_coverage = pebs_df.query(f'PAGE_NUMBER in {mem_layout}')['NUM_ACCESSES'].sum()
        expected_tlb_misses = self.total_misses - expected_tlb_coverage
        self.logger.debug(f'mem_layout of size {len(mem_layout)} has an expected-tlb-coverage={expected_tlb_coverage} and expected-tlb-misses={expected_tlb_misses}')
        return expected_tlb_misses

    def pebsTlbCoverage(self, mem_layout, pebs_df=None):
        if pebs_df is None:
            pebs_df = self.pebs_df

        assert pebs_df is not None
        df = pebs_df.query(f'PAGE_NUMBER in {mem_layout}')
        expected_tlb_coverage = df['TLB_COVERAGE'].sum()
        return expected_tlb_coverage

    def realMetricCoverage(self, layout_results, metric_name=None):
        if metric_name is None:
            metric_name = self.metric_name
        return self.realCoverage(layout_results[metric_name], metric_name)

    def realCoverage(self, layout_metric_val, metric_name=None):
        if not self.load_completed:
            return None
        if metric_name is None:
            metric_name = self.metric_name
        all_2mb_metric_val = self.all_2mb_r[metric_name]
        all_4kb_metric_val = self.all_4kb_r[metric_name]
        min_val = min(all_2mb_metric_val, all_4kb_metric_val)
        max_val = max(all_2mb_metric_val, all_4kb_metric_val)

        # either metric_val or metric_coverage will be provided
        delta = max_val - min_val
        coverage = ((max_val - layout_metric_val) / delta) * 100

        return coverage

    def run_endpoint_layouts(self):
        mem_layouts = [self.all_4kb_layout, self.all_2mb_layout, self.all_pebs_pages_layout]
        for i, mem_layout in enumerate(mem_layouts):
            self.logger.info('============================================================')
            self.logger.info(f'** Running endpoint memory layout #{i}')
            self.logger.info(f'\tlayout length: {len(mem_layout)} (x2MB) hugepages')
            self.logger.info('============================================================')
            self.run_next_layout(mem_layout)
        self.update_endpoint_results(self.results_df)
        return self.results_df

    def update_endpoint_results(self, res_df):
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

        self.load_completed = True
        assert self.all_2mb_r is not None
        assert self.all_4kb_r is not None

        all_2mb_metric_val = self.all_2mb_r[self.metric_name]
        all_4kb_metric_val = self.all_4kb_r[self.metric_name]
        self.metric_min_val = min(all_2mb_metric_val, all_4kb_metric_val)
        self.metric_max_val = max(all_2mb_metric_val, all_4kb_metric_val)

        # either metric_val or metric_coverage will be provided
        self.metric_range_delta = self.metric_max_val - self.metric_min_val

        all_pebs_real_misses_coverage = self.realMetricCoverage(self.all_pebs_r, 'stlb_misses')
        all_pebs_real_runtime_coverage = self.realMetricCoverage(self.all_pebs_r, 'cpu_cycles')
        if all_pebs_real_misses_coverage < 80 or all_pebs_real_runtime_coverage < 80:
            self.logger.error(f'PEBS could not sample all pages with major L2 TLB misses!')
            self.logger.error(f'Please try to rerun PEBS with higher frequency!')
            self.logger.error(f'Please make sure that PEBS is highly accurate and have no Errata in your CPU generation.')
            self.logger.error(f'You can try run PEBS on a newer CPU generation and use it for this CPU generation (as both have similar TLB structure)')
            self.logger.error(f'All 4KB layout results: {self.all_4kb_r}')
            self.logger.error(f'All 2MB layout results: {self.all_2mb_r}')
            self.logger.error(f'All PEBS layout results: {self.all_pebs_r}')
            assert False

        # add missing pages to pebs
        if self.rebuild_pebs:
            # backup pebs_df before updating it
            self.orig_pebs_df = self.pebs_df.copy()
            self.orig_total_misses = self.total_misses
            self.pebs_df = self.add_missing_pages_to_pebs(self.pebs_df)
            self.total_misses = self.pebs_df['NUM_ACCESSES'].sum()
            # print(f'saving update pebs to: {os.getcwd()}/updated_pebs.csv')
            # self.pebs_df.to_csv('updated_pebs.csv')
            # sys.exit(0)

    def select_layout_from_pebs_gradually(self, pebs_coverage, pebs_df):
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
            self.logger.debug(f'select_layout_from_pebs_gradually(): total_weight < (pebs_coverage - self.search_pebs_threshold): {total_weight} < ({pebs_coverage} - {self.search_pebs_threshold})')
            return []
        self.logger.debug(f'select_layout_from_pebs_gradually(): found layout of length: {len(mem_layout)}')
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
        if self.results_df is None:
            return None
        layout_results = self.results_df[self.results_df['layout'] == layout_name]
        if layout_results.empty:
            return None
        return layout_results.iloc[0]

    def get_surrounding_layouts(self, res_df, metric_name, metric_val):
        # sort pebs by stlb-misses
        df = res_df.sort_values(metric_name, ascending=True).reset_index(drop=True)
        lower_layout = None
        upper_layout = None
        for index, row in df.iterrows():
            row_metric_val = row[metric_name]
            if row_metric_val <= metric_val:
                lower_layout = row
            else:
                upper_layout = row
                break
        return lower_layout['hugepages'], upper_layout['hugepages']

    def add_missing_pages_to_pebs(self, pebs_df):
        pebs_pages = list(set(pebs_df['PAGE_NUMBER'].tolist()))
        missing_pages = list(set(self.all_2mb_layout) - set(pebs_pages))
        #self.total_misses
        all_pebs_real_coverage = self.realMetricCoverage(self.all_pebs_r, 'stlb_misses')
        all_pebs_real_coverage = min(100, all_pebs_real_coverage) # in case all_pebs layout is a little bit better than all_2mb layout
        # normalize pages recorded by pebs based on their real coverage
        ratio = all_pebs_real_coverage / 100
        pebs_df['TLB_COVERAGE'] *= ratio
        pebs_df['NUM_ACCESSES'] = (pebs_df['NUM_ACCESSES'] * ratio).astype(int)
        # add missing pages with a unified coverage ratio
        missing_pages_total_coverage = 100 - all_pebs_real_coverage
        total_missing_pages = len(missing_pages)
        if total_missing_pages == 0:
            return
        missing_pages_coverage_ratio = missing_pages_total_coverage / total_missing_pages
        # update total_misses acording to the new ratio
        old_total_misses = self.total_misses
        self.total_misses = int(self.total_misses * ratio)
        missing_pages_total_misses = old_total_misses - self.total_misses
        missing_pages_misses_ratio = missing_pages_total_misses / total_missing_pages
        # update pebs_df dataframe
        missing_pages_df = pd.DataFrame(
            {'PAGE_NUMBER': missing_pages,
             'NUM_ACCESSES': missing_pages_misses_ratio,
             'TLB_COVERAGE': missing_pages_coverage_ratio})
        pebs_df = pd.concat([pebs_df, missing_pages_df], ignore_index=True)
        return pebs_df

    def write_layout(self, layout_name, mem_layout):
        self.logger.info(f'writing {layout_name} with {len(mem_layout)} hugepages')
        LayoutUtils.write_layout(layout_name, mem_layout, self.exp_root_dir, self.brk_footprint, self.mmap_footprint)
        self.layouts.append(mem_layout)
        self.layout_names.append(layout_name)

    def isPagesListUnique(self, pages_list, all_layouts, debug=False):
        pages_set = set(pages_list)
        i=0
        for l in all_layouts:
            i += 1
            if set(l) == pages_set:
                if debug:
                    logging.info('-----------------------------')
                    logging.info(f'the compared pages list is equal to layout {i}')
                    logging.info('-----------------------------')
                return False
        return True

    def find_layout_results(self, layout):
        full_results_df, _ = self.collect_results(False)
        if full_results_df is None or full_results_df.empty:
            return False, None
        for index, row in full_results_df.iterrows():
            prev_layout_hugepages = row['hugepages']
            if set(prev_layout_hugepages) == set(layout):
                return True, row
        return False, None

    def layout_was_run(self, layout_name, mem_layout):
        full_results_df, _ = self.collect_results(False)
        if full_results_df is None or full_results_df.empty:
            return False, None

        prev_layout_res = None
        prev_layout_res = full_results_df.query(f'layout == "{layout_name}"')
        # prev_layout_res = full_results_df[full_results_df['layout'] == layout_name]
        if prev_layout_res is None or prev_layout_res.empty:
            # the layout does not exist in the results file
            return False, None
        prev_layout_res = prev_layout_res.iloc[0]
        prev_layout_hugepages = prev_layout_res['hugepages']
        if set(prev_layout_hugepages) != set(mem_layout):
            # the existing layout has different hugepages set than the new one
            return False, prev_layout_res

        # the layout exists and has the same hugepages set
        return True, prev_layout_res

    def custom_log_layout_result(self, layout_res, old_result=False):
        pass

    def log_layout_result(self, layout_res, old_result=False):
        tlb_misses = layout_res['stlb_misses']
        tlb_hits = layout_res['stlb_hits']
        walk_cycles = layout_res['walk_cycles']
        runtime = layout_res['cpu_cycles']

        self.logger.info('-------------------------------------------')
        self.logger.info(f'Results:')
        self.logger.info(f'\tstlb-misses={Utils.format_large_number(tlb_misses)}')
        self.logger.info(f'\tstlb-hits={Utils.format_large_number(tlb_hits)}')
        self.logger.info(f'\twalk-cycles={Utils.format_large_number(walk_cycles)}')
        self.logger.info(f'\truntime={Utils.format_large_number(runtime)}')
        self.custom_log_layout_result(layout_res, old_result)
        # self.logger.info(f'\treal-coverage: {self.realMetricCoverage(layout_res, self.metric_name)}')
        self.logger.info('-------------------------------------------')

    def run_workload(self, mem_layout, layout_name):
        found, prev_res = self.layout_was_run(layout_name, mem_layout)
        # if the layout's measurements were found
        if found and prev_res is not None:
            self.logger.info(f'+++ {layout_name} already exists, skip running it +++')
            self.layouts.append(mem_layout)
            self.layout_names.append(layout_name)
            self.log_layout_result(prev_res, True)
            self.phase_layout_names.append(layout_name)
            return prev_res
        elif not found and prev_res is not None:
            found_content, content_prev_res = self.find_layout_results(mem_layout)
            # if the layout was found but under different layout_name
            if found_content:
                self.last_layout_num -= 1
                # assert False
                self.logger.warning(f'--- {layout_name} already exists but under different name [{content_prev_res["layout"]}]. ---')
                self.phase_layout_names.append(content_prev_res["layout"])
                return content_prev_res

            self.logger.warning(f'--- {layout_name} already exists but its content is changed. ---')
            if self.rerun_modified_layouts:
                self.logger.warning(f'--- Overwriting {layout_name} and rerunning ---')
            else:
                self.logger.warning(f'--- Skipping rerunning {layout_name} ---')
                prev_mem_layout = prev_res['hugepages']
                self.layouts.append(prev_mem_layout)
                self.layout_names.append(layout_name)
                self.log_layout_result(prev_res, True)
                self.phase_layout_names.append(layout_name)
                return prev_res

        self.last_run_layout_counter = 0
        self.num_generated_layouts += 1
        self.phase_layout_names.append(layout_name)
        self.write_layout(layout_name, mem_layout)
        out_dir = f'{self.exp_root_dir}/{layout_name}'
        run_cmd = f'{self.run_experiment_cmd} {layout_name}'

        self.logger.info('=======================================================')
        self.logger.info(f'==> start running {layout_name} (#{len(mem_layout)} hugepages)')

        self.logger.info('------------------------------------------')
        self.logger.info(f'*** start running {out_dir} ***')
        self.logger.info(f'\t experiment: {self.exp_root_dir}')
        self.logger.info(f'\t layout: {layout_name}')
        self.logger.info(f'\t #hugepages: {len(mem_layout)}')
        self.logger.info(f'\t pebs-coverage: {round(self.pebsTlbCoverage(mem_layout), 2)}%')
        self.logger.debug(f'\t script: {run_cmd}')
        self.logger.info('------------------------------------------')

        found_outliers = True
        while found_outliers:
            ret_code = self.run_command(run_cmd, out_dir)
            if ret_code != 0:
                raise RuntimeError(f'Error: running {layout_name} failed with error code: {ret_code}')
            self.results_df, found_outliers = self.collect_results()
            if self.skip_outliers:
                break

        layout_res = self.get_layout_results(layout_name)
        if 'hugepages' not in layout_res:
            layout_res['hugepages'] = mem_layout
        if self.load_completed:
            layout_res['pebs_coverage'] = self.pebsTlbCoverage(mem_layout)
            layout_res['real_coverage'] = self.realMetricCoverage(layout_res)
        self.log_layout_result(layout_res)

        self.logger.info(f'<== completed running {layout_name}')
        self.logger.info('=======================================================')
        return layout_res

    def run_next_layout(self, mem_layout):
        self.last_layout_num += 1
        layout_name = f'layout{self.last_layout_num}'

        last_result = self.run_workload(mem_layout, layout_name)

        return last_result

    def run_layouts(self, layouts):
        results_df = pd.DataFrame()
        for l in layouts:
            r = self.run_next_layout(l)
            results_df = pd.concat([results_df, r], ignore_index=True)
        return results_df

    def reset_budget(self, new_budget):
        self.num_generated_layouts = 0
        self.budget = new_budget

    def disable_budget(self):
        self.budget = None

    def has_budget(self):
        if self.budget is None:
            return True
        remaining_budget = self.get_remaining_budget()
        return remaining_budget > 0

    def consumed_budget(self):
        return not self.has_budget()

    def get_remaining_budget(self):
        return self.budget - self.num_generated_layouts

    def reset_phase_layout_names(self):
        self.phase_layout_names = []
