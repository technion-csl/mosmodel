#!/usr/bin/env python3
# import cProfile
import itertools
import os, sys
import logging
import random
import pandas as pd

curr_file_dir = os.path.dirname(os.path.abspath(__file__))
experiments_root_dir = os.path.join(curr_file_dir, "..")
sys.path.append(experiments_root_dir)
from Utils.selector_utils import Selector

class InitLayoutsSelector(Selector):
    def __init__(
        self,
        memory_footprint_file,
        pebs_mem_bins_file,
        exp_root_dir,
        results_dir,
        run_experiment_cmd,
        num_layouts,
        num_repeats,
        metric_name,
        metric_coverage,
        debug=False
    ) -> None:
        super().__init__(
            memory_footprint_file,
            pebs_mem_bins_file,
            exp_root_dir,
            results_dir,
            run_experiment_cmd,
            num_layouts,
            num_repeats,
            metric_name,
            rebuild_pebs=True,
            skip_outliers=False,
            generate_endpoints=True,
            rerun_modified_layouts=False
        )
        self.metric_coverage = metric_coverage
        # Set the seed for reproducibility (optional)
        random.seed(42)
        self.logger = logging.getLogger(__name__)
        self.debug = debug

    # =================================================================== #
    #   Initial layouts - utilities
    # =================================================================== #

    def find_layout(self, pebs_df, pebs_coverage, epsilon=0.2, hot_to_cold=True):
        # if hot_to_cols is None then skip sorting PEBS dataframe
        if hot_to_cold is not None:
            df = pebs_df.sort_values("TLB_COVERAGE", ascending=(not hot_to_cold))

        pages = []
        min_pebs_coverage = pebs_coverage
        max_pebs_coverage = pebs_coverage + epsilon
        total_weight = 0
        for index, row in df.iterrows():
            page = row["PAGE_NUMBER"]
            weight = row["TLB_COVERAGE"]
            updated_total_weight = total_weight + weight
            if updated_total_weight <= max_pebs_coverage:
                pages.append(page)
                total_weight = updated_total_weight
            if min_pebs_coverage <= total_weight <= max_pebs_coverage:
                break
        if len(pages) == 0:
            return None, 0
        pages.sort()
        new_pebs_coverage = pebs_df.query(f"PAGE_NUMBER in {pages}")["TLB_COVERAGE"].sum()

        return pages, new_pebs_coverage

    def createSubgroups(self, group):
        init_layouts = []
        # 1.1.2. create eight layouts as all subgroups of these three group layouts
        for subset_size in range(len(group)+1):
            for subset in itertools.combinations(group, subset_size):
                subset_pages = list(itertools.chain(*subset))
                init_layouts.append(subset_pages)
        init_layouts.append(self.all_2mb_layout)
        return init_layouts

    def fillBuckets(self, df, buckets_weights, start_from_tail=False, fill_min_buckets_first=True):
        group_size = len(buckets_weights)
        group = [ [] for _ in range(group_size) ]
        df = df.sort_values('TLB_COVERAGE', ascending=start_from_tail)

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

    # =================================================================== #
    #   Initial layouts Algorithms
    # =================================================================== #

    # (A) initial layouts algorithm: Moselect, all subgroups of distinct three groups
    def get_moselect_init_layouts(self):
        pebs_df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        # desired weights for each group layout
        buckets_weights = [56, 28, 14]
        group = self.fillBuckets(pebs_df, buckets_weights)
        return self.createSubgroups(group)
   
    # (B) initial layouts algorithm: subgroups of four groups similiarly to moselect (3 groups)
    def get_moselect_init_layouts_4_groups(self):
        pebs_df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        # desired weights for each group layout
        buckets_weights = [52, 26, 13, 6.5]
        group = self.fillBuckets(pebs_df, buckets_weights)
        return self.createSubgroups(group)
    
    # (C) initial layouts algorithm: similar to Moselect but fill buckets from large to small
    def get_moselect_init_layouts_H2C_L2S(self):
        pebs_df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        # desired weights for each group layout
        buckets_weights = [56, 28, 14]
        group = self.fillBuckets(pebs_df, buckets_weights, start_from_tail=False, fill_min_buckets_first=False)
        return self.createSubgroups(group)

    # (D) initial layouts algorithm: similar to Moselect but fill buckets using pages from cold to hot
    def get_moselect_init_layouts_C2H_S2L(self):
        pebs_df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        # desired weights for each group layout
        buckets_weights = [56, 28, 14]
        group = self.fillBuckets(pebs_df, buckets_weights, start_from_tail=True, fill_min_buckets_first=True)
        return self.createSubgroups(group)
    
    # (E) initial layouts algorithm: all subgroups of three distinct groups, filled from large to small using pages from cold to hot
    def get_moselect_init_layouts_C2H_L2S(self):
        pebs_df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        # desired weights for each group layout
        buckets_weights = [56, 28, 14]
        group = self.fillBuckets(pebs_df, buckets_weights, start_from_tail=True, fill_min_buckets_first=False)
        return self.createSubgroups(group)
    
    # (F) initial layouts algorithm: moselect complement
    def get_moselect_distinct_init_layouts(self):
        pebs_df_v1 = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        # desired weights for each group layout
        buckets_weights_v1 = [56, 28, 14]
        group_v1 = self.fillBuckets(pebs_df_v1, buckets_weights_v1)

        exclude_pages = group_v1[2]
        pebs_df_v2_no_14 = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False).query(f'PAGE_NUMBER not in {exclude_pages}')
        group_v2_14 = self.fillBuckets(pebs_df_v2_no_14, [14])

        exclude_pages = group_v1[1] + group_v2_14[0]
        pebs_df_v2_no_28 = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False).query(f'PAGE_NUMBER not in {exclude_pages}')
        group_v2_28 = self.fillBuckets(pebs_df_v2_no_28, [28])

        exclude_pages = group_v2_14[0] + group_v2_28[0]
        pebs_df_v2 = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False).query(f'PAGE_NUMBER not in {exclude_pages}')
        group_v2_56 = self.fillBuckets(pebs_df_v2, [56])

        group_v2 = [group_v2_56[0], group_v2_28[0], group_v2_14[0]]
        return self.createSubgroups(group_v2)
    
    # (G/H) initial layouts algorithm:  complement of a given layout
    def get_complement_surrounding_layouts(self, layout, layout_result):
        layout_real_coverage = self.realMetricCoverage(layout_result)
        alpha = set(layout)
        alpha_layout = list(alpha)
        beta = set(self.all_pebs_pages_layout) - alpha
        beta_layout = list(beta)
        beta_df = self.pebs_df.query(f'PAGE_NUMBER in {beta_layout}')
        beta_pebs = self.pebsTlbCoverage(beta_layout)
        left_pebs_coverage = min(self.metric_coverage + 10, (100 + self.metric_coverage) / 2)

        res_layouts = []
        if beta_pebs >= left_pebs_coverage:
            new_layout, new_pebs = self.find_layout(beta_df, left_pebs_coverage, epsilon=0.2, hot_to_cold=True)
            if new_layout is None:
                alpha_df = self.pebs_df.query(f'PAGE_NUMBER in {alpha_layout}')
                pebs_df = pd.concat([beta_df, alpha_df], ignore_index=True)
                new_layout, new_pebs = self.find_layout(pebs_df, left_pebs_coverage, epsilon=1, hot_to_cold=None)
        else:
            new_layout, new_pebs = self.add_pages_to_base_layout(beta_layout, alpha_layout, None, left_pebs_coverage, tail=False)
            if new_layout is None:
                alpha_df = self.pebs_df.query(f'PAGE_NUMBER in {alpha_layout} and TLB_COVERAGE > 0.01')
                filtered_alpha_layout = list(set(alpha_df['PAGE_NUMBER'].tolist()))
                new_layout, new_pebs = self.add_pages_to_base_layout(beta_layout, filtered_alpha_layout, None, left_pebs_coverage, tail=True)
                if new_layout is None:
                    self.logger.warning(f"get_complement_surrounding_layouts: using tail pages to select layout based on beta pages")
                    new_layout, new_pebs = self.add_pages_to_base_layout(beta_layout, alpha_layout, None, left_pebs_coverage, tail=True)

        assert new_layout is not None
        res_layouts.append(new_layout)

        new_layout_result = self.run_next_layout(new_layout)
        new_layout_real_coverage = self.realMetricCoverage(new_layout_result)

        if new_layout_real_coverage > layout_real_coverage:
            # the new layout fall at left side, then we can take off some pages
            # from it to get a new layout at the right of the base layout
            for i in range(1, 10):
                expected_pebs = max(0, new_pebs - (10*i))
                right_layout, _ = self.remove_pages_in_order(new_layout, None, expected_pebs)
                if right_layout is None:
                    continue
                right_layout_result = self.run_next_layout(right_layout)
                right_layout_real = self.realMetricCoverage(right_layout_result)
                if right_layout_real <= layout_real_coverage:
                    res_layouts.append(right_layout)
                    break
        else: # new_layout_real_coverage <= layout_real_coverage
            # the new layout fall at right side, then we need to add some pages
            # to it from alpha set
            for i in range(1, 10):
                expected_pebs = min(100, new_pebs + (10*i))
                # 1st trial: use alpha pages in hot->cold order
                left_layout, _ = self.add_pages_to_base_layout(new_layout, alpha_layout, None, expected_pebs, tail=False)
                if left_layout is None:
                    non_zero_df = self.pebs_df.query(f'TLB_COVERAGE > 0.01')
                    non_zero_pages = list(set(non_zero_df['PAGE_NUMBER'].tolist()))
                    # 2nd trial: use all non-zero pages in hot->cold order
                    left_layout, _ = self.add_pages_to_base_layout(new_layout, non_zero_pages, None, expected_pebs, tail=False)
                    if left_layout is None:
                        # 3rd trial: use all non-zero pages in cold->hot order
                        left_layout, _ = self.add_pages_to_base_layout(new_layout, non_zero_pages, None, expected_pebs, tail=True)
                        if left_layout is None:
                            # 4th trial: use all pages in cold->hot order
                            left_layout, _ = self.add_pages_to_base_layout(new_layout, self.all_pebs_pages_layout, None, expected_pebs)
                if left_layout is None:
                    continue
                left_layout_result = self.run_next_layout(left_layout)
                left_layout_real = self.realMetricCoverage(left_layout_result)
                if left_layout_real >= layout_real_coverage:
                    res_layouts.append(left_layout)
                    break
        assert len(res_layouts) == 2
        return res_layouts

   