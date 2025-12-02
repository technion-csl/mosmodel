#!/usr/bin/env python3
# import cProfile
import itertools
import os, sys
import logging
import random
import numpy as np
from pathlib import Path

curr_file_dir = os.path.dirname(os.path.abspath(__file__))
experiments_root_dir = os.path.join(curr_file_dir, "..")
sys.path.append(experiments_root_dir)
from Utils.utils import Utils
from Utils.selector_utils import Selector


class MosrangeSelector(Selector):
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
        metric_val,
        metric_coverage,
        range_epsilon=0.01,
        absolute_range_epsilon=False,
        debug=False
    ) -> None:
        self.num_generated_layouts = 0
        self.metric_val = metric_val
        self.metric_coverage = metric_coverage
        self.range_epsilon = range_epsilon
        self.absolute_range_epsilon = absolute_range_epsilon
        self.search_pebs_threshold = 0.5
        self.last_lo_layout = None
        self.last_hi_layout = None
        self.last_layout_result = None
        self.last_runtime_range = 0
        self.head_pages_coverage_threshold = 2
        self.num_initial_layouts = 0
        rerun_modified_layouts = not debug
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
            rerun_modified_layouts=rerun_modified_layouts
        )
        # Set the seed for reproducibility (optional)
        random.seed(42)
        self.logger = logging.getLogger(__name__)
        self.update_metric_values()
        self.debug = debug

        current_file_path = Path(__file__).resolve()
        current_directory = current_file_path.parent
        self.log_file_path = current_directory / "log.csv"

    def update_metric_values(self):
        if self.metric_val is None:
            self.metric_val = self.metric_max_val - (
                self.metric_range_delta * (self.metric_coverage / 100)
            )
        else:
            self.metric_coverage = (
                (self.metric_max_val - self.metric_val) / self.metric_range_delta
            ) * 100

        self.metric_pebs_coverage = self.metric_coverage
        if self.metric_name == 'stlb_hits':
            self.metric_pebs_coverage = 100 - self.metric_coverage

    def get_surrounding_pair(
        self,
        res_df,
        layout_pair_idx=0,
        by="cpu_cycles",
        ascending=False,
        return_full_result=False,
    ):
        self.logger.debug(f"entry - get_surrounding_pair(layout_pair_idx={layout_pair_idx})")

        all_pairs_df = self.get_surrounding_layouts(res_df, by, ascending)

        if layout_pair_idx >= len(all_pairs_df):
            layout_pair_idx = 0
        selected_pair = all_pairs_df.iloc[layout_pair_idx]

        self.logger.debug(f"-------------------------------------------------------")
        self.logger.debug(
            f'get_surrounding_layouts: selected layouts [{selected_pair["layout_lo"]} , {selected_pair["layout_hi"]}]'
        )
        self.logger.debug(f"\t selected layouts:")
        self.logger.debug(f"\n{selected_pair}")
        self.logger.debug(f"-------------------------------------------------------")

        lo_layout = selected_pair[f"hugepages_lo"]
        hi_layout = selected_pair[f"hugepages_hi"]
        if return_full_result:
            cols = selected_pair.keys()
            # remove duplicates
            cols = list(set(cols))
            lo_cols = [c for c in cols if c.endswith("_lo")]
            hi_cols = [c for c in cols if c.endswith("_hi")]
            # split the result to two serieses
            lo_layout = selected_pair[lo_cols]
            hi_layout = selected_pair[hi_cols]
            # remove suffixes
            lo_cols = [c.replace("_lo", "") for c in lo_cols]
            hi_cols = [c.replace("_hi", "") for c in hi_cols]
            # rename columns (by removing the _lo and _hi suffixes)
            lo_layout = lo_layout.set_axis(lo_cols)
            hi_layout = hi_layout.set_axis(hi_cols)

        self.logger.debug(f"exit - get_surrounding_pair(layout_pair_idx={layout_pair_idx})")

        return lo_layout, hi_layout

    def get_surrounding_layouts(self, res_df, by="cpu_cycles", ascending=False):
        self.logger.debug(f"entry - get_surrounding_layouts")

        df = res_df.sort_values(self.metric_name, ascending=True).reset_index(drop=True)
        lo_layouts_df = df.query(f"{self.metric_name} < {self.metric_val}")
        assert len(lo_layouts_df) > 0

        hi_layouts_df = df.query(f"{self.metric_name} >= {self.metric_val}")
        assert len(hi_layouts_df) > 0

        all_pairs_df = lo_layouts_df.merge(
            hi_layouts_df, how="cross", suffixes=["_lo", "_hi"]
        )
        all_pairs_df[f"{by}_diff"] = abs(
            all_pairs_df[f"{by}_lo"] - all_pairs_df[f"{by}_hi"]
        )
        all_pairs_df = all_pairs_df.sort_values(
            f"{by}_diff", ascending=ascending
        ).reset_index(drop=True)

        self.logger.debug(f"exit - get_surrounding_layouts --> {len(all_pairs_df)} pairs")
        return all_pairs_df

    def calculate_runtime_range(self):
        self.logger.debug(f"entry -calculate_runtime_range")

        metric_low_val = self.metric_val * (1 - self.range_epsilon)
        metric_hi_val = self.metric_val * (1 + self.range_epsilon)
        range_df = self.results_df.query(
            f"{metric_hi_val} >= {self.metric_name} >= {metric_low_val}"
        )

        if len(range_df) < 2:
            self.logger.debug(f"exit -calculate_runtime_range --> 0 (less than two points)")
            return 0

        max_runtime = range_df["cpu_cycles"].max()
        min_runtime = range_df["cpu_cycles"].min()
        range_percentage = (max_runtime - min_runtime) / min_runtime
        range_percentage = round(range_percentage * 100, 2)

        self.logger.debug(f"exit -calculate_runtime_range --> {range_percentage}")
        return range_percentage

    def pause():
        print("=============================")
        print("press any key to continue ...")
        print("=============================")
        input()

    def log_metadata(self):
        self.logger.info(
            "================================================================="
        )
        self.logger.info(f"** Metadata: **")
        self.logger.info(f"\t metric_name: {self.metric_name}")
        self.logger.info(f"\t metric_coverage: {round(self.metric_coverage, 2)}%")
        self.logger.info(f"\t metric_val: {Utils.format_large_number(self.metric_val)}")
        self.logger.info(
            f"\t metric_min_val: {Utils.format_large_number(self.metric_min_val)}"
        )
        self.logger.info(
            f"\t metric_max_val: {Utils.format_large_number(self.metric_max_val)}"
        )
        self.logger.info(
            f"\t metric_range_delta: {Utils.format_large_number(self.metric_range_delta)}"
        )
        self.logger.info(f"\t #pages_in_pebs: {len(self.pebs_pages)}")
        self.logger.info(f"\t #pages_not_in_pebs: {len(self.pages_not_in_pebs)}")
        self.logger.info(f"\t #layouts: {self.num_layouts}")
        self.logger.info(f"\t #repeats: {self.num_repeats}")
        self.logger.info(
            "================================================================="
        )

    def custom_log_layout_result(self, layout_res, old_result=False):
        if old_result:
            return
        if not self.load_completed:
            return
        # if endpoints were not run already, then skip
        if not hasattr(self, "all_2mb_r"):
            return
        self.logger.info(
            f"\texpected-coverage={Utils.format_large_number(self.metric_coverage)}"
        )
        self.logger.info(
            f"\treal-coverage={Utils.format_large_number(self.realMetricCoverage(layout_res))}"
        )
        self.logger.info(
            f"\texpected-{self.metric_name}={Utils.format_large_number(self.metric_val)}"
        )
        self.logger.info(
            f"\treal-{self.metric_name}={Utils.format_large_number(layout_res[self.metric_name])}"
        )
        self.logger.info(
            f"\tis_result_within_target_range={self.is_result_within_target_range(layout_res)}"
        )
        prev_runtime_range = self.last_runtime_range
        curr_runtime_range = self.calculate_runtime_range()
        self.last_runtime_range = curr_runtime_range
        runtime_range_improvement = curr_runtime_range - prev_runtime_range
        self.logger.info(f"\tprev_runtime_range={prev_runtime_range}%")
        self.logger.info(f"\tcurr_runtime_range={curr_runtime_range}%")
        self.logger.info(f"\truntime_range_improvement={runtime_range_improvement}%")

    # =================================================================== #
    #   Shake runtime
    # =================================================================== #

    def get_layounts_within_target_range(self):
        epsilon = self.range_epsilon
        rel_range_delta = epsilon * self.metric_range_delta
        min_val = self.metric_val - rel_range_delta
        max_val = self.metric_val + rel_range_delta
        res = self.results_df.query(f"{min_val} <= {self.metric_name} <= {max_val}")
        return res

    def add_tails_pages_func(self, base_pages, tested_tail_pages, subset):
        self.logger.debug(f"add_tails_pages_func")
        layout = list(set(set(base_pages + tested_tail_pages + subset)))
        return layout

    def remove_tails_pages_func(self, base_pages, tested_tail_pages, subset):
        self.logger.debug(f"remove_tails_pages_func")
        layout = list(set(set(base_pages) - set(tested_tail_pages) - set(subset)))
        return layout

    def binary_search_tail_pages_selector(
        self, base_pages, tail_pages, create_layout_func, max_iterations
    ):
        self.logger.debug(f"entry: binary_search_tail_pages_selector")

        zero_pages = []
        iterations = 0

        def evaluate_subset(subset, tested_tail_pages):
            nonlocal iterations
            layout = create_layout_func(base_pages, tested_tail_pages, subset)
            # if layout and self.isPagesListUnique(layout, self.layouts):
            if layout:
                self.last_layout_result = self.run_next_layout(layout)
                iterations += 1
                return self.is_result_within_target_range(self.last_layout_result)
            return False

        def search(left, right):
            nonlocal iterations

            self.logger.debug(f"add_tails_pages_func->search: iterations={iterations}")

            if left >= right:
                return

            mid = (left + right) // 2
            left_subset = tail_pages[left:mid]
            right_subset = tail_pages[mid:right]

            left_in_range = evaluate_subset(left_subset, zero_pages)
            if left_in_range:
                # If the left subset is under the threshold, add it to the zero_pages
                zero_pages.extend(left_subset)
            if iterations > max_iterations:
                return
            if not left_in_range:
                search(left, mid)

            right_in_range = evaluate_subset(right_subset, zero_pages)
            if right_in_range:
                # If the left subset is under the threshold, add it to the zero_pages
                zero_pages.extend(right_subset)
            if iterations > max_iterations:
                return
            if not right_in_range:
                search(mid, right)

            if left_in_range and right_in_range:
                return

        # Start the search with the entire list
        search(0, len(tail_pages))
        layout = create_layout_func(base_pages, zero_pages, [])

        self.logger.debug(f"exit - add_tails_pages_func --> layout:#{len(layout)} pages , zero:#{len(zero_pages)} pages")
        return zero_pages, layout

    def get_tail_pages(self, threshold=0.01, total_threshold=2):
        tail_pages_df = self.pebs_df.query(f"TLB_COVERAGE < {threshold}")
        tail_pages_df = tail_pages_df.sort_values("TLB_COVERAGE", ascending=True)
        tail_pages_df["tail_cumsum"] = tail_pages_df["TLB_COVERAGE"].cumsum()
        tail_pages_df = tail_pages_df.query(f"tail_cumsum <= {total_threshold}")
        tail_pages = tail_pages_df["PAGE_NUMBER"].to_list()

        self.logger.debug(f"get_tail_pages --> tail pages:#{len(tail_pages)}")
        return tail_pages

    def shake_all_layouts_in_range(self, max_iterations):
        self.logger.debug(f"entry - shake_all_layouts_in_range(max_iterations={max_iterations})")

        range_layouts_df = self.get_layounts_within_target_range()
        for index, row in range_layouts_df.iterrows():
            layout = row["hugepages"]
            self.shake_runtime(layout)

        self.logger.debug(f"exit - shake_all_layouts_in_range(max_iterations={max_iterations})")

    def shake_runtime(self, layout, max_iterations):
        self.logger.debug(f"entry - shake_runtime={max_iterations}")

        self.num_generated_layouts = 0
        tail_pages = self.get_tail_pages()
        range_layouts_df = self.get_layounts_within_target_range()
        for index, row in range_layouts_df.iterrows():
            layout = row["hugepages"]
            self.last_layout_result = self.run_next_layout(layout)
            zero_pages1, new_layout = self.binary_search_tail_pages_selector(
                layout, tail_pages, self.remove_tails_pages_func, (max_iterations // 2) - 1
            )
            zero_pages2, new_layout = self.binary_search_tail_pages_selector(
                layout, tail_pages, self.add_tails_pages_func, (max_iterations // 2) - 1
            )
            layout_with_all_zeroes = list(set(layout + zero_pages1 + zero_pages2))
            layout_without_zeroes = list(set(layout) - set(zero_pages1) - set(zero_pages2))
            self.run_next_layout(layout_with_all_zeroes)
            self.run_next_layout(layout_without_zeroes)

            zp_df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)
            zp_df['cumsum'] = zp_df['TLB_COVERAGE'].cumsum()
            zp_df = zp_df[zp_df['cumsum'] >= 99]
            zero_pages = zp_df['PAGE_NUMBER'].to_list()
            layout_with_all_zeroes = list(set(layout + zero_pages))
            layout_without_zeroes = list(set(layout) - set(zero_pages))
            self.run_next_layout(layout_with_all_zeroes)
            self.run_next_layout(layout_without_zeroes)

            if self.num_generated_layouts >= max_iterations:
                break

        self.logger.debug(f"exit - shake_runtime={max_iterations}")

    # =================================================================== #
    #   Initial layouts
    # =================================================================== #

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

    def select_fixed_intervals_init_layouts(self, num_layouts=6):
        init_layouts = [self.all_4kb_layout, self.all_2mb_layout, self.all_pebs_pages_layout]
        pebs_df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        step = 100 / num_layouts
        pebs_val = 0
        for i in range(num_layouts):
            pebs_val += step
            layout = self.select_layout_from_pebs_gradually(pebs_val, pebs_df)
            if layout:
                init_layouts.append(layout)
        return init_layouts

    def get_moselect_init_layouts_v4(self):
        pebs_df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        # desired weights for each group layout
        buckets_weights = [56, 28, 14]
        group = self.fillBuckets(pebs_df, buckets_weights, start_from_tail=True, fill_min_buckets_first=False)
        return self.createSubgroups(group)

    def get_moselect_init_layouts_v3(self):
        pebs_df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        # desired weights for each group layout
        buckets_weights = [56, 28, 14]
        group = self.fillBuckets(pebs_df, buckets_weights, start_from_tail=True, fill_min_buckets_first=True)
        return self.createSubgroups(group)

    def get_moselect_init_layouts_v2(self):
        pebs_df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        # desired weights for each group layout
        buckets_weights = [56, 28, 14]
        group = self.fillBuckets(pebs_df, buckets_weights, start_from_tail=False, fill_min_buckets_first=False)
        return self.createSubgroups(group)

    def get_moselect_init_layouts_4_groups(self):
        pebs_df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        # desired weights for each group layout
        buckets_weights = [52, 26, 13, 6.5]
        group = self.fillBuckets(pebs_df, buckets_weights)
        return self.createSubgroups(group)

    def get_moselect_init_layouts(self):
        pebs_df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        # desired weights for each group layout
        buckets_weights = [56, 28, 14]
        group = self.fillBuckets(pebs_df, buckets_weights)
        return self.createSubgroups(group)

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
    #   General utilities
    # =================================================================== #

    def get_expected_pebs_coverage(self, left_offset=0, right_offset=0):
        results_df = self.results_df.sort_values(self.metric_name)
        insert_pos = np.searchsorted(results_df[self.metric_name].values, self.metric_val)
        # Get the surrounding rows
        if insert_pos == 0:
            # If the insertion point is at the beginning, take the first two rows
            left_idx, right_idx = 0, 1
        elif insert_pos == len(results_df):
            # If the insertion point is at the end, take the last two rows
            left_idx, right_idx = -2, -1
        else:
            # Otherwise, take the rows immediately before and after the insertion point
            left_idx, right_idx = insert_pos - 1, insert_pos

        if (left_idx - left_offset) >= 0:
            left_idx -= left_offset
        if (right_idx + right_offset) < len(results_df):
            right_idx += right_offset
        surrounding_rows = results_df.iloc[[left_idx, right_idx]]
        left = results_df.iloc[left_idx]
        right = results_df.iloc[right_idx]
        # prev_row, next_row = surrounding_rows.iloc[0], surrounding_rows.iloc[1]
        # left = prev_row # left has smaller misses TODO: for hits convert the order
        # right = next_row
        expected_pebs = self.calc_pebs_coverage_proportion(right, left)
        return expected_pebs, right, left

    def calc_pebs_coverage_proportion(self, base_layout_r, next_layout_r):
        prev_real = base_layout_r['real_coverage']
        next_real = next_layout_r['real_coverage']
        prev_pebs = base_layout_r['pebs_coverage']
        next_pebs = next_layout_r['pebs_coverage']
        real_coverage = self.metric_coverage
        # Calculate the proportion between the previous and next real values
        if (next_real - prev_real) == 0:
            proportion = 0.5
        else:
            proportion = (real_coverage - prev_real) / (next_real - prev_real)
        # Calculate the expected value using linear interpolation
        step = abs(proportion * (next_pebs - prev_pebs))
        base = min(prev_pebs, next_pebs)
        expected_pebs = base + step
        expected_pebs = min(100, expected_pebs)
        expected_pebs = max(0, expected_pebs)
        # expected_pebs = prev_pebs + abs(proportion) * (next_pebs - prev_pebs)

        return expected_pebs

    def add_pages_to_base_layout(
        self,
        base_layout_pages,
        add_working_set,
        remove_working_set,
        desired_pebs_coverage,
        tail=True,
    ):
        """
        Add pages to base_layout_pages to get a total pebs-coverage as close as
        possible to desired_pebs_coverage. The pages should be added from
        add_working_set. If cannot find pages subset from add_working_set
        that covers desired_pebs_coverage, then try to remove from the
        remove_working_set and retry finding a new pages subset.
        """
        if len(add_working_set) == 0:
            return None, 0

        if remove_working_set is None:
            remove_working_set = []

        # make sure that remove_working_set is a subset of the base-layout pages
        # assert len(set(remove_working_set) - set(base_layout_pages)) == 0
        remove_working_set = list(set(remove_working_set) & set(base_layout_pages))

        # sort remove_working_set pages by coverage ascendingly
        remove_pages_subset = (
            self.pebs_df.query(f"PAGE_NUMBER in {remove_working_set}")
            .sort_values("TLB_COVERAGE")["PAGE_NUMBER"]
            .to_list()
        )
        not_in_pebs = list(set(remove_working_set) - set(remove_pages_subset))
        remove_pages_subset += not_in_pebs

        i = 0
        pages = None
        threshold = 0.5
        tmp_base_layout = base_layout_pages.copy()
        for rp in remove_pages_subset:
            pages, pebs_coverage = self.add_pages_from_working_set(
                tmp_base_layout,
                add_working_set,
                desired_pebs_coverage,
                tail,
                threshold
            )
            if pages is not None and not self.layout_exist(pages):
                break
            # if cannot find pages subset with the expected coverage
            # then remove the page with least coverage and try again
            tmp_base_layout.remove(rp)

        if pages is None or self.layout_exist(pages):
            return None, 0

        num_common_pages = len(set(pages) & set(base_layout_pages))
        num_added_pages = len(pages) - num_common_pages
        num_removed_pages = len(base_layout_pages) - num_common_pages

        self.logger.debug(f"add_pages_to_base_layout:")
        self.logger.debug(f"\t layout has {len(base_layout_pages)} pages")
        self.logger.debug(
            f"\t the new layout has {len(pages)} pages with pebs-coverage: {pebs_coverage}"
        )
        self.logger.debug(f"\t {num_added_pages} pages were added")
        self.logger.debug(f"\t {num_removed_pages} pages were removed")

        return pages, pebs_coverage

    def remove_pages(self, base_layout, working_set, desired_pebs_coverage, tail=True):
        pages, pebs = self.remove_pages_in_order(
            base_layout, working_set, desired_pebs_coverage, tail
        )
        if pages is None or self.layout_exist(pages):
            pages, pebs = self.remove_pages_in_order(
                base_layout, None, desired_pebs_coverage, tail
            )
        if pages is None or self.layout_exist(pages):
            return None, 0
        return pages, pebs

    def layout_exist(self, layout_to_find):
        return not self.isPagesListUnique(layout_to_find, self.layouts)

    def remove_pages_in_order(
        self, base_layout, working_set, desired_pebs_coverage, tail=True
    ):
        self.logger.debug(f"entry - remove_pages_in_order()")

        base_layout_coverage = self.pebsTlbCoverage(base_layout)
        if working_set is None:
            working_set = base_layout
        df = self.pebs_df.query(f"PAGE_NUMBER in {working_set}")
        df = df.sort_values("TLB_COVERAGE", ascending=tail)
        self.logger.debug(
            f"remove_pages: base layout has {len(base_layout)} total pages, and {len(df)} pages in pebs as candidates to be removed"
        )

        removed_pages = []
        total_weight = base_layout_coverage
        epsilon = 0.2
        max_coverage = desired_pebs_coverage
        min_coverage = desired_pebs_coverage - epsilon
        for index, row in df.iterrows():
            page = row["PAGE_NUMBER"]
            weight = row["TLB_COVERAGE"]
            updated_total_weight = total_weight - weight
            if updated_total_weight > min_coverage:
                removed_pages.append(page)
                total_weight = updated_total_weight
            if max_coverage >= total_weight >= min_coverage:
                break
        if len(removed_pages) == 0:
            self.logger.debug(f"exit - iremove_pages_in_order() --> None")
            return None, 0
        new_pages = list(set(base_layout) - set(removed_pages))
        new_pages.sort()
        new_pebs_coverage = self.pebs_df.query(f"PAGE_NUMBER in {new_pages}")[
            "TLB_COVERAGE"
        ].sum()

        self.logger.debug(
            f"removed {len(removed_pages)} pages from total {len(base_layout)} in base layout"
        )
        self.logger.debug(f"new layout coverage: {new_pebs_coverage}")

        self.logger.debug(f"exit - remove_pages_in_order() --> layout: #{len(new_pages)} pages , coverage: {new_pebs_coverage}")
        return new_pages, new_pebs_coverage

    def add_pages_from_working_set(
        self, base_pages, working_set, desired_pebs_coverage, tail=True, epsilon=0.5
    ):
        base_pages_pebs = self.pebsTlbCoverage(base_pages)

        if desired_pebs_coverage < base_pages_pebs:
            self.logger.debug(f"exit - add_pages_from_working_set() --> None ({desired_pebs_coverage} < {base_pages_pebs})")
            return None, 0

        working_set_df = self.pebs_df.query(
            f"PAGE_NUMBER in {working_set} and PAGE_NUMBER not in {base_pages}"
        )
        if len(working_set_df) == 0:
            self.logger.debug(f"there is no more pages in pebs that can be added")
            self.logger.debug(f"exit - add_pages_from_working_set() --> None (len(working_set_df)==0)")
            return None, 0

        candidate_pebs_coverage = working_set_df["TLB_COVERAGE"].sum()
        if candidate_pebs_coverage + base_pages_pebs < desired_pebs_coverage:
            # self.logger.debug('maximal pebs coverage using working-set is less than desired pebs coverage')
            self.logger.debug(f"exit - add_pages_from_working_set() --> None ({candidate_pebs_coverage}+{base_pages_pebs} < {desired_pebs_coverage})")
            return None, 0

        df = working_set_df.sort_values("TLB_COVERAGE", ascending=tail)

        added_pages = []
        min_pebs_coverage = desired_pebs_coverage
        max_pebs_coverage = desired_pebs_coverage + epsilon
        total_weight = base_pages_pebs
        for index, row in df.iterrows():
            page = row["PAGE_NUMBER"]
            weight = row["TLB_COVERAGE"]
            updated_total_weight = total_weight + weight
            if updated_total_weight < max_pebs_coverage:
                added_pages.append(page)
                total_weight = updated_total_weight
            if max_pebs_coverage >= total_weight >= min_pebs_coverage:
                break
        if len(added_pages) == 0:
            self.logger.debug(f"exit - add_pages_from_working_set() --> None (len(added_pages) == 0)")
            return None, 0
        new_pages = base_pages + added_pages
        new_pages.sort()
        new_pebs_coverage = self.pebs_df.query(f"PAGE_NUMBER in {new_pages}")[
            "TLB_COVERAGE"
        ].sum()

        if (
            max_pebs_coverage < new_pebs_coverage
            or new_pebs_coverage < min_pebs_coverage
        ):
            self.logger.debug(
                f"Could not find pages subset with a coverage of {desired_pebs_coverage}"
            )
            self.logger.debug(f"\t pages subset that was found has:")
            self.logger.debug(
                f"\t\t added pages: {len(added_pages)} to {len(base_pages)} pages of the base layout"
            )
            self.logger.debug(f"\t\t pebs coverage: {new_pebs_coverage}")
            self.logger.debug(f"exit - add_pages_from_working_set() --> None")
            return None, 0

        self.logger.debug(
            f"Found pages subset with a coverage of {desired_pebs_coverage}"
        )
        self.logger.debug(f"\t pages subset that was found has:")
        self.logger.debug(
            f"\t\t added pages: {len(added_pages)} to {len(base_pages)} pages of the base layout ==> total pages: {len(new_pages)}"
        )

        self.logger.debug(f"exit - add_pages_from_working_set() --> layout: #{len(new_pages)} pages , coverage: {new_pebs_coverage}")
        return new_pages, new_pebs_coverage

    # =================================================================== #
    #   Select and run layout that hits the given input point
    # =================================================================== #

    def is_result_within_target_range(self, layout_res):
        if layout_res is None:
            return False
        epsilon = self.range_epsilon * 100
        min_coverage = self.metric_coverage - epsilon
        max_coverage = self.metric_coverage + epsilon
        layout_coverage = self.realMetricCoverage(layout_res, self.metric_name)
        return min_coverage <= layout_coverage <= max_coverage

    def get_working_sets(self, left_layout, right_layout):
        right_set = set(right_layout)
        left_set = set(left_layout)
        pebs_set = set(self.pebs_df["PAGE_NUMBER"].to_list())
        all_set = set(left_set | right_set | pebs_set)
        union_set = set(left_set | right_set)

        only_in_right = list(set(right_set - left_set))
        only_in_left = list(set(left_set - right_set))
        intersection = list(set(left_set & right_set))
        out_union = list(set(all_set - union_set))
        all_pages = list(all_set)
        return only_in_right, only_in_left, intersection, out_union, all_pages

    def select_layout_from_endpoints(self, left, right):
        self.logger.debug(f"entry - select_layout_from_endpoints()")

        alpha, beta, gamma, delta, U = self.get_working_sets(left, right)

        layout, pebs = self.add_pages_from_working_set(right, beta, self.metric_pebs_coverage)
        if layout is not None and not self.layout_exist(layout):
            self.logger.debug(f"exit - select_layout_from_endpoints() --> layout: #{len(layout)} pages")
            return layout

        layout, pebs = self.remove_pages(left, beta, self.metric_pebs_coverage)
        if layout is not None and not self.layout_exist(layout):
            self.logger.debug(f"exit - select_layout_from_endpoints() --> layout: #{len(layout)} pages")
            return layout

        layout, pebs = self.add_pages_from_working_set(right, delta, self.metric_pebs_coverage)
        if layout is not None and not self.layout_exist(layout):
            self.logger.debug(f"exit - select_layout_from_endpoints() --> layout: #{len(layout)} pages")
            return layout

        self.logger.debug(f"exit - select_layout_from_endpoints() --> None")
        return None

    def run_layout_from_virtual_surroundings(self, left, right):
        self.logger.debug(f"entry - run_layout_from_virtual_surroundings")

        next_layout = self.select_layout_from_endpoints(left, right)
        if next_layout is None:
            self.logger.debug(f"exit - run_layout_from_virtual_surroundings --> None")
            return None, None

        self.last_layout_result = self.run_next_layout(next_layout)

        self.logger.debug(f"exit - run_layout_from_virtual_surroundings --> layout: #{len(next_layout)} pages")
        return next_layout, self.last_layout_result

    def find_virtual_surrounding_initial_layouts(self, initial_layouts):
        self.logger.debug(f"entry - find_virtual_surrounding_initial_layouts")
        # sort layouts by their PEBS
        sorted_initial_layouts = sorted(
            initial_layouts, key=lambda layout: self.pebsTlbCoverage(layout)
        )
        right_i = 0
        right = sorted_initial_layouts[right_i]
        left_i = 0
        left = sorted_initial_layouts[left_i]
        for i in range(len(sorted_initial_layouts)):
            right_i = left_i
            right = left
            left_i = i
            left = sorted_initial_layouts[i]
            left_pebs = self.pebsTlbCoverage(left)
            right_pebs = self.pebsTlbCoverage(right)
            if right_pebs <= self.metric_coverage <= left_pebs:
                break
        return left, right

    def find_surrounding_initial_layouts(self, initial_layouts):
        def search_for_pair(candidate_results, target_coverage):
            # Sort candidate results by real coverage each time a new layout is added
            sorted_by_real = sorted(candidate_results, key=lambda x: x['real_coverage'])
            for i in range(len(sorted_by_real) - 1):
                lo = sorted_by_real[i]
                hi = sorted_by_real[i + 1]
                if lo['real_coverage'] <= target_coverage <= hi['real_coverage']:
                    return True, lo['layout'], hi['layout']
                if hi['real_coverage'] <= target_coverage <= lo['real_coverage']:
                    return True, hi['layout'], lo['layout']
            return False, None, None

        self.logger.debug("Starting to find surrounding layouts.")

        # Step 1: Sort layouts by their expected coverage
        sorted_layouts = sorted(initial_layouts, key=lambda layout: self.pebsTlbCoverage(layout))

        # Step 2: Identify candidate pairs
        for i in range(len(sorted_layouts) - 1):
            lo_idx, hi_idx = i, i+1
            lo = sorted_layouts[lo_idx]
            hi = sorted_layouts[hi_idx]
            if self.pebsTlbCoverage(lo) <= self.metric_coverage < self.pebsTlbCoverage(hi):
                break

        candidate_results = []
        # Step 3: Verify candidate pairs with real coverages
        for i in range(len(sorted_layouts) - 1):
            lo_real_coverage = self.realMetricCoverage(self.run_next_layout(lo))
            candidate_results.append({'layout': lo, 'real_coverage': lo_real_coverage})
            found, candidate_lo, candidate_hi = search_for_pair(candidate_results, self.metric_coverage)
            if found:
                return candidate_hi, candidate_lo

            hi_real_coverage = self.realMetricCoverage(self.run_next_layout(hi))
            candidate_results.append({'layout': hi, 'real_coverage': hi_real_coverage})
            found, candidate_lo, candidate_hi = search_for_pair(candidate_results, self.metric_coverage)
            if found:
                return candidate_hi, candidate_lo
            lo_idx = max(0, lo_idx - 1)
            hi_idx = min(hi_idx + 1, len(sorted_layouts) - 1)
            lo = sorted_layouts[lo_idx]
            hi = sorted_layouts[hi_idx]

        assert True

    def add_pages_virtually_to_find_desired_layout(self, base_layout, add_working_set):
        self.logger.debug(f"entry - add_pages_virtually_to_find_desired_layout")

        expected_pebs = self.metric_pebs_coverage
        base_real_coverage = None
        while True:
            layout, pebs = self.add_pages_to_base_layout(base_layout, add_working_set, None, expected_pebs)
            if layout is None or self.layout_exist(layout):
                self.logger.debug(f"exit - add_pages_virtually_to_find_desired_layout --> None")
                return None, None
            layout_result = self.run_next_layout(layout)
            if self.is_result_within_target_range(layout_result):
                self.logger.debug(f"exit - add_pages_virtually_to_find_desired_layout() --> layout:#{len(layout)} pages")
                return layout, layout_result
            base_pebs = self.pebsTlbCoverage(base_layout)
            last_real_coverage = self.realMetricCoverage(layout_result)
            if last_real_coverage < self.metric_coverage:
                if base_real_coverage is None or last_real_coverage > base_real_coverage:
                    base_layout = layout
                    base_real_coverage = last_real_coverage
                    expected_pebs += (self.metric_coverage - last_real_coverage)
            else:
                expected_pebs -= (last_real_coverage - self.metric_coverage)
            expected_pebs = min(100, expected_pebs)
            expected_pebs = max(0, expected_pebs)

    def add_pages_to_find_desired_layout(self, next_layout_r, base_layout_r, add_working_set, remove_working_set):
        self.logger.debug(f"entry - add_pages_to_find_desired_layout()")

        assert self.realMetricCoverage(next_layout_r) > self.realMetricCoverage(base_layout_r)
        base_layout = base_layout_r['hugepages']
        expected_pebs = self.calc_pebs_coverage_proportion(base_layout_r, next_layout_r)
        while True:
            layout, pebs = self.add_pages_to_base_layout(base_layout, add_working_set, remove_working_set, expected_pebs)
            if layout is None or self.layout_exist(layout):
                self.logger.debug(f"exit - add_pages_to_find_desired_layout() --> None")
                return None, None
            layout_result = self.run_next_layout(layout)
            if self.is_result_within_target_range(layout_result):
                self.logger.debug(f"exit - add_pages_to_find_desired_layout() --> layout:#{len(layout)} pages")
                return layout, layout_result
            base_pebs = self.pebsTlbCoverage(base_layout)
            last_pebs_step = expected_pebs - base_pebs
            base_real_coverage = self.realMetricCoverage(base_layout_r)
            next_real_coverage = self.realMetricCoverage(next_layout_r)
            last_real_coverage = self.realMetricCoverage(layout_result)
            if (last_real_coverage - base_real_coverage) < last_pebs_step:
                expected_pebs = (last_pebs_step * 1.5) + expected_pebs
                expected_pebs = min(100, expected_pebs)
                continue
            if last_real_coverage > next_real_coverage:
                expected_pebs = expected_pebs - (last_real_coverage - next_real_coverage) * 1.5
                expected_pebs = max(0, expected_pebs)
                continue
            if next_real_coverage >= last_real_coverage >= base_real_coverage:
                base_layout_r = layout_result
            expected_pebs = self.calc_pebs_coverage_proportion(base_layout_r, next_layout_r)

    def remove_pages_to_find_desired_layout(self, next_layout_r, base_layout_r, remove_working_set):
        self.logger.debug(f"entry - remove_pages_to_find_desired_layout()")

        assert self.realMetricCoverage(next_layout_r) < self.realMetricCoverage(base_layout_r)
        base_layout = base_layout_r['hugepages']
        expected_pebs = self.calc_pebs_coverage_proportion(base_layout_r, next_layout_r)
        while True:
            layout, pebs = self.remove_pages(base_layout, remove_working_set, expected_pebs)
            if layout is None or self.layout_exist(layout):
                self.logger.debug(f"exit - remove_pages_to_find_desired_layout() --> None")
                return None, None
            layout_result = self.run_next_layout(layout)
            if self.is_result_within_target_range(layout_result):
                self.logger.debug(f"exit - remove_pages_to_find_desired_layout() --> layout:#{len(layout)} pages")
                return layout, layout_result
            base_pebs = self.pebsTlbCoverage(base_layout)
            last_pebs_step = base_pebs - expected_pebs
            base_real_coverage = self.realMetricCoverage(base_layout_r)
            next_real_coverage = self.realMetricCoverage(next_layout_r)
            last_real_coverage = self.realMetricCoverage(layout_result)
            if (base_real_coverage - last_real_coverage) < last_pebs_step:
                expected_pebs = expected_pebs - (last_pebs_step * 1.5)
                expected_pebs = max(0, expected_pebs)
                continue
            if last_real_coverage < next_real_coverage:
                expected_pebs = expected_pebs + (next_real_coverage - last_real_coverage) * 1.5
                expected_pebs = min(100, expected_pebs)
                continue
            if next_real_coverage <= last_real_coverage <= base_real_coverage:
                base_layout_r = layout_result
            expected_pebs = self.calc_pebs_coverage_proportion(base_layout_r, next_layout_r)

    def find_desired_layout(self, initial_layouts, max_iterations=10):
        self.logger.debug(f"entry - find_desired_layout()")

        tested_layouts = []

        left, right = self.find_surrounding_initial_layouts(initial_layouts)
        assert left is not None
        assert right is not None

        right_r = self.run_next_layout(right)
        if self.is_result_within_target_range(right_r):
            return right, right_r
        tested_layouts.append(right_r)

        left_r = self.run_next_layout(left)
        if self.is_result_within_target_range(left_r):
            return left, left_r
        tested_layouts.append(left_r)

        if self.realMetricCoverage(right_r) < self.realMetricCoverage(left_r):
            base_layout = right
            base_layout_r = right_r
            next_layout = left
            next_layout_r = left_r
        else:
            base_layout = left
            base_layout_r = left_r
            next_layout = right
            next_layout_r = right_r

        layout_result = None
        while not self.is_result_within_target_range(layout_result):
            alpha, beta, gamma, delta, U = self.get_working_sets(next_layout, base_layout)

            layout, layout_result = self.add_pages_to_find_desired_layout(next_layout_r, base_layout_r, beta, None)
            if layout_result is None:
                layout, layout_result = self.add_pages_to_find_desired_layout(next_layout_r, base_layout_r, beta, alpha)
            if layout_result is None:
                layout, layout_result = self.add_pages_virtually_to_find_desired_layout(gamma, U)
            if layout_result is None:
                layout, layout_result = self.add_pages_to_find_desired_layout(next_layout_r, base_layout_r, delta, None)
            if layout_result is None:
                layout, layout_result = self.remove_pages_to_find_desired_layout(base_layout_r, next_layout_r, beta)

            if layout_result is not None:
                tested_layouts.append(layout_result)
            # limit the convergence processes to up to max_iterations trials
            if len(tested_layouts) > max_iterations:
                # find closest layout to desired coverage
                min_diff = 100
                closest_layout_r = None
                for layout_result in tested_layouts:
                    assert layout_result is not None
                    layout_coverage = self.realMetricCoverage(layout_result, self.metric_name)
                    diff = abs(layout_coverage - self.metric_coverage)
                    if diff < min_diff:
                        min_diff = diff
                        closest_layout_r = layout_result
                layout = closest_layout_r['hugepages']
                return layout, closest_layout_r

            last_base_layout = base_layout
            last_next_layout = next_layout

            found = False
            for i in range(len(self.results_df)):
                for j in range(len(self.results_df)):
                    _, base_layout_r, next_layout_r = self.get_expected_pebs_coverage(left_offset=i, right_offset=j)
                    base_layout = base_layout_r['hugepages']
                    next_layout = next_layout_r['hugepages']
                    if set(base_layout) != set(last_base_layout) or set(next_layout) != set(last_next_layout):
                        found = True
                        break
                if found:
                    break

        self.logger.debug(f"exit - find_desired_layout --> layout: #{len(layout)} pages")
        return layout, layout_result

    def full_run(self):
        self.log_metadata()

        if self.debug:
            breakpoint()

        num_layouts_round1 = self.num_layouts // 2
        num_layouts_round2 = self.num_layouts - num_layouts_round1

        self.logger.info("=====================================================")
        self.logger.info(f"Starting converging to required point")
        self.logger.info("=====================================================")
        initial_layouts = self.select_uni_dist_init_layouts()
        layout, layout_result = self.find_desired_layout(initial_layouts)
        self.logger.info("=====================================================")
        self.logger.info(f"Finished converging to required point")
        self.logger.info(f"Starting shaking runtime")
        self.logger.info("=====================================================")
        self.shake_runtime(layout, num_layouts_round1)

        self.logger.info("=====================================================")
        self.logger.info(f"Starting converging to required point")
        self.logger.info("=====================================================")
        initial_layouts = self.select_fixed_intervals_init_layouts()
        layout, layout_result = self.find_desired_layout(initial_layouts)
        self.logger.info("=====================================================")
        self.logger.info(f"Finished converging to required point")
        self.logger.info(f"Starting shaking runtime")
        self.logger.info("=====================================================")
        self.shake_runtime(layout, num_layouts_round2)

        if self.debug:
            breakpoint()

        self.logger.info(
            "================================================================="
        )
        self.logger.info(f"Finished running MosRange process for:\n{self.exp_root_dir}")
        self.logger.info(
            "================================================================="
        )
        # MosrangeSelector.pause()

    def quick_run(self):
        self.log_metadata()

        if self.debug:
            breakpoint()

        self.logger.info("=====================================================")
        self.logger.info(f"Running first layout")
        self.logger.info("=====================================================")
        initial_layouts = self.select_uni_dist_init_layouts()
        left, right = self.find_virtual_surrounding_initial_layouts(initial_layouts)
        layout = self.select_layout_from_endpoints(left, right)
        tail_pages = self.get_tail_pages(total_threshold=1)
        first_layout = list(set(layout) | set(tail_pages))
        layout_result = self.run_next_layout(first_layout)
        # update metric_coverage to the one got by the executed layout to save convergence time
        real_coverage = self.realMetricCoverage(layout_result)
        self.metric_coverage = real_coverage
        self.metric_val = None
        self.update_metric_values()

        self.logger.info("=====================================================")
        self.logger.info(f"Starting converging to required point")
        self.logger.info("=====================================================")
        initial_layouts = self.select_uni_dist_init_layouts_v2()
        layout, layout_result = self.find_desired_layout(initial_layouts)
        self.logger.info("=====================================================")
        self.logger.info(f"Finished converging to required point")
        self.logger.info(f"Starting shaking runtime")
        self.logger.info("=====================================================")
        rem_layouts = self.num_layouts - self.last_layout_num
        num_layouts = max(rem_layouts, 5)
        self.shake_runtime(layout, num_layouts)

        if self.debug:
            breakpoint()

        self.logger.info(
            "================================================================="
        )
        self.logger.info(f"Finished running MosRange process for:\n{self.exp_root_dir}")
        self.logger.info(
            "================================================================="
        )
        # MosrangeSelector.pause()

    def log(self, msg):
        with open(self.log_file_path, 'a+') as f:
            f.write(msg)
            f.write('\n')

    def run_with_custom_init_layouts(self, layout_name, initial_layouts, shake_budget=5, first_group=False):
        if first_group:
            self.log(f"points_group,layout_name")

        self.logger.info("=====================================================")
        self.logger.info(f"==> {layout_name}: Starting converging")
        self.log(f"{layout_name}_start_converge,layout{self.last_layout_num+1}")

        if first_group:
            left, right = self.find_virtual_surrounding_initial_layouts(initial_layouts)
            layout = self.select_layout_from_endpoints(left, right)
            layout_result = self.run_next_layout(layout)
        else:
            layout, layout_result = self.find_desired_layout(initial_layouts)
        self.log(f"{layout_name},layout{self.last_layout_num}")
        self.logger.info(f"<== {layout_name}: Finished converging")
        self.logger.info("=====================================================")

        self.logger.info("=====================================================")
        self.logger.info(f"Running {layout_name} with zero pages")
        self.logger.info("=====================================================")
        tail_pages = self.get_tail_pages(total_threshold=1)
        layout_zeroes = list(set(layout) | set(tail_pages))
        layout_result = self.run_next_layout(layout_zeroes)
        self.log(f"{layout_name}_with_zeroes,layout{self.last_layout_num}")

        if first_group:
            # update metric_coverage to the one got by the executed layout to save convergence time
            real_coverage = self.realMetricCoverage(layout_result)
            self.metric_coverage = real_coverage
            self.metric_val = None
            self.update_metric_values()

        self.logger.info("=====================================================")
        self.logger.info(f"==> {layout_name}: Starting shaking runtime")
        self.log(f"{layout_name}_start_shake_runtime,layout{self.last_layout_num+1}")
        self.shake_runtime(layout, shake_budget)
        self.log(f"{layout_name}_end_shake_runtime,layout{self.last_layout_num}")
        self.logger.info(f"<== {layout_name}: Finished shaking runtime")
        self.logger.info("=====================================================")

    def run_with_different_init_layouts(self):
        self.log_metadata()

        if self.debug:
            breakpoint()

        shake_budget = max(self.num_layouts//6, 5)
        self.logger.info(f"==> Shaking runtime budget: {shake_budget} <==")

        initial_layouts = self.get_moselect_init_layouts()
        self.run_with_custom_init_layouts('LayoutA', initial_layouts, shake_budget, True)

        initial_layouts = self.get_moselect_init_layouts_4_groups()
        self.run_with_custom_init_layouts('LayoutB', initial_layouts, shake_budget)

        initial_layouts = self.get_moselect_init_layouts_v2()
        self.run_with_custom_init_layouts('LayoutC', initial_layouts, shake_budget)

        initial_layouts = self.get_moselect_init_layouts_v3()
        self.run_with_custom_init_layouts('LayoutD', initial_layouts, shake_budget)

        initial_layouts = self.get_moselect_init_layouts_v4()
        self.run_with_custom_init_layouts('LayoutE', initial_layouts, shake_budget)

        if self.debug:
            breakpoint()

        self.logger.info("=================================================================")
        self.logger.info(f"Finished running MosRange process for:\n{self.exp_root_dir}")
        self.logger.info("=================================================================")
        # MosrangeSelector.pause()

    def run(self):
        self.run_with_different_init_layouts()

