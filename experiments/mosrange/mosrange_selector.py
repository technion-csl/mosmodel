#!/usr/bin/env python3
# import cProfile
import itertools
import os, sys
import logging
import random
import numpy as np
import pandas as pd
from pathlib import Path
import string
import shutil
from datetime import datetime

curr_file_dir = os.path.dirname(os.path.abspath(__file__))
experiments_root_dir = os.path.join(curr_file_dir, "..")
sys.path.append(experiments_root_dir)
from Utils.utils import Utils
from Utils.selector_utils import Selector


class MosrangeSelector(Selector):
    MAX_SEARCH_LAST_RUN_LAYOUT_COUNTER = 1024
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
        self.layout_group = 0
        self.layout_group_name = None

        current_file_path = Path(__file__).resolve()
        self.current_directory = current_file_path.parent
        self.log_file_path = self.current_directory / "log.csv"

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
            if iterations > max_iterations:
                return

            right_in_range = evaluate_subset(right_subset, zero_pages)
            if right_in_range:
                # If the left subset is under the threshold, add it to the zero_pages
                zero_pages.extend(right_subset)
            if iterations > max_iterations:
                return
            if not right_in_range:
                search(mid, right)
            if iterations > max_iterations:
                return

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

        self.reset_budget(max_iterations)
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

            if self.consumed_budget():
                self.disable_budget()
                break

        self.logger.debug(f"exit - shake_runtime={max_iterations}")

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
        left = results_df.iloc[left_idx]
        right = results_df.iloc[right_idx]
        # prev_row, next_row = surrounding_rows.iloc[0], surrounding_rows.iloc[1]
        # left = prev_row # left has smaller misses TODO: for hits convert the order
        # right = next_row
        expected_pebs = self.calc_pebs_coverage_proportion(right, left)
        return expected_pebs, right, left

    def calc_pebs_coverage_proportion(self, base_layout_r, next_layout_r):
        base_real = self.realMetricCoverage(base_layout_r)
        next_real = self.realMetricCoverage(next_layout_r)
        base_layout = base_layout_r['hugepages']
        next_layout = next_layout_r['hugepages']
        base_pebs = self.pebsTlbCoverage(base_layout)
        next_pebs = self.pebsTlbCoverage(next_layout)
        real_coverage = self.metric_coverage
        min_real = min(base_real, next_real)
        max_real = max(base_real, next_real)
        min_pebs = min(base_pebs, next_pebs)
        max_pebs = max(base_pebs, next_pebs)
        assert min_real <= real_coverage <= max_real

        # handle corner case to prevent division by zero
        if (max_real - min_real) == 0:
            return min_pebs + (real_coverage - min_real)

        # case 1) handle adding pages to base layout
        if base_pebs <= next_pebs:
            # Calculate the proportion between the previous and next real values
            proportion = (real_coverage - min_real) / (max_real - min_real)
            # Calculate the expected value using linear interpolation
            step = abs(proportion * (max_pebs - min_pebs))
            expected_pebs = min_pebs + step
        # case 2) handle removing pages from next layout
        else: # base_pebs > next_pebs:
            # Calculate the proportion between the previous and next real values
            proportion = (max_real - real_coverage) / (max_real - min_real)
            # Calculate the expected value using linear interpolation
            step = abs(proportion * (max_pebs - min_pebs))
            expected_pebs = max_pebs - step
        expected_pebs = min(100, expected_pebs)
        expected_pebs = max(0, expected_pebs)
        # expected_pebs = base_pebs + abs(proportion) * (next_pebs - base_pebs)

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

        if remove_pages_subset is None or not remove_pages_subset or len(remove_pages_subset) == 0:
            pages, pebs_coverage = self.add_pages_from_working_set(
                base_layout_pages, add_working_set, desired_pebs_coverage, tail, threshold)
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

        num_layouts_before = self.last_layout_num

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
                num_layouts_after = self.last_layout_num
                num_explored_layouts = num_layouts_after - num_layouts_before
                self.logger.info(f"found surrounding initial layouts:")
                self.logger.info(f"\tsizes: {len(candidate_hi)} , {len(candidate_lo)}")
                self.logger.info(f"\ttotal number of explored and run layouts: {num_explored_layouts}")
                return candidate_hi, candidate_lo

            hi_real_coverage = self.realMetricCoverage(self.run_next_layout(hi))
            candidate_results.append({'layout': hi, 'real_coverage': hi_real_coverage})
            found, candidate_lo, candidate_hi = search_for_pair(candidate_results, self.metric_coverage)
            if found:
                num_layouts_after = self.last_layout_num
                num_explored_layouts = num_layouts_after - num_layouts_before
                self.logger.info(f"found surrounding initial layouts:")
                self.logger.info(f"\tsizes: {len(candidate_hi)} , {len(candidate_lo)}")
                self.logger.info(f"\ttotal number of explored and run layouts: {num_explored_layouts}")
                return candidate_hi, candidate_lo
            lo_idx = max(0, lo_idx - 1)
            hi_idx = min(hi_idx + 1, len(sorted_layouts) - 1)
            lo = sorted_layouts[lo_idx]
            hi = sorted_layouts[hi_idx]

        assert True

    def add_pages_virtually_to_find_desired_layout(self, base_layout, add_working_set):
        self.logger.debug(f"entry - add_pages_virtually_to_find_desired_layout")

        num_layouts_before = self.last_layout_num
        expected_pebs = self.metric_pebs_coverage
        base_real_coverage = None
        while True:
            if self.consumed_budget():
                return None, None

            self.last_run_layout_counter += 1
            if self.last_run_layout_counter > MosrangeSelector.MAX_SEARCH_LAST_RUN_LAYOUT_COUNTER:
                return None, None

            layout, pebs = self.add_pages_to_base_layout(base_layout, add_working_set, None, expected_pebs)
            if layout is None or self.layout_exist(layout):
                self.logger.debug(f"exit - add_pages_virtually_to_find_desired_layout --> None")
                return None, None

            layout_result = self.run_next_layout(layout)
            if self.is_result_within_target_range(layout_result):
                self.logger.debug(f"exit - add_pages_virtually_to_find_desired_layout() --> layout:#{len(layout)} pages")
                return layout, layout_result

            num_layouts_after = self.last_layout_num
            num_explored_layouts = num_layouts_after - num_layouts_before
            self.logger.info(f"add_pages_virtually_to_find_desired_layout: explored and run {num_explored_layouts} layouts so far")

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

        num_layouts_before = self.last_layout_num

        assert self.realMetricCoverage(next_layout_r) >= self.metric_coverage >= self.realMetricCoverage(base_layout_r)
        base_layout = base_layout_r['hugepages']
        expected_pebs = self.calc_pebs_coverage_proportion(base_layout_r, next_layout_r)
        prev_real = None
        while True:
            if self.consumed_budget():
                return None, None

            self.last_run_layout_counter += 1
            if self.last_run_layout_counter > MosrangeSelector.MAX_SEARCH_LAST_RUN_LAYOUT_COUNTER:
                return None, None

            layout, pebs = self.add_pages_to_base_layout(base_layout, add_working_set, remove_working_set, expected_pebs)
            if layout is None or self.layout_exist(layout):
                self.logger.debug(f"exit - add_pages_to_find_desired_layout() --> None")
                return None, None

            layout_result = self.run_next_layout(layout)
            if self.is_result_within_target_range(layout_result):
                self.logger.debug(f"exit - add_pages_to_find_desired_layout() --> layout:#{len(layout)} pages")
                return layout, layout_result

            num_layouts_after = self.last_layout_num
            num_explored_layouts = num_layouts_after - num_layouts_before
            self.logger.info(f"add_pages_to_find_desired_layout: explored and run {num_explored_layouts} layouts so far")

            base_pebs = self.pebsTlbCoverage(base_layout)
            last_pebs_step = expected_pebs - base_pebs
            base_real_coverage = self.realMetricCoverage(base_layout_r)
            if prev_real is None:
                prev_real = base_real_coverage
            next_real_coverage = self.realMetricCoverage(next_layout_r)
            last_real_coverage = self.realMetricCoverage(layout_result)
            expected_increment = self.metric_coverage - base_real_coverage
            epsilon = max(5, expected_increment / 2)

            if self.metric_coverage > last_real_coverage:
                # case 1) minor real increment or decrement
                if abs(last_real_coverage - prev_real) <= epsilon:
                    expected_pebs = base_pebs + (last_pebs_step * 1.5)
                    expected_pebs = min(100, expected_pebs)
                    continue
            else:
                # case 2) major real increment
                if abs(last_real_coverage - prev_real) > epsilon:
                    expected_pebs = base_pebs + (last_pebs_step * 0.7)
                    expected_pebs = min(100, expected_pebs)
                    continue
            prev_real = last_real_coverage
            '''
            if (last_real_coverage - base_real_coverage) < last_pebs_step:
                expected_pebs = (last_pebs_step * 1.5) + expected_pebs
                expected_pebs = min(100, expected_pebs)
                continue
            if last_real_coverage > next_real_coverage:
                expected_pebs = expected_pebs - (last_real_coverage - next_real_coverage) * 1.5
                expected_pebs = max(0, expected_pebs)
                continue
            '''
            if self.metric_coverage >= last_real_coverage >= base_real_coverage:
                base_layout_r = layout_result
            expected_pebs = self.calc_pebs_coverage_proportion(base_layout_r, next_layout_r)

    def remove_pages_to_find_desired_layout(self, next_layout_r, base_layout_r, remove_working_set):
        self.logger.debug(f"entry - remove_pages_to_find_desired_layout()")

        num_layouts_before = self.last_layout_num

        assert self.realMetricCoverage(next_layout_r) < self.realMetricCoverage(base_layout_r)
        base_layout = base_layout_r['hugepages']
        expected_pebs = self.calc_pebs_coverage_proportion(base_layout_r, next_layout_r)
        while True:
            if self.consumed_budget():
                return None, None

            self.last_run_layout_counter += 1
            if self.last_run_layout_counter > MosrangeSelector.MAX_SEARCH_LAST_RUN_LAYOUT_COUNTER:
                return None, None

            layout, pebs = self.remove_pages(base_layout, remove_working_set, expected_pebs)
            if layout is None or self.layout_exist(layout):
                self.logger.debug(f"exit - remove_pages_to_find_desired_layout() --> None")
                return None, None

            layout_result = self.run_next_layout(layout)
            if self.is_result_within_target_range(layout_result):
                self.logger.debug(f"exit - remove_pages_to_find_desired_layout() --> layout:#{len(layout)} pages")
                return layout, layout_result

            num_layouts_after = self.last_layout_num
            num_explored_layouts = num_layouts_after - num_layouts_before
            self.logger.info(f"remove_pages_to_find_desired_layout: explored and run {num_explored_layouts} layouts so far")

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
            if self.metric_coverage <= last_real_coverage <= base_real_coverage:
                base_layout_r = layout_result
            expected_pebs = self.calc_pebs_coverage_proportion(base_layout_r, next_layout_r)

    def find_desired_layout(self, initial_layouts, max_iterations=20):
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

        self.reset_budget(max_iterations)
        layout_result = None
        while not self.is_result_within_target_range(layout_result):
            self.last_run_layout_counter += 1
            if self.last_run_layout_counter > (max_iterations * MosrangeSelector.MAX_SEARCH_LAST_RUN_LAYOUT_COUNTER):
                return None, None

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

            if self.consumed_budget():
                self.disable_budget()
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
            # if not found:
            #     break

        self.logger.debug(f"exit - find_desired_layout --> layout: #{len(layout)} pages")
        return layout, layout_result

    def find_closest_layout_to_required_coverage(self):
        results_df, _ = self.collect_results(False)
        results_df = results_df.query(f'layout in {self.phase_layout_names}')
        min_diff = 100
        closest_layout_r = None
        for index, layout_result in results_df.iterrows():
            assert layout_result is not None
            layout_coverage = self.realMetricCoverage(layout_result)
            diff = abs(layout_coverage - self.metric_coverage)
            if diff < min_diff:
                min_diff = diff
                closest_layout_r = layout_result
        layout = closest_layout_r['hugepages']
        return layout, closest_layout_r

    def backup_and_truncate_log(self, file_path):
        # Check if the file exists
        if not os.path.exists(file_path):
            return

        # Append a date-time suffix to the filename for the backup
        date_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{file_path}_{date_suffix}"

        # Copy the original file to the backup file
        shutil.copy(file_path, backup_file)

        # Truncate the original file
        with open(file_path, 'w') as file:
            pass  # Opening in 'w' mode truncates the file

    def log(self, key, val, truncate=False):
        if truncate:
            # self.backup_and_truncate_log(self.log_file_path)
            self.logger.warning('log was called with truncate...')
            assert False

        msg = f"{key},{val}"
        with open(self.log_file_path, 'a') as f:
            f.write(msg)
            f.write('\n')


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

    # =================================================================== #
    #   Multi-selection using different initial layouts algorithms
    # =================================================================== #

    def write_init_group_results(self, group_name, group_details, phase):
        results_df, _ = self.collect_results(False)
        results_df = results_df.query(f'layout in {self.phase_layout_names}')
        results_df['group_details'] = group_details
        results_df['phase'] = phase

        init_group_results_file = self.current_directory / f'{group_name}_results.csv'
        if init_group_results_file.exists():
            group_df = pd.read_csv(init_group_results_file)
            group_df = pd.concat([group_df, results_df], ignore_index=True)
        else:
            group_df = results_df
        group_df.to_csv(init_group_results_file, index=False)

        all_init_results_file = self.current_directory / 'all_init_groups_results.csv'
        if all_init_results_file.exists():
            all_groups_df = pd.read_csv(all_init_results_file)
            all_groups_df = pd.concat([all_groups_df, results_df], ignore_index=True)
        else:
            all_groups_df = results_df
        all_groups_df.to_csv(all_init_results_file, index=False)

        self.reset_phase_layout_names()

    def run_with_custom_init_layouts(
        self,
        initial_layouts,
        shake_budget=5,
        group_details="N/A",
        first_group=False,
        skip_first_group_convergence=False,
        enforce_convergence=True,
        layout_group_name=None
        ):
        if layout_group_name is not None:
            self.layout_group_name = layout_group_name
        else:
            self.layout_group_name = f'Layout{string.ascii_uppercase[self.layout_group]}'
        self.layout_group += 1

        self.reset_phase_layout_names()
        if first_group:
            self.log(f"points_group", "layout_name")
        self.log(f"{self.layout_group_name}_group_details", group_details)

        start_layout_num = self.last_layout_num+1
        start_layout_name = f"layout{start_layout_num}"
        self.log(f"{self.layout_group_name}_start_converge" ,start_layout_name)
        self.logger.info("=====================================================")
        self.logger.info(f"==> {self.layout_group_name}: Starting converging")

        if first_group and skip_first_group_convergence:
            left, right = self.find_virtual_surrounding_initial_layouts(initial_layouts)
            layout = self.select_layout_from_endpoints(left, right)
            layout_result = self.run_next_layout(layout)
        else:
            max_converge_budget = 30
            converge_budget = 10
            if first_group:
                converge_budget = 20
            while True:
                layout, layout_result = self.find_desired_layout(initial_layouts, max_iterations=converge_budget)
                if self.is_result_within_target_range(layout_result):
                    break
                converge_budget += 10
                if converge_budget > max_converge_budget:
                    self.logger.warning(f"consumed all given budget ({max_converge_budget} in total) but still could not converge!")
                    break
                self.logger.warning(f"could not converge yet, increasing the budget by 10 (to: {converge_budget})")


        self.logger.info(f"<== {self.layout_group_name}: Finished converging")
        self.logger.info("=====================================================")
        end_layout_name = f"layout{self.last_layout_num}"
        self.log(self.layout_group_name, end_layout_name)

        # update required coverage in case we failed to converge in the first group
        if first_group and not enforce_convergence:
            layout, layout_result = self.find_closest_layout_to_required_coverage()
            if not self.is_result_within_target_range(layout_result):
                # update metric_coverage to the one got by the executed layout to save convergence time
                prev_coverage = self.metric_coverage
                prev_val = self.metric_val
                real_coverage = self.realMetricCoverage(layout_result)
                self.metric_coverage = real_coverage
                self.metric_val = None
                self.update_metric_values()
                self.logger.info(f">>> Updating required coverage of {self.metric_name}: <<<")
                self.logger.info(f"\t>>> from: [{Utils.format_large_number(prev_val)} , {prev_coverage}%] <<<")
                self.logger.info(f"\t>>> to  : [{Utils.format_large_number(self.metric_val)} , {self.metric_coverage}%] <<<")
        elif enforce_convergence:
            layout, layout_result = self.find_closest_layout_to_required_coverage()
            # assert self.is_result_within_target_range(layout_result)
        closest_layout_name = layout_result['layout']
        self.log(f"{self.layout_group_name}_converged", closest_layout_name)

        self.write_init_group_results(self.layout_group_name, group_details, f'converging_to_{round(self.metric_coverage, 1)}_{self.metric_name}')
        self.phase_layout_names = [closest_layout_name]
        self.write_init_group_results(self.layout_group_name, group_details, 'converged_layout')

        self.logger.info("=====================================================")
        self.logger.info(f"Running {self.layout_group_name} with zero pages")
        self.logger.info("=====================================================")
        tail_pages = self.get_tail_pages(total_threshold=1)
        layout_zeroes = list(set(layout) | set(tail_pages))
        converged_layout_with_zeroes_r = self.run_next_layout(layout_zeroes)
        layout_name = converged_layout_with_zeroes_r['layout']
        self.log(f"{self.layout_group_name}_with_zeroes", layout_name)

        self.write_init_group_results(self.layout_group_name, group_details, 'converged_layout_with_all_zero_pages')

        self.log(f"{self.layout_group_name}_start_shake_runtime" ,f"layout{self.last_layout_num+1}")
        self.logger.info("=====================================================")
        self.logger.info(f"==> {self.layout_group_name}: Start shaking runtime")
        self.shake_runtime(layout, shake_budget)
        self.logger.info(f"<== {self.layout_group_name}: Finished shaking runtime")
        self.logger.info("=====================================================")
        self.log(f"{self.layout_group_name}_end_shake_runtime", f"layout{self.last_layout_num}")

        self.write_init_group_results(self.layout_group_name, group_details, 'shaking_runtime')

        return layout_result, converged_layout_with_zeroes_r

    def run_with_different_init_layouts(self):
        self.log_metadata()

        if self.debug:
            breakpoint()

        shake_budget = max(self.num_layouts//6, 5)
        self.logger.info(f"==> Shaking runtime budget: {shake_budget} <==")

        initial_layouts = self.get_moselect_init_layouts()
        layout_r, _ = self.run_with_custom_init_layouts(initial_layouts, shake_budget,
                                                        group_details="moselect",
                                                        first_group=True,
                                                        layout_group_name='LayoutA')

        layout = layout_r['hugepages']
        initial_layouts = self.get_complement_surrounding_layouts(layout, layout_r)
        self.run_with_custom_init_layouts(initial_layouts, shake_budget,
                                          group_details="complement moselect",
                                          layout_group_name='LayoutF')

        # initial_layouts = self.get_moselect_init_layouts_C2H_S2L()
        # self.run_with_custom_init_layouts(initial_layouts, shake_budget,
        #                                   group_details="moselect cold-to-hot small-to-large",
        #                                   layout_group_name = 'LayoutD')

        # initial_layouts = self.get_moselect_init_layouts_C2H_L2S()
        # self.run_with_custom_init_layouts(initial_layouts, shake_budget,
        #                                   group_details="moselect cold-to-hot large-to-small",
        #                                   layout_group_name = 'LayoutE')

        # initial_layouts = self.get_moselect_init_layouts_H2C_L2S()
        # self.run_with_custom_init_layouts(initial_layouts, shake_budget,
        #                                   group_details="moselect hot-to-cold large-to-small",
        #                                   layout_group_name = 'LayoutC')

        # initial_layouts = self.get_moselect_init_layouts_4_groups()
        # self.run_with_custom_init_layouts(initial_layouts, shake_budget,
        #                                   group_details="four groups",
        #                                   layout_group_name = 'LayoutB')

        if self.debug:
            breakpoint()

        self.logger.info("=================================================================")
        self.logger.info(f"Finished running MosRange process for:\n{self.exp_root_dir}")
        self.logger.info("=================================================================")
        # MosrangeSelector.pause()

    def run_with_main_init_layouts_algos(self):
        self.log_metadata()

        if self.debug:
            breakpoint()

        shake_budget = max(self.num_layouts//6, 5)
        self.logger.info(f"==> Shaking runtime budget: {shake_budget} <==")

        initial_layouts = self.get_moselect_init_layouts()
        moselect_layout_r, _ = self.run_with_custom_init_layouts(initial_layouts, shake_budget, group_details="moselect", first_group=True)

        initial_layouts = self.get_moselect_distinct_init_layouts()
        compmoselect_layout_r, _ = self.run_with_custom_init_layouts(initial_layouts, shake_budget, group_details="compmoselect", first_group=True)

        layout = moselect_layout_r['hugepages']
        initial_layouts = self.get_complement_surrounding_layouts(layout, moselect_layout_r)
        self.run_with_custom_init_layouts(initial_layouts, shake_budget, group_details="complement moselect layout")

        layout = compmoselect_layout_r['hugepages']
        initial_layouts = self.get_complement_surrounding_layouts(layout, compmoselect_layout_r)
        self.run_with_custom_init_layouts(initial_layouts, shake_budget, group_details="complement compmoselect layout")

        if self.debug:
            breakpoint()

        self.logger.info("=================================================================")
        self.logger.info(f"Finished running MosRange process for:\n{self.exp_root_dir}")
        self.logger.info("=================================================================")
        # MosrangeSelector.pause()

    def run(self):
        self.backup_and_truncate_log(self.log_file_path)
        self.run_with_different_init_layouts()

