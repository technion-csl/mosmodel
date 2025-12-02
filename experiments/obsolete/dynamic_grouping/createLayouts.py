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

def loadDataframe(mean_file):
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

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Log():
    def __init__(self, exp_dir, results_df, log_name, default_columns):
        self._exp_dir = exp_dir
        self._results_df = results_df.copy()
        self._log_file = self._exp_dir + '/' + log_name
        self._default_columns = default_columns
        self._df = self.readLog()

    def readLog(self):
        if not os.path.isfile(self._log_file):
            self._df = pd.DataFrame(columns=self._default_columns)
        else:
            self._df = pd.read_csv(self._log_file)
        return self._df

    def writeLog(self):
        self._df.to_csv(self._log_file, index=False)

    def clear(self):
        self._df = pd.DataFrame(columns=self._default_columns)

    def empty(self):
        return self._df.empty

    def getField(self, key_name, key_value, field_name):
        field_val = self._df.loc[self._df[key_name] == key_value, field_name]
        field_val = field_val.to_list()
        if field_val == []:
            return None
        else:
            return field_val[0]

    def getRealCoverage(self, layout):
        return self.getField('layout', layout, 'real_coverage')

    def getPebsCoverage(self, layout):
        return self.getField('layout', layout, 'pebs_coverage')

    def getLastRecord(self):
        if self.empty():
            return None
        return self._df.iloc[len(self._df)-1]

    def getRecord(self, key_name, key_value):
        record = self._df.query('{key} == "{value}"'.format(
            key=key_name,
            value=key_value))
        if record.empty:
            return None
        else:
            return record.iloc[0]

    def writeRealCoverage(self):
        max_walk_cycles = self._results_df['walk_cycles'].max()
        min_walk_cycles = self._results_df['walk_cycles'].min()
        delta_walk_cycles = max_walk_cycles - min_walk_cycles
        self._df['real_coverage'] = self._df['real_coverage'].astype(float)
        query = self._df.query('real_coverage == (-1)')
        for index, row in query.iterrows():
            layout = row['layout']
            walk_cycles = self._results_df.loc[self._results_df['layout'] == layout, 'walk_cycles'].iloc[0]
            real_coverage = (max_walk_cycles - walk_cycles) / delta_walk_cycles
            real_coverage *= 100
            self._df.loc[self._df['layout'] == layout, 'real_coverage'] = real_coverage
            self._df.loc[self._df['layout'] == layout, 'walk_cycles'] = walk_cycles
        self.writeLog()

class StaticLog(Log, metaclass=Singleton):
    def __init__(self, exp_dir, results_df):
        default_columns = [
            'layout', 'right_layout', 'left_layout',
            'pebs_coverage', 'real_coverage', 'walk_cycles']
        super().__init__(exp_dir, results_df, 'static_layouts.log', default_columns)

    def addRecord(self,
                  layout, right_layout, left_layout, pebs_coverage,
                  writeLog=False):
        self._df = self._df.append({
            'layout': layout,
            'right_layout': right_layout,
            'left_layout': left_layout,
            'pebs_coverage': pebs_coverage,
            'real_coverage': -1,
            'walk_cycles': -1
            }, ignore_index=True)
        if writeLog:
            self.writeLog()

class GroupsLog(Log, metaclass=Singleton):
    def __init__(self, exp_dir, results_df):
        default_columns = [
            'layout', 'total_budget', 'remaining_budget',
            'pebs_coverage', 'real_coverage', 'walk_cycles']
        super().__init__(exp_dir, results_df, 'groups.log', default_columns)

    def addRecord(self,
                  layout, pebs_coverage, writeLog=False):
        self._df = self._df.append({
            'layout': layout,
            'total_budget': -1,
            'remaining_budget': -1,
            'pebs_coverage': pebs_coverage,
            'real_coverage': -1,
            'walk_cycles': -1
            }, ignore_index=True)
        if writeLog:
            self.writeLog()

    def calculateBudget(self):
        query = self._df.query('real_coverage == (-1)')
        if len(query) > 0:
            raise Exception('GroupsLog.calculateBudget was called before updating the groups real_coverage.')
        query = self._df.query('total_budget < 0')
        if len(query) == 0:
            return
        # sort the group layouts by walk-cycles/real_coverage
        self._df = self._df.sort_values('real_coverage', ascending=True)
        # calculate the diff between each two adjacent layouts
        # (call it delta[i] for the diff between group[i] and group[i+1])
        self._df['delta'] = self._df['real_coverage'].diff().abs()
        self._df['delta'] = self._df['delta'].fillna(0)
        total_deltas = self._df.query('delta > 2.5')['delta'].sum()
        total_budgets = 46 # 55-9: num_layouts(55) - groups_layouts(9)
        for index, row in self._df.iterrows():
            delta = row['delta']
            # for each delta < 2.5 assign budget=0
            if delta <= 2.5:
                budget = 0
            else:
                budget = round((delta / total_deltas) * total_budgets)
            self._df.at[index, 'total_budget'] = budget
            self._df.at[index, 'remaining_budget'] = budget
        # fix total budgets due to rounding
        rounded_total_budgets = self._df['total_budget'].sum()
        delta_budget = total_budgets - rounded_total_budgets
        self._df.at[index, 'total_budget'] = budget + delta_budget
        self._df.at[index, 'remaining_budget'] = budget + delta_budget

        self.writeLog()

    def decreaseRemainingBudget(self, layout):
        self._df.loc[self._df['layout'] == layout, 'remaining_budget'] = self._df.loc[self._df['layout'] == layout, 'remaining_budget']-1
        self.writeLog()

    def writeLog(self):
        self._df.to_csv(self._log_file, index=False)

class StateLog(Log):
    def __init__(self, exp_dir, results_df, right_layout, left_layout, total_budget, remaining_budget):
        default_columns = [
            'layout',
            'scan_method', 'scan_direction', 'scan_value', 'scan_base',
            'pebs_coverage', 'target_coverage', 'real_coverage', 'walk_cycles']
        self._right_layout = right_layout
        self._left_layout = left_layout
        state_name = right_layout + '_' +left_layout
        self._total_budget = total_budget
        self._remaining_budget = remaining_budget
        super().__init__(exp_dir, results_df, state_name + '_state.log', default_columns)
        super().writeRealCoverage()

    def addRecord(self,
                  layout,
                  scan_method, scan_direction, scan_value, scan_base,
                  pebs_coverage, target_coverage,
                  writeLog=True):
        self._df = self._df.append({
            'layout': layout,
            'scan_method': scan_method,
            'scan_direction': scan_direction,
            'scan_value': scan_value,
            'scan_base': scan_base,
            'pebs_coverage': pebs_coverage,
            'target_coverage': target_coverage,
            'real_coverage': -1,
            'walk_cycles': -1
            }, ignore_index=True)
        if writeLog:
            self.writeLog()

    def getRightLayoutName(self):
        return self._right_layout

    def getLeftLayoutName(self):
        return self._left_layout

    def getRigthRecord(self):
        assert(not self.empty())
        return self.getRecord('layout', self.getRightLayoutName())

    def getLeftRecord(self):
        assert(not self.empty())
        return self.getRecord('layout', self.getLeftLayoutName())

    def getTotalBudget(self):
        return _total_budget

    def isLastRecordInRange(self):
        #self.writeRealCoverage()
        last_layout = self.getLastRecord()
        right_layout = self.getRecord('layout', self.getRightLayoutName())
        assert(right_layout is not None)
        left_layout = self.getRecord('layout', self.getLeftLayoutName())
        assert(left_layout is not None)

        return right_layout['walk_cycles'] >= last_layout['walk_cycles'] and \
            last_layout['walk_cycles'] >= left_layout['walk_cycles']

    def getGapFromBase(self, layout, base_layout):
        layout_coverage = self.getRealCoverage(layout)
        assert(layout_coverage is not None)
        base_coverage = self.getRealCoverage(base_layout)
        assert(base_coverage is not None)

        print('[DEBUG]--------------------------')
        print('[DEBUG]last-layout = '+str(layout)+' , base_layout= '+str(base_layout))
        print('[DEBUG]last-layout-coverage = '+str(layout_coverage)+' , base_layout_coverage= '+str(base_coverage))
        print('[DEBUG]--------------------------')

        return layout_coverage - base_coverage

    def lastLayoutImprovedMaxGap(self):
        """
        Calculates if the last layout contributed to reducing the
        maximal gap in the current group

        Returns
        -------
        bool
            Returns True if the last layout reduced the maximal gap
            Returns False otherwise
        """
        #self.writeRealCoverage()
        including_df_diffs = self._df.sort_values('real_coverage', ascending=True)
        including_max_diff = including_df_diffs['real_coverage'].diff().max()

        excluding_df = self._df.iloc[0:len(self._df)-1]
        excluding_df_diffs = excluding_df.sort_values('real_coverage', ascending=True)
        excluding_max_diff = excluding_df_diffs['real_coverage'].diff().max()

        return including_max_diff < excluding_max_diff

    def lastLayoutImprovedGaps(self):
        """
        Calculates if the last layout contributed to reducing the total
        'big gaps', which are greater than 2.5%.

        Returns
        -------
        bool
            Returns True if the last layout reduced the sum of the '
            big gaps' or splitted one 'big gap' to two more smalled gaps.
            Returns False otherwise
        """
        #self.writeRealCoverage()
        including_df_diffs = self._df.sort_values('real_coverage', ascending=True)
        including_df_diffs['diff'] = including_df_diffs['real_coverage'].diff()
        including_df_diffs = including_df_diffs.query('diff > 2.5')

        excluding_df = self._df.iloc[0:len(self._df)-1]
        excluding_df_diffs = excluding_df.sort_values('real_coverage', ascending=True)
        excluding_df_diffs['diff'] = excluding_df_diffs['real_coverage'].diff()
        excluding_df_diffs = excluding_df_diffs.query('diff > 2.5')

        sum_diff = excluding_df_diffs['diff'].sum() - including_df_diffs['diff'].sum()
        count_diff = excluding_df_diffs['diff'].count() - including_df_diffs['diff'].count()

        return count_diff > 0 or sum_diff > 0

    def getGapBetweenLastRecordAndBase(self):
        #self.writeRealCoverage()
        last_layout = self.getLastRecord()
        base_layout = last_layout['scan_base']
        return self.getGapFromBase(last_layout['layout'], base_layout)

    def getBaseLayout(self, layout_name):
        return self.getField('layout', layout_name, 'scan_base')

    def getRootBaseLayout(self, layout):
        """
        Searches the state log, starting from the last record, to find the
        root base_layout that was used to get to the last record.

        Returns a layout that has no base layout, which the last record was
        derived from it
        """
        current_layout = layout
        base_layout = current_layout
        while base_layout != 'none':
            current_layout = base_layout
            base_layout = self.getBaseLayout(current_layout)
        return current_layout

    def getNewBaseLayoutName(self):
        """
        Returns a new layout to be used as a base for scanning the space
        and closing the gap between the right and left layouts of current
        state.
        The new base_layout is found by looking for a layout with the
        maximal gap that is greater than 2.5%
        """
        #self.writeRealCoverage()
        diffs = self._df.sort_values('real_coverage', ascending=True)
        diffs['diff'] = diffs['real_coverage'].diff().abs()

        idx_label = diffs['diff'].idxmax()
        idx = diffs.index.get_loc(idx_label)
        right = diffs.iloc[idx-1]
        left = diffs.iloc[idx]
        if max(right['diff'], left['diff']) <= 2.5:
            return None, None
        return right['layout'], left['layout']

class deprcatedScanMethods():
    def __createNextDynamicLayoutStatically(pebs_df, results_df, layout, exp_dir):
        # 2.2. create additional 15 layouts dynamically (in runtime):
        # 2.2.1 collect their results
        # results_df contains these results

        # 2.2.2. update the real-coverage in the log
        log = StaticLog(args.exp_dir, results_df)
        log.writeRealCoverage()

        # 2.2.3 calculate gaps between measurements
        #results_df = results_df.sort_values('walk_cycles', ascending=False)
        df = log._df.sort_values('real_coverage', ascending=True)

        # 2.2.4 find the maximum gap
        #idx = results_df['walk_cycles'].diff().abs().argmax()
        idx = df['real_coverage'].diff().abs().argmax()

        # 2.2.4.1 find the two layouts of the gap’s two measurements (right and left)
        #right_layout = results_df.iloc[idx-1]
        #left_layout = results_df.iloc[idx]
        right_layout = df.iloc[idx-1]
        left_layout = df.iloc[idx]

        # 2.2.4.2 scale the expected coverage according to the ratio between real coverages
        #left_real_coverage = log.getRealCoverage(left_layout['layout'])
        #right_real_coverage = log.getRealCoverage(right_layout['layout'])
        #real_coverage_delta = left_real_coverage - right_real_coverage
        left_exp_coverage = log.getPebsCoverage(left_layout['layout'])
        right_exp_coverage = log.getPebsCoverage(right_layout['layout'])
        #exp_coverage_delta = left_exp_coverage - right_exp_coverage
        #scale = exp_coverage_delta / real_coverage_delta
        #new_delta = scale * 2.5

        # 2.2.4.3 add new_delta coverage to the right layout
        tlb_coverage_percentage = abs(left_exp_coverage + right_exp_coverage)/2
        pages = findTlbCoveragePages(pebs_df, tlb_coverage_percentage, 0.1)
        print('[DEBUG]TLB-coverage = {coverage} - Paegs = {pages}'.format(coverage=tlb_coverage_percentage, pages=pages))
        writeLayout(layout, pages, exp_dir)
        log.addRecord(layout, right_layout['layout'], left_layout['layout'], tlb_coverage_percentage, True)

    def __buildGroupsSparsely(pebs_df, exp_dir, desired_weights):
        pebs_df.sort_values('NUM_ACCESSES', ascending=False, inplace=True)
        groups = []
        current_group = []
        current_total_weight = 0
        i = 0
        for index, row in pebs_df.iterrows():
            page_number = int(row['PAGE_NUMBER'])
            weight = row['NUM_ACCESSES']
            if current_total_weight >= desired_weights[i]:
                groups.append(current_group)
                current_group = []
                current_total_weight = 0
                i += 1
            if i == len(desired_weights):
                break
            if current_total_weight < desired_weights[i]:
                current_total_weight += weight
                current_group.append(page_number)
        return groups


def writeLayoutAll2mb(layout, output):
    page_size = 1 << 21
    brk_pool_size = Utils.round_up(brk_footprint, page_size)
    configuration = Configuration()
    configuration.setPoolsSize(
            brk_size=brk_pool_size,
            file_size=1*Utils.GB,
            mmap_size=mmap_footprint)
    configuration.addWindow(
            type=configuration.TYPE_BRK,
            page_size=page_size,
            start_offset=0,
            end_offset=brk_pool_size)
    configuration.exportToCSV(output, layout)

def writeLayout(layout, pages, output, sliding_index=0):
    page_size = 1 << 21
    hugepages_start_offset = sliding_index * 4096
    brk_pool_size = Utils.round_up(brk_footprint, page_size) + hugepages_start_offset
    configuration = Configuration()
    configuration.setPoolsSize(
            brk_size=brk_pool_size,
            file_size=1*Utils.GB,
            mmap_size=mmap_footprint)
    for w in pages:
        configuration.addWindow(
                type=configuration.TYPE_BRK,
                page_size=page_size,
                start_offset=(w * page_size) + hugepages_start_offset,
                end_offset=((w+1) * page_size) + hugepages_start_offset)
    configuration.exportToCSV(output, layout)

def getLayoutHugepages(layout_name, exp_dir):
    layout_file = str.format('{exp_root}/layouts/{layout_name}.csv',
            exp_root=exp_dir,
            layout_name=layout_name)
    df = pd.read_csv(layout_file)
    df = df[df['type'] == 'brk']
    df = df[df['pageSize'] == 2097152]
    pages = []
    offset_deviation = 0
    for index, row in df.iterrows():
        start_page = int(row['startOffset'] / 2097152)
        end_page = int(row['endOffset'] / 2097152)
        offset_deviation = int(row['startOffset'] % 2097152)
        for i in range(start_page, end_page, 1):
            pages.append(i)
    start_deviation = offset_deviation / 4096
    return pages, start_deviation

def calculateTlbCoverage(pebs_df, pages):
    total_weight = pebs_df.query(
            'PAGE_NUMBER in {pages}'.format(pages=pages))\
                    ['NUM_ACCESSES'].sum()
    return total_weight

def buildGroupsSequentially(orig_pebs_df, exp_dir, desired_weights, all_pages):
    pebs_df = orig_pebs_df[['PAGE_NUMBER', 'NUM_ACCESSES']]
    pebs_df = pebs_df.sort_values('NUM_ACCESSES', ascending=False)
    groups = []
    i = 0
    for index, row in pebs_df.iterrows():
        current_total_weight = 0
        current_group = []

        page_number = int(row['PAGE_NUMBER'])
        if page_number in all_pages:
            continue

        weight = row['NUM_ACCESSES']

        epsilon = desired_weights[i] * 0.1
        if weight > (desired_weights[i] + epsilon):
            continue

        query = pebs_df.query(
            'PAGE_NUMBER <= {max_page} and PAGE_NUMBER >= {min_page} and PAGE_NUMBER not in {all_pages}'.format(
                max_page=page_number+25,
                min_page=page_number-25,
                all_pages=all_pages))
        if query['NUM_ACCESSES'].sum() < (desired_weights[i] - epsilon):
            continue
        query_df = query.sort_values('PAGE_NUMBER', ascending=True).reset_index()
        query_df = query_df[['PAGE_NUMBER', 'NUM_ACCESSES']]

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

def createGroups(pebs_df, results_df, exp_dir, write_layouts=True):
    i = 1
    desired_weights = [50, 20, 10]
    groups = []
    groups_pages = []
    log = GroupsLog(exp_dir, results_df)
    # 1.1.1. create three groups of pages that are responsible for (50%, 20%, 10%)
    while len(groups) != 3:
        g = buildGroupsSequentially(pebs_df, exp_dir, desired_weights, groups_pages)
        groups += g
        desired_weights = desired_weights[len(g):len(desired_weights)]
        # if we could not find the required groups with current weights
        # then try to lower bound the desired weights
        desired_weights = [0.9*w for w in desired_weights]
    # 1.1.2. create eight layouts as all sub-groups of these three groups
    for subset_size in range(len(groups)+1):
        for subset in itertools.combinations(groups, subset_size):
            pages = []
            for l in subset:
                pages += l
            layout_name = 'layout' + str(i)
            i += 1
            pebs_coverage = calculateTlbCoverage(pebs_df, pages)
            print(layout_name)
            print('[DEBUG]#hugepages: '+ str(len(pages)))
            print('[DEBUG]weight: ' + str(pebs_coverage))
            print('[DEBUG]hugepages: ' + str(pages))
            print('[DEBUG]---------------')
            if write_layouts:
                writeLayout(layout_name, pages, exp_dir)
            log.addRecord(layout_name, pebs_coverage)
    # 1.1.3. create additional layout in which all pages are backed with 2MB
    layout_name = 'layout' + str(i)
    print(layout_name)
    print('[DEBUG]weight: 100%')
    print('[DEBUG]hugepages: all pages')
    if write_layouts:
        writeLayoutAll2mb(layout_name, exp_dir)
    log.addRecord(layout_name, 100)
    log.writeLog()

def findTlbCoveragePages(df, tlb_coverage_percentage, epsilon, base_pages=[]):
    pages = None
    while pages is None:
        pages = _findTlbCoveragePages(df, tlb_coverage_percentage, epsilon, base_pages)
        epsilon += 0.1
    return pages

def _findTlbCoveragePages(df, tlb_coverage_percentage, epsilon, base_pages=[]):
    pages = base_pages.copy()
    total_weight = calculateTlbCoverage(df, base_pages)
    assert tlb_coverage_percentage >= total_weight,'findTlbCoveragePages: the required tlb-coverage is less than base pages coverage'
    coverage_needed = tlb_coverage_percentage - total_weight
    df = df.query('NUM_ACCESSES < {c}'.format(c=coverage_needed))
    for index, row in df.iterrows():
        weight = row['NUM_ACCESSES']
        page_number = row['PAGE_NUMBER']
        if page_number in base_pages:
            continue
        if (total_weight + weight) <= (tlb_coverage_percentage + epsilon):
            total_weight += weight
            pages.append(page_number)
        if total_weight >= (tlb_coverage_percentage - epsilon):
            break

    if total_weight > (tlb_coverage_percentage + epsilon) \
            or total_weight < (tlb_coverage_percentage - epsilon):
        return None
    return pages

def createStatisLayouts(pebs_df, results_df, exp_dir, step_size):
    """ Creates 40 layouts statically:

    2.1.	Create 40 layouts statically:
        2.1.1.	such that each layout covers 2.5% of TLB-misses more than previous layout (according to PEBS)
        2.1.2.	for each layout log the following record:
            layout,	      right-layout, left-layout,	expected-coverage,	real-coverage
            <layout-name>,	0,	          0,	       <expected %>,	filled later (in 2.2)
    2.2.    createNextStaticLayout()
    """
    df = pebs_df.sort_values('NUM_ACCESSES', ascending=False)
    tlb_coverage_percentage = 0
    num_layout = 1
    log = StaticLog(args.exp_dir, results_df)
    # 2.1. Create 40 layouts statically
    while tlb_coverage_percentage < 100:
        # 2.1.1. such that each layout covers 2.5% of TLB-misses more than previous layout (according to PEBS)
        pages = findTlbCoveragePages(df, tlb_coverage_percentage, 0.1)
        print('[DEBUG]TLB-coverage = {coverage} - Paegs = {pages}'.format(coverage=tlb_coverage_percentage, pages=pages))
        layout_name = 'layout'+str(num_layout)
        writeLayout(layout_name, pages, exp_dir)
        # 2.1.2. for each layout log the following record:
        log.addRecord(layout_name, 'TBD', 'TBD',
                       tlb_coverage_percentage)
        num_layout += 1
        tlb_coverage_percentage += step_size
    log.writeLog()

import re
def __natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def __layoutsOrderedOrdinally(results_df, layouts_list):
    sorted_layouts = __natural_sort(layouts_list)
    interesting_results = results_df.query('layout in {layouts_list}'.format(
        layouts_list=layouts_list))
    sorted_walks = interesting_results.sort_values('walk_cycles', ascending=False)
    return sorted_layouts == sorted_walks

def createNextStaticLayout(pebs_df, results_df, layout, exp_dir):
    """ Creates layout based on the previous statically-created layouts
        (when the first-10-pages weight < 30%)
    2.1.	createStatisLayouts()
    2.2.	Create additional 15 layouts dynamically (in runtime):
        2.2.1.	collect their results
        2.2.2.	update the real-coverage in the log
        2.2.3.	calculate gaps between measurements
        2.2.4.	find the maximum gap:
            2.2.4.1.	find the two layouts of the gap’s two measurements (right and left)
            2.2.4.2.	scale the expected coverage according to the ratio between real coverages:
            real_coverage_delta = left_real_coverage - right_real_coverage
            exp_coverage_delta = left_exp_coverage - right_exp_coverage
            scale = exp_coverage_delta / real_coverage_delta
            new_delta = scale * 2.5
            2.2.4.3.	add new_delta coverage to the right layout
    """

    # 2.2. create additional 15 layouts dynamically (in runtime):

    # 2.2.1 collect their results
    if results_df.empty:
        sys.exit('results csv file is empty but it should \
                 contain the results of the first 40 points!')

    # 2.2.2. update the real-coverage in the log
    log = StaticLog(args.exp_dir, results_df)
    if results_df.empty:
        sys.exit('The static.log is empty!')
    log.writeRealCoverage()

    # check if the first 40 layouts were not produced ordinally as we expect
    # then run the dynamic algorithm to create the rest layouts
    layouts_list = log._df['layout'].to_list()
    if not __layoutsOrderedOrdinally(results_df, layouts_list):
        createNextLayoutDynamically(pebs_df, results_df, layout, exp_dir)
        return

    # 2.2.3 calculate gaps between measurements
    results_df = results_df.sort_values('walk_cycles', ascending=False)
    # 2.2.4 find the maximum gap
    idx_label = results_df['walk_cycles'].diff().abs().idxmax()
    idx = results_df.index.get_loc(idx_label)
    # 2.2.4.1 find the two layouts of the gap’s two measurements (right and left)
    right_layout = results_df.iloc[idx-1]
    left_layout = results_df.iloc[idx]
    # 2.2.4.2(v2) decrease the expected-coverage by half
    left_exp_coverage = log.getPebsCoverage(left_layout['layout'])
    right_exp_coverage = log.getPebsCoverage(right_layout['layout'])
    new_coverage = (left_exp_coverage - right_exp_coverage) / 2
    assert(new_coverage > 0)
    pages = findTlbCoveragePages(pebs_df, new_coverage, 0.1)
    print('[DEBUG]TLB-coverage = {coverage} - Paegs = {pages}'.format(coverage=new_coverage, pages=pages))
    writeLayout(layout, pages, exp_dir)
    log.addRecord(layout, right_layout['layout'], left_layout['layout'], new_coverage, True)

def getRandomLayout():
    assert False,'getRandomLayout not implemented'

def getNextSlidingDirection(state, last_layout, last_direction, last_value):
    if last_direction == 'increment':
        return 'decrement', 128, state.getLastRecord()['layout']
    elif last_direction == 'decrement':
        root_base_layout = state.getRootBaseLayout(last_layout)
        if root_base_layout == state.getRightLayoutName():
            return 'increment', 128, state.getLeftLayoutName()
        elif root_base_layout == state.getLeftLayoutName():
            return getRandomLayout()
        else:
            assert False,'unexpected root base-layout: '+root_base_layout
    else:
        assert False,'unkown scan-direction: '+last_direction

def predictSlidingDirection(pebs_df, state, base_layout, exp_dir):
    pages, start_deviation = getLayoutHugepages(base_layout, exp_dir)
    incremented_pages = [p+1 for p in pages if p < last_page]
    decremented_pages = [p-1 for p in pages if p > 0]
    incremented_tlb_coverage = calculateTlbCoverage(pebs_df, incremented_pages)
    decremented_tlb_coverage = calculateTlbCoverage(pebs_df, decremented_pages)

    coverage = calculateTlbCoverage(pebs_df, pages)
    #coverage = state.getPebsCoverage(base_layout)
    #if coverage == None or coverage == -1:
    #    coverage = state.getRealCoverage(base_layout)

    if incremented_tlb_coverage > coverage:
        direction = 'increment'
        value = min((2.5 / (incremented_tlb_coverage - coverage)) * 511, 511)
    elif decremented_tlb_coverage > coverage:
        direction = 'decrement'
        value = min((2.5 / (decremented_tlb_coverage - coverage)) * 511, 511)
    else:
        assert False,'sliding to both sides does not increase the coverage'

    return direction, value

def getSlidingScanningParameters(pebs_df, state, exp_dir):
    last_record = state.getLastRecord()
    last_method = last_record['scan_method']
    last_direction = last_record['scan_direction']
    last_value = last_record['scan_value']
    last_base = last_record['scan_base']
    last_layout = last_record['layout']

    method = 'sliding'
    direction = last_direction
    value = last_value
    base = last_base
    target_value = -1

    if last_method == 'tail':
        direction, value = predictSlidingDirection(pebs_df, state, base, exp_dir)
    elif last_method == 'sliding':
        pages, deviation_offset = getLayoutHugepages(last_layout, exp_dir)
        if state.isLastRecordInRange():
            gap = state.getGapBetweenLastRecordAndBase()
            if gap <= 0:
                direction, value, base = getNextSlidingDirection(state, last_direction, last_value)
            elif gap <= 2.5:
                base = last_layout
                inc_factor = 1 + (1-(gap / 2.5)) *0.5
                value = int(last_value * inc_factor)
            else: # gap > 2.5
                dec_factor = gap / 2.5
                value = int(last_value / dec_factor)
        else: #last-record not in range
            direction, value, base = getNextSlidingDirection(state, last_direction, last_value)
    else:
        assert False,'unkown scan-method: '+last_method
    return method, direction, value, base, target_value

def getTlbCoverageScale(state, right_layout, left_layout):
    diff_real = state.getRealCoverage(left_layout) - state.getRealCoverage(right_layout)
    diff_pebs = state.getPebsCoverage(left_layout) - state.getPebsCoverage(right_layout)

def getTailScanningParameters(pebs_df, state, exp_dir):
    print('[DEBUG] --entry--> getTailScanningParameters')
    last_record = state.getLastRecord()
    method = 'tail'
    direction = 'increment'
    base = last_record['scan_base']
    desired_real_coverage = last_record['target_coverage']
    scaling_base = last_record['layout']

    print(state._df)
    right_base, left_base = state.getNewBaseLayoutName()
    if last_record['scan_base'] == 'none' or state.lastLayoutImprovedMaxGap():
        base = right_base
        desired_real_coverage = (state.getRealCoverage(right_base) + state.getRealCoverage(left_base)) / 2
        scaling_base = right_base
        if state.getRealCoverage(scaling_base) <= 0:
            scaling_base = left_base

    scale_factor = state.getRealCoverage(scaling_base) / state.getPebsCoverage(scaling_base)
    #scale_factor = last_record['real_coverage'] / last_record['pebs_coverage']

    pebs_coverage = desired_real_coverage / scale_factor
    value = pebs_coverage
    target_value = desired_real_coverage

    print('[DEBUG] the new base: ' + base)
    print('[DEBUG] target-pebs-coverage: {pebs} , target-real-coverage: {real}'.format(
        pebs=value, real=target_value))
    print('[DEBUG] <--exit-- getTailScanningParameters')

    return method, direction, value, base, target_value

def findNextScanMethod(pebs_df, state, exp_dir):
    last_record = state.getLastRecord()
    last_scan_method = last_record['scan_method']
    if last_scan_method == 'tail' or last_scan_method == 'none':
        return getTailScanningParameters(pebs_df, state, exp_dir)
    elif last_scan_method == 'sliding':
        return getSlidingScanningParameters(pebs_df, state, exp_dir)
    elif last_scan_method == 'random':
        return getRandomLayout()
    else:
        sys.exit('unexpected scanning method at this stage: ' + last_scan_method)

def applyScanMethod(pebs_df, exp_dir, state,
        layout, base_layout_name,
        method, direction, value):
    # find the base-layout in the state log
    base_layout = state.getRecord('layout', base_layout_name)
    assert base_layout is not None,'applyScanMethod: base-layout could not be found in the state log'

    # static method: create a new layout by incrementing/decrementing
    # the base-layout tlb-coverage with a fixed value
    if method == 'tail':
        found = True
        upper_bound = lower_bound = value
        while found:
            upper_found  = not state._df.query(
                    'scan_value > {min_val} and scan_value < {max_val}'.format(
                        min_val=upper_bound-0.6, max_val=upper_bound+0.6)).empty
            if upper_found:
                upper_bound += 0.6
                upper_bound = min(100, upper_bound)
            else:
                value = upper_bound
                break
            lower_found  = not state._df.query(
                    'scan_value > {min_val} and scan_value < {max_val}'.format(
                        min_val=lower_bound-0.6, max_val=lower_bound+0.6)).empty
            if lower_found:
                lower_bound -= 0.6
                lower_bound = max(0, lower_bound)
            else:
                value = lower_bound
                break

        value = min(100, value)
        base_pages, offset = getLayoutHugepages(base_layout_name, exp_dir)
        pages = findTlbCoveragePages(pebs_df, value, 0.1, base_pages)
        pebs_coverage = calculateTlbCoverage(pebs_df, pages)
        print('[DEBUG] trying to find pages with TLB-coverage: {c} --> found pages with coverage: {f}'.format(
            c=value, f=pebs_coverage))
        print('[DEBUG] TLB-coverage = {coverage} - Paegs = {pages}'.format(coverage=pebs_coverage, pages=pages))
        return pages, 0
    elif method == 'sliding':
        if direction == 'increment':
            sliding = value
        elif direction == 'decrement':
            sliding = -value
        else:
            sys.exit('Error: applyScanMethod was called with unexpected scanning direction for static method: ' + direction)
        pages, offset = getLayoutHugepages(base_layout['layout'], exp_dir)
        print('[DEBUG] sliding offset = {sliding} - Paegs = {pages}'.format(sliding=sliding, pages=pages))
        return pages, sliding
    else:
        sys.exit('Error: applyScanMethod was called with unexpected scanning method: ' + method)

def createNextLayoutDynamically(pebs_df, results_df, layout, exp_dir):

    # calculate the real-coverage for each group and update the log
    # if the groups-log was not created yet then create it based on the
    # current results (this could happen if we started with the static
    # method and then decided to move to the dynamic method due to irregual
    # layouts real coverage, i.e. increasing the coverage causes to increasing
    # the walk cycles)
    groups_log = GroupsLog(exp_dir, results_df)
    if groups_log.empty():
        results_df_sorted = results_df.sort_values('walk_cycles',
                                                   ascending=False)
        for index, row in results_df_sorted.iterrows():
            groups_log.addRecord(row['layout'], -1)
        groups_log.writeRealCoverage()
        # filter out all layouts that already have small gaps
        groups_log._df = groups_log._df.query('real_coverage > 2.5')
        groups_log._df = groups_log._df.sort_values('real_coverage')
        groups_log.writeLog()
    else:
        groups_log.writeRealCoverage()
    # calculate the budget that will be given for each group
    groups_log.calculateBudget()

    # find the first group that still has a remaining budget
    for i in range(len(groups_log._df)-1):
        right_layout = groups_log._df.iloc[i]
        left_layout = groups_log._df.iloc[i+1]
        if left_layout['remaining_budget'] > 0:
            break
    assert left_layout['remaining_budget'] > 0, 'already consumed all groups budgest but still have additional layouts to create!'

    state = StateLog(exp_dir, results_df, right_layout['layout'], left_layout['layout'], left_layout['total_budget'], left_layout['remaining_budget'])
    # if the state was not created yet then create it and add all
    # layouts that in the range [left_layout - right_layout]
    if state.empty():
        # if the state was not created before then this layout should
        # have a full budget (its budget should still unused)
        assert(left_layout['remaining_budget'] == left_layout['total_budget'])
        state_layouts = results_df.query(
            'walk_cycles >= {left} and walk_cycles <= {right}'.format(
                left=left_layout['walk_cycles'],
                right=right_layout['walk_cycles']))
        for index, row in state_layouts.iterrows():
            pages, offset = getLayoutHugepages(row['layout'], exp_dir)
            pebs_coverage = calculateTlbCoverage(pebs_df, pages)
            state.addRecord(row['layout'], 'none', 'none', -1, 'none', pebs_coverage, -1)
        state.writeLog()
        state.writeRealCoverage()

    # decide how to find the next layout
    method, direction, value, base, target_value = findNextScanMethod(pebs_df, state, exp_dir)
    # apply the previous decesion to create layout pages
    pages, sliding = applyScanMethod(pebs_df, exp_dir, state, layout, base, method, direction, value)
    pebs_coverage = calculateTlbCoverage(pebs_df, pages)
    # write layout configuration file
    writeLayout(layout, pages, exp_dir, sliding)
    # add a new state record
    state.addRecord(layout,
                    method, direction, value,
                    base, pebs_coverage, target_value)
    # decrease current group's budget by 1
    groups_log.decreaseRemainingBudget(left_layout['layout'])


def __normalizePebsAccesses(pebs_mem_bins):
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
    pebs_df['NUM_ACCESSES'] = pebs_df['NUM_ACCESSES'].mul(100).divide(total_access)
    pebs_df = pebs_df.sort_values('NUM_ACCESSES', ascending=False)
    return pebs_df

import argparse
def __parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--memory_footprint', default='memory_footprint.txt')
    parser.add_argument('-p', '--pebs_mem_bins', default='mem_bins_2mb.csv')
    parser.add_argument('-l', '--layout', required=True)
    parser.add_argument('-d', '--exp_dir', required=True)
    parser.add_argument('-n', '--mean_file', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = __parseArguments()

    # read memory-footprints
    footprint_df = pd.read_csv(args.memory_footprint)
    mmap_footprint = footprint_df['anon-mmap-max'][0]
    brk_footprint = footprint_df['brk-max'][0]
    last_page = int(brk_footprint / 4096)

    results_df = loadDataframe(args.mean_file)

    pebs_df = __normalizePebsAccesses(args.pebs_mem_bins)

    num_pages_to_examine = 10
    head_pages_weight_threshold = 30
    headPagesWeight = pebs_df.head(num_pages_to_examine)['NUM_ACCESSES'].sum()
    # 1. If first-10-pages weight > 30% then
    if headPagesWeight > head_pages_weight_threshold:
        print('[DEBUG]: sub-groups (dynamic) method')
        if args.layout == 'layout1':
            # 1.1. create nine layouts statically (using PEBS output):
            createGroups(pebs_df, results_df, args.exp_dir)
        else:
            # 1.2. create other layouts dynamically
            createNextLayoutDynamically(pebs_df, results_df,
                                      args.layout, args.exp_dir)
    # 2. else (first-10-pages weight < 30%) then
    else:
        print('[DEBUG]: static method')
        if args.layout == 'layout1':
            # 2.1. create 40 layouts statically
            createStatisLayouts(pebs_df, results_df, args.exp_dir, 2.5)
        else:
            # 2.2. create additional 15 layouts dynamically (in runtime)
            createNextStaticLayout(pebs_df, results_df,
                                   args.layout, args.exp_dir)

