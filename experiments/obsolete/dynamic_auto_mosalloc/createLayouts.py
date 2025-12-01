#!/usr/bin/env python3
import sys
import os
import pandas as pd
import itertools
import os.path
from ast import literal_eval

sys.path.append(os.path.dirname(sys.argv[0])+"/..")
from Utils.utils import Utils
from Utils.ConfigurationFile import Configuration

sys.path.append(os.path.dirname(sys.argv[0])+"/../../analysis")
from performance_statistics import PerformanceStatistics

MAX_GAP = 4
INCREMENT = MAX_GAP - 0.5
LOW_GAP = MAX_GAP - 1
MAX_TRIALS = 50


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args,
                                                                 **kwargs)
        return cls._instances[cls]


class Log():

    def __init__(self,
                 exp_dir, results_df, log_name,
                 default_columns, converters=None):
        self.exp_dir = exp_dir
        self.results_df = results_df
        self.log_file = self.exp_dir + '/' + log_name
        self.default_columns = default_columns
        self.df = self.readLog(converters)

    def readLog(self, converters=None):
        if not os.path.isfile(self.log_file):
            self.df = pd.DataFrame(columns=self.default_columns)
        else:
            self.df = pd.read_csv(self.log_file, converters=converters)
        return self.df

    def writeLog(self):
        self.df.to_csv(self.log_file, index=False)

    def clear(self):
        self.df = pd.DataFrame(columns=self.default_columns)

    def empty(self):
        return self.df.empty

    def getField(self, layout_name, field_name):
        field_val = self.df.loc[self.df['layout'] == layout_name, field_name]
        field_val = field_val.to_list()
        if field_val == []:
            return None
        else:
            return field_val[0]

    def getRealCoverage(self, layout):
        return self.getField(layout, 'real_coverage')

    def getExpectedRealCoverage(self, layout):
        return self.getField(layout, 'expected_real_coverage')

    def getPebsCoverage(self, layout):
        return self.getField(layout, 'pebs_coverage')

    def getLastRecord(self):
        if self.empty():
            return None
        return self.df.iloc[len(self.df)-1]

    def getLastLayoutName(self):
        """
        Returns
        -------
        string
            returns the name of the last layout in the state log.
        """
        last_record = self.getLastRecord()
        assert last_record is not None,'getLastLayoutName: there is no state records'
        return last_record['layout']

    def getRecord(self, key_name, key_value):
        record = self.df.query('{key} == "{value}"'.format(
            key=key_name,
            value=key_value))
        if record.empty:
            return None
        else:
            return record.iloc[0]

    def writeRealCoverage(self):
        max_walk_cycles = self.results_df['walk_cycles'].max()
        min_walk_cycles = self.results_df['walk_cycles'].min()
        delta_walk_cycles = max_walk_cycles - min_walk_cycles
        self.df['real_coverage'] = self.df['real_coverage'].astype(float)
        query = self.df.query('real_coverage == (-1)')
        for index, row in query.iterrows():
            layout = row['layout']
            walk_cycles = self.results_df.loc[self.results_df['layout'] == layout, 'walk_cycles'].iloc[0]
            real_coverage = (max_walk_cycles - walk_cycles) / delta_walk_cycles
            real_coverage *= 100
            self.df.loc[self.df['layout'] == layout, 'real_coverage'] = real_coverage
            self.df.loc[self.df['layout'] == layout, 'walk_cycles'] = walk_cycles
        self.writeLog()


class SubgroupsLog(Log, metaclass=Singleton):
    def __init__(self, exp_dir, results_df):
        default_columns = [
            'layout', 'total_budget', 'remaining_budget',
            'pebs_coverage', 'real_coverage', 'walk_cycles']
        super().__init__(exp_dir, results_df, 'subgroups.log', default_columns)

    def addRecord(self,
                  layout, pebs_coverage, writeLog=False):
        self.df = self.df.append({
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
        query = self.df.query('real_coverage == (-1)')
        assert len(query) == 0, 'SubgroupsLog.calculateBudget was called before updating the subgroups real_coverage.'
        query = self.df.query('total_budget < 0')
        if len(query) == 0:
            return
        # sort the group layouts by walk-cycles/real_coverage
        self.df = self.df.sort_values('real_coverage', ascending=True)
        # calculate the diff between each two adjacent layouts
        # (call it delta[i] for the diff between group[i] and group[i+1])
        self.df['delta'] = self.df['real_coverage'].diff().abs()
        self.df['delta'] = self.df['delta'].fillna(0)
        total_deltas = self.df.query(f'delta > {MAX_GAP}')['delta'].sum()
        # budgest = 50-9: num_layouts(50) - subgroups_layouts(9)
        total_budgets = MAX_TRIALS - 9
        for index, row in self.df.iterrows():
            delta = row['delta']
            # for each delta < MAX_GAP assign budget=0
            if delta <= MAX_GAP:
                budget = 0
            else:
                budget = round((delta / total_deltas) * total_budgets)
                # we have 41 layouts in total to create dynamically (each can get 2.5 in the best case)
                budget = max(budget, int(delta / 2.5))
            self.df.at[index, 'total_budget'] = budget
            self.df.at[index, 'remaining_budget'] = budget
        # fix total budgets due to rounding
        rounded_total_budgets = self.df['total_budget'].sum()
        delta_budget = total_budgets - rounded_total_budgets
        self.df.at[index, 'total_budget'] = budget + delta_budget
        self.df.at[index, 'remaining_budget'] = budget + delta_budget

        self.writeLog()

    def decreaseRemainingBudget(self, layout):
        self.df.loc[self.df['layout'] == layout, 'remaining_budget'] = self.df.loc[self.df['layout'] == layout, 'remaining_budget']-1
        self.writeLog()

    def zeroBudget(self, layout):
        total = self.getField(layout, 'total_budget')
        remaining = self.getField(layout, 'remaining_budget')
        self.df.loc[self.df['layout'] == layout, 'total_budget'] = total - remaining
        self.df.loc[self.df['layout'] == layout, 'remaining_budget'] = 0
        self.writeLog()

    def addExtraBudget(self, layout, extra_budget):
        self.df.loc[self.df['layout'] == layout, 'remaining_budget'] = self.df.loc[self.df['layout'] == layout, 'remaining_budget']+extra_budget
        self.df.loc[self.df['layout'] == layout, 'total_budget'] = self.df.loc[self.df['layout'] == layout, 'total_budget']+extra_budget
        self.writeLog()

    def getRightmostLayout(self):
        self.writeRealCoverage()
        df = self.df.sort_values('walk_cycles', ascending=False)
        return df.iloc[0]

    def getLeftmostLayout(self):
        self.writeRealCoverage()
        df = self.df.sort_values('walk_cycles', ascending=True)
        return df.iloc[0]

    def getTotalRemainingBudget(self):
        return self.df['remaining_budget'].sum()

    def getTotalBudget(self):
        return self.df['total_budget'].sum()


class StateLog(Log):
    def __init__(self, exp_dir, results_df, right_layout, left_layout):
        default_columns = [
            'layout', 'scan_base', 'increment_base',
            'scan_method', 'scan_direction', 'scan_factor',
            'scan_value', 'pebs_coverage', 'increment_real_coverage',
            'expected_real_coverage', 'real_coverage',
            'walk_cycles', 'offset']
        self.right_layout = right_layout
        self.left_layout = left_layout
        state_name = right_layout + '_' + left_layout
        super().__init__(exp_dir, results_df,
                         state_name + '_state.log', default_columns)
        super().writeRealCoverage()
        self.pages_log_name = self.exp_dir + '/layout_pages.log'
        if not os.path.isfile(self.pages_log_name):
            self.pages_df = pd.DataFrame(columns=[
                'layout', 'base_layout',
                'added_pages', 'pages'])
        else:
            self.pages_df = pd.read_csv(self.pages_log_name, converters={
                "pages": literal_eval, "added_pages": literal_eval})

    def addRecord(self,
                  layout,
                  scan_method, scan_direction,
                  scan_value, scan_factor, scan_base,
                  pebs_coverage, expected_real_coverage, increment_base,
                  pages, offset,
                  writeLog=True):
        base_pages = []
        if scan_base != 'none' and scan_base != 'other':
            base_pages = self.getLayoutPages(scan_base)
        added_pages = list(set(pages) - set(base_pages))
        added_pages.sort()
        self.df = self.df.append({
            'layout': layout,
            'scan_method': scan_method,
            'scan_direction': scan_direction,
            'scan_value': scan_value,
            'scan_factor': scan_factor,
            'scan_base': scan_base,
            'pebs_coverage': pebs_coverage,
            'expected_real_coverage': expected_real_coverage,
            'increment_base': increment_base,
            'increment_real_coverage': self.getRealCoverage(increment_base),
            'real_coverage': -1,
            'walk_cycles': -1,
            'offset': offset
            }, ignore_index=True)
        if writeLog:
            self.writeLog()
        if layout not in self.pages_df['layout']:
            self.pages_df = self.pages_df.append({
                'layout': layout,
                'base_layout': scan_base,
                'added_pages': added_pages,
                'pages': pages
                }, ignore_index=True)
            self.pages_df.to_csv(self.pages_log_name, index=False)

    def getLayoutPages(self, layout):
        pages = self.pages_df.loc[self.pages_df['layout'] == layout, 'pages'].iloc[0]
        return pages

    def getLayoutAddedPages(self, layout):
        return self.getField(layout, 'added_pages')

    def hasOnlyBaseLayouts(self):
        df = self.df.query(f'scan_method != "none" and scan_method != "other"')
        return len(df) == 0

    def getRightLayoutName(self):
        return self.right_layout

    def getLeftLayoutName(self):
        return self.left_layout

    def getRigthRecord(self):
        assert(not self.empty())
        return self.getRecord('layout', self.getRightLayoutName())

    def getLeftRecord(self):
        assert(not self.empty())
        return self.getRecord('layout', self.getLeftLayoutName())

    def getGapFromBase(self, layout, base_layout):
        layout_coverage = self.getRealCoverage(layout)
        assert(layout_coverage is not None)
        base_coverage = self.getRealCoverage(base_layout)
        assert(base_coverage is not None)

        print('--------------------------')
        print('last-layout = '+str(layout)+' , base_layout= '+str(base_layout))
        print('last-layout-coverage = '+str(layout_coverage)+' , base_layout_coverage= '+str(base_coverage))
        print('--------------------------')

        return layout_coverage - base_coverage

    def getGapBetweenLastRecordAndBase(self):
        #self.writeRealCoverage()
        last_layout = self.getLastRecord()
        base_layout = last_layout['increment_base']
        return self.getGapFromBase(last_layout['layout'], base_layout)

    def getBaseLayout(self, layout_name):
        return self.getField(layout_name, 'scan_base')

    def getIncBaseLayout(self, layout_name):
        return self.getField(layout_name, 'increment_base')

    def getLayoutScanFactor(self, layout_name):
        return self.getField(layout_name, 'scan_factor')

    def getNextLayoutToIncrement(self, start_layout):
        start_layout_coverage = self.getRealCoverage(start_layout)
        max_coverage = self.getRealCoverage(self.getLeftLayoutName())
        df = self.df.query(f'real_coverage >= {start_layout_coverage}')
        df = df.sort_values('real_coverage', ascending=True)
        base_layout = start_layout
        current_coverage = start_layout_coverage
        current_layout = start_layout
        print(start_layout)
        for index, row in df.iterrows():
            if row['real_coverage'] <= (current_coverage + MAX_GAP):
                current_layout = row['layout']
                if row['scan_base'] != 'other':
                    base_layout = current_layout
                current_coverage = row['real_coverage']
                if current_coverage >= max_coverage:
                    return None, None
            else:
                break
        return current_layout, base_layout

    def getMaxGapLayouts(self):
        diffs = self.df.sort_values('real_coverage', ascending=True)
        diffs['diff'] = diffs['real_coverage'].diff().abs()

        idx_label = diffs['diff'].idxmax()
        idx = diffs.index.get_loc(idx_label)
        right = diffs.iloc[idx-1]
        left = diffs.iloc[idx]
        return right['layout'], left['layout']

    def isLastLayoutIrregular(self, layout):
        assert len(self.df) > 2
        last_record = self.df.iloc[len(self.df)-1]
        last_layout_base = self.getBaseLayout(last_record['layout'])
        prev_last_record = self.df.iloc[len(self.df)-2]
        if last_record['scan_base'] == prev_last_record['scan_base'] and last_record['scan_method'] == 'right-tail':
            if last_record['pebs_coverage'] > prev_last_record['pebs_coverage'] and last_record['real_coverage'] < prev_last_record['real_coverage']:
                new_pebs_coverage = last_record['pebs_coverage'] + last_record['scan_value']
                return new_pebs_coverage
            if last_record['pebs_coverage'] < prev_last_record['pebs_coverage'] and last_record['real_coverage'] > prev_last_record['real_coverage']:
                new_pebs_coverage = (last_record['pebs_coverage'] + self.getPebsCoverage(last_layout_base)) / 2
                return new_pebs_coverage
        return None

class LayoutGenerator():
    def __init__(self, pebs_df, results_df, layout, exp_dir):
        self.pebs_df = pebs_df
        self.results_df = results_df
        self.layout = layout
        self.exp_dir = exp_dir
        self.subgroups_log = SubgroupsLog(exp_dir, results_df)
        self.state_log = None

    def generateLayout(self):
        if self.layout == 'layout1':
            # 1.1. create nine layouts statically (using PEBS output):
            self.createInitialLayoutsStatically()
        else:
            # 1.2. create other layouts dynamically
            self.createNextLayoutDynamically()

    def createInitialLayoutsStatically(self):
        group = self.createGroup()
        self.createSubgroups(group)

    def createGroup(self):
        # desired weights for each group layout
        desired_weights = [56, 28, 14]
        df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        return self.fillBuckets(df, desired_weights, False)
        '''
        g1 = self.fillBuckets(df, desired_weights, True)
        g1_pages = g1[0] + g1[1] + g1[2]

        df = df.query(f'PAGE_NUMBER not in {g1_pages}')
        df = df.sort_values('TLB_COVERAGE', ascending=True)
        g2 = self.fillBuckets(df, desired_weights, False)

        group = [g1[i]+g2[i] for i in range(len(g1))]

        return group
        '''

    def fillBuckets(self, df, buckets, fill_only_one_slot=False):
        group = [[], [], []]
        i = 0
        for index, row in df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            completed_buckets = [0, 0, 0]
            for k in range(3):
                # skip and count buckets that already filled out
                if buckets[i] <= 0:
                    completed_buckets[i] = 1
                    i = (i + 1) % 3
                # if current page can be added to current bucket then add it
                elif buckets[i] - weight >= -2:
                    if fill_only_one_slot:
                        if completed_buckets[i] == 1:
                            continue
                        completed_buckets[i] = 1
                    group[i].append(page)
                    buckets[i] -= weight
                    i = (i + 1) % 3
                    break
            if sum(completed_buckets) == 3:
                break
        return group

    def createSubgroups(self, group):
        total_pages = len(self.pebs_df)
        i = 1
        # 1.1.2. create eight layouts as all subgroups of these three group layouts
        for subset_size in range(len(group)+1):
            for subset in itertools.combinations(group, subset_size):
                windows = []
                for l in subset:
                    windows += l
                layout_name = f'layout{i}'
                i += 1
                pebs_coverage = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, windows)
                print(layout_name)
                print('#hugepages: {num} (~{percent}%) out of {total}'.format(
                    num=len(windows),
                    percent=round(len(windows)/total_pages * 100),
                    total=total_pages))
                print(f'weight: {pebs_coverage}')
                print(f'hugepages: {windows}')
                print('---------------')
                LayoutGeneratorUtils.writeLayout(layout_name, windows, self.exp_dir)
                self.subgroups_log.addRecord(layout_name, pebs_coverage)
        # 1.1.3. create additional layout in which all pages are backed with 2MB
        layout_name = f'layout{i}'
        print(layout_name)
        print('weight: 100%')
        print('hugepages: all pages')
        LayoutGeneratorUtils.writeLayoutAll2mb(layout_name, self.exp_dir)
        self.subgroups_log.addRecord(layout_name, 100)
        self.subgroups_log.writeLog()

    def updateSubgroupsLog(self):
        # calculate the real-coverage for each group and update the log
        # if the subgroups-log was not created yet then create it based on the
        # current results
        subgroups_layouts = ['layout1', 'layout2','layout3', 'layout4','layout5','layout6','layout7','layout8','layout9']
        if self.subgroups_log.empty():
            results_df_sorted = self.results_df.query(
                    f'layout in {subgroups_layouts}').sort_values(
                            'walk_cycles', ascending=False)
            for index, row in results_df_sorted.iterrows():
                #TODO update pebs coverage for subgroups
                self.subgroups_log.addRecord(row['layout'], -1)
            self.subgroups_log.writeRealCoverage()
            self.subgroups_log.df = self.subgroups_log.df.sort_values('real_coverage')
            self.subgroups_log.writeLog()
        else:
            self.subgroups_log.writeRealCoverage()
        # calculate the budget that will be given for each group
        self.subgroups_log.calculateBudget()

    def findNextSubgroup(self):
        found = False
        # find the first group that still has a remaining budget
        for i in range(len(self.subgroups_log.df)-1):
            right_layout = self.subgroups_log.df.iloc[i]
            left_layout = self.subgroups_log.df.iloc[i+1]
            # initialize the state-log for the current group
            self.state_log = StateLog(self.exp_dir,
                                      self.results_df,
                                      right_layout['layout'],
                                      left_layout['layout'])
            # if the state log is empty then it seems just now we are
            # about to start scanning this group
            if self.state_log.empty():
                self.initializeStateLog(right_layout, left_layout)
            # update state log real coverage for current and previous completed subsubgroups
            else:
                self.state_log.writeRealCoverage()
            # if we already closed all gaps in this group then move the
            # left budget to the next group
            next_layout, base_layout = self.state_log.getNextLayoutToIncrement(
                right_layout['layout'])
            if next_layout is None and base_layout is None:
                print('===========================================================')
                print(f'[DEBUG] closed all gaps for subgroup: {right_layout["layout"]} - {left_layout["layout"]}')
                print('===========================================================')
                self.subgroups_log.zeroBudget(left_layout['layout'])
                continue
            extra_budget = MAX_TRIALS - (self.subgroups_log.getTotalBudget() + 9)
            # if already consumed the total budget then move to next group
            remaining_budget = left_layout['remaining_budget']
            if (remaining_budget + extra_budget) == 0:
                print('===========================================================')
                print(f'[DEBUG] consumed all budget (but still gaps to close) for subgroup: {right_layout["layout"]} - {left_layout["layout"]}')
                print('===========================================================')
                continue
            # if there is an extra budget that remained---and not used---from
            # previous group, then add it to current group
            if extra_budget > 0:
                self.subgroups_log.addExtraBudget(left_layout['layout'], extra_budget)
            found = True
            break
        print('===========================================================')
        if found:
            print(f'[DEBUG] star closing gaps for subgroup: {right_layout["layout"]} - {left_layout["layout"]}')
        else:
            print('[DEBUG] could not find subgroup to close its gaps')
        print('===========================================================')

        return found

    def updateLogs(self):
        self.updateSubgroupsLog()
        found = self.findNextSubgroup()
        if not found:
            # second trial to make sure that passed remaining budget
            # can be reused again for some previous subgroup
            found = self.findNextSubgroup()
        if not found:
            extra_budget = MAX_TRIALS - (self.subgroups_log.getTotalBudget() + 9)
            print(f'finished the last group but there is still ({extra_budget}) remaining budget.')
            print('using the remaining budget to look for previous groups that have more gaps to close')
            right = self.subgroups_log.getRightmostLayout()
            left = self.subgroups_log.getLeftmostLayout()

            if extra_budget > 0:
                self.subgroups_log.addExtraBudget(left['layout'], 1)

            self.state_log = StateLog(self.exp_dir,
                                      self.results_df,
                                      right['layout'], left['layout'])
            if self.state_log.empty():
                self.initializeStateLog(right, left)
            self.improveMaxGapFurthermore()
            return False
        # return True, which means that there is some subgroup to be improved (close its gaps)
        return True

    def improveMaxGapFurthermore(self):
        print(self.state_log.df)
        method = 'reduce-max'
        how = 'increment'
        right, left = self.state_log.getMaxGapLayouts()
        max_gap = abs(self.state_log.getRealCoverage(right) - self.state_log.getRealCoverage(left))
        print(f'[DEBUG]: >>>>>>>>>> current max-gap: {max_gap} <<<<<<<<<<')
        base_layout = right
        last_record = self.state_log.getLastRecord()
        if last_record['scan_method'] == 'reduce-max' and abs(last_record['expected_real_coverage'] - last_record['real_coverage']) >= max_gap:
            factor = last_record['scan_factor'] * 2
        else:
            factor = self.state_log.getLayoutScanFactor(base_layout) + 1
        expected_real_coverage = (self.state_log.getRealCoverage(right) + self.state_log.getRealCoverage(left)) / 2
        pages, pebs_coverage = self.addCutComplementPages(left, base_layout, factor)
        if pages is None:
            desired_real_coverage = (self.state_log.getRealCoverage(right) + self.state_log.getRealCoverage(left)) / 2
            pages, pebs_coverage = self.removeTailPagesBasedOnRealCoverage(
                    left, None, desired_real_coverage)
            how = f'decrement ({left})'
        assert pages is not None
        LayoutGeneratorUtils.writeLayout(self.layout, pages, self.exp_dir)
        increment_value = pebs_coverage - self.state_log.getPebsCoverage(right)
        self.state_log.addRecord(self.layout, method, how,
                                 increment_value, factor, base_layout,
                                 pebs_coverage, expected_real_coverage,
                                 base_layout, pages, 0)
        # decrease current group's budget by 1
        self.subgroups_log.decreaseRemainingBudget(
            self.state_log.getLeftLayoutName())

    def initializeStateLog(self, right_layout, left_layout):
        # if the state was not created yet then create it and add all
        # layouts that in the range [left_layout - right_layout]
        if self.state_log.empty():
            # if the state was not created before then this layout should
            # have a full budget (its budget should still unused)
            #assert(left_layout['remaining_budget'] == left_layout['total_budget'])
            state_layouts = results_df.query(
                'walk_cycles >= {left} and walk_cycles <= {right}'.format(
                    left=left_layout['walk_cycles'],
                    right=right_layout['walk_cycles']))
            state_layouts.sort_values('walk_cycles', ascending=False)
            #for layout_name in [right_layout['layout'], left_layout['layout']]:
            for index, row in state_layouts.iterrows():
                layout_name = row['layout']
                pages, offset = LayoutGeneratorUtils.getLayoutHugepages(
                    layout_name, self.exp_dir)
                pebs_coverage = LayoutGeneratorUtils.calculateTlbCoverage(
                    self.pebs_df, pages)
                base = 'other'
                if layout_name == self.state_log.getRightLayoutName() or layout_name == self.state_log.getLeftLayoutName():
                    base = 'none'
                self.state_log.addRecord(layout_name,
                                         'none', 'none', -1, 1, base,
                                         pebs_coverage, -1, 'none',
                                         pages, offset)
            self.state_log.writeLog()
            self.state_log.writeRealCoverage()

    def handleOverMaxPebsCoverageCornerCase(self):
        last_layout = self.state_log.getLastLayoutName()
        last_layout_base = self.state_log.getBaseLayout(last_layout)
        right, left = self.state_log.getMaxGapLayouts()
        base_layout = right
        factor = 2
        if base_layout == last_layout_base:
            factor = self.state_log.getLayoutScanFactor(last_layout) + 1
            factor = max(factor, 2)
        pages, pebs_coverage = self.addCutComplementPages(left, base_layout, factor)
        return pages, pebs_coverage

    def addPages(self, base_layout, desired_coverage):
        if (desired_coverage + INCREMENT) >= 100:
            print('[WARNING]: trying to add pages to get pebs coverage >= 100%')
            print('[DEBUG]: move to remove pages from the left layout instead')
            return self.handleOverMaxPebsCoverageCornerCase()
        print('====== add tail pages based on pebs=====')
        pages, pebs_coverage = self.addPagesBasedOnPebsCoverage(base_layout, desired_coverage, tail=True)
        if pages is None:
            print('====== add tail pages based on real=====')
            desired_real_coverage = self.state_log.getRealCoverage(base_layout) + INCREMENT
            pages, pebs_coverage = self.addTailPagesBasedOnRealCoverage(base_layout, desired_real_coverage, tail=True)
        if pages is None:
            print('====== add head pages based on pebs=====')
            pages, pebs_coverage = self.addPagesBasedOnPebsCoverage(base_layout, desired_coverage, tail=False)
        if pages is None:
            print('====== add head pages based on real=====')
            pages, pebs_coverage = self.addTailPagesBasedOnRealCoverage(base_layout, desired_real_coverage, tail=False)
        print('===== DONE =====')
        return pages, pebs_coverage

    def getPebsCoverageBasedOnRealCoverage(self, base_layout, desired_real_coverage):
        base_layout_real_coverage = self.state_log.getRealCoverage(base_layout)
        base_layout_pebs_coverage = self.state_log.getPebsCoverage(base_layout)
        base_layout_real_to_pebs_scale = base_layout_pebs_coverage / base_layout_real_coverage
        scaled_desired_coverage = base_layout_real_to_pebs_scale * desired_real_coverage

        print('[DEBUG]: scaling real-coverage to pebs coverage')
        print(f'[DEBUG]: {base_layout} pebs coverage: {base_layout_pebs_coverage}')
        print(f'[DEBUG]: desired real coverage: {desired_real_coverage}')
        print(f'[DEBUG]: scaled desired pebs coverage: {scaled_desired_coverage}')

        return scaled_desired_coverage

    def addTailPagesBasedOnRealCoverage(self, base_layout, desired_real_coverage, tail=True):
        scaled_desired_coverage = self.getPebsCoverageBasedOnRealCoverage(base_layout, desired_real_coverage)
        print(f'[DEBUG]: trying to add tail pages to layout: {base_layout}')
        #assert scaled_desired_coverage > base_layout_pebs_coverage
        return self.addPagesBasedOnPebsCoverage(base_layout, scaled_desired_coverage, tail)

    def addPagesBasedOnPebsCoverage(self, base_layout, desired_coverage, tail=True):
        base_layout_pebs_coverage = self.state_log.getPebsCoverage(base_layout)

        pages_order='tail' if tail else 'head'
        print(f'[DEBUG]: addPagesBasedOnPebsCoverage: trying to add {pages_order} pages to {base_layout} to get a coverage of : {desired_coverage}')

        assert desired_coverage > base_layout_pebs_coverage

        base_pages, offset = LayoutGeneratorUtils.getLayoutHugepages(base_layout, self.exp_dir)
        df = self.pebs_df.query(f'PAGE_NUMBER not in {base_pages}')
        df = df.sort_values('TLB_COVERAGE', ascending=tail)

        assert len(df) > 0

        added_pages = []
        total_weight = base_layout_pebs_coverage
        epsilon = 0.2
        max_coverage = desired_coverage + epsilon
        min_coverage = desired_coverage
        for index, row in df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            updated_total_weight = total_weight + weight
            if updated_total_weight < max_coverage:
                added_pages.append(page)
                total_weight = updated_total_weight
            if max_coverage >= total_weight >= min_coverage:
                break
        if len(added_pages) == 0:
            return None, 0
        new_pages = base_pages + added_pages
        new_pages.sort()
        new_pebs_coverage = self.pebs_df.query(f'PAGE_NUMBER in {new_pages}')['TLB_COVERAGE'].sum()

        print(f'[DEBUG]: total added pages to {base_layout}: {len(added_pages)}')
        print(f'[DEBUG]: new layout coverage: {new_pebs_coverage}')

        return new_pages, new_pebs_coverage

    def _old_addTailPagesBasedOnPebsCoverage(self, base_layout, desired_coverage):
        # TODO get the base pages from the state log instead of reading them
        # get the base
        print('----------------------------------------------')
        print(f'[DEBUG]: trying to add tail pages to base layout: {base_layout}')
        print(f'[DEBUG]: desired tlb-coverage:{desired_coverage}')
        print('----------------------------------------------')
        df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=True)
        base_layout_pages, offset = LayoutGeneratorUtils.getLayoutHugepages(
            base_layout, self.exp_dir)
        new_layout_pages, actual_pebs_coverage = LayoutGeneratorUtils.findTlbCoverageWindows(
            df, desired_coverage, base_layout_pages)
        return new_layout_pages, actual_pebs_coverage

    def removeTailPagesBasedOnRealCoverage(self, layout, base_layout, desired_real_coverage):
        scaled_desired_coverage = self.getPebsCoverageBasedOnRealCoverage(layout, desired_real_coverage)

        print(f'[DEBUG]: desired real coverage: {desired_real_coverage}')
        print(f'[DEBUG]: scaled desired pebs coverage: {scaled_desired_coverage}')

        return self.removeTailPagesBasedOnPebsCoverage(
            layout, base_layout, scaled_desired_coverage)

    def removeTailPagesBasedOnPebsCoverage(self, layout, base_layout, desired_coverage):
        pages, offset = LayoutGeneratorUtils.getLayoutHugepages(
            layout, self.exp_dir)
        layout_coverage = self.pebs_df.query(f'PAGE_NUMBER in {pages}')['TLB_COVERAGE'].sum()

        if base_layout is not None:
            base_pages, offset = LayoutGeneratorUtils.getLayoutHugepages(
                base_layout, self.exp_dir)
            base_layout_coverage = self.pebs_df.query(f'PAGE_NUMBER in {base_pages}')['TLB_COVERAGE'].sum()
            print(f'[DEBUG]: {base_layout} coverage: {base_layout_coverage}')
        else:
            base_pages = []
            base_layout_coverage =  None

        df = self.pebs_df.query(f'PAGE_NUMBER in {pages} and PAGE_NUMBER not in {base_pages}')
        df = df.sort_values('TLB_COVERAGE', ascending=True)

        assert layout_coverage > desired_coverage

        print(f'[DEBUG]: start removing tail pages from {layout}')
        print(f'[DEBUG]: {layout} coverage: {layout_coverage}')

        removed_pages = []
        total_weight = layout_coverage
        epsilon = 0.2
        max_coverage = desired_coverage
        min_coverage = desired_coverage - epsilon
        for index, row in df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            updated_total_weight = total_weight - weight
            if updated_total_weight > min_coverage:
                removed_pages.append(page)
                total_weight = updated_total_weight
            if max_coverage >= total_weight >= min_coverage:
                break
        if len(removed_pages) == 0:
            return None, 0
        df = self.pebs_df.query(f'PAGE_NUMBER in {pages} and PAGE_NUMBER not in {removed_pages}')
        new_pages = df['PAGE_NUMBER'].to_list()
        new_pages.sort()
        new_pebs_coverage = df['TLB_COVERAGE'].sum()

        print(f'[DEBUG]: total removed tail pages from the leftmost layout: {len(removed_pages)}')
        print(f'[DEBUG]: new layout coverage: {new_pebs_coverage}')

        return new_pages, new_pebs_coverage

    def addCutComplementPages(self, layout, base_layout, factor=2):
        assert factor >= 2
        factor = int(factor)
        pages, offset = LayoutGeneratorUtils.getLayoutHugepages(layout, self.exp_dir)
        base_pages, offset = LayoutGeneratorUtils.getLayoutHugepages(base_layout, self.exp_dir)
        pages_coverage = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, pages)
        base_coverage = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, base_pages)

        candidate_pages = list(set(pages) - set(base_pages))
        candidate_pages.sort()
        remove_pages = candidate_pages[::factor]
        for p in remove_pages:
            candidate_pages.remove(p)
        new_pages = base_pages + candidate_pages
        new_coverage = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, new_pages)
        return new_pages, new_coverage

    def removeTailPagesByFactor(self, layout, base_layout, factor=2):
        assert factor >= 2
        factor = int(factor)
        pages, offset = LayoutGeneratorUtils.getLayoutHugepages(layout, self.exp_dir)
        base_pages, offset = LayoutGeneratorUtils.getLayoutHugepages(base_layout, self.exp_dir)
        pages_coverage = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, pages)
        base_coverage = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, base_pages)

        candidates = self.pebs_df.query(f'PAGE_NUMBER in {pages} and PAGE_NUMBER not in {base_pages}')
        print(f'[DEBUG]: removing pages from {layout} while skipping {base_layout} pages with a factor={factor}')
        print(f'[DEBUG]: number of candidate pages to be removed: {len(candidates)} with total-coverage={candidates["TLB_COVERAGE"].sum()}')
        sorted_candidates = candidates.sort_values('TLB_COVERAGE')
        i = 0
        removed_pages = []
        for p in sorted_candidates['PAGE_NUMBER']:
            if (i % factor) != 0:
                removed_pages.append(p)
            i += 1

        df = self.pebs_df.query(f'PAGE_NUMBER in {pages} and PAGE_NUMBER not in {removed_pages}')

        next_coverage = df['TLB_COVERAGE'].sum()
        print('----------------------------------------------')
        print(f'[DEBUG]: removeTailPages: current layout coverage: {pages_coverage}')
        print(f'[DEBUG]: removeTailPages: next layout coverage: {next_coverage}')
        print(f'[DEBUG]: removeTailPages: base layout coverage: {base_coverage}')
        print(f'[DEBUG]: removeTailPages: total removed pages: {len(removed_pages)}')
        print('----------------------------------------------')
        #assert next_coverage < pages_coverage
        #assert next_coverage > base_coverage
        if next_coverage >= pages_coverage or next_coverage <= base_coverage:
            return None, 0

        next_pages = df['PAGE_NUMBER'].to_list()
        return next_pages, next_coverage

    def concludeCoverageForCommonBase(self, layout, desired_real_coverage):
        base_layout = self.state_log.getBaseLayout(layout)
        base = self.state_log.getRecord('layout', base_layout)
        inc_base_layout = self.state_log.getIncBaseLayout(layout)
        inc_base = self.state_log.getRecord('layout', inc_base_layout)
        df = self.state_log.df.query(f'scan_base == "{base_layout}" and increment_base == "{inc_base_layout}"')
        df = df.sort_values('real_coverage', ascending=True)
        low = high = None
        for index, row in df.iterrows():
            real_coverage = row['real_coverage']
            if real_coverage < desired_real_coverage:
                low = row
            if real_coverage > desired_real_coverage and high is None:
                high = row
                break
        if low is not None and low['scan_method'] == 'left-tail':
            low = None
        if high is not None and high['scan_method'] == 'left-tail':
            high = None
        if low is None and high is None:
            return None, None
        # there is only higher layout
        elif low is None and high is not None:
            if high['layout'] == layout:
                return None, None
            ratio = (high['real_coverage'] - desired_real_coverage) / (high['real_coverage'] - base['real_coverage'])
            pebs_delta = ratio * (high['pebs_coverage'] - base['pebs_coverage'])
            desired_coverage = high['pebs_coverage'] - pebs_delta
            factor = pebs_delta / INCREMENT
            return factor, desired_coverage
        # there is only lower layout
        elif low is not None and high is None:
            if low['layout'] == layout:
                return None, None
            base_layout = low['layout']
            distance = desired_real_coverage - low['real_coverage']
            factor = distance / INCREMENT + 1
            if low['scan_direction'] == 'increment':
                factor += low['scan_factor']
            desired_coverage = low['pebs_coverage'] + (INCREMENT * factor)
            return factor, desired_coverage
        # there are two layouts around the desired_real_coverage
        else:
            if high['pebs_coverage'] <= low['pebs_coverage']:
                return None, None
            ratio = (desired_real_coverage - low['real_coverage']) / (high['real_coverage'] - low['real_coverage'])
            pebs_delta = ratio * (high['pebs_coverage'] - low['pebs_coverage'])
            desired_coverage = low['pebs_coverage'] + pebs_delta
            factor = pebs_delta / INCREMENT
            base_layout = low['layout']
            return factor, desired_coverage

    def createNextLayoutDynamically(self):
        assert self.results_df is not None,'results file does not exist'
        # fill or update SubgroupsLog and StateLog
        if not self.updateLogs():
            return
        print('==============================================')
        print(self.state_log.df)
        print('----------------------------------------------')

        # initialize required values with None
        base_layout = desired_coverage = pages = pebs_coverage = how = expected_real_coverage = increment_base = factor = None
        method = 'right-tail'
        last_layout_method = None

        # is this the first layout to be generated for the current group
        if self.state_log.hasOnlyBaseLayouts():
            increment_base = base_layout = self.state_log.getRightLayoutName()
            desired_coverage = self.state_log.getPebsCoverage(base_layout) + INCREMENT
            next_layout, base_layout = self.state_log.getNextLayoutToIncrement(base_layout)
            factor = self.state_log.getLayoutScanFactor(base_layout)
            # next_layout could be different than the returned base_layout in case that
            # the next_layout is from a previous run of some previous subgroup, in this
            # case we will use it to calculate the next desired-coverage but will not
            # use it as our base_layout because we do not want to use its pages since
            # it was created by a different subgroup
            if next_layout != base_layout:
                increment_base = next_layout
                desired_coverage = None # will be updated below
            else:
                desired_coverage = self.state_log.getPebsCoverage(base_layout) + (INCREMENT * factor)
            how = 'increment'
        else: # this is not the first layout in the subgroup
            last_layout = self.state_log.getLastLayoutName()
            last_layout_base = self.state_log.getBaseLayout(last_layout)
            last_layout_inc_base = self.state_log.getIncBaseLayout(last_layout)
            last_layout_pebs = self.state_log.getPebsCoverage(last_layout)
            last_layout_direction = self.state_log.getField(last_layout, 'scan_direction')
            last_layout_method = self.state_log.getField(last_layout, 'scan_method')
            if last_layout_direction == 'increment':
                last_layout_factor = self.state_log.getLayoutScanFactor(last_layout)
            else:
                last_layout_factor = 1

            last_increment = self.state_log.getGapBetweenLastRecordAndBase()
            print(f'[DEBUG]: gap between {last_layout} real-coverage and its increment-base real-coverage is: {last_increment}')

            base_layout = last_layout_base
            increment_base = last_layout_inc_base

            next_layout, base_layout = self.state_log.getNextLayoutToIncrement(self.state_log.getRightLayoutName())
            assert base_layout is not None
            if base_layout is None:
                base_layout = self.state_log.getRightLayoutName()
                deisred_coverage = self.state_log.getPebsCoverage(self.state_log.getLeftLayoutName())
            elif next_layout != last_layout_inc_base and next_layout != last_layout:
                print(f'[DEBUG]: moving to new base to increment: {next_layout} and base_layot: {base_layout}')
                increment_base = next_layout
                how = 'increment'
                factor = self.state_log.getLayoutScanFactor(base_layout)
                if next_layout != base_layout:
                    desired_coverage = None # will be updated below
                else:
                    desired_coverage = self.state_log.getPebsCoverage(base_layout) + (INCREMENT * factor)
            elif self.state_log.isLastLayoutIrregular(last_layout) is not None:
                desired_coverage = self.state_log.isLastLayoutIrregular(last_layout)
                how = last_layout_direction
                factor = last_layout_factor
            elif last_increment <= 0:
                print(f'[DEBUG]: {last_layout} got a lower real-coverage than expected.')
                base_layout = last_layout_base
                increment_base = last_layout_inc_base
                expected_real_coverage = self.state_log.getExpectedRealCoverage(last_layout)
                factor, desired_coverage = self.concludeCoverageForCommonBase(last_layout, expected_real_coverage)
                if factor is None:
                    factor = (-last_increment) / INCREMENT + 1
                    if last_layout_direction == 'increment':
                        factor += last_layout_factor
                    desired_coverage = last_layout_pebs + (INCREMENT * factor)
                how = 'increment'
            # last laout was incremented by < MAX_GAP%
            # there are two cases here: less than LOW_GAP% or btween LOW_GAP% and MAX_GAP%
            elif last_increment <= MAX_GAP:
                increment_base = next_layout
                how = 'increment'
                if last_layout_direction != 'increment':
                    factor = 1
                elif last_increment < LOW_GAP:
                    #scale = self.state_log.df['pebs_coverage'].mean() / self.state_log.df['real_coverage'].mean()
                    scale = INCREMENT / last_increment
                    scale = min(scale, 2)
                    factor = (last_layout_factor - 1) + scale
                    print(f'[DEBUG]: last layout closed a small gap (less than {LOW_GAP}%) --> scaling increment value by: {scale}')
                else:
                    factor = last_layout_factor
                desired_coverage = last_layout_pebs + (INCREMENT * factor)
            else:
            # last layout was incremented by > MAX_GAP%
                increment_base = last_layout_inc_base
                last_layout_factor = self.state_log.getLayoutScanFactor(last_layout)
                base_layout = last_layout_base
                base_layout_pebs = self.state_log.getPebsCoverage(base_layout)
                if last_layout_method == 'right-tail':
                    expected_real_coverage = self.state_log.getExpectedRealCoverage(last_layout)
                    #increment_value = self.state_log.getField(last_layout, 'scan_value')
                    #factor = last_layout_factor * (INCREMENT / last_increment)
                    #factor = last_layout_factor / 2
                    factor = last_layout_factor / max(2, (last_increment / INCREMENT))
                    desired_coverage = base_layout_pebs + (INCREMENT * factor)
                    pages, pebs_coverage = self.addPages(base_layout, desired_coverage)
                    method = 'right-tail'
                    how = 'increment'
                    # if add tail pages does not work then try to remove tail
                    if pages is None:
                        last_layout_real_coverage = self.state_log.getRealCoverage(last_layout)
                        base_layout_real_coverage = self.state_log.getRealCoverage(base_layout)
                        pebs_diff = last_layout_pebs - base_layout_pebs
                        real_diff = last_layout_real_coverage - base_layout_real_coverage
                        pebs_to_real_coverage_ratio = pebs_diff / real_diff
                        expected_real_diff = self.state_log.getExpectedRealCoverage(last_layout) - self.state_log.getRealCoverage(base_layout)
                        desired_coverage = base_layout_pebs + (expected_real_diff * pebs_to_real_coverage_ratio)
                        #desired_coverage = pebs_to_real_coverage_ratio * self.state_log.getExpectedRealCoverage(last_layout)
                        print(f'[DEBUG]: starting to remove tail pages from: {last_layout}')
                        print(f'[DEBUG]: trying to reduce coverage from: {last_layout_pebs} to: {desired_coverage}')
                        factor = (last_layout_pebs - desired_coverage) / last_layout_pebs
                        method = 'right-tail'
                        how = f'decrement ({last_layout})'
                        pages, pebs_coverage = self.removeTailPagesBasedOnPebsCoverage(
                                last_layout, base_layout, desired_coverage)
                if last_layout_method == 'left-tail':
                    factor = (last_layout_factor - 1) + last_increment / INCREMENT
                    desired_coverage = last_layout_pebs - (INCREMENT * factor)
                    pages, pebs_coverage = self.removeTailPagesBasedOnPebsCoverage(
                            last_layout, None, desired_coverage)
                    method = 'left-tail'
                    how = f'decrement ({last_layout})'

        if expected_real_coverage is None:
            expected_real_coverage = self.state_log.getRealCoverage(increment_base) + INCREMENT
        if factor is None:
            factor = self.state_log.getLayoutScanFactor(base_layout)
        if desired_coverage is None:
            desired_coverage = self.getPebsCoverageBasedOnRealCoverage(base_layout, expected_real_coverage)

        if (desired_coverage + INCREMENT) >= 100:
            pages, pebs_coverage = self.handleOverMaxPebsCoverageCornerCase()
            how = 'increment'
        elif how == 'increment' and pages is None:
            pages, pebs_coverage = self.addPages(base_layout, desired_coverage)

        # if all previous methods did not work, then try to remove tail pages from the leftmost layout
        if pages is None:
            print('[DEBUG]: cannot add pages, trying removing tail pages from the left layout...')
            desired_real_coverage = self.state_log.getRealCoverage(base_layout) + INCREMENT
            leftmost = self.state_log.getLeftLayoutName()
            pages, pebs_coverage = self.removeTailPagesBasedOnRealCoverage(
                    leftmost, None, desired_real_coverage)
            factor = 1
            method = 'left-tail'
            how = f'decrement ({leftmost})'

        assert base_layout is not None
        assert desired_coverage is not None
        assert pages is not None
        assert pebs_coverage is not None
        assert how is not None
        assert increment_base is not None
        if last_layout_method is not None and last_layout_method == 'left-tail':
            increment_value = -INCREMENT * factor
        else:
            increment_value = pebs_coverage - self.state_log.getPebsCoverage(base_layout)
        self.state_log.addRecord(self.layout, method, how,
                                 increment_value, factor, base_layout,
                                 pebs_coverage, expected_real_coverage,
                                 increment_base, pages, 0)
        print('----------------------------------------------')
        print(self.state_log.df)
        print('==============================================')
        print(self.layout)
        print('#hugepages: '+ str(len(pages)))
        print('weight: ' + str(pebs_coverage))
        print('==============================================')
        # write the layout configuration file
        LayoutGeneratorUtils.writeLayout(self.layout, pages, self.exp_dir)
        # decrease current group's budget by 1
        self.subgroups_log.decreaseRemainingBudget(
            self.state_log.getLeftLayoutName())

class LayoutGeneratorUtils(metaclass=Singleton):
    HUGE_PAGE_2MB_SIZE = 2097152
    BASE_PAGE_4KB_SIZE = 4096

    def __init__(self):
        pass

    def loadDataframe(results_file):
        if not os.path.isfile(results_file):
            return None
        results_ps = PerformanceStatistics(results_file)
        results_df = results_ps.getDataFrame()
        results_df['cpu-cycles'] = results_ps.getRuntime()
        results_df['walk_cycles'] = results_ps.getWalkDuration()
        results_df['stlb_hits'] = results_ps.getStlbHits()
        results_df['stlb_misses'] = results_ps.getStlbMisses()
        df = results_df[['layout', 'walk_cycles', 'stlb_hits', 'stlb_misses', 'cpu-cycles']]
        # drop duplicated rows
        important_columns = list(df.columns)
        important_columns.remove('layout')
        #df.drop_duplicates(inplace=True, subset=important_columns)
        df = df.drop_duplicates(subset=important_columns)
        return df

    def findTlbCoverageWindows(
            pebs_df, tlb_coverage_percentage, base_pages=[]):

        # start from the given base layout
        windows = base_pages.copy()
        total_weight = LayoutGeneratorUtils.calculateTlbCoverage(pebs_df, windows)
        assert tlb_coverage_percentage >= total_weight,'findTlbCoverageWindows: the required tlb-coverage is less than base pages coverage'
        remainder_coverage = tlb_coverage_percentage - total_weight

        max_epsilon = 0.5
        # filter-out pages that are in the base-pages
        df = pebs_df.query(
            'TLB_COVERAGE <= {target_coverage} and PAGE_NUMBER not in {pages}'.format(
                target_coverage=remainder_coverage+max_epsilon,
                pages=windows))

        epsilon = 0.1
        pages = None
        coverage = 0
        while pages is None:
            pages, coverage = LayoutGeneratorUtils.__findTlbCoverageWindows(
                df, remainder_coverage, epsilon)
            epsilon += 0.1
            if epsilon > max_epsilon:
                break
        if pages is None:
            return None, 0
        windows += pages
        total_weight += coverage
        return windows, total_weight

    def __findTlbCoverageWindows(
            df, tlb_coverage_percentage, epsilon):
        total_weight = 0
        windows = []
        for index, row in df.iterrows():
            weight = row['TLB_COVERAGE']
            page_number = row['PAGE_NUMBER']
            if (total_weight + weight) <= (tlb_coverage_percentage + epsilon):
                total_weight += weight
                windows.append(page_number)
            if total_weight >= (tlb_coverage_percentage - epsilon):
                break

        if total_weight > (tlb_coverage_percentage + epsilon) \
                or total_weight < (tlb_coverage_percentage - epsilon):
            return None, 0
        return windows, total_weight

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
                type=configuration.TYPE_BRK,
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

    def getLayoutHugepages(layout_name, exp_dir):
        page_size = LayoutGeneratorUtils.HUGE_PAGE_2MB_SIZE
        layout_file = str.format('{exp_root}/layouts/{layout_name}.csv',
                exp_root=exp_dir,
                layout_name=layout_name)
        df = pd.read_csv(layout_file)
        df = df[df['type'] == 'brk']
        df = df[df['pageSize'] == page_size]
        pages = []
        offset_deviation = 0
        for index, row in df.iterrows():
            start_page = int(row['startOffset'] / page_size)
            end_page = int(row['endOffset'] / page_size)
            offset_deviation = int(row['startOffset'] % page_size)
            pages += list(range(start_page, end_page))
        start_deviation = offset_deviation / LayoutGeneratorUtils.BASE_PAGE_4KB_SIZE
        return pages, start_deviation

    def calculateTlbCoverage(pebs_df, pages):
        selected_pages = pebs_df.query(
                'PAGE_NUMBER in {pages}'.format(pages=pages))
        return selected_pages['TLB_COVERAGE'].sum()

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
    parser.add_argument('-l', '--layout', required=True)
    parser.add_argument('-d', '--exp_dir', required=True)
    parser.add_argument('-n', '--results_file', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArguments()

    # read memory-footprints
    footprint_df = pd.read_csv(args.memory_footprint)
    mmap_footprint = footprint_df['anon-mmap-max'][0]
    brk_footprint = footprint_df['brk-max'][0]

    results_df = LayoutGeneratorUtils.loadDataframe(args.results_file)

    pebs_df = LayoutGeneratorUtils.normalizePebsAccesses(args.pebs_mem_bins)

    layout_generator = LayoutGenerator(pebs_df, results_df, args.layout, args.exp_dir)
    layout_generator.generateLayout()
