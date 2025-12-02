#!/usr/bin/env python3
import os
import pandas as pd
from ast import literal_eval
import os.path

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
                 max_gap, max_budget, dry_run,
                 default_columns, converters=None):
        self.exp_dir = exp_dir
        self.results_df = results_df
        self.log_file = self.exp_dir + '/' + log_name
        self.max_gap = max_gap
        self.max_budget = max_budget
        self.default_columns = default_columns
        self.df = self.readLog(converters)
        self.dry_run = dry_run

    def readLog(self, converters=None):
        if not os.path.isfile(self.log_file):
            self.df = pd.DataFrame(columns=self.default_columns)
        else:
            self.df = pd.read_csv(self.log_file, converters=converters)
        return self.df

    def writeLog(self):
        if not self.dry_run:
            self.df.to_csv(self.log_file, index=False)

    def clear(self):
        if not self.dry_run:
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

    def layoutExist(self, layout):
        return len(self.df.query(f'layout == "{layout}"')) > 0

    def getRealCoverage(self, layout):
        return self.getField(layout, 'real_coverage')

    def getExpectedRealCoverage(self, layout):
        return self.getField(layout, 'expected_real_coverage')

    def getPebsCoverage(self, layout):
        return self.getField(layout, 'pebs_coverage')

    def getLastRecord(self):
        if self.empty():
            return None

        df = self.df.query('scan_base != "other"')
        if len(df) == 0:
            return None
        return df.iloc[-1]

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
    def __init__(self, exp_dir, results_df, max_gap, max_budget, dry_run):
        default_columns = [
            'layout', 'total_budget', 'remaining_budget',
            'pebs_coverage', 'real_coverage', 'walk_cycles']
        super().__init__(exp_dir, results_df, 'subgroups.log', max_gap, max_budget, dry_run, default_columns)

    def addRecord(self,
                  layout, pebs_coverage, writeLog=False):
        new_row = pd.Series({
            'layout': layout,
            'total_budget': -1,
            'remaining_budget': -1,
            'pebs_coverage': pebs_coverage,
            'real_coverage': -1,
            'walk_cycles': -1 })
        self.df = pd.concat([self.df, new_row.to_frame().T], ignore_index=True)
        if writeLog:
            self.writeLog()

    def getSubgroup(self, num):
        right = self.df.iloc[num]
        left = self.df.iloc[num+1]
        return right, left

    def sortByRealCoverage(self):
        #self.df = self.df.sort_values('real_coverage', ascending=True)
        self.df = self.df.sort_values('walk_cycles', ascending=False)

    def getExtraBudget(self):
        return self.max_budget - (self.getTotalBudget() + len(self.df))

    def calculateBudget(self):
        query = self.df.query('real_coverage == (-1)')
        assert len(query) == 0, 'SubgroupsLog.calculateBudget was called before updating the subgroups real_coverage.'
        query = self.df.query('total_budget < 0')
        if len(query) == 0:
            return
        # sort the group layouts by walk-cycles/real_coverage
        self.sortByRealCoverage()
        # calculate the diff between each two adjacent layouts
        # (call it delta[i] for the diff between group[i] and group[i+1])
        self.df['delta'] = self.df['real_coverage'].diff().abs()
        self.df['delta'] = self.df['delta'].fillna(0)
        total_deltas = self.df.query(f'delta > {self.max_gap}')['delta'].sum()
        # budgest = 50-9: num_layouts(50) - subgroups_layouts(9)
        total_budgets = self.max_budget - len(self.df)
        for index, row in self.df.iterrows():
            delta = row['delta']
            # for each delta < self.max_gap assign budget=0
            if delta <= self.max_gap:
                budget = 0
            else:
                budget = int((delta / total_deltas) * total_budgets)
                budget = max(budget, int(delta / 3.5))
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

    def zeroAllBudgets(self):
        remaining = 0
        for index, row in self.df.iterrows():
            layout = row['layout']
            remaining += self.zeroBudget(layout)
        return remaining

    def zeroBudget(self, layout):
        total = self.getField(layout, 'total_budget')
        remaining = self.getField(layout, 'remaining_budget')
        self.df.loc[self.df['layout'] == layout, 'total_budget'] = total - remaining
        self.df.loc[self.df['layout'] == layout, 'remaining_budget'] = 0
        self.writeLog()
        return remaining

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

    def getRemainingBudget(self, left_layout):
        return self.getField(left_layout, 'remaining_budget')

    def getTotalRemainingBudget(self):
        return self.df['remaining_budget'].sum()

    def getTotalBudget(self):
        return self.df['total_budget'].sum()


class StateLog(Log):
    def __init__(self, exp_dir, results_df, right_layout, left_layout, max_gap, max_budget, dry_run):
        default_columns = [
            'layout', 'scan_base', 'increment_base',
            'scan_direction', 'scan_order', 'scan_value',
            'pebs_coverage', 'increment_real_coverage',
            'expected_real_coverage', 'real_coverage',
            'walk_cycles']
        self.right_layout = right_layout
        self.left_layout = left_layout
        state_name = right_layout + '_' + left_layout
        super().__init__(exp_dir, results_df,
                         state_name + '_state.log',
                         max_gap, max_budget, dry_run,
                         default_columns)
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
                  scan_direction,
                  scan_order,
                  scan_value, scan_base,
                  pebs_coverage, expected_real_coverage, increment_base,
                  pages,
                  writeLog=True):
        base_pages = []
        if scan_base != 'none' and scan_base != 'other':
            base_pages = self.getLayoutPages(scan_base)
        added_pages = list(set(pages) - set(base_pages))
        added_pages.sort()
        new_row = pd.Series({
            'layout': layout,
            'scan_direction': scan_direction,
            'scan_order': scan_order,
            'scan_value': scan_value,
            'scan_base': scan_base,
            'pebs_coverage': pebs_coverage,
            'expected_real_coverage': expected_real_coverage,
            'increment_base': increment_base,
            'increment_real_coverage': self.getRealCoverage(increment_base),
            'real_coverage': -1,
            'walk_cycles': -1
            })
        self.df = pd.concat([self.df, new_row.to_frame().T], ignore_index=True)
        if writeLog:
            self.writeLog()
        if layout not in self.pages_df['layout']:
            new_row = pd.Series({
                'layout': layout,
                'base_layout': scan_base,
                'added_pages': added_pages,
                'pages': pages
                })
            self.pages_df = pd.concat([self.pages_df, new_row.to_frame().T], ignore_index=True)
            if not self.dry_run:
                self.pages_df.to_csv(self.pages_log_name, index=False)

    def getLayoutPages(self, layout):
        pages = self.pages_df.loc[self.pages_df['layout'] == layout, 'pages'].iloc[0]
        return pages

    def getLayoutAddedPages(self, layout):
        return self.getField(layout, 'added_pages')

    def hasOnlyBaseLayouts(self):
        df = self.df.query(f'scan_base != "none" and scan_base != "other"')
        return len(df) == 0

    def hasOnlyOneNewLayout(self):
        df = self.df.query(f'scan_base != "none" and scan_base != "other"')
        return len(df) == 1

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

    def getPebsCoverageDeltaBetweenLayoutAndItsBase(self, layout):
        base_layout = self.getBaseLayout(layout)
        if base_layout is None or base_layout == 'none':
            return None

        layout_coverage = self.getPebsCoverage(layout)
        assert(layout_coverage is not None)
        base_coverage = self.getPebsCoverage(base_layout)
        assert(base_coverage is not None)

        delta = layout_coverage - base_coverage
        return delta

    def getGapBetweenLayoutAndItsBase(self, layout):
        base_layout = self.getBaseLayout(layout)
        if base_layout is None or base_layout == 'none':
            return None
        return self.getGapFromBase(layout, base_layout)

    def getGapFromBase(self, layout, base_layout):
        layout_coverage = self.getRealCoverage(layout)
        assert(layout_coverage is not None)
        base_coverage = self.getRealCoverage(base_layout)
        assert(base_coverage is not None)

        gap = layout_coverage - base_coverage
        print(f'{layout} real-coverage: {layout_coverage} , {base_layout} real-coverage: {base_coverage} ==> gap: {gap}')
        return gap

    def getGapBetweenLastRecordAndIncrementBase(self):
        #self.writeRealCoverage()
        last_layout = self.getLastRecord()
        base_layout = last_layout['increment_base']
        return self.getGapFromBase(last_layout['layout'], base_layout)

    def getBaseLayout(self, layout_name):
        return self.getField(layout_name, 'scan_base')

    def getIncBaseLayout(self, layout_name):
        return self.getField(layout_name, 'increment_base')

    def getLayoutScanOrder(self, layout_name):
        return self.getField(layout_name, 'scan_order')

    def getLayoutScanDirection(self, layout_name):
        return self.getField(layout_name, 'scan_direction')

    def getLayoutScanValue(self, layout_name):
        return self.getField(layout_name, 'scan_value')

    def getNextBaseLayout(self, scan_direction, scan_order):
        start_layout = self.getRightLayoutName()
        start_layout_coverage = self.getRealCoverage(start_layout)
        max_coverage = self.getRealCoverage(self.getLeftLayoutName())
        increment_base = self.getNextIncrementBase()
        if increment_base is None:
            return None
        increment_layout_coverage = self.getRealCoverage(increment_base)

        df = self.df.query(f'real_coverage >= {start_layout_coverage} and real_coverage <= {increment_layout_coverage}')
        df = df.query(f'scan_base == "none" or (scan_direction == "{scan_direction}" and scan_order == "{scan_order}")')
        df = df.sort_values('real_coverage', ascending=True)
        assert len(df) > 0
        return df.iloc[-1]['layout']

    def getNextIncrementBase(self):
        start_layout = self.getRightLayoutName()
        start_layout_coverage = self.getRealCoverage(start_layout)
        max_coverage = self.getRealCoverage(self.getLeftLayoutName())
        df = self.df.query(f'real_coverage >= {start_layout_coverage}')
        df = df.sort_values('real_coverage', ascending=True)
        current_coverage = start_layout_coverage
        current_layout = start_layout
        for index, row in df.iterrows():
            if row['real_coverage'] <= (current_coverage + self.max_gap):
                current_layout = row['layout']
                current_coverage = row['real_coverage']
                if current_coverage >= max_coverage:
                    return None
            else:
                break
        return current_layout

    def getNextExpectedRealCoverage(self):
        expected_increment = (7.0/8.0) * self.max_gap

        next_increment = self.getNextIncrementBase()
        inc_real_coverage = self.getRealCoverage(next_increment)
        max_expected_real = inc_real_coverage + 2 * expected_increment

        df = self.df.query(f'{inc_real_coverage} < real_coverage <= {max_expected_real}')

        if len(df) == 0:
            return inc_real_coverage + expected_increment

        df = df.sort_values('real_coverage')
        upper_real_coverage = df.iloc[0]['real_coverage']
        avg_real_coverage = (inc_real_coverage + upper_real_coverage) / 2
        return avg_real_coverage

    def getMaxGapLayouts(self, include_other_layouts=True):
        left_coverage = self.getRealCoverage(self.getLeftLayoutName())
        right_coverage = self.getRealCoverage(self.getRightLayoutName())
        if include_other_layouts:
            query = self.df.query(f'{right_coverage} <= real_coverage <= {left_coverage}')
        else:
            query = self.df.query(f'{right_coverage} <= real_coverage <= {left_coverage} and scan_base != "other"')
        diffs = query.sort_values('real_coverage', ascending=True)
        diffs['diff'] = diffs['real_coverage'].diff().abs()

        idx_label = diffs['diff'].idxmax()
        idx = diffs.index.get_loc(idx_label)
        right = diffs.iloc[idx-1]
        left = diffs.iloc[idx]
        return right['layout'], left['layout']

    def getMaxGap(self):
        right, left = self.getMaxGapLayouts()
        max_gap = abs(self.getRealCoverage(left) - self.getRealCoverage(right))
        print(f'=========> the maximal gap was found between: {right} - {left}, which is: {max_gap:.2f} <=========')
        return max_gap

    def getAllLayouts(self):
        return self.df['layout'].to_list()
