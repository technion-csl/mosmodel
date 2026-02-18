#! /usr/bin/env python3

import pandas as pd
from experiment import Experiment

class ExperimentList:
    def __init__(self, layouts, experiments_root):
        if (not layouts):
            raise ValueError('layouts is empty')

        self._experiments = [
                Experiment(layout, experiments_root)
                for layout in layouts]
        self._index_label = 'layout'

    def collect(self, repeat, instruction_count=None):
        dataframe_list = []
        for experiment in self._experiments:
            try:
                if instruction_count is None:
                    df = experiment.collect(repeat)
                else:
                    df = experiment.collect_interpolated(repeat, instruction_count)
            except Exception:
                raise ValueError('Could not collect the results of ' + str(experiment))
                
            df.index = [experiment._layout]
            dataframe_list.append(df)
            
        df = pd.concat(dataframe_list, sort=False)
        df.index.name = self._index_label
        return df

