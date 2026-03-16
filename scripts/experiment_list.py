#! /usr/bin/env python3

import pandas as pd
from experiment import Experiment


class ExperimentList:
    def __init__(self, layouts, experiments_root):
        if not layouts:
            raise ValueError('layouts is empty')

        self._experiments = [
            Experiment(layout, experiments_root)
            for layout in layouts
        ]
        self._index_label = 'layout'

    def collect(self, repeat, I_start=None, I_end=None):
        dataframe_list = []
        if (I_start is None) != (I_end is None):
            raise ValueError('Both I_start and I_end must be provided together')

        for experiment in self._experiments:
            try:
                df = experiment.collect_interval(repeat, I_start=I_start, I_end=I_end)
            except Exception:
                raise ValueError('Could not collect the results of ' + str(experiment))

            df.index = [experiment._layout]
            dataframe_list.append(df)

        df = pd.concat(dataframe_list, sort=False)
        df.index.name = self._index_label
        return df
