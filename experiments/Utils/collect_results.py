import pandas as pd
import numpy as np
import os
import logging
import datetime
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from Utils.utils import Utils

class OutlierDetector:
    def __init__(self, method=None, threshold=3, factor=1.5, contamination=0.1):
        if method is None:
            self.method = 'std'
        else:
            self.method = method
        self.threshold = threshold
        self.factor = factor
        self.contamination = contamination

    def detect_outliers(self, df):
        if self.method == 'zscore':
            return self._detect_outliers_zscore(df)
        elif self.method == 'iqr':
            return self._detect_outliers_iqr(df)
        elif self.method == 'isolation_forest':
            return self._detect_outliers_isolation_forest(df)
        elif self.method == 'lof':
            return self._detect_outliers_lof(df)
        elif self.method == 'std':
            return self._detect_outliers_std(df)
        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")

    def _detect_outliers_zscore(self, df):
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = (z_scores > self.threshold).any(axis=1)
        return outliers

    def _detect_outliers_iqr(self, df):
        Q1 = df.groupby(df.index).quantile(0.25)
        Q3 = df.groupby(df.index).quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df.groupby(df.index).median() < (Q1 - factor * IQR)) | (df.groupby(df.index).median() > (Q3 + factor * IQR))).any(axis=1)
        return outliers

    def _detect_outliers_isolation_forest(self, df):
        clf = IsolationForest(contamination=self.contamination, random_state=42)
        clf.fit(df)
        outliers = clf.predict(df) == -1
        return outliers

    def _detect_outliers_lof(self, df):
        clf = LocalOutlierFactor(n_neighbors=20, contamination=self.contamination)
        outliers = clf.fit_predict(df) == -1
        return outliers

    def _detect_outliers_std(self, df):
        # mean_df = df.groupby(df.index).mean()
        median_df = df.groupby(df.index).median()
        std_df = df.groupby(df.index).std()
        variation = std_df / median_df
        outlier_threshold = 0.05
        outliers = (variation > outlier_threshold).any(axis=1)
        return outliers

class CollectResults:
    def __init__(self, experiments_root, output_dir, num_repeats, outlier_method=None) -> None:
        self.experiments_root = experiments_root
        self.output_dir = output_dir
        self.num_repeats = num_repeats
        self.outlier_detector = OutlierDetector(method=outlier_method)
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    class Experiment:
        def __init__(self, layout, experiments_root):
            self._layout = layout
            self._experiments_root = experiments_root

        @staticmethod
        def __readSingleFile(file_name, metrics_column=0, stats_column=1):
            try:
                metrics, stats = np.loadtxt(file_name, delimiter=',', dtype=str, unpack=True, usecols=[metrics_column, stats_column])
                df = pd.DataFrame({'stats': stats}, index=metrics)
                df['stats'] = pd.to_numeric(df['stats'], errors='coerce')
            except IOError:
                return None
            except Exception as e:
                raise ValueError(f'Could not read the CSV file: {file_name}, Error: {e}')
            return df

        def collect(self, repeat):
            experiment_dir = f"{self._experiments_root}/{self._layout}/repeat{repeat}"
            perf_file_name = f"{experiment_dir}/perf.out"
            perf_df = CollectResults.Experiment.__readSingleFile(perf_file_name, metrics_column=2, stats_column=0)
            time_file_name = f"{experiment_dir}/time.out"
            time_df = CollectResults.Experiment.__readSingleFile(time_file_name)
            df = pd.concat([perf_df, time_df])
            df = df.transpose()
            return df

        def __repr__(self):
            return f'experiment with {str(self.__dict__)}'

    class ExperimentList:
        def __init__(self, layouts, experiments_root):
            if not layouts:
                raise ValueError('layouts is empty')
            self._experiments = [CollectResults.Experiment(layout, experiments_root) for layout in layouts]
            self._index_label = 'layout'

        def collect(self, repeat):
            dataframe_list = []
            for experiment in self._experiments:
                try:
                    df = experiment.collect(repeat)
                except Exception as e:
                    raise ValueError(f'Could not collect the results of {str(experiment)}. Error: {e}')
                df.index = [experiment._layout]
                dataframe_list.append(df)
            df = pd.concat(dataframe_list, sort=False)
            df.index.name = self._index_label
            return df

    @staticmethod
    def __writeDataframeToCsv(df, file_name):
        df.to_csv(file_name, na_rep='NaN')

    @staticmethod
    def __getLayouts(experiments_root):
        layout_list = []
        for f in os.scandir(experiments_root):
            if f.is_dir() and f.name.startswith('layout') and not f.name == 'layouts' and 'outlier' not in f.name:
                layout_list.append(f.name)
        return layout_list

    def __collectRawResults(self):
        layout_list = CollectResults.__getLayouts(self.experiments_root)
        if not layout_list:
            return None
        # Build the output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not self.output_dir.endswith('/'):
            self.output_dir += '/'

        # Collect the results, one dataframe for each repetition
        dataframe_list = []
        for repeat in range(1, self.num_repeats + 1):
            experiment_list = CollectResults.ExperimentList(layout_list, self.experiments_root)
            df = experiment_list.collect(repeat)
            csv_file_name = f'repeat{repeat}.csv'
            if len(layout_list) > 1:
                CollectResults.__writeDataframeToCsv(df, self.output_dir + csv_file_name)
            dataframe_list.append(df)

        df = pd.concat(dataframe_list)
        return df

    def __handleOutliers(self, df, remove_outliers, skip_outliers):
        found = False
        # Detect outliers
        interesting_metrics = ['seconds-elapsed', 'ref-cycles', 'cpu-cycles']
        interesting_metrics = [metric for metric in interesting_metrics if metric in df.columns]

        # Detect outliers using the specified method
        outliers = self.outlier_detector.detect_outliers(df[interesting_metrics])
        if outliers.any():
            found = True
            logging.warning(f"Error: the results in {self.experiments_root} showed considerable variation")
            logging.warning(outliers)
            if remove_outliers:
                now = str(datetime.datetime.now())[:19].replace(" ", "_").replace(":", "-")
                for layout in df[outliers].index.drop_duplicates():
                    l_old_path = f"{self.experiments_root}/{layout}"
                    l_new_path = f"{l_old_path}.outlier.{now}"
                    print(f'remove outlier: {l_old_path} --> {l_new_path}')
                    os.rename(l_old_path, l_new_path)
                print('The results with outliers have been removed, please try to run them again')
            elif not skip_outliers:
                raise ValueError('CollectResults: Cells marked with True are the outliers.')
        return found

    def collectResults(self, write_results=True, remove_outliers=True, skip_outliers=False):
        logging.debug(f'collecting results to the directory: {self.output_dir}')
        df = self.__collectRawResults()
        if df is None or df.empty:
            logging.debug('there is no results to collect, skipping...')
            return None, False

        mean_df = df.groupby(df.index).mean()
        median_df = df.groupby(df.index).median()
        std_df = df.groupby(df.index).std()

        found_outliers = self.__handleOutliers(df, remove_outliers, skip_outliers)

        if write_results:
            # If there are no outliers, write the aggregated results
            CollectResults.__writeDataframeToCsv(mean_df, self.output_dir + 'mean.csv')
            CollectResults.__writeDataframeToCsv(median_df, self.output_dir + 'median.csv')
            CollectResults.__writeDataframeToCsv(df, self.output_dir + 'all_repeats.csv')
            CollectResults.__writeDataframeToCsv(std_df, self.output_dir + 'std.csv')

        logging.info(f'** results of {len(median_df)} layouts were collected and written **')

        res_df = Utils.load_dataframe(median_df.reset_index())
        return res_df, found_outliers
