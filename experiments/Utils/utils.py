#! /usr/bin/env python3

import pandas as pd
import numpy as np
import os, sys
from Utils.ConfigurationFile import Configuration
from sklearn.utils import shuffle, resample
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import math

curr_file_dir = os.path.dirname(os.path.abspath(__file__))
experiments_root_dir = os.path.join(curr_file_dir, '..')
analysis_root_dir = os.path.join(experiments_root_dir, '../analysis')
sys.path.append(analysis_root_dir)
from performance_statistics import PerformanceStatistics

# TODO: go over all files that import Utils and modify them to use
# the Utils class
kb = 1024
mb = 1024*kb
gb = 1024*mb

def round_up(x, base):
    return int(base * math.ceil(x/base))

def round_down(x, base):
    return (int(x / base) * base)

def isPowerOfTwo(number):
    return (number != 0) and ((number & (number - 1)) == 0)

class Utils:
    KB = 1024
    MB = 1024*kb
    GB = 1024*mb

    def round_up(x, base):
        return int(base * math.ceil(x/base))

    def round_down(x, base):
        return (int(x / base) * base)

    def isPowerOfTwo(number):
        return (number != 0) and ((number & (number - 1)) == 0)

    def load_dataframe(perf_file, sort=False, drop_duplicates=True, shuffle=False):
        if type(perf_file) is str:
            df = pd.read_csv(perf_file)
        elif type(perf_file) is pd.DataFrame:
            df = perf_file
        else:
            assert False

        if 'walk_cycles' not in df.columns:
            ps = PerformanceStatistics(perf_file)
            df = ps.getDataFrame().reset_index()
            df['cpu_cycles'] = ps.getRuntime()
            df['stlb_misses'] = ps.getStlbMisses()
            df['stlb_hits'] = ps.getStlbHits()

            walk_cycles = ps.getWalkDuration()
            df['walk_cycles'] = walk_cycles

            walk_active = ps.getWalkActive()
            if walk_active is not None:
                df['walk_active'] = walk_active
            else:
                df['walk_active'] = walk_cycles
            walk_pending = ps.getWalkPending()
            if walk_pending is not None:
                df['walk_pending'] = walk_pending
            else:
                df['walk_pending'] = walk_cycles

            df = df[['layout', 'walk_cycles', 'stlb_misses', 'stlb_hits', 'cpu_cycles', 'walk_active', 'walk_pending']]

        if sort:
            df = df.sort_values('cpu_cycles').reset_index(drop=True)

        if drop_duplicates:
            cols = list(df.columns)
            if 'layout' in cols:
                cols.remove('layout')
            df = df.drop_duplicates(subset=cols)

        if shuffle:
            df = Utils.shuffle_dataframe(df)

        return df.reset_index(drop=True)
        # return df

    def _shuffle_dataframe(df):
        shuffled_df = shuffle(df, random_state=42).reset_index(drop=True)
        return shuffled_df

    def shuffle_dataframe(df):
        # return df
        return Utils._shuffle_dataframe(df)

    def load_pebs(pebs_mem_bins, normalize=True):
        # read mem-bins
        pebs_df = pd.read_csv(pebs_mem_bins, delimiter=',')

        if not normalize:
            pebs_df = pebs_df[['PAGE_NUMBER', 'NUM_ACCESSES']]
            pebs_df = pebs_df.reset_index()
            pebs_df = pebs_df.sort_values('NUM_ACCESSES', ascending=False)
            return pebs_df

        # filter and eep only brk pool accesses
        pebs_df = pebs_df[pebs_df['PAGE_TYPE'].str.contains('brk')]
        if pebs_df.empty:
            sys.exit('Input file does not contain page accesses information about the brk pool!')
        pebs_df = pebs_df[['PAGE_NUMBER', 'NUM_ACCESSES']]

        # transform NUM_ACCESSES from absolute number to percentage
        total_access = pebs_df['NUM_ACCESSES'].sum()
        pebs_df['TLB_COVERAGE'] = pebs_df['NUM_ACCESSES'].mul(100).divide(total_access)
        pebs_df = pebs_df.sort_values('TLB_COVERAGE', ascending=False)

        pebs_df = pebs_df.reset_index()

        return pebs_df

    def load_layout_hugepages(layout_name, exp_dir):
        hugepage_size = 1 << 21
        base_page_size = 1 << 12
        layout_file = str.format('{exp_root}/layouts/{layout_name}.csv',
                exp_root=exp_dir,
                layout_name=layout_name)
        df = pd.read_csv(layout_file)
        df = df[df['type'] == 'brk']
        df = df[df['pageSize'] == hugepage_size]
        pages = []
        offset = 0
        for index, row in df.iterrows():
            start_page = int(row['startOffset'] / hugepage_size)
            end_page = int(row['endOffset'] / hugepage_size)
            offset = int(row['startOffset'] % hugepage_size)
            pages += list(range(start_page, end_page))
        start_offset = offset / base_page_size
        return pages

    def write_layout(layout, pages, output, brk_footprint, mmap_footprint, sliding_index=0):
        hugepage_size= 1 << 21
        base_page_size = 1 << 12
        hugepages_start_offset = sliding_index * base_page_size
        brk_pool_size = Utils.round_up(brk_footprint, hugepage_size) + hugepages_start_offset
        configuration = Configuration()
        configuration.setPoolsSize(
                brk_size=brk_pool_size,
                file_size=1*Utils.GB,
                mmap_size=mmap_footprint)
        for p in pages:
            configuration.addWindow(
                    type=configuration.TYPE_BRK,
                    page_size=hugepage_size,
                    start_offset=(p * hugepage_size) + hugepages_start_offset,
                    end_offset=((p+1) * hugepage_size) + hugepages_start_offset)
        configuration.exportToCSV(output, layout)

    def format_large_number(num) -> str:
        factor, suffix = Utils.get_large_number_format_suffix(num)
        num_str = str(round(num/factor, 2))
        if suffix.lower == 'billions':
            suffix = 'B'
        elif suffix.lower == 'millions':
            suffix = 'M'
        else:
            suffix = suffix.capitalize()

        return f'{num_str} {suffix}'

    def get_large_number_format_suffix(num):
        suffix = ''
        factor = 1
        if 0.1*1e9 >= num >= 0.1*1e6:
            factor = 1e6
            suffix = 'millions'
        elif num > 0.1*1e9:
            factor = 1e9
            suffix = 'billions'
        return factor, suffix

    def get_format_suffix_for_large_number(dataframes, metric):
        max_val = 0
        min_val = None
        for df in dataframes:
            max_val = max(max_val, df[metric].max())
            if min_val is None:
                min_val = df[metric].min()
            min_val = min(min_val, df[metric].min())
        suffix = ''
        factor = 1
        if min_val > 0.1*1e6 and max_val < 1e9:
            factor = 1e6
            suffix = 'millions'
        elif min_val > 0.1*1e9:
            factor = 1e9
            suffix = 'billions'
        return factor, suffix

    def get_file_per_benchmark(root_dir, file_name, searched_benchmark=None):
        result = []
        for root, dirs, files in os.walk(root_dir):
            # read all relevant files under the results root directory
            if file_name not in files:
                continue
            # remove trailing / from the benchmark name
            if root.startswith(root_dir + '/'):
                benchmark = root.replace(root_dir + '/', '', 1)
            else:
                benchmark = root.replace(root_dir, '', 1)
            if searched_benchmark is not None and benchmark != searched_benchmark:
                continue
            # use short names for the benchmarks
            benchmark = Utils.shortenBenchmarkName(benchmark)
            file_path = os.path.join(root, file_name)
            result.append({'benchmark': benchmark, 'file': file_path})
        return result

    def shortenBenchmarkName(benchmark_name):
        benchmark_name = benchmark_name.replace('_cpu20', '')
        benchmark_name = benchmark_name.replace('my_gups', 'gups')
        benchmark_name = benchmark_name.replace('sequential-', '')
        benchmark_name = benchmark_name.replace('unionized-', '')
        benchmark_name = benchmark_name.replace('graph500-2.1', 'graph500')
        if '/' in benchmark_name and '.' in benchmark_name:
            benchmark_name = benchmark_name[:benchmark_name.find('/')+1] + benchmark_name[benchmark_name.find('.')+1:]
        return benchmark_name

    def shortenBenchmarkNameToMinimal(benchmark_name):
        benchmark_name = Utils.shortenBenchmarkName(benchmark_name)
        benchmark_name = benchmark_name.replace('gapbs/', '')
        benchmark_name = benchmark_name.replace('graph500', 'graph')
        if 'spec06' in benchmark_name:
            benchmark_name = benchmark_name.replace('spec06/', '') + '_06'
        if 'spec17' in benchmark_name:
            benchmark_name = benchmark_name.replace('spec17/', '') + '_17'
        benchmark_name = benchmark_name.replace('omnetpp_s', 'omnet')
        benchmark_name = benchmark_name.replace('omnetpp', 'omnet')
        benchmark_name = benchmark_name.replace('xalancbmk_s', 'xalan')
        return benchmark_name

class DeltaXCalculator:

    DELTA_X_THRESHOLD = 4.1

    def calculate_deltaX(perf_results, metric='walk_cycles', sort_gaps=False):
        if type(perf_results) is pd.DataFrame:
            df = perf_results.copy()
        else:
            df = Utils.load_dataframe(perf_results)
        return DeltaXCalculator.calculate_deltaX_from_dataframe(df, metric, sort_gaps)

    def calculate_deltaX_from_dataframe(df, metric='walk_cycles', sort_gaps=False):
        df = df.sort_values(metric, ascending=False).reset_index()
        min_val = df[metric].min()
        max_val = df[metric].max()
        delta = max_val - min_val
        saving_col = f'saving-{metric}'
        df[saving_col] = ((max_val - df[metric]) / delta) * 100
        df['deltaX'] = df[saving_col].diff()
        df = df.dropna(subset=['deltaX'])
        if sort_gaps:
            df = df.sort_values('deltaX', ascending=False)
        return df

    def get_max_deltaX_from_dataframe(df, metric='walk_cycles'):
        df = DeltaXCalculator.calculate_deltaX_from_dataframe(df, metric)
        max_deltaX = DeltaXCalculator.calculate_max_deltaX(df)
        missing_samples = DeltaXCalculator.get_missing_samples(df)
        return max_deltaX, missing_samples

    def get_max_deltaX(perf_file, metric='walk_cycles'):
        df = Utils.load_dataframe(perf_file)
        return DeltaXCalculator.get_max_deltaX_from_dataframe(df, metric)

    def get_missing_samples(df, metric='walk_cycles'):
        df = df.copy()
        if 'deltaX' not in df.columns:
            df = DeltaXCalculator.calculate_deltaX_from_dataframe(df, metric)
        return DeltaXCalculator.calculate_missing_samples(df)

    def calculate_max_deltaX(df):
        max_deltaX = df['deltaX'].max()
        return max_deltaX

    def calculate_missing_samples(df):
        df = df.copy()
        df['missing_samples'] = df['deltaX'] // DeltaXCalculator.DELTA_X_THRESHOLD
        missing_samples = df['missing_samples'].sum()
        return missing_samples

    def calculate_uncovered_range(df):
        maximal_coverage_delta = 4
        df = df.query(f'deltaX > {maximal_coverage_delta}')
        df = df.assign(uncovered_range = df['deltaX'] - maximal_coverage_delta)
        uncovered_range = df['uncovered_range'].sum()
        return uncovered_range

    def calculate_largest_three_deltaX(df):
        df = df.sort_values('deltaX', ascending=False)
        deltaX = df['deltaX']
        return deltaX.iloc[0], deltaX.iloc[1], deltaX.iloc[2]

    def get_largest_three_deltaX(perf_file, metric='walk_cycles'):
        df = DeltaXCalculator.calculate_deltaX(perf_file, metric)
        df = df.sort_values('deltaX', ascending=False)
        deltaX = df['deltaX']
        return deltaX.iloc[0], deltaX.iloc[1], deltaX.iloc[2]

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LassoLarsCV, ElasticNetCV
from sklearn.preprocessing import MaxAbsScaler, PolynomialFeatures, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, LeaveOneOut
from enum import Enum, auto
class CrossValidation:
    DEFAULT_FEATURES = ['walk_cycles', 'stlb_misses', 'stlb_hits']
    default_training_repeated_k_fold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    default_validation_repeated_k_fold = RepeatedKFold(n_splits=5, n_repeats=30, random_state=42)
    leave_one_out = LeaveOneOut()

    def get_coefficients(model, features):
        poly_features = ['1'] + list(model['poly'].get_feature_names_out(features))
        poly_coefficients = [model['linear'].intercept_] + list(model['linear'].coef_)
        poly_df = pd.DataFrame([poly_coefficients], columns=poly_features)
        return poly_df

    def getPolyModel(degree):
        poly_model = Pipeline([
            ('scale', MaxAbsScaler()),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear', LinearRegression(fit_intercept=True))])
        return poly_model

    def getElasticModel(degree, cv=10):
        if cv is None:
            cv_method = None
        elif type(cv) is int:
            cv_method = RepeatedKFold(n_splits=5, n_repeats=cv, random_state=42)
        else:
            cv_method = cv
        mosmodel = Pipeline([
            ('scale', MaxAbsScaler()),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear', ElasticNetCV(
                #l1_ratio=[0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                #l1_ratio=[0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.7, 0.95],
                l1_ratio=[0.05, 0.5, 0.95],
                alphas=(0.1, 1.0, 10.0),
                #alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
                fit_intercept=True,
                cv=cv_method,
                eps=1e-4,
                max_iter=int(1e6),
                #tol=1e-3,
                selection='random',
                random_state=42,
                n_jobs=-1))
            ])
        return mosmodel

    def _getUnshuffledElasticModel(degree):
        return CrossValidation.getElasticModel(degree, None)

    def getLassoModel(degree, cv=10):
        if type(cv) is int:
            cv_method = RepeatedKFold(n_splits=5, n_repeats=cv, random_state=42)
        else:
            cv_method = cv
        mosmodel = Pipeline([
            ('scale', MaxAbsScaler()),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear', LassoLarsCV(fit_intercept=True, cv=cv_method, eps=1e-4, n_jobs=-1))])
        return mosmodel

    def _getUnshuffledLassoModel(degree):
        mosmodel = Pipeline([
            ('scale', MaxAbsScaler()),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear', LassoLarsCV(fit_intercept=True, eps=1e-4, n_jobs=-1))])
        return mosmodel

    def getRidgeModel(degree, cv=10):
        if type(cv) is int:
            cv_method = RepeatedKFold(n_splits=5, n_repeats=cv, random_state=42)
        else:
            cv_method = cv
        mosmodel = Pipeline([
            ('scale', MaxAbsScaler()),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear', RidgeCV(fit_intercept=True, cv=cv_method))])
        return mosmodel

    def _getUnshuffledRidgeModel(degree):
        mosmodel = Pipeline([
            ('scale', MaxAbsScaler()),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear', RidgeCV(fit_intercept=True))])
        return mosmodel

    def getSplineModel(degree, cv=10):
        if type(cv) is int:
            cv_method = RepeatedKFold(n_splits=5, n_repeats=cv, random_state=42)
        else:
            cv_method = cv
        mosmodel = Pipeline([
            ('scale', MaxAbsScaler()),
            ('poly', SplineTransformer(n_knots=5, degree=degree, include_bias=False, extrapolation='continue', knots='quantile')),
            ('linear', LassoCV(fit_intercept=True, cv=cv_method))])
        return mosmodel

    moselect_mosmodel = getElasticModel(3)
    micro20_mosmodel = getLassoModel(3)
    spline_model = getSplineModel(1)

    ridge_model = getRidgeModel(3)
    lasso_model = getLassoModel(3)
    elastic_model = getElasticModel(3)

    def getTestErrors(model, train_df, test_df, features):
        x_test = test_df[features]
        y_true = test_df['cpu_cycles']
        y_pred = CrossValidation.predictRuntime(model, train_df, features, x_test)
        error = CrossValidation.relativeError(y_true, y_pred)

        res_df = x_test.copy()
        res_df['true_y'] = y_true
        res_df['predicted_y'] = y_pred
        res_df['error'] = error

        return res_df

    def calculateModelError(model, train_df, test_df, features):
        df = CrossValidation.getTestErrors(model, train_df, test_df, features)
        return df['error']

    def trainModel(model, features, train_df):
        # train_df = Utils.shuffle_dataframe(train_df)
        x_train = train_df[features]
        y_train = train_df['cpu_cycles']
        model.fit(x_train, y_train)

    def predictRuntime(model, train_df, features, x):
        CrossValidation.trainModel(model, features, train_df)
        return model.predict(x)

    def relativeError(y_true, y_pred):
        return (y_pred-y_true)/y_true

    def printR2Score(model, train_df, test_df, features, method):
        train_X = train_df[features]
        train_Y = train_df['cpu_cycles']
        test_X = train_X
        test_Y = train_Y
        if test_df is not None:
            test_X = test_df[features]
            test_Y_true = test_df['cpu_cycles']
        reg = model.fit(train_X, train_Y)
        test_Y_pred = model.predict(test_X)
        score = r2_score(test_Y_true, test_Y_pred)
        print(f'Score: {score.mean():.2f}')

    def printScore(model, train_df, test_df, features, method):
        train_X = train_df[features]
        train_Y = train_df['cpu_cycles']
        test_X = train_X
        test_Y = train_Y
        if test_df is not None:
            test_X = test_df[features]
            test_Y = test_df['cpu_cycles']
        scores = cross_val_score(model, train_X, train_Y, cv=method, scoring='r2')
        print(f'R2 score: {scores.mean():.2f}')
        scores = cross_val_score(model, train_X, train_Y, cv=method, scoring='neg_mean_absolute_percentage_error')
        print(f'mean_absolute_percentage_error score: {scores.mean():.2f}')

    def crossValidate(df, model, features, method):
        df = Utils.shuffle_dataframe(df)
        res = pd.Series(dtype=float)
        max_err = 0
        max_err_train_set = max_err_test_set = None
        for train_index, test_index in method.split(df):
            error = CrossValidation.calculateModelError(model, df.iloc[train_index], df.iloc[test_index], features)
            res = pd.concat([res, error])
            if error.abs().max() > max_err:
                max_err = error.abs().max()
                max_err_train_set = df.iloc[train_index]
                max_err_test_set = df.iloc[test_index]
        error_df = pd.DataFrame(res, columns=['cv_error'])
        return error_df, max_err_train_set, max_err_test_set

    def doubleCrossValidate(train_df, test_df, model, features, method):
        if method is None:
            error = CrossValidation.testTrainSet(train_df, test_df, model, features)
            return error, train_df, test_df

        train_df = Utils.shuffle_dataframe(train_df)
        res = pd.Series(dtype=float)
        max_err = 0
        max_err_train = max_err_test = None
        for train_index, test_index in method.split(train_df):
            combined_test_df = pd.concat([test_df, train_df.iloc[test_index]])
            error = CrossValidation.calculateModelError(model, train_df.iloc[train_index], combined_test_df, features)
            res = pd.concat([res, error])
            if error.abs().max() > max_err:
                max_err = error.abs().max()
                max_err_train = train_df.iloc[train_index]
                max_err_test = combined_test_df
                # max_err_test = train_df.iloc[test_index]
        error_df = pd.DataFrame(res, columns=['cv_error'])
        return error_df, max_err_train, max_err_test

    def doubleCrossValidateKFold(train_df, test_df, model, features, test_cv_n_repeats=30):
        method = RepeatedKFold(n_splits=5, n_repeats=test_cv_n_repeats, random_state=42)
        return CrossValidation.doubleCrossValidate(train_df, test_df, model, features, method)

    def doubleCrossValidateLeaveOneOut(train_df, test_df, model, features):
        method = LeaveOneOut()
        return CrossValidation.doubleCrossValidate(train_df, test_df, model, features, method)

    def trainAndTestUsingAllSamples(df, model, features):
        df = Utils.shuffle_dataframe(df)
        error_df = df.copy().reset_index(drop=True)
        error = CrossValidation.calculateModelError(model, error_df, error_df, features)
        error_df['cv_error'] = error
        return error_df[['layout', 'cv_error']]

    def crossValidateKFold(df, model, features, test_cv_n_repeats=30):
        method = RepeatedKFold(n_splits=5, n_repeats=test_cv_n_repeats, random_state=42)
        return CrossValidation.crossValidate(df, model, features, method)

    def crossValidateLeaveOneOut(df, model, features):
        method = LeaveOneOut()
        return CrossValidation.crossValidate(df, model, features, method)

    def testTrainSet(train_df, test_df, model, features):
        train_df = Utils.shuffle_dataframe(train_df)
        error = CrossValidation.calculateModelError(model, train_df, test_df, features)

        res_df = test_df.copy()
        res_df['cv_error'] = error
        return res_df[['layout', 'cv_error']]

    class CV_METHOD(Enum):
        KFOLD = auto()
        LEAVE_ONE_OUT = auto()
        DOUBLE_KFOLD = auto()
        DOUBLE_CV = DOUBLE_KFOLD
        DOUBLE_LEAVE_ONE_OUT = auto()
        TEST_ALL = auto()

    def getMaxAbsErrorPercentage(error_df):
        return round(error_df['cv_error'].abs().max() * 100, 2)

    def calculateMaximalCrossValidationError(file_path, model, cv_method,
            features=DEFAULT_FEATURES, external_test_set_path=None):
        error_df, _, __ = CrossValidation.crossValidateBenchmark(file_path, model, cv_method, features, external_test_set_path)
        return round(error_df['cv_error'].abs().max() * 100, 2)

    def getModelByItsName(model_str, cv_n_repeats=10, shuffle=True):
        if model_str == 'lasso':
            if shuffle:
                return CrossValidation.getLassoModel(3, cv_n_repeats)
            else:
                return CrossValidation._getUnshuffledLassoModel(3)
        elif model_str == 'ridge':
            if shuffle:
                return CrossValidation.getRidgeModel(3, cv_n_repeats)
            else:
                return CrossValidation._getUnshuffledRidgeModel(3)
        elif model_str == 'elastic':
            if shuffle:
                return CrossValidation.getElasticModel(3, cv_n_repeats)
            else:
                return CrossValidation._getUnshuffledElasticModel(3)
        else:
            raise ValueError(f'unsupported model: {model_str}')

    def getCvMethodByItsName(cv_method_str):
        cv_method_str = cv_method_str.lower()
        if cv_method_str == 'loo' or cv_method_str == 'leave_one_out':
            return CrossValidation.CV_METHOD.LEAVE_ONE_OUT
        elif cv_method_str == 'kfold':
            return CrossValidation.CV_METHOD.KFOLD
        if cv_method_str == 'double_loo' or cv_method_str == 'double_leave_one_out':
            return CrossValidation.CV_METHOD.DOUBLE_LEAVE_ONE_OUT
        elif cv_method_str == 'double_kfold':
            return CrossValidation.CV_METHOD.DOUBLE_KFOLD
        elif cv_method_str == 'double_cv':
            return CrossValidation.CV_METHOD.DOUBLE_CV
        elif cv_method_str == 'test_all':
            return CrossValidation.CV_METHOD.TEST_ALL
        else:
            raise ValueError(f'unsupported CV-method: {cv_method_str}')

    def crossValidateBenchmark(train_data_set, model, cv_method,
            features=DEFAULT_FEATURES, test_data_set=None,
            train_cv_n_repeats=10, test_cv_n_repeats=30, shuffle=True):

        if type(model) is str:
            model = CrossValidation.getModelByItsName(model, train_cv_n_repeats, shuffle)
        if type(cv_method) is str:
            cv_method = CrossValidation.getCvMethodByItsName(cv_method)

        if type(train_data_set) is pd.DataFrame:
            train_df = train_data_set
        else:
            train_df = Utils.load_dataframe(train_data_set, sort=False, drop_duplicates=True)
            print(f'{train_data_set} ==> {len(train_df)} layouts')

        if test_data_set is not None:
            if  type(test_data_set) is pd.DataFrame:
                test_df = test_data_set
            else:
                test_df = Utils.load_dataframe(test_data_set, sort=False, drop_duplicates=True)
        else:
            test_df = None

        error_df = max_err_train_set = max_err_test_set = None
        if cv_method == CrossValidation.CV_METHOD.KFOLD:
            error_df, max_err_train_set, max_err_test_set = CrossValidation.crossValidateKFold(train_df, model, features, test_cv_n_repeats)
        elif cv_method == CrossValidation.CV_METHOD.DOUBLE_CV or cv_method == CrossValidation.CV_METHOD.DOUBLE_KFOLD:
            if test_df is not None:
                error_df, max_err_train_set, max_err_test_set = CrossValidation.doubleCrossValidateKFold(train_df, test_df, model, features, test_cv_n_repeats)
        elif cv_method == CrossValidation.CV_METHOD.LEAVE_ONE_OUT:
            error_df, max_err_train_set, max_err_test_set = CrossValidation.crossValidateLeaveOneOut(train_df, model, features)
        elif cv_method == CrossValidation.CV_METHOD.DOUBLE_LEAVE_ONE_OUT:
            if test_df is not None:
                error_df, max_err_train_set, max_err_test_set = CrossValidation.doubleCrossValidateLeaveOneOut(train_df, test_df, model, features)
        elif cv_method == CrossValidation.CV_METHOD.TEST_ALL:
            if test_df is not None:
                error_df = CrossValidation.testTrainSet(train_df, test_df, model, features)
        else:
            raise ValueError(f'invalid parameter: {cv_method}')

        return error_df, max_err_train_set, max_err_test_set
