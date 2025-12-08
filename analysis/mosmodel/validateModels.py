#! /usr/bin/env python3

import pandas as pd
import my_models
import utility

def calculateBasuError(df, basu_A, basu_B, conf):
    '''
    Basu_runtime = A * TLB_MISSES + B
    A = PAGE_WALK_LATENCY / TLB_MISSES
    B = 4kb_runtime - PAGE_WALK_LATENCY
    '''
    y_pred = basu_A * df.loc[conf]['stlb_misses'] + basu_B
    y_true = df.loc[conf]['cpu-cycles']
    return utility.relativeError(y_true, y_pred)

def calculateAlamError(df, alam_B, conf):
    '''
    Alam_runtime = PAGE_WALK_LATENCY + B
    '''
    y_pred = df.loc[conf]['walk_cycles'] + alam_B
    y_true = df.loc[conf]['cpu-cycles']
    return utility.relativeError(y_true, y_pred)

def calculatePhamError(df, pham_B, conf):
    '''
    Pham_runtime = 1xDTLB_HITS + 7xSTLB_HITS + PAGE_WALK_LATENCY + B
    DTLB_HITS can be ignored, therefore:
    '''
    y_pred = 7*df.loc[conf]['stlb_hits'] + \
            df.loc[conf]['walk_cycles'] + pham_B
    y_true = df.loc[conf]['cpu-cycles']
    return utility.relativeError(y_true, y_pred)

def calculateGandhiError(df, A, B, conf):
    '''
    runtime = B + A*stlb_misses
    Where:
    B = runtime[2mb] - walk_cycles[2mb]
    A = walk_cycles / stlb_misses
    '''
    y_pred = B + A*df.loc[conf]['stlb_misses']
    y_true = df.loc[conf]['cpu-cycles']
    return utility.relativeError(y_true, y_pred)

def calculateYanivError(df, yaniv_A, yaniv_B, conf):
    y_pred = yaniv_A * df.loc[conf]['walk_cycles'] + yaniv_B
    y_true = df.loc[conf]['cpu-cycles']
    return utility.relativeError(y_true, y_pred)

def print_coefficients(model, features, output_file):
    # Use get_feature_names_out() for scikit-learn >= 1.0, get_feature_names() for older versions
    try:
        poly_features = ['1'] + list(model['poly'].get_feature_names_out(features))
    except AttributeError:
        poly_features = ['1'] + model['poly'].get_feature_names(features)
    poly_coefficients = [model['linear'].intercept_] + list(model['linear'].coef_)
    poly_df = pd.DataFrame([poly_features, poly_coefficients])
    poly_df.to_csv(output_file, index=False, header=False)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train_dataset', default='train_mean.csv')
parser.add_argument('-v', '--test_dataset', default='test_mean.csv')
parser.add_argument('-c', '--coeffs_file', default='linear_models_coeffs.csv')
parser.add_argument('-o', '--output', required=True)
parser.add_argument('-p', '--poly',  default='poly.csv',
        help='the coefficients of the polynomial models')
args = parser.parse_args()

train_df = utility.loadDataframe(args.train_dataset)
test_df = utility.loadDataframe(args.test_dataset)

coeffs = pd.read_csv(args.coeffs_file)

res_df = pd.DataFrame(columns=['layout', 'basu_error', 'alam_error', 'pham_error', 'yaniv_error', 'gandhi_error', 'poly1_error', 'poly2_error', 'poly3_error', 'mosmodel_error'])
for i in test_df.index.values:
    layout = test_df.loc[i]['layout']
    basu_err = calculateBasuError(test_df, coeffs['basu_A'][0], \
            coeffs['basu_B'][0], i)
    alam_err = calculateAlamError(test_df, coeffs['alam_B'][0], i)
    pham_err = calculatePhamError(test_df, coeffs['pham_B'][0], i)
    gandhi_err = calculateGandhiError(test_df, coeffs['gandhi_A'][0], \
            coeffs['gandhi_B'][0], i)
    yaniv_err = calculateYanivError(test_df, coeffs['yaniv_A'][0], \
            coeffs['yaniv_B'][0], i)
    res_df.loc[i] = {'layout':layout,
            'basu_error':basu_err,
            'alam_error':alam_err,
            'pham_error':pham_err,
            'yaniv_error':yaniv_err,
            'gandhi_error':gandhi_err,
            'poly1_error': -1,
            'poly2_error': -1,
            'poly3_error': -1,
            'mosmodel_error': -1}

poly1_err = my_models.calculateModelError(my_models.poly1, train_df, test_df, ['walk_cycles'])
poly2_err = my_models.calculateModelError(my_models.poly2, train_df, test_df, ['walk_cycles'])
poly3_err = my_models.calculateModelError(my_models.poly3, train_df, test_df, ['walk_cycles'])
print_coefficients(my_models.poly3, ['walk_cycles'], args.poly)
mosmodel_err = my_models.calculateModelError(my_models.mosmodel, train_df, test_df,
        ['walk_cycles', 'stlb_misses', 'stlb_hits'])
res_df['poly1_error'] = poly1_err
res_df['poly2_error'] = poly2_err
res_df['poly3_error'] = poly3_err
res_df['mosmodel_error'] = mosmodel_err

res_df.to_csv(args.output)


