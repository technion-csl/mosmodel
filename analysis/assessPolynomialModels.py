#!/usr/bin/env python3

import numpy as np
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', help='input CSV file')
parser.add_argument('-o', '--output_file', help='output PDF file')
parser.add_argument('-k', '--num_of_folds', default=3, type=int,
                    help='number of cross-validation folds')
parser.add_argument('-d', '--max_degree', default=3, type=int,
                    help='maximum degree of polynomials')
parser.add_argument('-m', '--metric', default='cpu-cycles',
                    help='the metric to use (e.g., cpu-cycles)')
parser.add_argument('-x', '--x_metric', default='walk_cycles',
                    help='the metric to use on X-axis (e.g., walk_cycles/tlb-misses)')
args = parser.parse_args()

df = pd.read_csv(args.input_file, index_col='layout')
x = df[[args.x_metric]]
y = df[args.metric]

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') #use a non-interactive backend

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4,3))
ax.scatter(x, y, marker='o', s=10, label='measurements')

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

k_fold = KFold(n_splits=args.num_of_folds, shuffle=True, random_state=0)

line_styles = ['dashed', 'dotted', 'solid']
for degree in range(1, args.max_degree+1):
    print('=====', 'degree =', degree, '=====')
    polynomial_model = Pipeline(
            [('poly', PolynomialFeatures(degree=degree)),
             ('linear', LinearRegression(fit_intercept=False))])
    scores = cross_val_score(polynomial_model, x, y, cv=k_fold)
    print('residual sum of squares with 95% confidence level:',
          '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    polynomial_model.fit(x, y)
    linear_model = polynomial_model.named_steps['linear']
    print('model coefficients (1, x, x^2, ...):', linear_model.coef_)
    prediction = polynomial_model.predict(x)
    error = prediction - y
    print('max error = ', '{:.1%}'.format(max(abs(error))))
    x_high = x.max(axis='index')[0]
    x_poly = np.linspace(0.0, x_high, num=100).reshape(-1, 1)
    y_poly = polynomial_model.predict(x_poly)
    ax.plot(x_poly, y_poly, linestyle=line_styles[degree-1],
            color='k', label='degree='+str(degree))

# set x, y labels
plt.xlabel('relative ' + args.x_metric)
plt.ylabel('relative ' + args.metric)

# set legend
handles, labels = ax.get_legend_handles_labels()
legend_order = [args.max_degree] + list(range(args.max_degree))
labels = [labels[i] for i in legend_order]
handles = [handles[i] for i in legend_order]
ax.legend(handles, labels, loc=(0.12, 0.6))

# set x, y spines
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
#ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')

# save to pdf
fig.savefig(args.output_file, bbox_inches='tight')
plt.close('all')

