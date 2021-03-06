# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 10:29:52 2019

@author: sahil
"""

import pandas as pd
import numpy as np
import pandas_datareader.data as web

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet


# Get the data
def get_data():
	return_period = 21

	stk_tickers = ['MSFT', 'IBM', 'GOOGL']
	ccy_tickers = ['DEXJPUS', 'DEXUSUK']
	idx_tickers = ['SP500', 'DJIA', 'VIXCLS']

	stk_data = web.DataReader(stk_tickers, 'yahoo')
	ccy_data = web.DataReader(ccy_tickers, 'fred')
	idx_data = web.DataReader(idx_tickers, 'fred')

	Y = np.log(stk_data.loc[:, ('Adj Close', 'MSFT')]).diff(return_period).shift(-return_period)
	Y.name = Y.name[-1]+'_pred'

	X1 = np.log(stk_data.loc[:, ('Adj Close', ('GOOGL', 'IBM'))]).diff(return_period)
	X1.columns = X1.columns.droplevel()
	X2 = np.log(ccy_data).diff(return_period)
	X3 = np.log(idx_data).diff(return_period)

	X4 = pd.concat([Y.diff(i) for i in [21, 63, 126,252]], axis=1).dropna()
	X4.columns = ['1M', '3M', '6M', '1Y']

	X = pd.concat([X1, X2, X3, X4], axis=1)

	data = pd.concat([Y, X], axis=1).dropna()
	Y = data.loc[:, Y.name]
	X = data.loc[:, X.columns]
	return X, Y

X, Y = get_data()

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
test_size=validation_size, random_state=seed)


# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'



# Grid Search : ElasticNet
'''
alpha : float, optional
    Constant that multiplies the penalty terms. Defaults to 1.0.
    See the notes for the exact mathematical meaning of this
    parameter.``alpha = 0`` is equivalent to an ordinary least square,
    solved by the :class:`LinearRegression` object. For numerical
    reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
    Given this, you should use the :class:`LinearRegression` object.

l1_ratio : float
    The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
    ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
    is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
    combination of L1 and L2.
'''
param_grid = {'alpha': [0.01, 0.1, 0.3, 0.7, 1, 1.5, 3, 5],
              'l1_ratio': [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]}
model = ElasticNet()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X_train, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

