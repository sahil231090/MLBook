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

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


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

def simple_nn_model(nn_shape=None):
    input_dim = X.shape[1]
    if nn_shape is None:
        # By default we add only 1 layer = input dim
        nn_shape = [input_dim]
    # create model
    model = Sequential()
    for idx, layer_shape in enumerate(nn_shape):
        if idx == 0:
            model.add(Dense(layer_shape, input_dim=input_dim,
                            kernel_initializer='normal',
                            activation='relu'))
        else:
            model.add(Dense(layer_shape,
                kernel_initializer='normal',
                activation='relu'))
    # Final output layer
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def KerasNNRegressor(nn_shape = None, epochs=100, batch_size=5, verbose=0):
    return KerasRegressor(build_fn=simple_nn_model, epochs=epochs,
                          batch_size=batch_size, verbose=verbose,
                          nn_shape=nn_shape)

# Grid search : KerasNNRegressor 
'''
nn_shape : tuple, length = n_layers - 2, default (100,)
    The ith element represents the number of neurons in the ith
    hidden layer.
''' 
param_grid={'nn_shape': [(20,), (50,), (20,20), (20, 30, 20)]}
model = KerasNNRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X_train, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
