# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 10:41:49 2019

@author: sahil
"""

import pandas as pd
import numpy as np
import pandas_datareader.data as web

from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import LinearRegression


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

class SimpleTsModel():
    def __init__(self, ts_model, linear_model=LinearRegression):
        self.ts_model = ts_model()
        self.linear_model = linear_model()
    
    def fit(self, Y, X=None):
        if X is None:
            X = np.ones_like(Y)
        self.linear_model.fit(X, Y)
        Y_hat = self.linear_model.predict(X)
        eps = Y - Y_hat
        #Apply TS Model to remaining data
        self.ts_model.fit(eps)

    
    def predict(self, Y, X, X_test):
        Y_hat = self.linear_model.predict(X)
        eps = Y - Y_hat
        eps_hat = self.ts_model(eps)
        Y_hat = eps_hat + self.linear_model.predict(X_test)
        return Y_hat


class ArimaModel():
    def __init__(self, order):
        self.order = order

    def fit(self, Y, X):
        self.model = ARIMA(Y.values, order=self.order, exog=X.values, freq=None).fit(disp=0)

    def predict(self, Y, X, X_test):
        Y_hat = pd.Series(data=self.model.predict(start=0,
                                                  end=X_test.shape[0]-1,
                                                  exog=X_test),
                          index=X_test.index)
        return Y_hat

        
def scoring_function(Y, Y_hat):
    return np.sum(np.square(Y-Y_hat))


#Split the training and testing data
split_ratio = 0.7
idx_split = int(split_ratio * len(Y))
X_train, X_test = X.iloc[:idx_split, :], X.iloc[idx_split:, :]
Y_train, Y_test = Y.iloc[:idx_split], Y.iloc[idx_split:]



#  Optimize paramters over the 
n_fold = 5
scores = []
for i in range(n_fold):
    model = ArimaModel(order=(1,0,0))
    idx_fold = int((i+1.0)/n_fold * len(X_train))
    X_fold, Y_fold = X_train.iloc[:idx_fold, :], Y.iloc[:idx_fold]
    
    # Split furether into training and testing
    idx_fold_split = int(split_ratio * len(Y_fold))
    X_fold_train, X_fold_test = X_fold.iloc[:idx_fold_split, :], X_fold.iloc[idx_fold_split:, :]
    Y_fold_train, Y_fold_test = Y_fold.iloc[:idx_fold_split], Y_fold.iloc[idx_fold_split:]
    
    model.fit(Y_fold_train, X_fold_train)
    
    Y_fold_predict = model.predict(Y_fold_train, X_fold_train, X_fold_test)

    score = scoring_function(Y_fold_test, Y_fold_predict)
    scores.append(score)
average_score = np.mean(scores)

