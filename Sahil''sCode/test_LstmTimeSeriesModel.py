# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 10:41:49 2019

@author: sahil
"""

import pandas as pd
import numpy as np
import pandas_datareader.data as web

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

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

def simple_lstm_model(seq_len, input_dim, hidden_dim):
    # create model
    model = Sequential()
    model.add(LSTM(hidden_dim, input_shape=(seq_len,input_dim)))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


class LstmModel():
    def __init__(self, seq_len, dim_size):
        self.seq_len = seq_len
        self.dim_size = dim_size

    def fit(self, Y, X):
        input_dim = X.shape[1]
        self.input_dim = input_dim
        self.model = simple_lstm_model(self.seq_len, self.input_dim, self.dim_size)
        Y_lstm = Y.iloc[(self.seq_len-1):].values
        X_lstm = np.zeros((len(Y_lstm), self.seq_len, input_dim))
        for i in range(self.seq_len):
            X_lstm[:, i, :] = X.iloc[i:len(Y_lstm)+i, :].values
        self.model.fit(X_lstm, Y_lstm)

    def predict(self, Y, X, X_test):
        X_total = pd.concat([X.tail(self.seq_len-1), X_test])
        X_lstm = np.zeros((len(X_test), self.seq_len, self.input_dim))
        for i in range(self.seq_len):
            X_lstm[:, i, :] = X_total.iloc[i:len(X_test)+i, :].values
        Y_hat = pd.Series(data=self.model.predict(X_lstm)[:, 0],
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
    model = LstmModel(seq_len=5, dim_size=20)
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

