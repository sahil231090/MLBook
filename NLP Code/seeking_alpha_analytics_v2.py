# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 23:19:30 2020

@author: sahil
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize

df = pd.read_csv(r'C:\Users\sahil\Documents\Projects\SignalReplication\Factors\idiovol\Data\StockReturnsDaily.csv',
                 dtype={'TICKER':str, 'RET':str, 'date':str}, usecols=['TICKER', 'date', 'RET'])

ret_mat = df.drop_duplicates(subset=['date', 'TICKER']).pivot(index='date', columns='TICKER', values='RET')
df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
df['date'] = pd.to_datetime(df['date'])
df = df.loc[df['date'] > '2011-01-01'].dropna()
df['date'] = df['date'].apply(lambda x: x.date())

df = df.drop_duplicates(subset=['date', 'TICKER'])
ret_spy = df.loc[df['TICKER'] == 'SPY', ['date', 'RET']].set_index('date').loc[:, 'RET']
ret_mat = df.pivot(index='date', columns='TICKER', values='RET')

del df
r_df = pd.read_csv(r'C:\Users\sahil\Documents\Data\SeekingAlpha\Reduced.csv', sep='|')
rr_df = r_df[ (r_df['ret'] < r_df['ret'].quantile(0.95)) & \
              (r_df['ret'] > r_df['ret'].quantile(0.05)) & \
              (r_df['ret_f'] < r_df['ret_f'].quantile(0.95)) & \
              (r_df['ret_f'] > r_df['ret_f'].quantile(0.05)) ]
rr_df['const'] = 1
del r_df

#features = ['av_{0}'.format(i+1) for i in range(300)]
features = ['av_{0}'.format(i) for i in [123,264,62,203,204,180,272,274,26,140,197,270,34,15,48]]
f_df = rr_df.drop_duplicates(subset=['date', 'ticker']).loc[:, ['const'] + features]

f_df.shape
f_df = rr_df.drop_duplicates(subset=['date', 'ticker']).loc[:, ['date', 'ticker', 'const'] + features]
f_df.date.min()
f_df.date.max()
train_f = f_df[f_df['date'] < '2017-01-01']
test_f = f_df[f_df['date'] >= '2017-01-01']

theta_0 = np.zeros((len(features)+1))

w_df = train_f.loc[:, ['date', 'ticker']]
w_df['weights'] = np.NaN
w_df['date'] = w_df['date'].apply(lambda x: pd.to_datetime(x).date())

w_mat = w_df.pivot(index='date', columns='ticker', values='weights')
train_mat = ret_mat.loc[w_mat.index, w_mat.columns]

def f(theta):
    #w_df['weights'] = (2/(1+np.exp(-train_f.iloc[:, 2:].dot(theta))) - 1)
    w_df['weights'] = np.minimum(np.ones((train_f.shape[0])),
                        np.maximum(-np.ones((train_f.shape[0])), 
                                   train_f.iloc[:, 2:].dot(theta)))
    
    w_mat = w_df.pivot(index='date', columns='ticker', values='weights').shift(2).ffill(limit=10)
    ret_s = (train_mat * w_mat).sum(axis=1)
    sr = ret_s.mean()/(1e-10+ret_s.std())
    print(sr*np.sqrt(252))
    return -sr

res_opt = minimize(f, theta_0)
theta_opt = res_opt['x']

w_df = test_f.loc[:, ['date', 'ticker']]
w_df['weights'] = np.NaN
w_df['date'] = w_df['date'].apply(lambda x: pd.to_datetime(x).date())

w_mat = w_df.pivot(index='date', columns='ticker', values='weights')

w_df['weights'] = np.minimum(np.ones((test_f.shape[0])),
                        np.maximum(-np.ones((test_f.shape[0])), 
                                   test_f.iloc[:, 2:].dot(theta_opt)))
w_mat = w_df.pivot(index='date', columns='ticker', values='weights').shift(2).ffill(limit=10)

test_mat = ret_mat.loc[w_mat.index, w_mat.columns]
ret_s = (test_mat * w_mat).sum(axis=1)
sr = ret_s.mean()/(1e-10+ret_s.std())
print(sr*np.sqrt(252))

252*ret_s.cumsum().plot(figsize=(10,6)); plt.title('Test Period Performance'); plt.xlabel('Annualized Return');







