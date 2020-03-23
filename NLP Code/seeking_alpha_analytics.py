# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 22:16:03 2020

@author: sahil
"""

import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\sahil\Documents\Projects\SignalReplication\Factors\idiovol\Data\StockReturnsDaily.csv',
                 dtype={'TICKER':str, 'RET':str, 'date':str}, usecols=['TICKER', 'date', 'RET'])
                 
ret_mat = df.drop_duplicates(subset=['date', 'TICKER']).pivot(index='date', columns='TICKER', values='RET')
df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
df['date'] = pd.to_datetime(df['date'])
df = df.loc[df['date'] > '2011-01-01'].dropna()
df['date'] = df['date'].apply(lambda x: x.date())

df = df.drop_duplicates(subset=['date', 'TICKER'])

ret_mat = df.pivot(index='date', columns='TICKER', values='RET')
del df
r_df = pd.read_csv(r'C:\Users\sahil\Documents\Data\SeekingAlpha\Reduced.csv', sep='|')
rr_df = r_df[ (r_df['ret'] < r_df['ret'].quantile(0.95)) & \
              (r_df['ret'] > r_df['ret'].quantile(0.05)) & \
              (r_df['ret_f'] < r_df['ret_f'].quantile(0.95)) & \
              (r_df['ret_f'] > r_df['ret_f'].quantile(0.05)) ]
rr_df['const'] = 1
del r_df

features = ['av_{0}'.format(i+1) for i in range(300)]
f_df = rr_df.drop_duplicates(subset=['date', 'ticker']).loc[:, ['const'] + features]


