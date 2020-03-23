# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:09:12 2020

@author: sahil
"""
import statsmodels.api as sm
import seaborn as sns
from lxml import etree
import json
from io import StringIO
from os import listdir
from os.path import isfile, join
import pandas as pd
from datetime import date
from pandas.tseries.offsets import BDay
from textblob import TextBlob
from scipy.stats.mstats import winsorize

import spacy
nlp = spacy.load("en_core_web_lg")

root_path = 'C:/Users/sahil/Documents/Data/SeekingAlpha/OnTheGoRaw/'

json_files = [join(root_path, f) for f in listdir(root_path) if isfile(join(root_path, f))]

df = pd.read_csv(r'C:\Users\sahil\Documents\Projects\SignalReplication\Factors\idiovol\Data\StockReturnsDaily.csv',
                 dtype={'TICKER':str, 'RET':str, 'date':str}, usecols=['TICKER', 'date', 'RET'])
df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
df['date'] = pd.to_datetime(df['date'])
df = df.loc[df['date'] > '2011-01-01'].dropna()
df['date'] = df['date'].apply(lambda x: x.date())


df_arr = []
for json_file in json_files:
    try:
        print('Running {}'.format(json_file))
        json_data = json.load(open(json_file))
        
        file_date = json_file.split('/')[-1].replace('.json', '')
        file_date = date(int(file_date[:4]), int(file_date[5:7]), int(file_date[8:]))
        
        tm1 = (file_date - BDay(1)).date()
        tp1 = (file_date + BDay(1)).date()
        tp11 = (file_date + BDay(11)).date()
    
        t_df = df.loc[(df['date'] >= tm1) & (df['date'] <= tp1), :]
        tf_df = df.loc[(df['date'] > tp1) & (df['date'] <= tp11), :]
    
        if json_data.get('count', 0)> 10:
            xml_data = json_data['content']
            
            tree = etree.parse(StringIO(xml_data), parser=etree.HTMLParser())
            
            headlines = tree.xpath("//h4[contains(@class, 'media-heading')]/a/text()")
            assert len(headlines) == json_data['count']
            
            main_tickers = list(map(lambda x: x.replace('/symbol/', ''), tree.xpath("//div[contains(@class, 'media-left')]//a/@href")))
            assert len(main_tickers) == json_data['count']
            
       
            #comments = list(map(lambda x: 0 if x == '|' else int(x.replace('\xa0', '')), filter(lambda x: not x.startswith('Comment'), tree.xpath("//div[contains(@class, 'media-body')]/div[contains(@class, 'tiny-share-widget')]/following-sibling::span[last()]//text()"))))
            #assert len(comments) == json_data['count']
        
            first_sent = [''.join(f.xpath('.//text()')) for f in tree.xpath("//div[contains(@class, 'media-body')]/ul/li[1]")]
            if len(first_sent) == 0:
                first_sent = [''.join(f.xpath('.//text()')) for f in tree.xpath("//div[contains(@class, 'media-body')]")]
                first_sent = [f.replace(h, '').split('\xa0')[0].strip() for f,h in zip (first_sent, headlines)]
            if len(first_sent) != json_data['count']:
                continue
        
            
            ret = []
            for main_ticker in main_tickers:
                tt_df = t_df.loc[t_df['TICKER'] == main_ticker, :]
                if tt_df.empty:
                    ret.append(pd.np.NaN)
                else:
                    ret.append(tt_df.loc[:, 'RET'].sum())
            assert len(ret) == json_data['count']
            
            ret_f = []
            for main_ticker in main_tickers:
                ttf_df = tf_df.loc[tf_df['TICKER'] == main_ticker, :]
                if ttf_df.shape[0] <= 5:
                    ret_f.append(pd.np.NaN)
                else:
                    ret_f.append(ttf_df.loc[:, 'RET'].sum())
            assert len(ret_f) == json_data['count']
            
            sentiments = [TextBlob(s).sentiment.polarity for s in first_sent]
            verb_vectors = pd.np.array([pd.np.array([token.vector for token in nlp(s) if token.pos_ == 'VERB']).mean(axis=0)*pd.np.ones((300)) for s in first_sent])
            all_vectors = pd.np.array([pd.np.array([token.vector for token in nlp(s) ]).mean(axis=0)*pd.np.ones((300)) for s in first_sent])
        
            df_dict = {'ticker': main_tickers,
                       'date': [file_date] * len(main_tickers),
                       #'n_comments': comments,
                       'sentiment': sentiments,
                       'ret': ret,
                       'ret_f': ret_f}
            for i in range(300):
                df_dict['vv_{0}'.format(i+1)] = verb_vectors[:, i].tolist()
                df_dict['av_{0}'.format(i+1)] = all_vectors[:, i].tolist()
    
            df_f = pd.DataFrame(df_dict)
            df_arr.append(df_f)
    except:
        pass

data_df =  pd.concat(df_arr).reset_index()
data_df.to_csv(r'C:\Users\sahil\Documents\Data\SeekingAlpha\Clean.csv', sep='|', index=False)
data_df.dropna().to_csv(r'C:\Users\sahil\Documents\Data\SeekingAlpha\Reduced.csv', sep='|', index=False)
r_df = data_df.dropna()
rr_df = r_df[ (r_df['ret'] < r_df['ret'].quantile(0.95)) & \
              (r_df['ret'] > r_df['ret'].quantile(0.05)) & \
              (r_df['ret_f'] < r_df['ret_f'].quantile(0.95)) & \
              (r_df['ret_f'] > r_df['ret_f'].quantile(0.05)) ]

train_df = rr_df.loc[rr_df['date'] < date(2017,1,1), :]
test_df = rr_df.loc[rr_df['date'] >= date(2017,1,1), :]
ret_spy = df.loc[df['TICKER'] == 'SPY', ['date', 'RET']].set_index('date').loc[:, 'RET'].loc[test_df.index]

Y_col = 'ret_f'
X_cols = ['sentiment']

Y_train = train_df.loc[:, Y_col]; Y_test = test_df.loc[:, Y_col];
X_train = train_df.loc[:, X_cols]; X_test = test_df.loc[:, X_cols];
X_train = sm.add_constant(X_train); X_test = sm.add_constant(X_test)

model = sm.OLS(Y_train,X_train)
results = model.fit()

Y_hat = results.predict(X_test)
test_df['ret_pred'] = Y_hat.values

test_df['s_ret'] = test_df.apply(lambda x: x['ret_f'] if x['ret_pred'] > 0.0 else -x['ret_f'] if x['ret_pred'] < -0.0 else 0, axis=1)
ret_s = (test_df.groupby('date')['s_ret'].mean())
pd.DataFrame({'Stg': ret_s, 'SPY': ret_spy}).cumsum().plot()


Y = ret_s
X = ret_spy
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
results.summary()


test_df['s_ret'] = test_df.apply(lambda x: x['ret_f'] if x['sentiment'] > 0.25 else -x['ret_f'] if x['sentiment'] < -0.25 else 0, axis=1)
ret_s = (test_df.groupby('date')['s_ret'].mean())
pd.DataFrame({'Stg': ret_s, 'SPY': ret_spy}).cumsum().plot()

Y = ret_s
X = ret_spy
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
results.summary()


rr_df['s_ret'] = rr_df.apply(lambda x: x['ret_f'] if x['sentiment'] > 0.25 else -x['ret_f'] if x['sentiment'] < -0.25 else 0, axis=1)
ret_s = (rr_df.groupby('date')['s_ret'].mean())
ret_spy = df.loc[df['TICKER'] == 'SPY', ['date', 'RET']].set_index('date').loc[:, 'RET'].loc[ret_s.index]
pd.DataFrame({'Stg': ret_s, 'SPY': ret_spy}).cumsum().plot()

Y = ret_s
X = ret_spy
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
results.summary()





