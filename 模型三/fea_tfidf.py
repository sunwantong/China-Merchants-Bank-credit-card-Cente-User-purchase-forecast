# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 17:22:07 2018

@author: Alan Yan
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def get_str(a_df):
    str_list = a_df['EVT_LBL'].values
    return ' '.join(str_list)

df1 = pd.read_csv('../orig_data/train_log.csv', sep='\t')
df2 = pd.read_csv('../orig_data/test_log.csv', sep='\t')
df = pd.concat([df1, df2], axis=0)
df.set_index(['USRID', 'OCC_TIM'], inplace=True, drop=False)
df = df.sort_index()
df = df.reset_index(drop=True)
str_df = pd.pivot_table(df, index='USRID', values=['EVT_LBL'], aggfunc=get_str)

vectorizer = CountVectorizer(min_df=2, token_pattern=r"\b\w+\b") # 保留单字
corpus = list(str_df['EVT_LBL'].values)
X_tfidf = vectorizer.fit_transform(corpus)
fea_name = vectorizer.get_feature_names()
X_tfidf = X_tfidf.todense()
count_fea_df = pd.DataFrame(X_tfidf)
count_fea_df.columns = ['w_'+item for item in  fea_name]
count_fea_df['USRID'] = list(str_df.index)
count_fea_df.to_csv('../fea/fea_tfidf_all.csv', index=False)

