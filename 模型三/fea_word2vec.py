# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 21:30:39 2018

@author: Alan Yan
"""

import pandas as pd
import gensim
import re
import numpy as np


def get_str(a_df):
    str_list = a_df['EVT_LBL'].values
    return ' '.join(str_list)


def get_word2vec_fea(content):
    content = re.sub(r"\s{2,}", " ", content)
    content_list = content.strip().split(' ')
    fea_vec_one = np.zeros(100)
    for item in content_list:
        if item in word_set:
            fea_vec_one += model.wv[item]
    fea_vec_one = fea_vec_one / len(content_list)
    fea_vec_one = [int(item*1000)/1000 for item in fea_vec_one]
    return fea_vec_one

df1 = pd.read_csv('../orig_data/train_log.csv', sep='\t')
df2 = pd.read_csv('../orig_data/test_log.csv', sep='\t')
df = pd.concat([df1, df2], axis=0)
df.set_index(['USRID', 'OCC_TIM'], inplace=True, drop=False)
df = df.sort_index()
df = df.reset_index(drop=True)

#user_id = list(set(df['USRID'].values))

str_df = pd.pivot_table(df, index='USRID', values=['EVT_LBL'], aggfunc=get_str)
data = str_df['EVT_LBL'].values
data = [item.split(' ') for item in data]
model = gensim.models.Word2Vec(data, min_count=1, size=100)
word_set = set(model.wv.index2word)
w2v_fea_mat = list(str_df['EVT_LBL'].apply(get_word2vec_fea).values)
w2v_df = pd.DataFrame(w2v_fea_mat)   
w2v_df['USRID'] = list(str_df.index)
w2v_df.to_csv('../fea/fea_w2v_all.csv', index=False)