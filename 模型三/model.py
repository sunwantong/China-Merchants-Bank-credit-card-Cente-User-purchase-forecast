# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 19:49:57 2018

@author: Alan Yan
"""

import xgboost as xgb
import pandas as pd
from sklearn import metrics
import numpy as np
from scipy.sparse import csr_matrix
from util_fs import xgb_fea_select


def xgb_clf(train_x, train_y, test_x):
    dtrain=xgb.DMatrix(train_x,label=train_y)
    dtest=xgb.DMatrix(test_x)
    params = {'booster':'gbtree',
              'max_depth': 3,
              'colsample_bytree': 0.7,
              'subsample': 0.7, 
              'eta': 0.03,
              'silent': 1,
#              'objective': 'binary:logistic',
              'objective': 'rank:pairwise',
              'min_child_weight': 6,  # 这儿不是3就是6
              'seed': 10,
              'eval_metric':'auc',
              'scale_pos_weight': 3176 / 76824}
    watchlist = [(dtrain,'train')]
    bst=xgb.train(params,dtrain,num_boost_round=1000,evals=watchlist, 
                  early_stopping_rounds=100)
    ypred=bst.predict(dtest)
    return ypred

train_log_df = pd.read_csv('../fea/fea_log_train.csv')
train_agg_df = pd.read_csv('../orig_data/train_agg.csv', sep='\t')
train_df = pd.merge(train_agg_df, train_log_df, on='USRID', how='left')
train_label_df = pd.read_csv('../orig_data/train_flg.csv', sep='\t')
train_df = pd.merge(train_df, train_label_df, on='USRID', how='left')
tfidf_df = pd.read_csv('../fea/fea_tfidf_all.csv')
train_df = pd.merge(tfidf_df, train_df, on='USRID', how='right')
w2v_df = pd.read_csv('../fea/fea_w2v_all.csv')
train_df = pd.merge(w2v_df, train_df, on='USRID', how='right')

test_log_df = pd.read_csv('../fea/fea_log_test.csv')
test_agg_df = pd.read_csv('../orig_data/test_agg.csv', sep='\t')
test_df = pd.merge(test_agg_df, test_log_df, on='USRID', how='left')
test_df = pd.merge(tfidf_df, test_df, on='USRID', how='right')
test_df = pd.merge(w2v_df, test_df, on='USRID', how='right')


# 特征选择
y_train = train_df['FLAG'].values
del train_df['FLAG'], train_df['USRID']
X_train = train_df.values
fea_name_list = train_df.columns
X_train = X_train.astype(np.float64)
X_train = csr_matrix(X_train)
fea_name_new = xgb_fea_select(X_train, y_train, fea_name_list)
train_df = train_df[fea_name_new]
print('特征选择完成。')


test_id = test_df['USRID'].values
del test_df['USRID']
test_df = test_df[fea_name_new]

X_train = train_df.values
X_train = X_train.astype(np.float64)
X_train = csr_matrix(X_train)
X_test = test_df.values
X_test = X_test.astype(np.float64)
X_test = csr_matrix(X_test)

y_pred_prob = xgb_clf(X_train, y_train, X_test)
pd.Series(np.sort(y_pred_prob)).plot()

result_df = pd.DataFrame()
result_df['USRID'] = test_id
result_df['RST'] = y_pred_prob
result_df.to_csv('../result/test_result.csv', index=False, sep='\t')