# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 17:48:05 2018

@author: Alan Yan
"""

import xgboost as xgb

def xgb_fea_select(train_x, train_y, f_name_list):
    rate_fea = 0.5
    dtrain=xgb.DMatrix(train_x, label=train_y)
    params = {'booster':'gbtree',
              'max_depth': 3,
              'colsample_bytree': 0.7,
              'subsample': 0.7, 
              'eta': 0.03,
              'silent': 1,
              # 'objective': 'binary:logistic',
             'objective': 'rank:pairwise',
              'min_child_weight': 3,
              'seed': 10,
              'eval_metric':'auc',
              'scale_pos_weight': 3176 / 76824}
    watchlist = [(dtrain,'train')]
    bst=xgb.train(params,dtrain,num_boost_round=1000,evals=watchlist, 
                  early_stopping_rounds=100)
    fscore_dict = bst.get_fscore()
    sorted_fs_dict = sorted(fscore_dict.items(),key = lambda x:x[1],reverse = True)
    fea_id_set = set([int(item[0][1:]) for item in sorted_fs_dict[:int(len(sorted_fs_dict)*rate_fea)]])
    f_name_list = [item for i, item in enumerate(f_name_list) if i in fea_id_set]
    return f_name_list