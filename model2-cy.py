# -*- coding: utf-8 -*-
#用来划分训练集和验证集
import pandas as pd

import xgboost as xgb
import operator
from functools import reduce
import numpy as np
import scipy.stats as sp
import pandas as pd
train_user_info = pd.read_csv(r'train_agg.csv',sep='\t')
train_app_log = pd.read_csv(r'train_log.csv',sep='\t')

train_flag = pd.read_csv(r'train_flg.csv',sep='\t')

test_user_info = pd.read_csv(r'test_agg.csv',sep='\t')
test_app_log = pd.read_csv(r'test_log.csv',sep='\t')

#添加week
train_app_log['week'] =pd.to_datetime(train_app_log.OCC_TIM)
train_app_log.week = list(map(lambda x:x.weekday(),train_app_log.week))
test_app_log['week'] =pd.to_datetime(test_app_log.OCC_TIM)
test_app_log.week = list(map(lambda x:x.weekday(),test_app_log.week))


# 切割字符串--训练集
temp=train_app_log['EVT_LBL'].str.split('-')
temp1 = list(map(lambda line: line[0],temp))
temp2 = list(map(lambda line: line[1],temp))
temp3 = list(map(lambda line: line[2],temp))
train_app_log['EVT_LBL_1'] = temp1
train_app_log['EVT_LBL_2'] = temp2
train_app_log['EVT_LBL_3'] = temp3


temptemp=train_app_log['OCC_TIM'].str.split(' ')
# 加日
temp = list(map(lambda line: line[0], temptemp))
train_app_log['time'] = temp
time = train_app_log['time'].str.split('-')
day = list(map(lambda line: line[2], time))
train_app_log['day'] = day
del train_app_log['time']
# 加时分秒
temp = list(map(lambda line: line[1], temptemp))
train_app_log['time'] = temp
time = train_app_log['time'].str.split(':')
hour = list(map(lambda line: line[0], time))
minu = list(map(lambda line: line[1], time))
sec = list(map(lambda line: line[2], time))
train_app_log['hour'] = hour
train_app_log['minu'] = minu
train_app_log['sec'] = sec
del train_app_log['time']
train_app_log.hour = list(map(lambda x:int(x),train_app_log.hour))
train_app_log.minu = list(map(lambda x:int(x),train_app_log.minu))
train_app_log.sec = list(map(lambda x:int(x),train_app_log.sec))
train_app_log.day = list(map(lambda x:int(x),train_app_log.day))

# 切割字符串--测试集
temp=test_app_log['EVT_LBL'].str.split('-')
temp1 = list(map(lambda line: line[0], temp))
temp2 = list(map(lambda line: line[1], temp))
temp3 = list(map(lambda line: line[2], temp))
test_app_log['EVT_LBL_1'] = temp1
test_app_log['EVT_LBL_2'] = temp2
test_app_log['EVT_LBL_3'] = temp3


temptemp=test_app_log['OCC_TIM'].str.split(' ')
# 加日
temp = list(map(lambda line: line[0], temptemp))
test_app_log['time'] = temp
time = test_app_log['time'].str.split('-')
day = list(map(lambda line: line[2], time))
test_app_log['day'] = day
del test_app_log['time']
# 加时分秒
temp = list(map(lambda line: line[1], temptemp))
test_app_log['time'] = temp
time = test_app_log['time'].str.split(':')
hour = list(map(lambda line: line[0], time))
minu = list(map(lambda line: line[1], time))
sec = list(map(lambda line: line[2], time))
test_app_log['hour'] = hour
test_app_log['minu'] = minu
test_app_log['sec'] = sec
del test_app_log['time']
test_app_log.hour = list(map(lambda x:int(x),test_app_log.hour))
test_app_log.minu = list(map(lambda x:int(x),test_app_log.minu))
test_app_log.sec = list(map(lambda x:int(x),test_app_log.sec))
test_app_log.day = list(map(lambda x:int(x),test_app_log.day))
def get_av_time_dis(x):#apply

    x=x.str_day
    if x!=x:
        x='-1'
    # print(x)
    day = x.split(':')
    day = list(set(day))
    day = list(map(lambda x:float(x),day))
    day.sort()
    if day is None or len(day) == 0:
        return 0
    m={}
    res = 0
    for i in day:
        if i not in m:
            l=0
            r=0
            if i-1 in m:
                l = m[i-1]
            if i+1 in m:
                r = m[i+1]
            m[i] = 1+r+l
            m[i+r] = 1+r+l
            m[i-l] = 1+r+l
            res = max(res,m[i])
    return res
# 训练
# xgboost
def xgboosts(df_train,df_test,df_eval):


    print('xgb---training')
    # XGB  'shop_star_level','shop_review_num_level','context_page_id','item_pv_level','item_collected_level','item_sales_level','item_price_level','user_star_level','user_occupation_id','user_age_level','item_category_list3','item_category_list2','item_category_list1','item_city_id','item_brand_id','context_id',
    feature1 = [x for x in df_train.columns if x not in ['USRID','EVT_LBL','OCC_TIM','TCH_TYP','FLAG']]
    feature2 = [x for x in df_test.columns if x not in ['USRID','EVT_LBL','OCC_TIM','TCH_TYP','FLAG']]
    feature = [v for v in feature1 if v in feature2]

    dtrain = xgb.DMatrix(df_train[feature].values,df_train['FLAG'].values)
    dpre = xgb.DMatrix(df_test[feature].values)
    deva = xgb.DMatrix(df_eval[feature].values,df_eval['FLAG'].values)
    deva2 = xgb.DMatrix(df_eval[feature].values)
    param = {'max_depth': 5,
             'eta': 0.02,
             # 'objective': 'binary:logistic',
             'objective': 'rank:pairwise',
             'eval_metric': 'auc',
             'colsample_bytree':0.8,
             'subsample':0.8,
             'scale_pos_weight':1,
             # 'booster':'gblinear',
             'silent':1,
             'min_child_weight':18
             }
    # param['nthread'] =5
    print('xxxxxx')
    watchlist = [(deva, 'eval'), (dtrain, 'train')]
    num_round =600
    bst = xgb.train(param, dtrain, num_round, watchlist)
    print('xxxxxx')
    # 进行预测
    # dtest= xgb.DMatrix(predict)
    preds2 = bst.predict(dpre)
    # 保存整体结果。
    predict = df_test[['USRID']]
    predict['rst'] = preds2
    # temp = predict.drop_duplicates(['user_id'])  # 去重
    predict.to_csv('test_result.csv', encoding='utf-8', index=None,sep='\t')
# 提取特征
def getF(user_info,data):
    # 1、用户总的点击次数
    temp = data.groupby(['USRID'])['EVT_LBL'].agg({'user_sum': np.size})  #
    temp = temp.reset_index()
    result = pd.merge(user_info, temp, on=['USRID'], how='left')  #
#     2、取第一级模块,用户对各个模块的点击数
    data['EVT_LBL_1_new1'] = list(map(lambda x: 'EVT_LBL_1_new1' + str(x), data.EVT_LBL_1))
    temp = pd.crosstab(data.USRID,data.EVT_LBL_1_new1).reset_index()
    del data['EVT_LBL_1_new1']
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    # 3、取第二级模块,用户对各个模块的点击数
    data['EVT_LBL_2_new1'] = list(map(lambda x: 'EVT_LBL_2_new1' + str(x), data.EVT_LBL_1))
    temp = pd.crosstab(data.USRID,data.EVT_LBL_2_new1).reset_index()
    del data['EVT_LBL_2_new1']
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    # 3、取第三级模块,用户对各个模块的点击数
    data['EVT_LBL_3_new1'] = list(map(lambda x: 'EVT_LBL_3_new1' + str(x), data.EVT_LBL_1))
    temp = pd.crosstab(data.USRID,data.EVT_LBL_3_new1).reset_index()
    del data['EVT_LBL_3_new1']
    result = pd.merge(result, temp, on=['USRID'], how='left')  #

# 6、各个用户在各个小时的点击量，离散
    data['hour_new1'] = list(map(lambda x: 'hour_new1' + str(x), data.hour))
    temp = pd.crosstab(data.USRID,data.hour_new1).reset_index()
    del data['hour_new1']
    result = pd.merge(result, temp, on=['USRID'], how='left')  #



# 7、各个用户在各个星期几的点击量，离散
    data['week_new1'] = list(map(lambda x: 'week_new1' + str(x), data.week))
    temp = pd.crosstab(data.USRID,data.week_new1).reset_index()
    del data['week_new1']
    result = pd.merge(result, temp, on=['USRID'], how='left')  #


# 9、用户的平均点击时间间隔，最大时间间隔，最小时间间隔，
    temp = data.sort_values(['OCC_TIM'], ascending=True)
    temp['OCC_TIM'] =pd.to_datetime(temp.OCC_TIM)
    temp['next_time'] = temp.groupby(['USRID'])['OCC_TIM'].diff(1)
    temp['next_time'] = temp['next_time']/np.timedelta64(1,'s')
    temp2=temp
    # average
    temp = temp.groupby(['USRID'])['next_time'].agg({'avg_time': np.mean})
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    # median
    temp = temp2.groupby(['USRID'])['next_time'].agg({'medain_time': np.median})
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
#     max
    temp = temp2.groupby(['USRID'])['next_time'].agg({'max_time': np.max})
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
#     min
    temp = temp2.groupby(['USRID'])['next_time'].agg({'min_time': np.min})
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    #     sp.stats.skew 偏度
    temp = temp2.groupby(['USRID'])['next_time'].agg({'skew_time': sp.stats.skew})
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    #     sp.stats.kurtosis 峰度
    temp = temp2.groupby(['USRID'])['next_time'].agg({'kurt_time': sp.stats.kurtosis})
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #

# 10、用户有多少天点击/有多少天点击多次
    temp = data.drop_duplicates(['USRID','day'])#去重
    temp = temp.groupby(['USRID'])['day'].agg({'howmany_day_click': np.size})
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #

# 12、用户是否重复点击过同一个模块
# 14、用户对于各个事件类型历史发生的数量。
    data['TCH_TYP_new1'] = list(map(lambda x: 'TCH_TYP_new1' + str(x), data.TCH_TYP))
    temp = pd.crosstab(data.USRID,data.TCH_TYP_new1).reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #

    # 用户每天平均点击量
    temp = data.groupby(['USRID','day'])['EVT_LBL'].agg({'day_user_sum': np.size})  #
    temp = temp.reset_index()
    data = pd.merge(data, temp, on=['USRID','day'], how='left')  #
    temp = data.drop_duplicates(['USRID','day'])#去重
    temp = temp.groupby(['USRID'])['day_user_sum'].agg({'day_user_mean': np.mean})  #
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    # 用户每天点击量方差
    temp = data.drop_duplicates(['USRID','day'])#去重
    temp = temp.groupby(['USRID'])['day_user_sum'].agg({'day_user_var': np.var})  #
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    # 用户每天点击量标准差
    temp = data.drop_duplicates(['USRID','day'])#去重
    temp = temp.groupby(['USRID'])['day_user_sum'].agg({'day_user_std': np.std})  #
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    # 用户每天点击量中位数
    temp = data.drop_duplicates(['USRID','day'])#去重
    temp = temp.groupby(['USRID'])['day_user_sum'].agg({'day_user_median': np.median})  #
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    # 用户每天点击量max
    temp = data.drop_duplicates(['USRID','day'])#去重
    temp = temp.groupby(['USRID'])['day_user_sum'].agg({'day_user_max': np.max})  #
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    # 用户每天点击量min
    temp = data.drop_duplicates(['USRID','day'])#去重
    temp = temp.groupby(['USRID'])['day_user_sum'].agg({'day_user_min': np.min})  #
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    # 用户每天点击量sp.stats.skew 偏度
    temp = data.drop_duplicates(['USRID','day'])#去重
    temp = temp.groupby(['USRID'])['day_user_sum'].agg({'day_user_skew': sp.stats.skew})  #
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    # 用户每天点击量sp.stats.kurtosis 峰度
    temp = data.drop_duplicates(['USRID','day'])#去重
    temp = temp.groupby(['USRID'])['day_user_sum'].agg({'day_user_kurt': sp.stats.kurtosis})  #
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    # 用户最大连续点击天数
    temp = data[['USRID', 'day']]
    temp['day'] = temp['day'].astype('str')
    temp = temp.groupby(['USRID'])['day'].agg(lambda x: ':'.join(x)).reset_index()
    temp = temp.drop_duplicates(['USRID', 'day'])  # 去重
    temp.rename(columns={'day': 'str_day'}, inplace=True)
    temp['max_continue_day'] = temp.apply(get_av_time_dis, axis=1)#apply
    temp = temp[['USRID','max_continue_day']]
    result = pd.merge(result, temp, on=['USRID'], how='left')  #

     # 118、区间内最后一次活跃距离区间末端的天数

    temp = data.groupby(['USRID'])['day'].agg({'last_day': np.max}).reset_index()  #
    temp.last_day = 30-temp.last_day
    result = pd.merge(result, temp, on=['USRID'], how='left')  #



    # 最后一天的统计值
    Fregion1 = data[data.day==30]
    # 1、用户总的点击次数
    temp = Fregion1.groupby(['USRID'])['EVT_LBL'].agg({'user_sum_30': np.size})  #
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
#     2、取第一级模块,用户对各个模块的点击数
    Fregion1['EVT_LBL_1_new1'] = list(map(lambda x: 'EVT_LBL_1_new1_' + str(x), Fregion1.EVT_LBL_1))
    temp = pd.crosstab(Fregion1.USRID,Fregion1.EVT_LBL_1_new1).reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    # 3、取第二级模块,用户对各个模块的点击数
    Fregion1['EVT_LBL_2_new1'] = list(map(lambda x: 'EVT_LBL_2_new1_' + str(x), Fregion1.EVT_LBL_2))
    temp = pd.crosstab(Fregion1.USRID,Fregion1.EVT_LBL_2_new1).reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    # 3、取第三级模块,用户对各个模块的点击数
    Fregion1['EVT_LBL_3_new1'] = list(map(lambda x: 'EVT_LBL_3_new1_' + str(x), Fregion1.EVT_LBL_3))
    temp = pd.crosstab(Fregion1.USRID,Fregion1.EVT_LBL_3_new1).reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #

# 6、各个用户在各个小时的点击量，离散
    Fregion1['hour_new1'] = list(map(lambda x: 'hour_new1_' + str(x), Fregion1.hour))
    temp = pd.crosstab(Fregion1.USRID,Fregion1.hour_new1).reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
# 14、用户对于各个事件类型历史发生的数量。
    Fregion1['TCH_TYP_new1'] = list(map(lambda x: 'TCH_TYP_new1_' + str(x), Fregion1.TCH_TYP))
    temp = pd.crosstab(Fregion1.USRID,Fregion1.TCH_TYP_new1).reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
#
    # 最后2天的统计值
    Fregion1 = data[data.day>=29]
    # 1、用户总的点击次数
    temp = Fregion1.groupby(['USRID'])['EVT_LBL'].agg({'user_sum_29': np.size})  #
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
#     2、取第一级模块,用户对各个模块的点击数
    Fregion1['EVT_LBL_1_new2'] = list(map(lambda x: 'EVT_LBL_1_new2_' + str(x), Fregion1.EVT_LBL_1))
    temp = pd.crosstab(Fregion1.USRID,Fregion1.EVT_LBL_1_new2).reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    # 3、取第二级模块,用户对各个模块的点击数
    Fregion1['EVT_LBL_2_new2'] = list(map(lambda x: 'EVT_LBL_2_new2_' + str(x), Fregion1.EVT_LBL_2))
    temp = pd.crosstab(Fregion1.USRID,Fregion1.EVT_LBL_2_new2).reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    # 3、取第三级模块,用户对各个模块的点击数
    Fregion1['EVT_LBL_3_new2'] = list(map(lambda x: 'EVT_LBL_3_new2_' + str(x), Fregion1.EVT_LBL_3))
    temp = pd.crosstab(Fregion1.USRID,Fregion1.EVT_LBL_3_new2).reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #

# 6、各个用户在各个小时的点击量，离散
    Fregion1['hour_new2'] = list(map(lambda x: 'hour_new2_' + str(x), Fregion1.hour))
    temp = pd.crosstab(Fregion1.USRID,Fregion1.hour_new2).reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
# 14、用户对于各个事件类型历史发生的数量。
    Fregion1['TCH_TYP_new2'] = list(map(lambda x: 'TCH_TYP_new2_' + str(x), Fregion1.TCH_TYP))
    temp = pd.crosstab(Fregion1.USRID,Fregion1.TCH_TYP_new2).reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #

    # ################################比例特征##################################################################################
    #1，用户的各种行为类型占用户的总行为比例(总的)
    result['tch_type_0_rate'] = result['TCH_TYP_new10']/result['user_sum']
    # result['tch_type_1_rate'] = result['TCH_TYP_new11'] / result['user_sum']
    result['tch_type_2_rate'] = result['TCH_TYP_new12'] / result['user_sum']
    # 2,用户各个星期点击量占比（总的）
    result['week_1_rate'] = result['week_new11'] / result['user_sum']
    result['week_2_rate'] = result['week_new12'] / result['user_sum']
    result['week_3_rate'] = result['week_new13'] / result['user_sum']
    result['week_4_rate'] = result['week_new14'] / result['user_sum']
    result['week_5_rate'] = result['week_new15'] / result['user_sum']
    result['week_6_rate'] = result['week_new16'] / result['user_sum']
    result['week_7_rate'] = result['week_new10'] / result['user_sum']
    # 3,用户各个小时的点击量占比（总的）
    result['hour_0_rate'] = result['hour_new10'] / result['user_sum']
    result['hour_1_rate'] = result['hour_new11'] / result['user_sum']
    result['hour_2_rate'] = result['hour_new12'] / result['user_sum']
    result['hour_3_rate'] = result['hour_new13'] / result['user_sum']
    result['hour_4_rate'] = result['hour_new14'] / result['user_sum']
    result['hour_5_rate'] = result['hour_new15'] / result['user_sum']
    result['hour_6_rate'] = result['hour_new16'] / result['user_sum']
    result['hour_7_rate'] = result['hour_new17'] / result['user_sum']
    result['hour_8_rate'] = result['hour_new18'] / result['user_sum']
    result['hour_9_rate'] = result['hour_new19'] / result['user_sum']
    result['hour_10_rate'] = result['hour_new110'] / result['user_sum']
    result['hour_11_rate'] = result['hour_new111'] / result['user_sum']
    result['hour_12_rate'] = result['hour_new112'] / result['user_sum']
    result['hour_13_rate'] = result['hour_new113'] / result['user_sum']
    result['hour_14_rate'] = result['hour_new114'] / result['user_sum']
    result['hour_15_rate'] = result['hour_new115'] / result['user_sum']
    result['hour_16_rate'] = result['hour_new116'] / result['user_sum']
    result['hour_17_rate'] = result['hour_new117'] / result['user_sum']
    result['hour_18_rate'] = result['hour_new118'] / result['user_sum']
    result['hour_19_rate'] = result['hour_new119'] / result['user_sum']
    result['hour_20_rate'] = result['hour_new120'] / result['user_sum']
    result['hour_21_rate'] = result['hour_new121'] / result['user_sum']
    result['hour_22_rate'] = result['hour_new122'] / result['user_sum']
    result['hour_23_rate'] = result['hour_new123'] / result['user_sum']

    # 1，用户的各种行为类型占用户的总行为比例(最后一天)
    result['TCH_TYP_new1_0_rate'] = result['TCH_TYP_new1_0'] / result['user_sum_30']
    # result['TCH_TYP_new1_1_rate'] = result['TCH_TYP_new1_1'] / result['user_sum_30']
    result['TCH_TYP_new1_2_rate'] = result['TCH_TYP_new1_2'] / result['user_sum_30']

   

    # 1，用户的各种行为类型占用户的总行为比例(最后2天)
    result['TCH_TYP_new2_0_rate'] = result['TCH_TYP_new2_0'] / result['user_sum_29']
    # result['TCH_TYP_new2_1_rate'] = result['TCH_TYP_new2_1'] / result['user_sum_29']
    result['TCH_TYP_new2_2_rate'] = result['TCH_TYP_new2_2'] / result['user_sum_29']

   
    ####################################################################################################################
    # 分为前10天，中间10天，后十天
    # 前10的统计值
    Fregion1 = data[data.day <=10]
    # 1、用户总的点击次数
    temp = Fregion1.groupby(['USRID'])['EVT_LBL'].agg({'user_sum_one': np.size})  #
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
   
    # 14、用户对于各个事件类型历史发生的数量。
    Fregion1['TCH_TYP_new3'] = list(map(lambda x: 'TCH_TYP_new3_' + str(x), Fregion1.TCH_TYP))
    temp = pd.crosstab(Fregion1.USRID, Fregion1.TCH_TYP_new3).reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
    ##########################3
    # 中间10的统计值
    Fregion1 = data[(data.day > 10)&(data.day<=20)]
    # 1、用户总的点击次数
    temp = Fregion1.groupby(['USRID'])['EVT_LBL'].agg({'user_sum_two': np.size})  #
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
   
    # 14、用户对于各个事件类型历史发生的数量。
    Fregion1['TCH_TYP_new4'] = list(map(lambda x: 'TCH_TYP_new4_' + str(x), Fregion1.TCH_TYP))
    temp = pd.crosstab(Fregion1.USRID, Fregion1.TCH_TYP_new4).reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #

    ##################################3
    # 最后10的统计值
    Fregion1 = data[data.day > 20]
    # 1、用户总的点击次数
    temp = Fregion1.groupby(['USRID'])['EVT_LBL'].agg({'user_sum_three': np.size})  #
    temp = temp.reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
   
    # 14、用户对于各个事件类型历史发生的数量。
    Fregion1['TCH_TYP_new5'] = list(map(lambda x: 'TCH_TYP_new5_' + str(x), Fregion1.TCH_TYP))
    temp = pd.crosstab(Fregion1.USRID, Fregion1.TCH_TYP_new5).reset_index()
    result = pd.merge(result, temp, on=['USRID'], how='left')  #
  




    print(result)
    return result

train = getF(train_user_info,train_app_log)
train = pd.merge(train, train_flag, on=['USRID'], how='left')  #
test = getF(test_user_info,test_app_log)
train.to_csv('train_last.csv', encoding='utf-8', index=None)
test.to_csv('test_last.csv', encoding='utf-8', index=None)
train = pd.read_csv(r'train_last.csv')
test = pd.read_csv(r'test_last.csv')
xgboosts(train,test,train)
