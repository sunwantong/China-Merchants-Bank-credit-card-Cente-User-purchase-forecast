import pandas as pd
import numpy as np
from collections import Counter
import scipy.stats as sp
import time
import datetime



def get_continue_launch_count(strs,parm):
    time = strs.split(":")
    time = dict(Counter(time))
    time = sorted(time.items(), key=lambda x: x[0], reverse=False)
    key_list = []
    value_list = []
    if len(time) == 1:
        return -2
    for key,value in dict(time).items():
        key_list.append(int(key))
        value_list.append(int(value))

    if np.mean(np.diff(key_list, 1)) == 1:
        if parm == '1':
            return np.mean(value_list)
        elif parm == '2':
            return np.max(value_list)
        elif parm == '3':
            return np.min(value_list)
        elif parm == '4':
            return np.sum(value_list)
        elif parm == '5':
            return np.std(value_list)
    else:
        return -1



def get_time_gap(strs,parm):
    time = strs.split(":")
    time = list(set(time))
    time = sorted(list(map(lambda x:int(x),time)))
    time_gap = []
    #用户只在当天活跃
    if len(time) == 1:
        return -20

    for index, value in enumerate(time):
        if index <= len(time) - 2:
            gap = abs(time[index] - time[index + 1])
            time_gap.append(gap)

    if parm == '1':
        return np.mean(time_gap)
    elif parm == '2':
        return np.max(time_gap)
    elif parm == '3':
        return np.min(time_gap)
    elif parm == '4':
        return np.std(time_gap)
    elif parm == '5':
        return sp.stats.skew(time_gap)
    elif parm == '6':
        return sp.stats.kurtosis(time_gap)


def get_week(day):
    day = int(day)
    if day >= 1 and day <= 7:
        return 1

    if day >= 8  and  day <= 14:
        return 2

    if day >= 15 and day <= 21:
        return 3

    if day >= 22 and day <= 28:
        return 4

    if day >= 28:
        return 5


def cur_day_repeat_count(strs):
    time = strs.split(":")
    time = dict(Counter(time))
    time = sorted(time.items(), key=lambda x: x[1], reverse=False)
    # 一天一次启动
    if (len(time) == 1) & (time[0][1] == 1):
        return 0
    # 一天多次启动
    elif (len(time) == 1) & (time[0][1] > 1):
        return 1
    # 多天多次启动
    elif (len(time) > 1) & (time[0][1] >= 2):
        return 2
    else:
        return 3


def get_lianxu_day(day_list):
    time = day_list.split(":")
    time = list(map(lambda x:int(x),time))
    m = np.array(time)
    if len(set(m)) == 1:
        return -1
    m = list(set(m))
    if len(m) == 0:
        return -20
    n = np.where(np.diff(m) == 1)[0]
    i = 0
    result = []
    while i < len(n) - 1:
        state = 1
        while n[i + 1] - n[i] == 1:
            state += 1
            i += 1
            if i == len(n) - 1:
                break
        if state == 1:
            i += 1
            result.append(2)
        else:
            i += 1
            result.append(state + 1)
    if len(n) == 1:
        result.append(2)
    if len(result) != 0:
        # print(result)
        return np.max(result)


def load_csv():
    train_agg = pd.read_csv('../orig_data/train_agg.csv',sep='\t')
    train_log = pd.read_csv('../orig_data/train_log.csv', sep='\t')
    train_flg = pd.read_csv('../orig_data/train_flg.csv', sep='\t')

    test_agg = pd.read_csv('../orig_data/test_agg.csv', sep='\t')
    test_log = pd.read_csv('../orig_data/test_log.csv', sep='\t')

    return train_agg,train_log,train_flg,test_agg,test_log




def merge_table(train_agg, train_log, train_flg, test_agg, test_log):
    train_log['label'] = 1
    test_log['label'] = 0

    data = pd.concat([train_log,test_log],axis=0)
    data = extract_feature(data)

    train_log = data[data.label == 1]
    test_log = data[data.label == 0]

    del train_log['label']
    del test_log['label']

    all_train = pd.merge(train_flg, train_agg, on=['USRID'], how='left')
    train = pd.merge(all_train,train_log,on='USRID',how='left')
    test = pd.merge(test_agg,test_log,on='USRID',how='left')

    return train,test


def extract_feature(data):
    data['cate_1'] = data['EVT_LBL'].apply(lambda x: int(x.split('-')[0]))
    data['cate_2'] = data['EVT_LBL'].apply(lambda x: int(x.split('-')[1]))
    data['cate_3'] = data['EVT_LBL'].apply(lambda x: int(x.split('-')[2]))
    data['day'] = data['OCC_TIM'].apply(lambda x: int(x[8:10]))
    data['hour'] = data['OCC_TIM'].apply(lambda x: int(x[11:13]))
    data['week'] = data['day'].apply(get_week)


    feat1 = data.groupby(['USRID'], as_index=False)['OCC_TIM'].agg({"user_count": "count"})
    feat2 = data.groupby(['USRID'], as_index=False)['day'].agg({"user_act_day_count": "nunique"})
    feat3 = data[['USRID', 'day']]
    feat3['day'] = feat3['day'].astype('str')
    feat3 = feat3.groupby(['USRID'])['day'].agg(lambda x: ':'.join(x)).reset_index()
    feat3.rename(columns={'day': 'act_list'}, inplace=True)
    # 用户是否多天有多次启动(均值)
    feat3['time_gap_mean'] = feat3['act_list'].apply(get_time_gap,args=('1'))
    # 最大
    feat3['time_gap_max'] = feat3['act_list'].apply(get_time_gap,args=('2'))
    # 最小
    feat3['time_gap_min'] = feat3['act_list'].apply(get_time_gap,args=('3'))
    # 方差
    feat3['time_gap_std'] = feat3['act_list'].apply(get_time_gap,args=('4'))
    # 锋度
    feat3['time_gap_skew'] = feat3['act_list'].apply(get_time_gap, args=('5'))
    # 偏度
    feat3['time_gap_kurt'] = feat3['act_list'].apply(get_time_gap, args=('6'))
    # 平均行为次数
    feat3['mean_act_count'] = feat3['act_list'].apply(lambda x: len(x.split(":")) / len(set(x.split(":"))))
    # 平均行为日期
    feat3['act_mean_date'] = feat3['act_list'].apply(lambda x: np.sum([int(ele) for ele in x.split(":")]) / len(x.split(":")))
    # 活动天数占当月的比率
    # feat3['act_rate'] = feat3['act_list'].apply(lambda x: len(list(set(x.split(":")))) / 31)
    # 用户是否当天有多次启动
    feat3['cur_day_repeat_count'] = feat3['act_list'].apply(cur_day_repeat_count)
    # 连续几天启动次数的均值，
    feat3['con_act_day_count_mean'] = feat3['act_list'].apply(get_continue_launch_count, args=('1'))
    # 最大值，
    feat3['con_act_day_count_max'] = feat3['act_list'].apply(get_continue_launch_count, args=('2'))
    # 最小值
    feat3['con_act_day_count_min'] = feat3['act_list'].apply(get_continue_launch_count, args=('3'))
    # 次数
    feat3['con_act_day_count_total'] = feat3['act_list'].apply(get_continue_launch_count, args=('4'))
    # 方差
    feat3['con_act_day_count_std'] = feat3['act_list'].apply(get_continue_launch_count, args=('5'))
    feat3['con_act_max'] = feat3['act_list'].apply(get_lianxu_day)
    del feat3['act_list']

    # 用户发生行为的天数
    feat4 = data.groupby(['USRID'], as_index=False)['cate_1'].agg({'user_cate_1_count': "count"})
    feat5 = data.groupby(['USRID'], as_index=False)['cate_2'].agg({'user_cate_2_count': "count"})
    feat6 = data.groupby(['USRID'], as_index=False)['cate_3'].agg({'user_cate_3_count': "count"})

    # 判断时期是否为高峰日
    higt_act_day_list = [7, 14, 21, 28]
    feat8 = data[['USRID', 'day']]
    feat8['is_higt_act'] = feat8['day'].apply(lambda x: 1 if x in higt_act_day_list else 0)
    feat8 = feat8.drop_duplicates(subset=['USRID'])


    feat10 = data.groupby(['USRID','day'], as_index=False)['TCH_TYP'].agg({'user_per_count': "count"})
    feat10_copy = feat10.copy()
    # 用户平均每天启动次数
    feat11 = feat10_copy.groupby(['USRID'],as_index=False)['user_per_count'].agg({"user_per_count_mean":"mean"})
    # 用户启动次数最大值
    feat12 = feat10_copy.groupby(['USRID'], as_index=False)['user_per_count'].agg({"user_per_count_max": "max"})
    # 用户启动次数最小值
    feat13 = feat10_copy.groupby(['USRID'], as_index=False)['user_per_count'].agg({"user_per_count_min": "min"})
    # 用户每天启动次数的众值
    feat14 = feat10_copy.groupby(['USRID'], as_index=False)['user_per_count'].agg({"user_mode_count":lambda x: x.value_counts().index[0]})
    # 方差
    feat15 = feat10_copy.groupby(['USRID'], as_index=False)['user_per_count'].agg({"user_std_count":np.std})
    # 峰度
    feat16 = feat10_copy.groupby(['USRID'], as_index=False)['user_per_count'].agg({"user_skew_count": sp.stats.skew})
    # 偏度
    feat17 = feat10_copy.groupby(['USRID'], as_index=False)['user_per_count'].agg({"user_kurt_count": sp.stats.kurtosis})
    # 中位数
    feat18 = feat10_copy.groupby(['USRID'], as_index=False)['user_per_count'].agg({"user_median_count": np.median})

    feat27 = data[['USRID', 'OCC_TIM']]
    feat27['OCC_TIM'] = feat27['OCC_TIM'].apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
    log = feat27.sort_values(['USRID', 'OCC_TIM'])
    log['next_time'] = log.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)
    log = log.groupby(['USRID'], as_index=False)['next_time'].agg({
        'next_time_mean': np.mean,
        'next_time_std': np.std,
        'next_time_min': np.min,
        'next_time_max': np.max
    })

    # 每周的平均消费次数
    feat28_sp = data.groupby(['USRID','week'], as_index=False)['TCH_TYP'].agg({'user_per_week_count': "count"})
    feat28_sp_copy = feat28_sp.copy()
    # 用户平均每天启动次数
    feat11_sp = feat28_sp_copy.groupby(['USRID'], as_index=False)['user_per_week_count'].agg({"user_per_week_count_mean": "mean"})
    # 用户启动次数最大值
    feat12_sp = feat28_sp_copy.groupby(['USRID'], as_index=False)['user_per_week_count'].agg({"user_per_week_count_max": "max"})
    # 用户启动次数最小值
    feat13_sp = feat28_sp_copy.groupby(['USRID'], as_index=False)['user_per_week_count'].agg({"user_per_week_count_min": "min"})
    # 用户每天启动次数的众值
    feat14_sp = feat28_sp_copy.groupby(['USRID'], as_index=False)['user_per_week_count'].agg({"user_per_week_count_mode": lambda x: x.value_counts().index[0]})
    # 方差
    feat15_sp = feat28_sp_copy.groupby(['USRID'], as_index=False)['user_per_week_count'].agg({"user_per_week_count_std": np.std})
    # 峰度
    feat16_sp = feat28_sp_copy.groupby(['USRID'], as_index=False)['user_per_week_count'].agg({"user_per_week_count_skew": sp.stats.skew})
    # 偏度
    feat17_sp = feat28_sp_copy.groupby(['USRID'], as_index=False)['user_per_week_count'].agg({"user_per_week_count_kurt": sp.stats.kurtosis})
    # 中位数
    feat18_sp = feat28_sp_copy.groupby(['USRID'], as_index=False)['user_per_week_count'].agg({"user_per_week_count_median": np.median})




    # 离周末越近，越消费的可能性比较大，统计前2天的特征
    before_three = data[(data.day >= 28) & (data.day <= 31)]
    before_three_copy = before_three.copy()

    feat1_before = before_three_copy.groupby(['USRID'], as_index=False)['OCC_TIM'].agg({"user_count_before": "count"})
    feat2_before = before_three_copy.groupby(['USRID'], as_index=False)['day'].agg({"user_act_day_count_before": "nunique"})
    feat3_before = before_three_copy[['USRID', 'day']]
    feat3_before['day'] = feat3_before['day'].astype('str')
    feat3_before = feat3_before.groupby(['USRID'])['day'].agg(lambda x: ':'.join(x)).reset_index()
    feat3_before.rename(columns={'day': 'act_list'}, inplace=True)
    # 用户是否多天有多次启动(均值)
    feat3_before['before_time_gap_mean'] = feat3_before['act_list'].apply(get_time_gap, args=('1'))
    # 最大
    feat3_before['before_time_gap_max'] = feat3_before['act_list'].apply(get_time_gap, args=('2'))
    # 最小
    feat3_before['before_time_gap_min'] = feat3_before['act_list'].apply(get_time_gap, args=('3'))
    # 方差
    feat3_before['before_time_gap_std'] = feat3_before['act_list'].apply(get_time_gap, args=('4'))
    # 锋度
    feat3_before['before_time_gap_skew'] = feat3_before['act_list'].apply(get_time_gap, args=('5'))
    # 偏度
    feat3_before['before_time_gap_kurt'] = feat3_before['act_list'].apply(get_time_gap, args=('6'))
    # 平均行为次数
    feat3_before['before_mean_act_count'] = feat3_before['act_list'].apply(lambda x: len(x.split(":")) / len(set(x.split(":"))))
    # 平均行为日期
    feat3_before['before_act_mean_date'] = feat3_before['act_list'].apply(lambda x: np.sum([int(ele) for ele in x.split(":")]) / len(x.split(":")))
    # 用户是否当天有多次启动
    feat3_before['before_cur_day_repeat_count'] = feat3_before['act_list'].apply(cur_day_repeat_count)
    # 连续几天启动次数的均值，
    feat3_before['before_con_act_day_count_mean'] = feat3_before['act_list'].apply(get_continue_launch_count, args=('1'))
    # 最大值，
    feat3_before['before_con_act_day_count_max'] = feat3_before['act_list'].apply(get_continue_launch_count, args=('2'))
    # 最小值
    feat3_before['before_con_act_day_count_min'] = feat3_before['act_list'].apply(get_continue_launch_count, args=('3'))
    # 次数
    feat3_before['before_con_act_day_count_total'] = feat3_before['act_list'].apply(get_continue_launch_count, args=('4'))
    # 方差
    feat3_before['before_con_act_day_count_std'] = feat3_before['act_list'].apply(get_continue_launch_count, args=('5'))
    feat3_before['before_con_act_max'] = feat3_before['act_list'].apply(get_lianxu_day)
    del feat3_before['act_list']

    # 用户发生行为的天数
    feat4_before = before_three.groupby(['USRID'], as_index=False)['cate_1'].agg({'before_user_cate_1_count': "count"})
    feat5_before = before_three.groupby(['USRID'], as_index=False)['cate_2'].agg({'before_user_cate_2_count': "count"})
    feat6_before = before_three.groupby(['USRID'], as_index=False)['cate_3'].agg({'before_user_cate_3_count': "count"})


    feat28 = pd.crosstab(data['USRID'],data['TCH_TYP']).reset_index()
    feat29 = pd.crosstab(data.USRID,data.cate_1).reset_index()
    feat30 = pd.crosstab(data.USRID, data.cate_2).reset_index()
    feat31 = pd.crosstab(data.USRID, data.cate_3).reset_index()
    feat32 = pd.crosstab(data.USRID,data.hour).reset_index()
    feat34 = pd.crosstab(data.USRID,data.week).reset_index()












    data = data[['USRID','label']]
    data = data.drop_duplicates(subset='USRID')
    data = pd.merge(data, feat1, on=['USRID'], how='left')
    data = pd.merge(data, feat2, on=['USRID'], how='left')
    data = pd.merge(data, feat3, on=['USRID'], how='left')
    data = pd.merge(data, feat4, on=['USRID'], how='left')
    data = pd.merge(data, feat5, on=['USRID'], how='left')
    data = pd.merge(data, feat6, on=['USRID'], how='left')
    data = pd.merge(data, feat8, on=['USRID'], how='left')
    data = pd.merge(data, feat11, on=['USRID'], how='left')
    data = pd.merge(data, feat12, on=['USRID'], how='left')
    data = pd.merge(data, feat13, on=['USRID'], how='left')
    data = pd.merge(data, feat14, on=['USRID'], how='left')
    data = pd.merge(data, feat15, on=['USRID'], how='left')
    data = pd.merge(data, feat16, on=['USRID'], how='left')
    data = pd.merge(data, feat17, on=['USRID'], how='left')
    data = pd.merge(data, feat18, on=['USRID'], how='left')
    data = pd.merge(data, log, on=['USRID'], how='left')
    data = pd.merge(data, feat28, on=['USRID'], how='left')
    data = pd.merge(data, feat29, on=['USRID'], how='left')
    data = pd.merge(data, feat30, on=['USRID'], how='left')
    data = pd.merge(data, feat31, on=['USRID'], how='left')
    data = pd.merge(data, feat32, on=['USRID'], how='left')
    data = pd.merge(data, feat34, on=['USRID'], how='left')

    data = pd.merge(data, feat11_sp, on=['USRID'], how='left')
    data = pd.merge(data, feat12_sp, on=['USRID'], how='left')
    data = pd.merge(data, feat13_sp, on=['USRID'], how='left')
    data = pd.merge(data, feat14_sp, on=['USRID'], how='left')
    data = pd.merge(data, feat15_sp, on=['USRID'], how='left')
    data = pd.merge(data, feat16_sp, on=['USRID'], how='left')
    data = pd.merge(data, feat17_sp, on=['USRID'], how='left')
    data = pd.merge(data, feat18_sp, on=['USRID'], how='left')

    data = pd.merge(data, feat1_before, on=['USRID'], how='left')
    data = pd.merge(data, feat2_before, on=['USRID'], how='left')
    data = pd.merge(data, feat3_before, on=['USRID'], how='left')
    data = pd.merge(data, feat4_before, on=['USRID'], how='left')
    data = pd.merge(data, feat5_before, on=['USRID'], how='left')
    data = pd.merge(data, feat6_before, on=['USRID'], how='left')

    return data


def main():
    train_agg, train_log, train_flg, test_agg, test_log = load_csv()
    train, test = merge_table(train_agg, train_log, train_flg, test_agg, test_log)
    train.to_csv('../fea/train.csv',sep='\t',index=None)
    test.to_csv('../fea/test.csv', sep='\t', index=None)


if __name__ == '__main__':
    main()