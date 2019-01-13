# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 21:32:49 2018

@author: Alan Yan
"""

import pandas as pd
import numpy as np
from util_fea import get_stat_fea
import time


def time2stamp(a_datetime_str):
    a_datetime = time.strptime(a_datetime_str, "%Y-%m-%d %H:%M:%S")
    return time.mktime(a_datetime)

def date2stamp(a_datetime_str):
    a_datetime_str = a_datetime_str.split(' ')[0]
    a_datetime = time.strptime(a_datetime_str, "%Y-%m-%d")
    return time.mktime(a_datetime)

def time2hour(a_datetime_str):
    a_hour = int(a_datetime_str.split(' ')[1].split(':')[0])
    return a_hour

def time2period(a_datetime_str):
    a_hour = int(a_datetime_str.split(' ')[1].split(':')[0])
    if a_hour > 8 and a_hour < 13:
        return 1
    elif a_hour >= 13 and a_hour < 19:
        return 2
    elif a_hour >= 19 and a_hour < 23:
        return 3
    else:
        return 0


def day_list_fea(a_day_list):
    if len(a_day_list) > 1:
        a_day_list.sort()
        a_sub_day_list = [a_day_list[i] - a_day_list[i-1] for i in range(1, len(a_day_list))]
        a_fea_day_list = get_stat_fea(a_sub_day_list)
    else:
        a_fea_day_list = [0] * len(get_stat_fea([1, 1]))
    return a_fea_day_list

def fea_log(a_id):
    a_df = train_log_df.loc[a_id]

    a_date_list = list(set(a_df['OCC_DATE'].values))
    a_date_list.sort()
    a_date_rate = len(a_date_list) / len(a_df) # fea
    fea_date = get_stat_fea(a_date_list)
    
    a_date_list_1 = a_date_list[:int(len(a_date_list) / 2)]
    a_date_list_2 = a_date_list[int(len(a_date_list) / 2):]
    fea_date_sub_1 = day_list_fea(a_date_list_1)
    fea_date_sub_2 = day_list_fea(a_date_list_2)
    fea_date_sub_1_2 = list(np.array(fea_date_sub_2) - np.array(fea_date_sub_1))

    a_date_count = a_df['OCC_DATE'].value_counts()
    a_date_count = a_date_count.sort_index()
    a_date_count = list(a_date_count.values)
    fea_date_count = get_stat_fea(a_date_count)

    a_fea_all = fea_date  + fea_date_sub_1_2 + fea_date_count 
    a_fea_all.append(a_date_rate)
    a_fea_all.append(a_id)

    return a_fea_all


train_log_df = pd.read_csv('../orig_data/train_log.csv', sep='\t')
#train_log_df = pd.read_csv('../orig_data/test_log.csv', sep='\t')

train_log_df.set_index(['USRID', 'OCC_TIM'], drop=False, inplace=True)
train_log_df = train_log_df.sort_index()
user_id_list = list(set(train_log_df['USRID'].values))

train_log_df['OCC_STAMP'] = train_log_df['OCC_TIM'].apply(time2stamp)
train_log_df['OCC_DATE'] = train_log_df['OCC_TIM'].apply(date2stamp)
train_log_df['OCC_PERIOD'] = train_log_df['OCC_TIM'].apply(time2period)
train_log_df['OCC_HOUR'] = train_log_df['OCC_TIM'].apply(time2hour)
fea_mat = list(map(fea_log, user_id_list))

fea_name_list_1 = ['var_1', 'std_1', 'max_1', 'min_1', 't_max_min_1', 'ent_1', 'median_1',
                   'mode_1', 'rate_1_max', 'rate_1_min', 'len_1', 'sum_1']
fea_name_list_2 = [item.replace('_1', '_2') for item in fea_name_list_1]
fea_name_list_3 = [item.replace('_1', '_3') for item in fea_name_list_1]
fea_name_list = fea_name_list_1 + fea_name_list_2 + fea_name_list_3
fea_name_list.append('date_rate')
fea_name_list.append('USRID')

fea_all_df = pd.DataFrame(fea_mat)
fea_all_df.columns = fea_name_list

fea_all_df.to_csv('../fea/fea_log_train.csv', index=False)
#fea_all_df.to_csv('../fea/fea_log_test.csv', index=False)
