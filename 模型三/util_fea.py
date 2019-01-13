# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:43:50 2018

@author: Alan Yan
"""
from math import log
from scipy.stats import mode
import numpy as np


def cal_ent(a_list):
    item_count = {}
    for item in a_list:
        if item not in item_count.keys():
            item_count[item] = 0
        item_count[item] += 1
    ent = 0.0
    for key in item_count:
        prob = float(item_count[key]) / len(a_list)
        ent -= prob * log(prob, 2)
    return ent

def get_stat_fea(a_list):
    t_array = np.array(a_list)
    var_t = t_array.var()  # t序列的方差
    std_t = t_array.std()  # t序列的标准差
    max_t = t_array.max()  # t序列的最大值
    min_t = t_array.min()  # t序列的最小值
    t_max_min = t_array.max() - t_array.min()  # t序列的极差
    t_ent = cal_ent(t_array)  # t序列的熵
    median_t = np.median(t_array)  # t序列的中位数
    mode_t = mode(t_array)[0][0]  # t序列的众数
    rate_t_max = (t_array.argmax() + 1) * 1.0 / len(t_array)  # 最大值位置
    rate_t_min = (t_array.argmin() + 1) * 1.0 / len(t_array)  # 最小值位置
    len_t = len(t_array)  # t序列的长度
    sum_t = sum(t_array)  # t序列的和
    fea_stat_list = [var_t, std_t, max_t, min_t, t_max_min, t_ent, median_t,
                     mode_t, rate_t_max, rate_t_min, len_t, sum_t]
    return fea_stat_list


