# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:55:43 2021

@author: chc12
"""
import math
import pandas as pd
data = pd.read_csv('/Users/chenhanchuan/OneDrive/IDS/movieReplicationSet.csv', usecols = range(400))
mean_1 = data.mean(axis = 0, skipna = True)
meanOfMean = mean_1.mean()
std_1 = data.std(axis = 0, skipna = True)
meanOfStd = std_1.mean()

lst = []
lst_2 = []
count = data.count()
for i in range(400):
    width_1 = 1.96 * std_1[i] / math.sqrt(count[i])
    lst.append(width_1)
    width_2 = 2.58 * std_1[i] / math.sqrt(count[i])
    lst_2.append(width_2)
df = pd.DataFrame({"mean": mean_1,
                   "std": std_1,
                   "95%CI": lst,
                   "99%CI": lst_2})
