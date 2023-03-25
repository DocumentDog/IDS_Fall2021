# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:51:45 2021

@author: chc12
"""
from scipy import stats
import numpy as np
data1 = np.genfromtxt('C:/Users/chc12/OneDrive/Sadex1.csv', delimiter = ",")
data1_1 = data1[:,0]
total1 = 0
total2 = 0
for i in range(len(data1_1)):
    if i <= 29:
        total1 += data1_1[i]
    else:
        total2 += data1_1[i]
ave = total1 / 30 - total2 / 30 #difference in average

t1, p1 = stats.ttest_ind(data1_1[:30], data1_1[30:60]) #calculate the p value


data2 = np.genfromtxt('C:/Users/chc12/OneDrive/Sadex2.csv', delimiter = ",")
data2_1 = data2[:,0]
data2_2 = data2[:,1]
diff = np.mean(data2_1 - data2_2) 
t2, p2 = stats.ttest_ind(data2_1, data2_2)

data3 = np.genfromtxt('C:/Users/chc12/OneDrive/Sadex3.csv', delimiter = ",")
data3_1 = data3[:,0] #first column of Sadex3
data3_1_1 = data3_1[:90]  #first 90 element in the first column
data3_1_2 = data3_1[90:180] #last 90 elements in the first column

t3, p3 = stats.ttest_ind(data3_1_1, data3_1_2)

data4 = np.genfromtxt('C:/Users/chc12/OneDrive/Sadex4.csv', delimiter = ",")
data4_1 = data4[:,0] #first column of data4, which is before Sadex
data4_2 = data4[:,1] #second column of data4, which is after Sadex

diff4 = np.mean(data4_1) - np.mean(data4_2)

t4, p4 = stats.ttest_ind(data4_1, data4_2)























