# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 19:48:55 2021

@author: chc12
"""

import numpy as np
from scipy import stats
data =np.genfromtxt('C:/Users/chc12/OneDrive/movieReplicationSet.csv', delimiter=',', usecols=(range(400)), skip_header=1)
KB_1 = data[:,313]
KB_2 = data[:,176]
PF = data[:,297]

#Row Wise
temp = np.array([np.isnan(KB_1),np.isnan(KB_2),np.isnan(PF)],dtype=bool)
temp2 = temp*1 # convert boolean to int
temp2 = sum(temp2) # take sum of each participant
missingData = np.where(temp2>0) # find participants with missing data
KB_1 = np.delete(KB_1,missingData) # delete missing data from array
KB_2 = np.delete(KB_2,missingData) # delete missing data from array
PF = np.delete(PF,missingData)

combinedData = np.transpose(np.array([KB_1,KB_2,PF]))

meanKB_1 = np.mean(combinedData[:,0])
medianKB_1 = np.median(combinedData[:,0])

meanKB_2 = np.mean(combinedData[:,1])
medianKB_2 = np.median(combinedData[:,1])

meanPF = np.mean(combinedData[:,2])
medianPF = np.median(combinedData[:,2])


reltK_K, relpK_K = stats.ttest_rel(combinedData[:,0], combinedData[:,1]) #paired t test between KB1 and KB2
    
indtK_K, indpK_K = stats.ttest_ind(combinedData[:,0], combinedData[:,1]) #independent t test between KB1 and KB2

reltK_P, relpK_P = stats.ttest_rel(combinedData[:,1], combinedData[:,2]) #paired t test between KB2 and PF

indtK_P, indpK_P = stats.ttest_ind(combinedData[:,1], combinedData[:,2]) #independent t test between KB2 and PF

