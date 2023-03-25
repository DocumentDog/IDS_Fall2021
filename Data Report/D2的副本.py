#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 21:10:26 2021

@author: chenhanchuan
"""

import numpy as np
import pandas as pd
#1
data = np.genfromtxt('C:/Users/chc12/OneDrive/movieReplicationSet.csv', delimiter=',', usecols=(range(400)), skip_header=1)

#2
stdMovie = np.nanstd(data, axis = 0) 

#3
meanEachMovie = np.nanmean(data, axis=0)
meanOfMean = np.mean(meanEachMovie)

MADMovie = np.zeros(400)
for i in range(400):
    for j in range(1097):
        if np.isnan(data[j][i]):
            continue
        else:
            MADMovie[i] = np.mean(np.absolute(data[j][i] - meanEachMovie[i]))

#4
meanStdMovie = np.mean(stdMovie)
medianStdMovie = np.median(stdMovie)

#5
meanMADMovie = np.mean(MADMovie)
medianMADMovie = np.median(MADMovie)

#6
pdData = pd.read_csv('C:/Users/chc12/OneDrive/movieReplicationSet.csv', usecols=(range(400)))
correlation = pdData.corr()
mean1 = correlation.mean()
mean2 = mean1.mean()
MAD = pdData.mad(axis=0, skipna=True)
meanMAD = MAD.mean()
medianMAD = MAD.median()
























