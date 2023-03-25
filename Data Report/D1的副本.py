#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 18:08:43 2021

@author: chenhanchuan
"""

import numpy as np
from scipy import stats
data = np.genfromtxt('C:/Users/chc12/OneDrive/movieReplicationSet.csv', delimiter=',', usecols=(range(400)), skip_header=1) #1

meanEachMovie = np.nanmean(data, axis=0) #2
print(meanEachMovie)
minimum = meanEachMovie[0]
for i in range(0, len(meanEachMovie)):
    if meanEachMovie[i] > minimum:
        minimum = meanEachMovie[i]



medianEachMovie = np.nanmedian(data, axis=0) #3
maximum = medianEachMovie[0]
for j in range(0, len(medianEachMovie)):
    if medianEachMovie[j] > maximum:
        maximum = medianEachMovie[j]
        #print(j)
        #print(maximum)
modeEachMovie = stats.mode(data, axis=0) #4


meanOfMean = np.mean(meanEachMovie) #5
