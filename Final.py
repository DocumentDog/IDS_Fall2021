#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 21:39:30 2021

@author: chenhanchuan
"""

import numpy as np
from sklearn.linear_model import LinearRegression
#%% College Success
data = np.genfromtxt('/Users/chenhanchuan/Downloads/College Success.csv', delimiter = ',', skip_header = 1)
gpa = data[:,1]
hsm = data[:,2]
hss = data[:,3]
hse = data[:,4]
satm = data[:,5]
satv = data[:,6]

x = satv
model = LinearRegression().fit(x.reshape(len(x),1),gpa)

r_sq = model.score(x.reshape(len(x),1),gpa)

#%% Thanksgiving
from scipy import stats
data2 = np.genfromtxt('/Users/chenhanchuan/Downloads/How much you enjoy Thanksgiving based on turkey family and cooking.csv', delimiter = ',', skip_header = 1)
h,p = stats.kruskal(data2[:,0], data2[:,1], data2[:,2], data2[:,3])

#%% Dark Triad
data3 = np.genfromtxt('/Users/chenhanchuan/Downloads/DarkTriad.csv', delimiter = ',', skip_header = 1)
count = 0
for i in data3[:,2]:
    if i > 30 and i < 80:
        count += 1
#%% Sojow
data4 = np.genfromtxt('/Users/chenhanchuan/Downloads/Sojow.csv', delimiter = ',', skip_header = 1)
IQ = data4[:,1]
brainMass = data4[:,2]

model1 = LinearRegression().fit(brainMass.reshape(len(brainMass), 1), IQ)
yHat = model1.coef_ * 4000 + model1.intercept_