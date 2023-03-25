# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 17:06:19 2021

@author: chc12
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import math

data =np.genfromtxt('/Users/chenhanchuan/Desktop/DS112/movieReplicationSet.csv', delimiter=',', usecols=(range(400)), skip_header=1)
sata = data[:,[93,273]] #Star War 2 and 1
jointSata = sata[~np.isnan(sata).any(axis=1)] #2

x = jointSata[:,0].reshape(len(jointSata[:,0]),1)
y = jointSata[:,1]

model = LinearRegression().fit(x,y) #3

betas = model.coef_
r_sq = model.score(x,y)
yHat = betas * jointSata[:,0] + model.intercept_
residual = yHat.flatten() - y
RMSE = math.sqrt(np.mean(residual ** 2))



newSata = data[:,[273,292]] #star war 1 and Titanic
newJointSata = newSata[~np.isnan(newSata).any(axis=1)]

m = newJointSata[:,0].reshape(len(newJointSata),1)  #SW
n = newJointSata[:,1] #Titanic

model2 = LinearRegression().fit(m,n)

betas2 = model2.coef_
r_sq2 = model2.score(m,n)
nHat = betas2 * newJointSata[:,0] + model2.intercept_
residual2 = n - nHat.flatten()
RMSE2 = math.sqrt(np.mean(residual2 ** 2))



sata3 = data[:,[273,93,292]] #SW1, SW2, Titanic
jointSata3 = sata3[~np.isnan(sata3).any(axis=1)]

model3 = LinearRegression().fit(jointSata3[:,[0,1]],jointSata3[:,2])
r_sq3 = model3.score(jointSata3[:,[0,1]],jointSata3[:,2])





