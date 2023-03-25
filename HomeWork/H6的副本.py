#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 15:28:27 2021

@author: chenhanchuan
"""

import numpy as np
from sklearn.linear_model import LinearRegression

kepler = np.genfromtxt("/Users/chenhanchuan/Desktop/DS112/kepler.csv", delimiter = ",")

IQ = kepler[:,1]
caste = kepler[:,0]
brainMass = kepler[:,2]
income = kepler[:,4]
hoursWorked = kepler[:,3]

corrIQandCaste = np.corrcoef(IQ, caste) #11

brainMassRow = brainMass.reshape(len(brainMass),1)

brainMassAndIQ = LinearRegression().fit(brainMassRow, IQ)
slope = brainMassAndIQ.coef_
intercept = brainMassAndIQ.intercept_
yHatBMIQ = slope * brainMassRow + intercept
residualIQ = IQ - yHatBMIQ.flatten()

brainMassAndCaste = LinearRegression().fit(brainMassRow, caste)
yHatBMC = brainMassAndCaste.coef_ * brainMassRow + brainMassAndCaste.intercept_
residualCaste = caste - yHatBMC.flatten()

partCorr = np.corrcoef(residualIQ, residualCaste) #12

r_sq = brainMassAndIQ.score(brainMassRow, IQ) #13


yHat = slope * 3000 + intercept #14

corrCI = np.corrcoef(caste, income) #15

modelConToI = LinearRegression().fit(kepler[:,[1,3]], income)
a0, a1 = modelConToI.intercept_, modelConToI.coef_

yHatConToI = a1[0] * kepler[:,1] + a1[1] * kepler[:,3] + a0
residualConToI = income - yHatConToI.flatten()

modelConToC = LinearRegression().fit(kepler[:,[1,3]],caste)
c0, c1 = modelConToC.intercept_, modelConToC.coef_
yHatConToC = c1[0] * kepler[:,1] + c1[1] * kepler[:,3] + c0
residualConToC = caste - yHatConToC.flatten()

partCorr1 = np.corrcoef(residualConToI, residualConToC)
 

modelIQI = LinearRegression().fit(IQ.reshape(len(IQ),1), income)
r_sqIQI = modelIQI.score(IQ.reshape(len(IQ),1), income) #17

modelHWI = LinearRegression().fit(hoursWorked.reshape(len(hoursWorked),1), income)
r_sqHWI = modelHWI.score(hoursWorked.reshape(len(hoursWorked),1), income) #18

modelHWAndIQToI = LinearRegression().fit(kepler[:,[3,1]], income)
r_sqHWAndIQToI = modelHWAndIQToI.score(kepler[:,[3,1]], income) #19

b0, b1 = modelHWAndIQToI.intercept_, modelHWAndIQToI.coef_
yHatHWAndIQToI = b1[0] * 50 + b1[1] * 120 + b0

















