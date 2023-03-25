#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 21:07:21 2021

@author: chenhanchuan
"""

import numpy as np
data =np.genfromtxt('C:/Users/chc12/OneDrive/IDS/movieReplicationSet.csv', delimiter=',', usecols=(range(400)), skip_header=1)
PF = data[:,308]
MG = data[:,58]

#row wise
temp = np.array([np.isnan(PF),np.isnan(MG)],dtype=bool)
temp2 = temp*1 # convert boolean to int
temp2 = sum(temp2) # take sum of each participant
missingData = np.where(temp2>0) # find participants with missing data
PF = np.delete(PF,missingData) # delete missing data from array
MG = np.delete(MG,missingData) # delete missing data from array
combinedData = np.transpose(np.array([PF,MG]))

#doing the permutation test
empiricalData1 = combinedData[:,0] 
empiricalData2 = combinedData[:,1] 
ourTestStat = np.mean(empiricalData1) - np.mean(empiricalData2)


numReps = int(1e5) # This is how many times we'll draw WITHOUT replacement 

jointData = np.concatenate((empiricalData1,empiricalData2)) # Stack them on 

n1 = len(empiricalData1) # How long one of them is
n2 = len(jointData) # Overall length
shuffledStats = np.empty([numReps,1]) # Initialize empty array
shuffledStats[:] = np.NaN # Then convert to NaN

for i in range(numReps):
    shuffledIndices = np.random.permutation(n2) # shuffle indices 0 to 2985
    shuffledGroup1 = jointData[shuffledIndices[:n1]]
    shuffledGroup2 = jointData[shuffledIndices[n1:]]
    shuffledStats[i,0] = np.mean(shuffledGroup1) - np.mean(shuffledGroup2)

temp1 = np.argwhere(shuffledStats > ourTestStat)
temp2 = len(temp1)

# Compute the p-value:
exactPvalue = temp2/len(shuffledStats)
print(exactPvalue)

#Bootstrapping method
ZL = data[:,30]
ZL = ZL[np.isfinite(ZL)] #use element wise to remove nan in zoolanders

sampleMeans = np.mean(ZL,axis=0) 

numRepeats = int(1e4) # How many times do we want to resample the 1 empirical 
nSample = len(combinedData) # Number of data points in the sample

tychenicMeans = np.empty([numRepeats,1]) 
tychenicMeans[:] = np.NaN

tychenicIndices = np.random.randint(0,nSample,[numRepeats,numRepeats])

for i in range(numRepeats): # loop through each repeat
    tempIndices = tychenicIndices[:,i] # indices for this iteration
    tychenicMeans[i] = np.mean(ZL[tempIndices]) # compute the mean
 
estimateOffset = np.mean(tychenicMeans) - sampleMeans

confidenceLevel = 95 # What confidence level (probability of containing 
# the empirical mean) is desired? Also try 99%
lowerBoundPercent = (100 - confidenceLevel)/2 # lower bound
upperBoundPercent = 100 - lowerBoundPercent # upper bound
lowerBoundIndex = round(numRepeats/100*lowerBoundPercent)-1 # what index?
upperBoundIndex = round(numRepeats/100*upperBoundPercent)-1 # what index?
sortedSamples = np.sort(tychenicMeans,axis=0)
lowerBound = sortedSamples[lowerBoundIndex] # What tychenic value consistutes the lower bound?
upperBound = sortedSamples[upperBoundIndex]
print(lowerBound, upperBound)

