#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 19:04:32 2021

@author: chenhanchuan
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
data = np.genfromtxt('/Users/chenhanchuan/OneDrive/IDS/movieReplicationSet.csv', delimiter = ',', skip_header = 1)
senSeek = data[:,range(400,420)]
newSenSeek = senSeek[~np.isnan(senSeek).any(axis=1)]
per = data[:,range(420,464)]
newPer = per[~np.isnan(per).any(axis=1)]
movieExp = data[:,range(464,474)]
newMovieExp = movieExp[~np.isnan(movieExp).any(axis=1)]
#%%cell one
#correlation of sensation seeking
corrMatrix = np.corrcoef(newSenSeek, rowvar = False)
plt.imshow(corrMatrix)
plt.xlabel('questions')
plt.ylabel('questions')
plt.colorbar()
plt.show()

#run the pca for sensation seeking 
#1. z-score the data
zscoredData = stats.zscore(newSenSeek)

#2. initialize pca object and fit it into our data
pca = PCA().fit(zscoredData)

#3a. eigenvalue
eigVals = pca.explained_variance_

#3b. loading
loadings = pca.components_

#3c.rotated Data
rotatedData = pca.transform(zscoredData)

#4. covariance explained 
covarExplained = eigVals / sum(eigVals) * 100

for i in range(len(covarExplained)):
    print(covarExplained[i].round(3))
    
# scree plot
numClasses = 20
x = np.linspace(1, numClasses, numClasses)
plt.bar(x, eigVals, color = 'gray')
plt.plot([0,numClasses], [1,1], color = 'orange')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalues')
plt.show()

#interpreting the factors
whichPC = 0
plt.bar(x, loadings[whichPC,:] * -1)
plt.xlabel('questions')
plt.ylabel('Loadings')
plt.show()


#%%cell two
from sklearn.linear_model import LogisticRegression

saw = data[:,341]
newSaw = saw[~np.isnan(saw)]
median = np.nanmedian(saw)
label = np.zeros(len(newSaw))
for i in range(len(newSaw)):
    if newSaw[i] < median:
        label[i] = 0
    elif newSaw[i] > median:
        label[i] = 1
    else:
        label[i] = np.nan
rating = np.column_stack((newSaw, label))
newRating = rating[~np.isnan(rating).any(axis = 1)]

x = newRating[:,0].reshape(len(newRating),1)
y = newRating[:,1]

model = LogisticRegression().fit(x,y)















