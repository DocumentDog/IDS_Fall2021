#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 02:19:54 2021

@author: chenhanchuan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
data = np.genfromtxt('/Users/chenhanchuan/Desktop/DS112/Capstone/movieReplicationSet.csv', delimiter = ',', skip_header = 1)

#%% Question1
#dimension reduction first for sense seeking 
#data cleaning for sensation seeking and movie exp:combine two data and do row-wise so they have same rows
senSeek = data[:,range(400,420)]
movieExp = data[:,range(464,474)]
rawdata = np.concatenate((senSeek,movieExp), axis = 1)    
rawdata = rawdata[~np.isnan(rawdata).any(axis = 1)]
senSeek = rawdata[:,range(20)]
movieExp = rawdata[:,range(20,30)]
#correlation
corrMatrixS = np.corrcoef(senSeek, rowvar = False)
plt.imshow(corrMatrixS)
plt.xlabel('questions')
plt.ylabel('questions')
plt.colorbar()
plt.show()

#run the pca for sensation seeking 
#1. z-score the data
zscoredDataS = stats.zscore(senSeek)
#2. initialize pca object and fit it into our data
pca = PCA().fit(zscoredDataS)
#3a. eigenvalue
eigValsS = pca.explained_variance_
#3b. loading
loadingsS = pca.components_
#3c.rotated Data
rotatedDataS = pca.transform(zscoredDataS)
#4. covariance explained 
covarExplainedS = eigValsS / sum(eigValsS) * 100

for i in range(len(covarExplainedS)):
    print(covarExplainedS[i].round(3))
    
threshold = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigValsS > threshold))

#data cleanning for movie experience


#correlation
corrMatrixM = np.corrcoef(movieExp, rowvar = False)
plt.imshow(corrMatrixM)
plt.xlabel('questions')
plt.ylabel('questions')
plt.colorbar()
plt.show()

#run the pca for movie experience
#1. z-score the data
zscoredDataM = stats.zscore(movieExp)
#2. initialize pca object and fit it into our data
pca = PCA().fit(zscoredDataM)
#3a. eigenvalue
eigValsM = pca.explained_variance_
#3b. loading
loadingsM = pca.components_
#3c.rotated Data
rotatedDataM = pca.transform(zscoredDataM)
#4. covariance explained 
covarExplainedM = eigValsM / sum(eigValsM) * 100

for i in range(len(covarExplainedM)):
    print(covarExplainedM[i].round(3))
    
threshold = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigValsM > threshold))
#now we know that the sensation seeking have 6 variables after dimension reduction with kaiser rule
#and first principle is 17.477 and second is 8.864
#and the movie experience have 2 variables after dimension reduction with kaiser rule
#and first principle is 29.426 and second is 18.75
r = np.corrcoef(rotatedDataS[:,0]*-1,rotatedDataM[:,0]*-1)
plt.plot(rotatedDataS[:,0]*-1,rotatedDataM[:,0]*-1,'o',markersize=5)
plt.xlabel('Sensation Seeking')
plt.ylabel('Movie Experience')
plt.show()

#%% Question 2
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
personality = pd.read_csv('/Users/chenhanchuan/Desktop/DS112/Capstone/movieReplicationSet.csv', usecols = range(420,464))
personality = personality.dropna()
predictors = personality.to_numpy()

# Z-score the data:
zscoredData = stats.zscore(predictors)
# Initialize PCA object and fit to our data:
pca = PCA().fit(zscoredData)
# Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals = pca.explained_variance_
# Loadings (eigenvectors): Weights per factor in terms of the original data.
loadings = pca.components_*-1
# Rotated Data - simply the transformed data:
origDataNewCoordinates = pca.fit_transform(zscoredData)*-1
# Scree plot:
numPredictors = 44
plt.bar(np.linspace(1,numPredictors,numPredictors),eigVals)
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show()

#by using elbow rule, we can see the number of factors that explained most is 6

plt.subplot(1,2,1) # Factor 1: 
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:]) # "support"
plt.title('extroverse')
plt.subplot(1,2,2) # Factor 2:
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[1,:]) # "Support"
plt.title('generosity')
plt.show()

plt.plot(origDataNewCoordinates[:,0],origDataNewCoordinates[:,1],'o',markersize=1)
plt.xlabel('extroverse')
plt.ylabel('generosity')
plt.show()

x = np.column_stack((origDataNewCoordinates[:,0],origDataNewCoordinates[:,1]))
# Init:
numClusters = 9 # how many clusters are we looping over? (from 2 to 10)
Q = np.empty([numClusters,1])*np.NaN # init container to store sums
# Compute kMeans:
for ii in range(2, 11): # Loop through each cluster (from 2 to 10!)
    kMeans = KMeans(n_clusters = int(ii)).fit(x) # compute kmeans using scikit
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(x,cId) # compute the mean silhouette coefficient of all samples
    Q[ii-2] = sum(s) # take the sum
# Plot this to make it clearer what is going on
plt.plot(np.linspace(2,10,9),Q)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()

# Recompute kMeans:
numClusters = 2
kMeans = KMeans(n_clusters = numClusters).fit(x) 
cId = kMeans.labels_ 
cCoords = kMeans.cluster_centers_ 

# Plot the color-coded data:
for ii in range(numClusters):
    plotIndex = np.argwhere(cId == int(ii))
    plt.plot(x[plotIndex,0],x[plotIndex,1],'o',markersize=1)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')  
    plt.xlabel('Extroverse')
    plt.ylabel('generosity')

#%% question 3
#null hypothesis is that more popular movie has the same rate of less popular movie
#alternative hypothesis is more popular movie has higher rating than less popular movie
sata = data[:,range(400)]
numArray = np.zeros(400)
for i in range(len(sata[0])):
    numArray[i] = len(sata[:,i][~np.isnan(sata[:,i])])
    
median = np.median(numArray) #use code above to calculate the median 
highRate = []
lowRate = []
for j in range(len(numArray)):
    if len(sata[:,j][~np.isnan(sata[:,j])]) > median:
        highRate.append(np.nanmean(sata[:j]))
    else:
        lowRate.append(np.nanmean(sata[:,j])) #split the mean rate of each move above medain and below median
highRate = np.array(highRate)
lowRate = np.array(lowRate)
result = stats.ttest_ind(highRate, lowRate)

#set up descriptive container
desCon = np.empty([2,2])
desCon[:] = np.nan
desCon[0,0] = np.mean(highRate)
desCon[1,0] = np.mean(lowRate)
desCon[0,1] = np.std(highRate) / np.sqrt(len(highRate))
desCon[1,1] = np.std(lowRate) / np.sqrt(len(lowRate))

#plot
x = ['highRate', 'lowRate']
xPos = np.array([1,2])
plt.bar(xPos, desCon[:,0], width = 0.5, yerr = desCon[:,1])
plt.xticks(xPos,x)
plt.ylabel("mean rating")
plt.title('f = {:.3f}'.format(result[0]) + ', p = {:.3f}'.format(result[1]))
#from the test result, we get that t-test statistics is 29.53 and
#p-value is nearly approach to 0 for two tailed, so single tail is also 0 which is lower than 
#alpha = 0.05, so we reject the null hypothesis.
#more popular movie rated higher than less popular movie.

#%% question 4
#null hypothesis is that male and female viewer rate the same (Rm = Rf)
#alternative hypothesis is that male viewer rate differently than female viewer (Rm != Rf)
#first we need to clean the gender who is not male or female
shrek = data[:,87]
gender = data[:,474]
sata = np.column_stack((shrek,gender))
for i in range(len(sata[:,1])):
    if sata[:,1][i] == 3:
        sata[:,1][i] = np.nan #let gender who is 3 become NaN
sata = sata[~np.isnan(sata).any(axis=1)] #remove missing value

maleViewer = []
femaleViewer = []
for j in range(len(sata[:,0])):
    if sata[:,1][j] == 1:
        maleViewer.append(sata[:,0][j])
    else:
        femaleViewer.append(sata[:,0][j])

result = stats.ttest_ind(maleViewer,femaleViewer)

#descriptive containeer
desCon = np.empty([2,2])
desCon[:] = np.nan
desCon[0,0] = np.mean(maleViewer)
desCon[1,0] = np.mean(femaleViewer)
desCon[0,1] = np.std(maleViewer) / np.sqrt(len(maleViewer))
desCon[1,1] = np.std(femaleViewer) / np.sqrt(len(femaleViewer))

#plot
x = ['maleViewer', 'femaleViewer']
xPos = np.array([1,2])
plt.bar(xPos, desCon[:,0], width = 0.5, yerr = desCon[:,1])
plt.xticks(xPos,x)
plt.ylabel("mean rating")
plt.title('f = {:.3f}'.format(result[0]) + ', p = {:.3f}'.format(result[1]))
#this result shows that t-test statistics is 1.101 and p-value for two-tailed is 0.27
#p-value is still bigger than alpha = 0.05, so we fail to reject null hypothesis
#conclusion: male and female viewer rate the same

#%% question 5
#null hypothesis is that people who are child only enjoy the same as people with siblings
#alternative hypothesis is that people who are child only enjoy more than people with siblings
#first we clean the data with NaN and people who does not respond
lionKing = data[:,220]
onlyKid = data[:,475]
sata = np.column_stack((lionKing, onlyKid))
for i in range(len(sata[:,1])):
    if sata[:,1][i] == -1:
        sata[:,1][i] = np.nan #let people who does not respond become NaN
sata = sata[~np.isnan(sata).any(axis=1)] #remove missing value

oneKid = []
moreKid = []
for j in range(len(sata[:,1])):
    if sata[:,1][j] == 1:
        oneKid.append(sata[:,0][j])
    else:
        moreKid.append(sata[:,0][j])

f, p = stats.ttest_ind(oneKid, moreKid)
p = p / 2 #since it is a one-teiled t-test but it return two-tailed, we should half it

#descriptive containeer
desCon = np.empty([2,2])
desCon[:] = np.nan
desCon[0,0] = np.mean(oneKid)
desCon[1,0] = np.mean(moreKid)
desCon[0,1] = np.std(oneKid) / np.sqrt(len(oneKid))
desCon[1,1] = np.std(moreKid) / np.sqrt(len(moreKid))

#plot
x = ['oneKid', 'moreKid']
xPos = np.array([1,2])
plt.bar(xPos, desCon[:,0], width = 0.5, yerr = desCon[:,1])
plt.xticks(xPos,x)
plt.ylabel("mean rating")
plt.title('f = {:.3f}'.format(f) + ', p = {:.3f}'.format(p))

#this result shows that t-test statistic is -2.054 and p-value is 0.02 < alpha = 0.05
#so we reject the null hypothesis.
#conclusion: people who are child only enjoy more than people with siblings.

#%% question 6
#null hypothesis: people who like watch movie socially enjoy the same as people who watch alone
#alternative hypothesis: people who like watch movie socially enjoy more than people watch alone.
#as always we clean the data first
TWWS = data[:,357]
viewPre = data[:,476]
sata = np.column_stack((TWWS,viewPre))
for i in range(len(sata[:,1])):
    if sata[:,1][i] == -1:
        sata[:,1][i] = np.nan #let people who does not respond become NaN
sata = sata[~np.isnan(sata).any(axis=1)] #remove the missing value

social = []
alone = []

for j in range(len(sata[:,1])):
    if sata[:,1][j] == 1:
        alone.append(sata[:,0][j])
    else:
        social.append(sata[:,0][j])
    
f, p = stats.ttest_ind(social, alone)
p = p / 2 #also it is one-tailed t-test

#descriptive containeer
desCon = np.empty([2,2])
desCon[:] = np.nan
desCon[0,0] = np.mean(social)
desCon[1,0] = np.mean(alone)
desCon[0,1] = np.std(social) / np.sqrt(len(social))
desCon[1,1] = np.std(alone) / np.sqrt(len(alone))

#plot
x = ['social', 'alone']
xPos = np.array([1,2])
plt.bar(xPos, desCon[:,0], width = 0.5, yerr = desCon[:,1])
plt.xticks(xPos,x)
plt.ylabel("mean rating")
plt.title('f = {:.3f}'.format(f) + ', p = {:.3f}'.format(p))

#from the result we can see the t-test statistic is -1.568 and p-value is 0.059
#since p-value > alpha = 0.05, we fail to reject null hypothesis
#conclusion: people like watch movie socially enjoy more than people watch alone

#%% question 7
#define inconsistent quality:the mean rating of each season in its franchise have huge difference
#that means we should use ANOVA to do this hypothesis test
#null hypothesis: each movie in its franchise has no inconsistent quality (xbar all the same)
#alternative hypothesis: the movie in its franchise has inconsistent quality (xbar not the same)
starWar = data[:,[21,93,174,273,336,342]]
starWar = starWar[~np.isnan(starWar).any(axis=1)]
fs, ps = stats.f_oneway(starWar[0],starWar[1],starWar[2],starWar[3],starWar[4],starWar[5])

desCon = np.empty([6,4])
desCon[:] = np.nan
for i in range(6): #6 movies of star war
    desCon[i,0] = np.mean(starWar[:,i])
    desCon[i,1] = np.std(starWar[:,i])
    desCon[i,2] = len(starWar[:,i])
    desCon[i,3] = desCon[i,1] / np.sqrt(desCon[i,2])
    
#plot
x = ['starWar1', 'starWar2','starWar3','starWar4','starWar5','starWar6']
xPos = np.array([1,2,3,4,5,6])
plt.bar(xPos, desCon[:,0], width = 0.5, yerr = desCon[:,3])
plt.xticks(xPos,x)
plt.ylabel("mean rating")
plt.title('f = {:.3f}'.format(fs) + ', p = {:.3f}'.format(ps))
#f-test statistic is 2.096 and p-value for Star War series is 0.093 > alpha = 0.05. fail to reject null
# =============================================================================
HP = data[:,[230,258,387,394]]
HP = HP[~np.isnan(HP).any(axis=1)]
fh, ph = stats.f_oneway(HP[0], HP[1], HP[2], HP[3])

desCon = np.empty([4,4])
desCon[:] = np.nan
for i in range(4): #4 movies of harry potter
    desCon[i,0] = np.mean(HP[:,i])
    desCon[i,1] = np.std(HP[:,i])
    desCon[i,2] = len(HP[:,i])
    desCon[i,3] = desCon[i,1] / np.sqrt(desCon[i,2])
    
#plot
x = ['HP1', 'HP2','HP3','HP4']
xPos = np.array([1,2,3,4])
plt.bar(xPos, desCon[:,0], width = 0.5, yerr = desCon[:,3])
plt.xticks(xPos,x)
plt.ylabel("mean rating")
plt.title('f = {:.3f}'.format(fh) + ', p = {:.3f}'.format(ph))
#f-test statistics is 68 and p-value for harry potter is 8*10^-8 < alpha = 0.05. Reject null
# =============================================================================
Matrix = data[:,[35,177,306]]
Matrix = Matrix[~np.isnan(Matrix).any(axis=1)]
fm, pm = stats.f_oneway(Matrix[0], Matrix[1], Matrix[2])

desCon = np.empty([3,4])
desCon[:] = np.nan
for i in range(3): #3 movies of the matrix
    desCon[i,0] = np.mean(Matrix[:,i])
    desCon[i,1] = np.std(Matrix[:,i])
    desCon[i,2] = len(Matrix[:,i])
    desCon[i,3] = desCon[i,1] / np.sqrt(desCon[i,2])

#plot
x = ['Matrix1', 'Matrix2','Matrix3']
xPos = np.array([1,2,3])
plt.bar(xPos, desCon[:,0], width = 0.5, yerr = desCon[:,3])
plt.xticks(xPos,x)
plt.ylabel("mean rating")
plt.title('f = {:.3f}'.format(fm) + ', p = {:.3f}'.format(pm))
#f-test statistics is 0.881 and p-value for matrix is 0.462 > alpha. Fail to reject null.
# =============================================================================
indiJones = data[:,[4,32,33,142]]
indiJones = indiJones[~np.isnan(indiJones).any(axis=1)]

fi, pi = stats.f_oneway(indiJones[0], indiJones[1], indiJones[2], indiJones[3])

desCon = np.empty([4,4])
desCon[:] = np.nan
for i in range(4): #4 movies of indiana jones
    desCon[i,0] = np.mean(indiJones[:,i])
    desCon[i,1] = np.std(indiJones[:,i])
    desCon[i,2] = len(indiJones[:,i])
    desCon[i,3] = desCon[i,1] / np.sqrt(desCon[i,2])
    
#plot
x = ['indiJones1', 'indiJones2','indiJones3', 'indiJones4']
xPos = np.array([1,2,3,4])
plt.bar(xPos, desCon[:,0], width = 0.5, yerr = desCon[:,3])
plt.xticks(xPos,x)
plt.ylabel("mean rating")
plt.title('f = {:.3f}'.format(fi) + ', p = {:.3f}'.format(pi))
#f-test statistic is 0.576 and p-value is 0.642 for indiana jones. p-value > alpha. Fail to reject null
# =============================================================================
juraPark = data[:,[37,47,370]]
juraPark = juraPark[~np.isnan(juraPark).any(axis=1)]

fj, pj = stats.f_oneway(juraPark[0], juraPark[1], juraPark[2])

desCon = np.empty([3,4])
desCon[:] = np.nan
for i in range(3): #3 movies of jurassic park
    desCon[i,0] = np.mean(juraPark[:,i])
    desCon[i,1] = np.std(juraPark[:,i])
    desCon[i,2] = len(juraPark[:,i])
    desCon[i,3] = desCon[i,1] / np.sqrt(desCon[i,2])
    
#plot
x = ['juraPark1', 'juraPark2','juraPark3']
xPos = np.array([1,2,3])
plt.bar(xPos, desCon[:,0], width = 0.5, yerr = desCon[:,3])
plt.xticks(xPos,x)
plt.ylabel("mean rating")
plt.title('f = {:.3f}'.format(fj) + ', p = {:.3f}'.format(pj))
#f-test statistic is 1.722 and p-value is 0.256. p-value > alpha. Fail to reject null
# =============================================================================
pirate = data[:,[75,204,351]]
pirate = pirate[~np.isnan(pirate).any(axis = 1)]

fp, pp = stats.f_oneway(pirate[0], pirate[1], pirate[2])

desCon = np.empty([3,4])
desCon[:] = np.nan
for i in range(3): #3 movies of pirate of carribean
    desCon[i,0] = np.mean(pirate[:,i])
    desCon[i,1] = np.std(pirate[:,i])
    desCon[i,2] = len(pirate[:,i])
    desCon[i,3] = desCon[i,1] / np.sqrt(desCon[i,2])
    
#plot
x = ['pirate1', 'pirate2','pirate3']
xPos = np.array([1,2,3])
plt.bar(xPos, desCon[:,0], width = 0.5, yerr = desCon[:,3])
plt.xticks(xPos,x)
plt.ylabel("mean rating")
plt.title('f = {:.3f}'.format(fp) + ', p = {:.3f}'.format(pp))
#f-test statistic is 7 and p-value is 0.027. p-value < alpha. Reject null
# =============================================================================
toyStory = data[:,[157,176,276]]
toyStory = toyStory[~np.isnan(toyStory).any(axis=1)]

ft, pt = stats.f_oneway(toyStory[0], toyStory[1], toyStory[2])

desCon = np.empty([3,4])
desCon[:] = np.nan
for i in range(3): #3 movies of toy story
    desCon[i,0] = np.mean(toyStory[:,i])
    desCon[i,1] = np.std(toyStory[:,i])
    desCon[i,2] = len(toyStory[:,i])
    desCon[i,3] = desCon[i,1] / np.sqrt(desCon[i,2])
    
#plot
x = ['toyStory1', 'toyStory2','toyStory3']
xPos = np.array([1,2,3])
plt.bar(xPos, desCon[:,0], width = 0.5, yerr = desCon[:,3])
plt.xticks(xPos,x)
plt.ylabel("mean rating")
plt.title('f = {:.3f}'.format(ft) + ', p = {:.3f}'.format(pt))
#f-test statistic is 0.333 and p-value is 0.729. p-value > alpha. fail to reject null
# =============================================================================
batman = data[:,[46,181,235]]
batman = batman[~np.isnan(batman).any(axis=1)]

fb, pb = stats.f_oneway(batman[0],batman[1],batman[2])

desCon = np.empty([3,4])
desCon[:] = np.nan
for i in range(3): #3 movies of batman
    desCon[i,0] = np.mean(batman[:,i])
    desCon[i,1] = np.std(batman[:,i])
    desCon[i,2] = len(batman[:,i])
    desCon[i,3] = desCon[i,1] / np.sqrt(desCon[i,2])
    
#plot
x = ['batman1', 'batman2','batman3']
xPos = np.array([1,2,3])
plt.bar(xPos, desCon[:,0], width = 0.5, yerr = desCon[:,3])
plt.xticks(xPos,x)
plt.ylabel("mean rating")
plt.title('f = {:.3f}'.format(fb) + ', p = {:.3f}'.format(pb))
#f-test statistic is 3.042 and p-value is 0.122. p-value > alpha. Fail to reject null
#%% question8
#I find the mean rating of 400 movies for 1097 people, and mean personality for 1097 people
#so one mean rating corresponding to one mean personality
from sklearn.linear_model import LinearRegression

movieRating = data[:,range(400)]
personality = data[:,range(420,464)]
meanRating = []
meanPersonality = []
for i in range(len(movieRating)):
    meanRating.append(np.nanmean(movieRating[i,:]))
    meanPersonality.append(np.nanmean(personality[i,:])) #append mean personality and movie rating in lists

meanRating = np.array(meanRating)
meanPersonality = np.array(meanPersonality)
sata = np.column_stack((meanRating, meanPersonality))
sata = sata[~np.isnan(sata).any(axis=1)] #make it to array so we can reshape it


model = LinearRegression().fit(sata[:,1].reshape(len(sata[:,1]),1), sata[:,0]) #build the model 
r_sq = model.score(sata[:,1].reshape(len(sata[:,1]),1), sata[:,0]) #r^2 
slope = model.coef_
intercept = model.intercept_
yHat = slope * sata[:,1] + intercept #prediction

plt.plot(sata[:,1], sata[:,0], 'o', markersize = 1)
plt.xlabel('meanPersonality')
plt.ylabel('meanRating')
plt.plot(sata[:,1], yHat, color='orange', linewidth=0.5)
plt.title('R^2 = {:.3f}'.format(r_sq))
#since the r^2 is only 0.045, which means the mean personality only represent 4.5% of mean movie
#rating, so it is not a good prediction model

#%% question 9
from sklearn.linear_model import LinearRegression

movieRating = data[:,range(400)]
gender = data[:,474]
sibship = data[:,475]
social= data[:,476]

meanRating = []
for i in range(len(movieRating[:,0])):
    meanRating.append(np.nanmean(movieRating[i,:]))
    
gender = np.array(gender)
sibship = np.array(sibship)
social = np.array(social)

x = np.column_stack((gender,sibship,social,meanRating))
x = x[~np.isnan(x).any(axis=1)]
predictor = x[:,range(3)]
y = x[:,-1]

model = LinearRegression().fit(predictor,y)
b0, b1 = model.intercept_, model.coef_
yHat = b1[0] * predictor[:,0] + b1[1] * predictor[:,1] + b1[2] * predictor[:,2] + b0

plt.plot(yHat, y, 'o', markersize = 5)
plt.xlabel('predicted movie rating')
plt.ylabel('actual movie rating')
r_sq = model.score(predictor, y)  
plt.title('R^2 = {:.3f}'.format(r_sq))

#%% question 10
movieRating = data[:,range(400)]
predictors = data[:,range(401,477)]

meanRating = []
for i in range(len(movieRating[:,0])):
    meanRating.append(np.nanmean(movieRating[i,:]))
    
meanRating = np.array(meanRating)
sata = np.column_stack((predictors,meanRating))
sata = sata[~np.isnan(sata).any(axis=1)]

x = sata[:,range(76)]
y = sata[:,-1]

# Z-score the data:
zscoredData = stats.zscore(x)

# Initialize PCA object and fit to our data:
pca = PCA().fit(zscoredData)

# Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals = pca.explained_variance_

# Loadings (eigenvectors): Weights per factor in terms of the original data.
loadings = pca.components_*-1

# Rotated Data - simply the transformed data:
origDataNewCoordinates = pca.fit_transform(zscoredData)*-1

predictor = origDataNewCoordinates[:,0].reshape(len(origDataNewCoordinates),1)
model = LinearRegression().fit(predictor, y)
r_sq = model.score(predictor, y) #r^2 
slope = model.coef_
intercept = model.intercept_
yHat = slope * origDataNewCoordinates[:,0] + intercept #prediction

plt.plot(origDataNewCoordinates, y, 'o', markersize = 1)
plt.xlabel('all factors')
plt.ylabel('meanRating')
plt.plot(sata[:,1], yHat, color='orange', linewidth=0.5)
plt.title('R^2 = {:.3f}'.format(r_sq))

















