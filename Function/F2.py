#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 22:24:03 2021

@author: chenhanchuan
"""
#for the function of random variable, first we need to input array as parameter; then call unique function
#to sort the unique numbers. then create a dataset with first colon unique numbers and second colon count/total length of array
#then return the dataset. After that muktiply two arrays and sum up to get the expect value. 

import numpy as np

def Random_Variable(charles): #charles here is the paramater that we suppose to input the 1D array
    numData = np.unique(charles) #numData is the array that store unique value of input array
    countData = np.zeros(len(np.unique(charles))) #countData stores the count of each unique value, here we first initialize it.
    for i in range(len(numData)):
        for j in range(len(charles)):
            if numData[i] == charles[j]:
                countData[i] += 1
    probData = countData / len(charles) #probData is the probability of each unique value
    sata = np.stack([numData, probData], axis = 1) #sata is a 2D array that stores the numData and probData
    expectV = np.dot(numData, probData) #expectV is the expect value 
    return sata, expectV
