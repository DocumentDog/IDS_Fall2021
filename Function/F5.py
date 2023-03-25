# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 01:27:10 2021

@author: chc12
"""

#for this function, we only need to follow the hint ont the specific guideline and it is easy to build this function.
#the writtenBy value is the length of input array
import numpy as np
def empiricalSampleBound(array, percent):
    array = np.sort(array, axis=0) #sort the array so it start from smallest value to biggest
    writtenByCharles = len(array) 
    lowerBound = (100-percent) / 2 #lower bound is only at one side so we need to divided by 2
    upperBound = 100 - lowerBound #100 minus lower bound is the upper bound
    lowerIndex = round(writtenByCharles / 100 * lowerBound) - 1 #since python start index at 0, we should minus 1 to correct the index
    upperIndex = round(writtenByCharles / 100 * upperBound) - 1
    lowerValue = array[lowerIndex]
    upperValue = array[upperIndex]
    
    return round(lowerValue,3), round(upperValue,3)

data = np.genfromtxt('C:/Users/chc12/Downloads/sampleInput1.csv', delimiter = ',')
print(empiricalSampleBound(data, 99))
    