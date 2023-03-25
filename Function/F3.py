#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 21:03:03 2021

@author: chenhanchuan
"""
#for this function, the input is a 1D array, so first we should calculate the median of this array
#then I create the array with same length of original array and fill in with absolute value of deviation of each number from median
#when all deviations are in the new array, calculate the median of this new array and output it.
import numpy as np

def MAD(array):
    medianOfArray = np.median(array)
    writtenByCharles = np.zeros(len(array))
    for i in range(len(array)):
        writtenByCharles[i] = np.abs(array[i] - medianOfArray)
    medianAbsDev = np.median(writtenByCharles)
    return medianAbsDev
        
