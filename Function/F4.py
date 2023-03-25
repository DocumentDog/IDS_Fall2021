# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 17:00:00 2021

@author: chc12
"""

import numpy as np
def normalizedError(array, p):
    writtenByCharles = 0 #here the writtenBy value is the sum 
    for i in range(len(array)):
        writtenByCharles += np.abs(array[i][0] - array[i][1]) ** p #let prediction in first column minus measurement in second column, then absolute the value and power it.
    NE = (writtenByCharles / len(array)) ** (1 / p) #calculate the mean of sum and multiple the power with p as demoniator
    return round(NE,4) #round to 4 point
