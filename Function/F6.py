# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 17:18:48 2021

@author: chc12
"""
#written in Nov. 30th 2021 by Charles Chen
#for this function, the input are array(1D or 2D), flag(which is number between 1 to 3), and the size(number)
#the output is 1D array, and its content depend on which flag the users choose.

import numpy as np
def slidWinDescStats(array, flag, size):
    if flag == 1:
        lst = []
        for i in range(len(array) - 1):
            if (i + size) <= len(array):
                temp = []
                for j in range(size):
                    temp.append(array[i + j]) #append the number in array to temporary list
            else:
                break
            lst.append(np.mean(temp)) #calculate the average in temp and append to the lst
        return np.array(lst)
    elif flag == 2:
        lst = []
        for i in range(len(array) - 1):
            if (i + size) <= len(array):
                temp = []
                for j in range(size):
                    temp.append(array[i + j])
            else:
                break
            lst.append(np.std(temp, ddof = 1)) #we use sample standard deviation so the degree of freedom will be 1
        return np.array(lst)
    elif flag == 3:
        output = []
        for i in range(len(array) - 1):
            lst_1 = []
            lst_2 = []
            if (i + size) <= len(array):
                for j in range(size):
                    lst_1.append(array[i + j][0]) #append number in the first column in the 2D array
                    lst_2.append(array[i + j][1]) #append number in the second coumn in the 2D array
            else:
                break
            corr = np.corrcoef(lst_1, lst_2) 
            output.append(corr[0][1].round(3)) #append the correlation and round it to 3 digit
        return np.array(output)
    else:
        return "please enter the number between 1 to 3"
        
        
        
a = np.array([[1,1],[3,2],[5,4],[7,3],[9,5]])
print(slidWinDescStats(a, 0, 4))