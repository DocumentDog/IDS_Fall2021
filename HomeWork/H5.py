#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 20:15:38 2021

@author: chenhanchuan
"""

import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt("/Users/chenhanchuan/Desktop/DS112/corporateSales.csv", delimiter = ",")

IQ = data[:,0]
Complaint = data[:,1]
Ethical = data[:,2]

#r = np.corrcoef(IQ,Complaint)

#plt.plot(IQ, Complaint, "o")
#plt.xlabel("IQ")
#plt.ylabel("Complaint")
#plt.title("r = {:.3f}".format(r[0,1]))

medianComplaint = np.median(Complaint)
meanIQ = np.mean(IQ)
MADComplaint = np.mean(np.absolute(Complaint-np.mean(Complaint)))
medianIQ = np.median(IQ)
stdEthical = np.std(Ethical)
stdIQ = np.std(IQ)
medianEthical = np.median(Ethical)

#r1 = np.corrcoef(Complaint, Ethical)
#plt.plot(Complaint, Ethical, "o")
#plt.xlabel("Complaint")
#plt.ylabel("Ethical")
#plt.title("r1 = {:.3f}".format(r1[0,1]))

r2 = np.corrcoef(IQ, Ethical)
plt.plot(IQ, Ethical, "o")
plt.xlabel("IQ")
plt.ylabel("Ethical")
plt.title("r2 = {:.3f}".format(r2[0,1]))
