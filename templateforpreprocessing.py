#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 20:22:17 2017

@author: rishi
"""
#just copy and paste the content and change the vairable indexs as per your data
import numpy as np; 
import matplotlib.pyplot as plt;
import pandas as pd;

dataset=pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state= 0)
