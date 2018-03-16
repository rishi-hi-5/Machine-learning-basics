#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 21:10:42 2017

@author: rishi
"""

#a simple linear regression relies on a formula
#slope intercept form of line equation which is given as 
#follows y=mx+c 
#here
#   y-dependent variable (eg::how does salary change with years of exp)
#   x- independent variable causes why to change as it changes
#       relation between x and y could be direct or indirect
#   m is coefficent of independent variable it says
#   how a unit change in x affects unit change in y
#   acts as a connector between in x and y and tells the proportionin which both of them are dependent
#   c- constant -- it means the point on the vertical axis where the line cuts the vertical axis
#how to find the best fitting line
#or make simple linear regression make that find out for you
#the line can be found by ordinary least square method which states that
#min(sum(sq(y`- y)))
#y` is the orignal 
#y is the value that should be according to model

import numpy as np; 
import matplotlib.pyplot as plt;
import pandas as pd;

dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
#x here is the matrix of independent variable 
#y is vector of dependent variable
from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=1/3,random_state= 0)

#here we dont need to apply feature scaling as some of the library we used below will provide 
#feature scaling by default

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xtrain,ytrain)
#now we will create a vector of predicted value
ypred=regressor.predict(xtest)
#below function makes a scatter graph for the input provided
#train set
plt.scatter(xtrain,ytrain,color='red')
plt.plot(xtrain,regressor.predict(xtrain),color='blue')
plt.title('salary vs experience (training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

#test set
#dont need to change the value for 1st parameter in plot function
#as the liner regressor is trained on xtrain and we have already obtained a model on same

plt.scatter(xtest,ytest,color='red')
plt.plot(xtrain,regressor.predict(xtrain),color='blue')
plt.title('salary vs experience (test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

