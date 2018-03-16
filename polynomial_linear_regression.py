#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 21:51:09 2017

@author: rishi
"""

#why linear is mentioned in this one because it is in respect to the linearity of the coefficient and not the variables

import numpy as np; 
import matplotlib.pyplot as plt;
import pandas as pd;

dataset=pd.read_csv('Position_Salaries.csv')
#make sure that x is a matrix and y is a vector this for convenience
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

# we dont need to split the datasets . because we dont have enough information for training and testing

#comparision between linear regression model and polynomial regression model

#we will be using the same regressor class of linear regression

#linear regression model
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#polynomial regression model
#to include polynomial feature in linear regression class we include the below library

from sklearn.preprocessing import PolynomialFeatures

#below object is for transforming the matrix of x to poly in terms of y=b0+b1x1+b2*x2^2...
poly_reg=PolynomialFeatures(degree=4) 
##directly use the below fit transform function while we are plotting graph for simplicity and generalization
xpoly=poly_reg.fit_transform(x)

lin_reg2=LinearRegression()
lin_reg2.fit(xpoly,y)

#now lets visualize
#lets plot x any then build our predictions
#then we compare two predictions

#scatter gives us true observation
#below gives us a scatter graph
plt.scatter(x,y,color='red')
#below gives us the line graph
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Truth or bluff linear regression')

plt.xlabel('position level')
plt.ylabel('salaries')
plt.show()

#we create new x for smooth graph results
#we give the step value
xgrid=np.arange(min(x),max(x),0.1)
#we convert the xgrid vecto to matrix using the reshape function rows and columns specified
xgrid=xgrid.reshape(len(xgrid),1)

plt.scatter(x,y,color='red')
#below gives us the line graph
plt.plot(xgrid,lin_reg2.predict(poly_reg.fit_transform(xgrid)),color='blue')
plt.title('Truth or bluff polynomial linear regression')

plt.xlabel('position level')
plt.ylabel('salaries')
plt.show()

#now final prediction check whether the employee was bluffing using the model we have built . the employee said his salary was 160k

lin_reg.predict(6.5)

lin_reg2.predict(poly_reg.fit_transform(6.5))