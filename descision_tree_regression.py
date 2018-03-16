#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 15:56:14 2017

@author: rishi
"""

"""
types of regression model
1.linear regression model
2.non linear regression model
3.non linear and non continuous regression model

    Decision tree regression
        is the 3rd type of regression model
        theres a term is called CART
        meaning there are two types of decision 
        tree 
        1. classification tree
        2. regression tree:: these are bitmore 
        complex

        in this there are number of
        independent variables and a dependent 
        variables 
        
        the algorithm tries to split the data
        based on INFORMATION ENTROPY -- a 
        statistical term which tells about 
        amount of information on the basis of 
        which the information could be split 
        by the algorithm. each one of the 
        split is called leaves
        
        the predicted value of y for the new 
        observation added to split would be 
        the average of all the obsertvation in 
        that split
        
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Decision tree Model to the dataset
#criterion followed in below algorithm is mse.
#mean squared distance between prediction and actual result then we take sum of this difference to measure error
# Create your regressor here
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
# for a single variable the average would be a constant and therefore we should get a straight line for y in an interval
#you might face a problem due to resolution of the graph that we considered for plotting
regressor.fit(X,y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()