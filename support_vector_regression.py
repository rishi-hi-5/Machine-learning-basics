#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 22:34:31 2017

@author: rishi
"""

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
#the following model does not includes feature scaling to because it is not commonly used
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Fitting the Regression Model to the dataset
# Create your regressor here
from sklearn.svm import SVR
#linear kernel is used for linear model
#rbf and poly are used for non linear models
#rbf is the most commonly used
#there is something called as penalty which does not consider far distant point while creating a model ---- i did not understood this
regressor= SVR(kernel='rbf')
regressor.fit(X,y)
# Predicting a new result
#we have to take the inverse transform so that we ge the actual value instead of scaled predicted value for that we have to inverse transform
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (svr Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#svr are great model 