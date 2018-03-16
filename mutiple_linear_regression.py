#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 21:24:11 2017

@author: rishi
"""
#multiple linear regression
#it is same as simple linear regression but multiple variables
#are in volved. the formula for the this turns out to be
#y=b0+b1*x1+b2*x2+....+bn*xn
#y -- dependent variable
#x1, x2 ,x3 ... -- are independent variable
#b0 constants (basically i have seen this as an error)
#b1, b2 ,b3 ... -- are coefficents

#A note about sample size.  In Linear regression the sample size rule of thumb is that regression analysis requires at least 20 cases per independent variable in the analysis.
##$ we cannot have constant and all dummy variable at the same time
###$whenever we are building a model we always let go one dummy vairable why ?? 
#there assumptions to be considered for linear regression
#they are as follows
#1.linearity-- The linearity assumption can best be tested with scatter plots
#2.homoscedasticity -- variance at each x point .1st and 2nd thing can be tested using residual plot
#3.multivariate normality
#4.independece of errors
#5.lack of multicollinearity

#why do we throw out some models out?
#because it would become garbage model and it would not be able to predict
#only very imp one which are imp

#how do we construct model?
#there are some of the steps

#1. all in
## throw in all the variable . if we know that they are the true variables
## or if you have to use the variable without eliminating any
## preparing for backward elimination
# number 2 3 4 are step wise regression
#because they are true step by step
#but 4 is mostly considered the step wise regression
#2. backward elimination
## step 1: select a significance level to stay in model
## step 2: fit all the variables in model
## step 3: consider the predictor with highest p value
### if p is greater than significance level then go to step 4 else end the model is prepared
## step 4: remove predictor -- remove the variable highest p value
## step 5 : fit model without this variable then go back to step 5 to remove any more redundant variable
### meaning recreate the model
#3. forward selection
## step 1: select the significance level to enter the model
## step 2: fit all simple regression model. we take independent variable and make all the possible regression model then select the one with lowest p value
## step 3: keep the chosen predicted variable and fit all the possible models with one extra predictor added to the ones you already have
## step 4: consider the predictor with lowest p value if p<sl go to step 3 (meaning we consider the two variable model with the least p value and go for step 3 if we can add more variables one by one) else end you have your model(here keep the previous model not the current one)
#4. bidirectional elimination (combines both backward and forward selection methods)
## step 1: select the significance level to stay and and enter the model
## step 2: perform the next step of forward selection in which variables enter one by one until the significance level condition is violated :)
## step 3: perform all the steps of backward elimination now go to step2 again
## step 4: this step is encountered if we are neither able to add new variable nor eliminate the old one at this point we stop and we have our required model 
#5. all possible model
## step 1: select a criterion (eg:: akaike criterion)
## step 2: construct all the possible regression model:: 2^(N-1)--(N here total number of independent variables) total combination
## step 3: select one with best criterion and there you have your model
#6. score comparison


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

dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values
#profit is dependent variable

from sklearn.preprocessing import LabelEncoder

#we create object 
labelencoder_x=LabelEncoder()
#we fit the label encoder object
x[:,3]=labelencoder_x.fit_transform(x[:,3])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[3])

x=onehotencoder.fit_transform(x).toarray()
#removing dummy variable trap
x=x[:,1:]

from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state= 0)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(xtrain,ytrain)

ypred=regressor.predict(xtest)

import statsmodels.formula.api as sm
#library used below does not take into account the b0 constant in the linear equation given above so we will some how to add the constant 
 
#so to make it understand we will add one more column which contains 1's and then the model will come to know that it has x0 which it can use to comput the quation y=b0*x0+b1*x1+.... if we dont add the this 1's column then the b0 would not be considered the by the library that we are going to use

#np below is numpy
#we are going to use append to append the 1's column
#ones returns matrix of ones you just have to specify no of rows and no of columns

#as type int to avoid data type error from happening ## if you check ones function dock you would come to know there is a dtype mentioned in the function

#last parameter in append specifies the axis along which the new column would be added if axis=0 row 
#   else if axis =1 column
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1) 

#now we create our optimal matrix of feature . this contains the high imapact independent variable or you can say those which are statistically significant
x_opt=x[:,[0,1,2,3,4,5]]

#now we select a significance level.
#if p value of varaible is greater than significance level to stay in the model

#here oridinary least square method is used
regressor_ols=sm.OLS(endog=y , exog=x_opt).fit()
regressor_ols.summary()

x_opt=x[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=y , exog=x_opt).fit()
regressor_ols.summary()

x_opt=x[:,[0,3,4,5]]
regressor_ols=sm.OLS(endog=y , exog=x_opt).fit()
regressor_ols.summary()

x_opt=x[:,[0,3,5]]
regressor_ols=sm.OLS(endog=y , exog=x_opt).fit()
regressor_ols.summary()

x_opt=x[:,[0,3]]
regressor_ols=sm.OLS(endog=y , exog=x_opt).fit()
regressor_ols.summary()