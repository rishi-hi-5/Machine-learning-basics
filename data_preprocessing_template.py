#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:22:01 2017

@author: rishi
"""
"""
    Three essential libraries that are all ways required
"""

#below one includes mathematical rules
import numpy as np; 
#really essential library
#helps us to plot mathematical functions
import matplotlib.pyplot as plt;

#very much important to import the data set
#and manage them
import pandas as pd;
# to set the folder as working directory go to file explorer tab
# and go to directory where the csv file is located

#import dataset
dataset=pd.read_csv('Data.csv')
#we have to decide on matrix of feature
#also we decide on independent variable 
#which varies on the basis on dependent 
#variable available in our feature matrix

#creating matrix of feature
#by putting the : we say we take all the line
#and the :-1 states that we dont include the
#the last column as it is the column whoes 
#value needs to be predicted
#.values means we want to take all the values
#[which column to take, which column to concentrate on]
x=dataset.iloc[:,:-1].values

print(x)

#now create the depended variable vector
y=dataset.iloc[:,3].values
print(y)

#how to handle the missing data in dataset
#1. remove the observation which have empyt that
#   in there feature column
#   --quite dangerous to do the above
#2. take the mean of the values for the missing data
#we will be using the second one
#we are using the below lib to do this one
#imputer is used to preprocess data
from sklearn.preprocessing import Imputer
#now we create a object of imputer class
imputer=Imputer(missing_values='NaN',strategy = 'mean',axis=0)

#now we are going to fit the imputer object on our data
#to fit the imputer object we do the following
# we will do this only for missing data
imputer=imputer.fit(x[:,1:3])
#no we will replace the missing data
#to replace the missing data we use the transform function
x[:,1:3]=imputer.transform(x[:,1:3])

#how to encode categorical data
#why do we need the encoding for this
#country and purchased columns are categorical variable
#since machine learing model is based on mathematics
#so this would caues problem 
#to solve this we use encoding text => number
#to do this we do the following

from sklearn.preprocessing import LabelEncoder

#we create object 
labelencoder_x=LabelEncoder()
#we fit the label encoder object
x[:,0]=labelencoder_x.fit_transform(x[:,0])

print(x)

#problem : as the catogorical column is replaced by number
#mathematically ML we will think that one categorical data is greater than other
#as spain cannot be > germany
# to solve this we will use dummy encoding to make 
#one column for each categorical data
# so we will have three columns so the column will have 1 if its data
#lie in that row in original data set or else 0 in resultant column
#we create two new variables using one hot encoder

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0])

x=onehotencoder.fit_transform(x).toarray()
print(x)
#why do we use label encoder instead of one hot encoder on dependent variale?
#we do so because ML will know that it is category and it is dependent variable
#its category and there is no order between the two
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=labelencoder.fit_transform(y)

print(y)

#splitting data into training and test set
#why we need to do so?
#because this is about machine to learn something
#performance on test set should be same as training set.
#why so?
#it inidicates that our machine learning algos has understood the
#correlation perfectly
#we use cross validation library
from sklearn.cross_validation import train_test_split
#now lets split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state= 0)
#train set should not be huge as ML that is mathematical model built will not be able to predict as it then byhearts which is not good
#to solve this we use regularization which we will learn in future lectures

#feature scaling
#why do we need ?
#if the feature column is not on the same scale 
#then ML will face an issue
#why?
#because machine learning models are based on eucludean distance
#which is sq_root(sq(x2-x1)+sq(y2-y1))
#if scale of one column is bigger that is if it has wider range of value
#then the euclidean distance will be dominated by column having wider range of value
#to stop this dominator we need to do scaling
#sabki aukat ek line me lao bhai
#feature scaling formulas
#1. standardization
#xstand=(x-mean(x))/standard_deviation(x)
#2. Normalisation
#xnorm=(x-min(x))/(max(x)-min(x))

from sklearn.preprocessing import StandardScaler
scx=StandardScaler()

xtrain=scx.fit_transform(xtrain)
#dont need to fit scx to training set as it is already fitted to train set
xtest=scx.transform(xtest)
#do we need to fit and transform the dummy variables?
#author says that it depends upon the context
#all the variables seems to be between -1 and 1. this imporves ML.
#in descision tree which is based on the euclidean distance
#will run for longer time if we dont do feature scaling ;P
#we dont need to apply feature scaling on dependent variable for classification problem
#in regression we will need to apply feature scaling on dependent variable
