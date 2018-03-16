#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 21:04:51 2017

@author: rishi
"""
#in clustering problem we dont know the answer. we dont know what to look for we use this
# k means clustering :: here everything is based on centroid. centroid is adjusted until we get centroids where we dont need any adjustments

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset with panda

dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

#now we will find the number of clusters required

from sklearn.cluster import KMeans
#within cluster sum of squares to obtain the required value of cluster to used in the k means
wcss=[]
#now in each iteration we will fit x and append the calculated value to list
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
#what to plot on x and y
#on x we plot 1 to 11 and on y we plot wcss
plt.plot(range(1,11),wcss)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#now we have got the value of k after inspecting the elbow

kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)

#fit predict instead of fit because ?

ymeans=kmeans.fit_predict(X)

#visualizing the cluster and we are going to make scatter plot with centroid highlighted cluster number starts from 0
clustername=['careful','standard','target','careless','sensible']
colorlist=['red','blue','green','cyan','magenta']
for i in range(0,5):
    plt.scatter(X[ymeans==i,0],X[ymeans==i,1],s=100,c=colorlist[i],label=clustername[i])

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroids')

plt.title('clusters of client')
plt.xlabel('Annual income') 
plt.ylabel('spending score')
plt.legend()
plt.show()       