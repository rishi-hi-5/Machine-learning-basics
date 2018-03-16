#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:22:46 2017

@author: rishi
"""

#heirarcial clustering .
#this is similar to k means clustering
# two types:: 1.agglomerative -bottom up :- combine the clusters one by one till only one cluster remains. at the start each data is cluster
#             2. divisive -top bottom 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

#now we will use dendrogram to find the optimal number of clusters

#scipy contains tools for heriarchical clustering. we are using these here to build dendrogram
import scipy.cluster.hierarchy as sch
#ward method similar to kmeans . it does it on the basis of minimizing the variance within each clusters
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering

hc= AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')

y_hc=hc.fit_predict(X)

ymeans=y_hc
#visualizing the cluster and we are going to make scatter plot with centroid highlighted cluster number starts from 0
clustername=['careful','standard','target','careless','sensible']
colorlist=['red','blue','green','cyan','magenta']
for i in range(0,5):
    plt.scatter(X[ymeans==i,0],X[ymeans==i,1],s=100,c=colorlist[i],label=clustername[i])

plt.title('clusters of client')
plt.xlabel('Annual income') 
plt.ylabel('spending score')
plt.legend()
plt.show()      