#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)


# In[4]:


means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T


# In[5]:


def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()
    
kmeans_display(X, original_label)


# In[7]:



def initialize_centroids(X,k):
    return X[np.random.choice(X.shape[0], k, replace=False)]
def get_label(X, centroids): #
    label=[]
    for point in X:
        min_dist= float("inf")
        label1= None
        for i, centroid in enumerate(centroids):
            new_dist= get_distrance(point, centroid)
            if min_dist> new_dist:
                min_dist=new_dist
                label1=i
        label.append(label1)
    return labels
def get_distance(point_1, point_2):
    return ((point_1[0]-point_2[0]) **2 + (point_1[1]-point_2[1]) **2) ** 0.5

def update_centroids(X, label,k):
    new_centroids= np.zeros((k, X.shape[1]))
    for i in range(k):
        Xk=X[label==i, :]
        new_centroids[i,:]= np.mean(Xk, axis=0)
    return new_centroids

    


# In[8]:


def main(X,k):
    centroids = initialize_centroids(X,k)
    while True:
        old_centroid= centroids
        label=get_label(X, centroids)
        centroids= update_centroids(X, label,k)
        if 
        return label


# In[ ]:




