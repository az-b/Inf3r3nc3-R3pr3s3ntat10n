#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 18:00:27 2017

@author: BOY
"""

import numpy as np
from sklearn import decomposition  

R=np.random.normal(size=(10^5,3))

X1=np.zeros((10^5,3))
X1[:,0]=R[:,0] #v1=r1
X1[:,1]=R[:,0]+0.001*R[:,1] #v2=v1+eps*r2
X1[:,2]=100*R[:,2] #50*r3
	
print('cov(data):')
print(np.cov(X1.T))
print(' ')

#PCA:
pca=decomposition.PCA(n_components=2)
pca.fit(X1)
pca_comp = pca.components_
print('PCA:')
print(pca_comp)
print(' ')

#FA:
fa=decomposition.FactorAnalysis(n_components=2, max_iter=200)
fa.fit(X1)
fa_comp = fa.components_
print('Factor Analysis:')
print(fa_comp)
print(' ')


#%%############## PROBLEM 3.D ##########################

R2=np.random.normal(size=(10^5,3))

X2=np.zeros((10^5,3))
X2[:,0]=R2[:,0] #v1=r1
X2[:,1]=20*R2[:,1] #v2=15*r2
X2[:,2]=200*R2[:,2] #300*r3
	
print('cov(data2):')
print(np.cov(X2.T))
print(' ')

#FA:
fa=decomposition.FactorAnalysis(n_components=2, max_iter=200)
fa.fit(X2)
fa_comp = fa.components_
print('Factor Analysis:')
print(fa_comp)
print(' ')

#PCA:
pca=decomposition.PCA(n_components=2)
pca.fit(X2)
pca_comp = pca.components_
print('PCA:')
print(pca_comp)
print(' ')