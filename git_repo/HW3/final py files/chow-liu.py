#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:05:15 2017

@author: BOY
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


###########______MUTUAL INFO_______############################################ 

def mutual(ppair,px,py): #ppair:= p(x_i,x_j)
	(nx,ny)=ppair.shape
	tmp=np.zeros((nx,ny)) 
	for i in range(nx):
		for j in range(ny):
			if ppair[i,j] != 0.:
				tmp[i,j]=ppair[i,j]*(np.log(ppair[i,j])-np.log(px[i])-np.log(px[j]))
			else:
				tmp[i,j]=ppair[i,j]
	return np.sum(tmp)

#####reference:
#def calculate_mutual_information(X, u, j):
#    if i > j:
#        i, j = j, i
#    marginal_i = marginal_distribution(X, i)
#    marginal_j = marginal_distribution(X, j)
#    marginal_ij = marginal_pair_distribution(X, i, j)
#    I = 0.
#    for x_i, p_x_i in marginal_i.iteritems():
#        for x_j, p_x_j in marginal_j.iteritems():
#            if (x_i, x_j) in marginal_ij:
#                p_x_ij = marginal_ij[(x_i, x_j)]
#                I += p_x_ij * (N.log(p_x_ij) - N.log(p_x_i) - N.log(p_x_j))
#    return I


###########_______CALCULATE PPARIR=p(x_i,x_j)______############################
data=np.loadtxt('data/chowliu-input.txt').astype(int)
(imn,tn)=data.shape

ppair=np.zeros((tn,tn,2,2))
for i in range(tn):
	for j in range(tn):
		for k in range(imn):
			ppair[i,j,data[k][i],data[k][j]]+=1
ppair=ppair/imn


###########____
weight=np.zeros((tn,tn))
for i in range(tn):
	for j in range(tn):
		weight[i][j]=mutual(ppair[i,j,:,:],[ppair[i,i,k,k] for k in range(2)],[ppair[j,j,k,k] for k in range(2)])

G=csr_matrix(-weight) #make it sparse
tree=minimum_spanning_tree(G)

edges=np.array(tree.nonzero()).T
ne=edges.shape[0] #ne:= num of edges
potential=np.zeros((ne,2,2))

for k in range(ne):
	for i in range(2):
		for j in range(2):
			potential[k,i,j]=ppair[edges[k,0],edges[k,1],i,j]/(ppair[edges[k,0],edges[k,0],i,i]*ppair[edges[k,1],edges[k,1],j,j])

print(potential[0,:,:])

####reference:
#def build_chow_liu_tree(X, n):
#    """
#    Build a Chow-Liu tree from the data, X. n is the number of features. The weight on each edge is
#    the negative of the mutual information between those features. The tree is returned as a networkx
#    object.
#    """
#    G = nx.Graph()
#    for v in xrange(n):
#        G.add_node(v)
#        for u in xrange(v):
#            G.add_edge(u, v, weight=-calculate_mutual_information(X, u, v))
#    T = nx.minimum_spanning_tree(G)
#    return T

###########___ok, now the output part..____#######


ne=potential.shape[0]
nv=ne + 1 #nv:= num of var

f=open('data/output.txt','w')
f.write('MARKOV\n')
f.close()
f=open('data/output.txt','a')
f.write('%d\n'%nv)
for _ in range(nv-1): 
	f.write('2 ')
f.write('2\n')	
f.write('%d\n'%ne)
for k in range(ne):
	f.write('2 %d %d\n'%(edges[k,0],edges[k,1]))
for k in range(ne):
	f.write('4\n')
	for i in range(2):
		f.write(' %f %f\n'%(potential[k,i,0],potential[k,i,1]))
f.close()