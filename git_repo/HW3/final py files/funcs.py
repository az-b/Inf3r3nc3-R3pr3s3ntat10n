#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:39:15 2017

@author: BOY
"""
from make_debug_graph import *
import itertools as it
import pyldpc as pl

def getUnary(msg,eps,n):
    d=np.zeros((2))
    m=msg[n]
    d[int(m)]=1-eps 
    d[1-int(m)]=eps
    return d

def AddUF(fg,msg,eps):
    ufn=[]	#unary factor nodes
    for n in range(msg.size):
        ufn.append(nodes.FNode('uf'+str(n))) 
    fg.set_nodes([ufn[n] for n in range(msg.size)])
    for v in fg.get_vnodes(): #adding edges
        n=getVarNum(v)
        uf=ufn[n]
        fg.set_edge(v, uf)
    for v in fg.get_vnodes():	#adding potentials
        n=getVarNum(v)
        uf=ufn[n]
        tmp=getUnary(msg,eps,n)
        uf.factor = rv.Discrete(tmp,v)
    return fg

def getFP(dv):
	get=np.ones(tuple([2 for _ in range(dv)]))
	index=it.product(range(2),repeat=dv)
	for i in index:
		get[i]=np.float(1-(sum(list(i))%2))
	if int(np.sum(get))==0: 
		print("error")
	return get


def getInvalid(H):
	msg=np.zeros((H.shape[1]))
	msg[np.argwhere(H[0,:])[0][0]]=1
	return msg

def getNodes(fg,f):
	tmp=nx.all_neighbors(fg,f)
	return np.sort([getVarNum(v) for v in tmp])

def getFactor(fg,v):
	tmp=1.
	for f in fg.get_fnodes():
		factor=f.factor.pmf
		nodes=getNodes(fg,f)
		index=tuple([int(v[i]) for i in nodes])
		tmp *= 1.-factor[index]
	return tmp

def getVarNum(v):
    return int(v.__str__()[1:])

def getH(N,dv,dc):
	return pl.RegularH(2*N,dv,dc) #builds a regular parity check matrix H (n,dv,dc) 
    
def getG(H):
    return pl.CodingMatrix_systematic(H) #returns a tuple (Hp,tGS)

def getGraph(H):
	(N,K)=H.shape
	dv=int(sum(H[0,:]))
	
	fg=graphs.FactorGraph()#creating the factor graph
	
	vNodes=[]
	for n in range(K):
		vNodes.append(nodes.VNode('v'+str(n), rv.Discrete)) #creating variable nodes	  
	fNodes=[]
	for n in range(N):
		fNodes.append(nodes.FNode('f'+str(n)))	#creating factor nodes  
	
	fg.set_nodes([vNodes[n] for n in range(K)])	#adding nodes to the graph
	fg.set_nodes([fNodes[n] for n in range(N)]) #adding nodes to the graph
	
	for v in range(K):
		for f in range(N):
			if H[f,v] == 1:
				fg.set_edge(vNodes[v], fNodes[f]) #adding edges
	
	for f in fg.get_fnodes():
		fp=getFP(dv) #potentials
		fpx=nx.all_neighbors(fg, f)
		f.factor=rv.Discrete(fp,*fpx) #add  potentials
	return fg
		
def passMsg(msg,eps):
	corrupted_msg=[]
	for n in range(msg.size):
		corrupt=np.random.choice([0,1],p=[1-eps,eps])
		corrupted_msg.append((1-msg[n])*corrupt+msg[n]*(1-corrupt))
	return np.array(corrupted_msg)

def getPotential(fg,ni=50):
	vnodes=fg.get_vnodes()	
	tmp=np.zeros((2,len(vnodes)))
	bel=GetBeliefs(fg, ni, vnodes)
	for [v,b] in bel:
		nv=getVarNum(v)
		tmp[:,nv]=b.pmf
	return np.array(tmp)

def MsgEst(p):
	est=[]
	for n in range(p.shape[1]):
		est.append(np.argmax(p[:,n]))
	return est

def Hamming(v,w):
	return sum(np.abs(np.array(v)-np.array(w)))

def BinTo256(numpy_array):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        if numpy_array[i] ==1:
            numpy_array[i]= 0
        else:
            numpy_array[i] = 255
    return numpy_array