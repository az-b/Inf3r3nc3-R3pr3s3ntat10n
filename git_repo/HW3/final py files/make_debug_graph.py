# define the example tree-stuctured factor graph
# compare the results of my sum-product algorithm versus the results of fglib library sum-product algorithm 

import numpy as np 
import networkx as nx
from fglib import graphs, nodes, rv, inference
from funcs import *

def make_debug_graph():

    # Create factor graph
    fg = graphs.FactorGraph()

    # Create variable nodes
    x1 = nodes.VNode("x1", rv.Discrete)
    x2 = nodes.VNode("x2", rv.Discrete)
    x3 = nodes.VNode("x3", rv.Discrete)
    x4 = nodes.VNode("x4", rv.Discrete)

    # Create factor nodes
    f12 = nodes.FNode("f12")
    f234 = nodes.FNode("f234")
    f3 = nodes.FNode("f3")
    f4 = nodes.FNode("f4")

    # Add nodes to factor graph
    fg.set_nodes([x1, x2, x3, x4])
    fg.set_nodes([f12, f234, f3,f4 ])

    # Add edges to factor graph
    fg.set_edge(x1, f12)
    fg.set_edge(f12, x2)
    fg.set_edge(x2, f234)
    fg.set_edge(f234, x3)
    fg.set_edge(f234, x4)
    fg.set_edge(x3, f3)
    fg.set_edge(x4, f4)

    #add potential for f_3: p(x3)
    dist_f3 = [0.5, 0.5]
    f3.factor = rv.Discrete(dist_f3,x3)
    
    #add potential for f_4: p(x4)
    dist_f4 = [0.4,0.6]
    f4.factor = rv.Discrete(dist_f4,x4)
    
    # add potential for f_{234}: p(x2, x3, x4) = p(x2|x3,x4) p(x3,x4)
    px3x4=np.outer(dist_f3,dist_f4)
    px3x4=np.reshape(px3x4, np.shape(px3x4)+(1,))
    px2_conditioned_x3x4=[[[0.2,0.8],
                         [0.25,0.75],],
                         [[0.7,0.3],
                         [0.3,0.7]]]
    
    dist_f234 =px3x4*px2_conditioned_x3x4
    f234.factor = rv.Discrete(dist_f234,x3,x4,x2)
   
    # add potential for f_{12}:  p (x1,x2) = p(x1 | x2) p(x2)
    px1_conditioned_x2 = [[0.5,0.5],
                         [0.7,0.3]]
    px2= np.sum(dist_f234, axis=(0,1))
    dist_f12 =px2[:,np.newaxis]*px1_conditioned_x2
    f12.factor = rv.Discrete(dist_f12,x2,x1)


#########_____________ PROBLEM 2_____________############    
    fglib_beliefs = inference.belief_propagation(fg, query_node=x1)   
    return fglib_beliefs, fg, x1

(fglib_beliefs,fg,x1) = make_debug_graph() 


def SetMsg(g,a,b,value):
    g[a][b]['object'].set_message(a, b, value)

def SumProd(g,f,n):
    return f.spa(n)                                        
                                         
def GetBeliefs(fg, ni, q): #ni:= number of iterations, q:=query node
    for n in fg.get_vnodes():#initializing v-t-f msgs with value 1 for all states. 
        for f in nx.all_neighbors(fg, n):
            msg = rv.Discrete(np.array([1,1]),n)
            SetMsg(fg,n,f,msg)   
    
    for i in range(ni):# below is the parallel message updating:
        fg.get_fnodes()
        for f in fg.get_fnodes():# updating the factor-to-variable msgs given the current v-t-f
            for n in nx.all_neighbors(fg, f):
                SetMsg(fg,f,n,SumProd(fg,f,n).normalize())
        for n in fg.get_vnodes(): 	#updating the v-t-f msgs given the current f-t-v
            for f in nx.all_neighbors(fg, n):
               SetMsg(fg,n,f,SumProd(fg,n,f).normalize()) # here we normalize
	
    return [[n,n.belief()] for n in q]
