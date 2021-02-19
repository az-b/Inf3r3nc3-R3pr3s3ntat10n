import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
import seaborn as sns

from IPython.core.display import Image

import pymc3 as pm

#constants

def HMC(U,K,dUdq,N,q_0, p_0, epsilon=0.01, L=100):
    p_mu=0.
    p_sig=1.
    
    current_q = q_0
    current_p = p_0
    
    H = np.zeros(N)
    qall = np.zeros(N)
    accept=0
    for j in range(N):
    
        q = current_q
        p = current_p
        
        #draw a new p
        p = np.random.normal(p_mu, p_sig)
        
        current_p=p
        
        # leap frog
        
        # Make a half step for momentum at the beginning
        p = p - epsilon*dUdq(q)/2.0
        
        
        # alternate full steps for position and momentum
        for i in range(L):
            q = q + epsilon*p
            if (i != L-1):
                p = p - epsilon*dUdq(q)
    
        #make a half step at the end
        p = p - epsilon*dUdq(q)/2.

        # negate the momentum
        p= -p;
        current_U = U(current_q)
        current_K = K(current_p)

        proposed_U = U(q)
        proposed_K = K(p)
        A=np.exp( current_U-proposed_U+current_K-proposed_K)
    
        # accept/reject
        if np.random.rand() < A:
            current_q = q
            qall[j]=q
            accept+=1
        else:
            qall[j] = current_q
        
        H[j] = U(current_q)+K(current_p)
    print("accept=",accept/np.double(N))
    return H, qall


# functions
U = lambda q: q**2/2.
K = lambda p:  (p**2)/2.
dUdq= lambda q: q


H, qall= HMC(U=U,K=K,dUdq=dUdq,N=10000,q_0=0, p_0=-4, epsilon=0.01, L=200)
plt.hist(qall, bins=50, normed=True)
x = np.linspace(-4,4,100)
plt.plot(x, sp.stats.norm.pdf(x),'r')
plt.show()
accept= 1.0

