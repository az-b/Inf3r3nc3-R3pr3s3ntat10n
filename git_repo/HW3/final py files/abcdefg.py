#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:29:35 2017

@author: BOY
"""

from funcs import *
from make_debug_graph import *
import matplotlib.pyplot as plt 
from PIL import Image

#########_____________ PROBLEM 2______________############  
print('PROBLEM 2.')
beliefs = GetBeliefs(fg, 3, [x1])      
print("Belief of all query nodes [fglib] : ", fglib_beliefs)    
print("Belief of all query nodes [Implemented] : ", beliefs[0][1])

if (np.array_equal(fglib_beliefs,beliefs[0][1])):
    print("New sum product implementation is consistent with the legacy computation.")


#########_____________ PROBLEM 3.a_____________############  
print('PROBLEM 3.a')
N = 8
H = getH(N,2,4)
fg = getGraph(H)
inv_msg = getInvalid(H)
print('The probability of invalid codeword is: %f'%getFactor(fg,inv_msg))


#########_____________ PROBLEM 3.b_____________############  
print('PROBLEM 3.b')
eps = 0.05
N2 = 128
M = N2*2
H2 = getH(N2,4,8)
fg2 = getGraph(H2)
msg = np.zeros(M)
msg_sent = passMsg(msg,eps)
print('corrupted msg:',msg_sent, '(%d corrupted bits)'%np.sum(msg_sent))
#print('# of corrupted bits = %d'%np.sum(msg_sent))
fg2 = AddUF(fg2,msg_sent,eps)
pt = getPotential(fg2,ni=50)
print('posterior P(x_n=1):', pt[1,:])
msg_dec = MsgEst(pt) #decoded message
print('Hamming distance = %d'%Hamming(msg,msg_dec))#dist bw sent and decoded
plt.ylim([-1,1])
plt.plot(pt[1,:])

#########_____________ PROBLEM 3.c_____________############  
print('PROBLEM 3.c')
eps3c = 0.06
N3c = 128
M3c = N3c*2
hamming=[]
for i in range(10):
    H3c = getH(N3c,4,8)
    fg3c=getGraph(H3c)
    msg3c=np.zeros(M3c)
    msg_sent3c= passMsg(msg3c,eps3c)
    print('corrupted msg # %d'%(i+1),':',msg_sent3c)
    fg3c=AddUF(fg3c,msg_sent3c,eps3c)
    for ni in range(50):
        pt3c = getPotential(fg3c,ni+1)
        print('posterior P(x_n=1):', pt3c[1,:])
        msg_dec3c = MsgEst(pt3c) #decoded message
        hamming.append(Hamming(msg3c,msg_dec3c))
        print('Hamming distance = %d'%Hamming(msg3c,msg_dec3c))
plt.plot(hamming[0:49])
plt.plot(hamming[50:99])
plt.plot(hamming[100:149])
plt.plot(hamming[150:199])
plt.plot(hamming[200:249])
plt.plot(hamming[250:299])
plt.plot(hamming[300:349])
plt.plot(hamming[350:399])
plt.plot(hamming[400:449])
plt.plot(hamming[450:499])

#########_____________ PROBLEM 3.d.1_____________############  
print('PROBLEM 3.d')
eps3d = 0.08
N3d = 128
M3d = N3d*2
hamming3d=[]
for i in range(10):
    H3d = getH(N3d,4,8)
    fg3d=getGraph(H3d)
    msg3d=np.zeros(M3d)
    msg_sent3d= passMsg(msg3d,eps3d)
    print('corrupted msg # %d'%(i+1),':',msg_sent3d)
    fg3d=AddUF(fg3d,msg_sent3d,eps3d)
    for ni in range(50):
        pt3d = getPotential(fg3d,ni+1)
        print('posterior P(x_n=1):', pt3d[1,:])
        msg_dec3d = MsgEst(pt3d) #decoded message
        hamming3d.append(Hamming(msg3d,msg_dec3d))
        print('Hamming distance = %d'%Hamming(msg3d,msg_dec3d))
plt.plot(hamming3d[0:49])
plt.plot(hamming3d[50:99])
plt.plot(hamming3d[100:149])
plt.plot(hamming3d[150:199])
plt.plot(hamming3d[200:249])
plt.plot(hamming3d[250:299])
plt.plot(hamming3d[300:349])
plt.plot(hamming3d[350:399])
plt.plot(hamming3d[400:449])
plt.plot(hamming3d[450:499])

#########_____________ PROBLEM 3.d.2_____________############  
print('PROBLEM 3.e')
eps3e = 0.1
N3e = 128
M3e = N3e*2
hamming3e=[]
for i in range(10):
    H3e = getH(N3e,4,8)
    fg3e=getGraph(H3e)
    msg3e=np.zeros(M3e)
    msg_sent3e= passMsg(msg3e,eps3e)
    print('corrupted msg # %d'%(i+1),':',msg_sent3e)
    fg3e=AddUF(fg3e,msg_sent3e,eps3e)
    for ni in range(50):
        pt3e = getPotential(fg3e,ni+1)
        print('posterior P(x_n=1):', pt3e[1,:])
        msg_dec3e = MsgEst(pt3e) #decoded message
        hamming3e.append(Hamming(msg3e,msg_dec3e))
        print('Hamming distance = %d'%Hamming(msg3e,msg_dec3e))
plt.plot(hamming3e[0:49])
plt.plot(hamming3e[50:99])
plt.plot(hamming3e[100:149])
plt.plot(hamming3e[150:199])
plt.plot(hamming3e[200:249])
plt.plot(hamming3e[250:299])
plt.plot(hamming3e[300:349])
plt.plot(hamming3e[350:399])
plt.plot(hamming3e[400:449])
plt.plot(hamming3e[450:499])

#########_____________ PROBLEM 3.e_____________############  
print('PROBLEM 3.f')
eps3f = 0.06
N3f=1600
M3f = N3f*2
H3f = getH(N3f,4,8)
Hp,G3f=getG(H3f) # RETURNS a Nx(N/2-3) MATRIX. weird. 

G3f = np.delete(G3f, -1, 1)#I'll just cut the 3 extra columns..
G3f = np.delete(G3f, -1, 1)#.. manually..
G3f = np.delete(G3f, -1, 1)# .could use a data frame..

#now let's create the message:
img = Image.open('CLAUDE.png')
pre_msg = np.array(img)
shape = pre_msg.shape# record the original shape
msg3f = pre_msg.ravel()# make a 1-dimensional view of pre_msg


#ENCODE. modulo 2 prod of G and message:
coded=G3f.dot(msg3f)%2 

#now the coded msg is ready to be sent through the noisy channel:
fg3f = getGraph(H3f)
msg_sent3f= passMsg(coded,eps3f)
fg3f=AddUF(fg3f,msg_sent3f,eps3f)
pt3f = getPotential(fg3f,ni=30)

print('posterior P(x_n=1):', pt3f[1,:])
msg_dec3f = MsgEst(pt3f) #decoded message

#decoded=Hp.dot(msg_dec3f)%2 #now let's multiply by the transpose H:
decoded=msg_dec3f[0:1600]
decoded=BinTo256(decoded)
arr2 = np.asarray(decoded).reshape(shape)# reform a numpy array of the original shape
img2 = Image.fromarray(np.uint8(arr2))# make a PIL image
img2.show()

#########_____________ PROBLEM 3.f_____________############  
print('PROBLEM 3.g')
eps3g = 0.1
N3g=1600
M3g = N3g*2
H3g = getH(N3g,4,8)
Hp,G3g=getG(H3g) # RETURNS a Nx(N/2-3) MATRIX. weird. 

G3g = np.delete(G3g, -1, 1)#I'll just cut the 3 extra columns..
G3g = np.delete(G3g, -1, 1)#.. manually..
G3g = np.delete(G3g, -1, 1)# .could use a data frame..

#now let's create the message:
img = Image.open('CLAUDE.png')
pre_msg = np.array(img)
shape = pre_msg.shape# record the original shape
msg3g = pre_msg.ravel()# make a 1-dimensional view of pre_msg


#ENCODE. modulo 2 prod of G and message:
coded=G3g.dot(msg3g)%2 

#now the coded msg is ready to be sent through the noisy channel:
fg3g = getGraph(H3g)
msg_sent3g= passMsg(coded,eps3g)
fg3g=AddUF(fg3g,msg_sent3g,eps3g)
pt3g = getPotential(fg3g,ni=30)

print('posterior P(x_n=1):', pt3g[1,:])
msg_dec3g = MsgEst(pt3g) #decoded message

#decoded=Hp.dot(msg_dec3g)%2 #now let's multiply by the transpose H:
decoded=msg_dec3g[0:1600]
decoded=BinTo256(decoded)
arr2 = np.asarray(decoded).reshape(shape)# reform a numpy array of the original shape
img2 = Image.fromarray(np.uint8(arr2))# make a PIL image
img2.show()
