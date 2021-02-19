
# coding: utf-8

# In[1]:

"""
Inference & Representation HW3
Question 2 PCA and Non-negative matrix factorization.

"""

"""
Tools for loading the MNIST Data.
From Optimization Based Data Analysis HW1
@author: Brett
"""
import numpy as np
from mnist_tools import *
from plot_tools import *
import matplotlib.pyplot as plt
#from scipy.linalg import orth

datafile = "mnist_all.mat" #Change if you put the file in a different path
train = load_train_data(datafile)

trainarr=np.asarray(train)
trainarr = np.reshape(trainarr, (trainarr.shape[0]*trainarr.shape[1],-1))
trainarr = trainarr.astype(float)
trainarr=trainarr-trainarr.mean(axis=0)

# In[2]:

"""
Plot of the singular vectors corresponding 
to top 10 singular values of the data.
@author: Vlad 
"""
U, s, V = np.linalg.svd(trainarr, full_matrices=True)
n=10
imgs = [V[i,:] for i in range(n)]
plot_image_grid(imgs,"Singular vectors corresponding to top 10 singular values of the data")


# In[3]:

"""
Plot of the results of the nearest neighbour test applied 
to a principal component projection.
@author: Vlad 
"""

def project(V, Images) :
     return np.dot(V.T, np.dot(V, Images))
    
def compute_nearest_neighbors(train, testImage, V) :
    train=[np.array(i, dtype=float) for i in train]
    testImage= np.array(testImage, dtype=float)
    digit=0
    imageIdx=0
    dist=np.linalg.norm(project(V, train[digit][imageIdx])-project(V,testImage))
    for i in range(len(train)):
        for j in range (train[i].shape[0]):
            tempDist=np.linalg.norm(project(V,train[i][j])-project(V,testImage))
            if tempDist<dist:
                digit=i
                imageIdx =j
                dist= tempDist
    return digit, imageIdx 

#HERE IS VLAD's PCA:
n=8
U, s, V = np.linalg.svd(trainarr, full_matrices=False)
V2C = V[0:10,:] #I ADD THIS TO BE USED IN PROBLEM 2.C
V=V[0:n,:]

test,testLabels = load_test_data(datafile)

imgs = []
TestLabels = []
for i in range(len(testLabels)) :
    trueDigit = testLabels[i]
    testImage = test[i]
    (nnDig,nnIdx) = compute_nearest_neighbors(train,testImage,V)
    imgs.extend( [testImage,train[nnDig][nnIdx,:]] )
    TestLabels.append(nnDig)

row_titles = ['Test','Nearest']
col_titles = ['%d vs. %d'%(i,j) for i,j in zip(testLabels,TestLabels)]
### I AM RENAMING OUTPUT FILES for VLAD's code #####
plot_image_grid(imgs,
                    "PCA-NN",
                    (28,28),len(testLabels),2,True,row_titles=row_titles,col_titles=col_titles)


#%%########### MY CODE BEGINS #########################################

#### FIRST, LET'S PLOT THE TOP 10 PCA GRAPH REQUIRD FOR PART C. ####

for i in range(10):
	plt.plot(np.arange(10)+1,np.dot(V2C,trainarr[100*i+np.random.randint(0,99),:]),label=str(i))
plt.xlabel('Principal Component')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=-0.9)
plt.savefig('PCA 10 components')
plt.show()


#%%########### NOW MY NMF #############################################

datafile = "mnist_all.mat" 
train = load_train_data(datafile)

trainarr=np.asarray(train)
trainarr = np.reshape(trainarr, (trainarr.shape[0]*trainarr.shape[1],-1))
trainarr = trainarr.astype(float)

test, testLabels = load_test_data(datafile)

def orth(A):#constructing the orthonormal basis
    u,s,vh = np.linalg.svd(A)
    M,N = A.shape
    tol = max(M,N)*np.amax(s)*eps
    num = np.sum(s > tol,dtype=int)
    Q = u[:,:num]
    return Q


def nearest(train, test) :
    result=[]
    for n in range(len(list(train))):
        value=np.linalg.norm(train[n,:]-test) 
        result.append(value)
    return np.argmin(result)

eps = 1e-10

def NMF(x,r,	ni = 3000):
    x_range = np.arange(1,np.max(x)+1)
    w = np.random.choice(x_range,(x.shape[0],r))
    h = np.random.choice(x_range,(r,x.shape[1]))
    for _ in range(ni):
        h *= np.dot(w.T,x+eps)/np.maximum(np.dot(np.dot(w.T,w),h),eps)
        w *= np.dot(x+eps,h.T)/np.maximum(np.dot(np.dot(w,h),h.T),eps)
        #if (i % self.niter_test_conv == 0) and self.checkConvergence():
        #    print("NMF converged after %i iterations" % i)
        #    break
    return w, h


########### PROBLEM 2.A ################
for r in [3,6,10]:
    W, H = NMF(trainarr,r,ni = 3000)
    images = [H[i,:] for i in range(r)]
    #plotting the rows:
    plot_image_grid(images,'rows_n_'+str(r)) 
########### PROBLEM 2.B ################
    tr = np.dot(trainarr, orth(H.T)) #tr: train
    p = np.dot(test, orth(H.T)) #p: test
    images = []
    nearestLabels = []
    for i in range(len(testLabels)):
        index = nearest(tr,p[i])
        images.extend([test[i],trainarr[index,:]])
        nearestLabels.append(index / 100)
        
    row_titles = ['Test','Nearest']
    col_titles = ['%d vs. %d'%(i,j) for i,j in zip(testLabels,nearestLabels)]
    plot_image_grid(images,"EX2_output_n"+str(r),(28,28),len(testLabels),2,True,row_titles=row_titles,col_titles=col_titles) 

########### PROBLEM 2.C ################


for i in range(10):
	plt.plot(np.arange(10)+1,tr[i*100+np.random.randint(0,99),:],label=str(i))
plt.xlabel('Principal Component')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=-0.9)
plt.savefig('NMF 10 components')
plt.show()

