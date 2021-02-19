
# coding: utf-8

# In[97]:

"""
HW3
Question 4 Factor Analsys.
@author: Vlad, Rahul
"""

get_ipython().magic('matplotlib inline')
from matplotlib.pylab import plt
import os,sys
import numpy as np
#from sklearn import decomposition  
#you can only use sklearn to test/debug your own implementation of FactorAnalysis


# ## Factor Analysis on self-reported personality data
# Reference: http://web.stanford.edu/class/psych253/tutorials/FactorAnalysis.html
# 

# In[98]:

if not os.path.exists('personality0.txt'):
    ret = os.system('wget http://web.stanford.edu/class/psych253/data/personality0.txt .')
    if ret!=0:
        ret = os.system('curl -o personality0.txt http://web.stanford.edu/class/psych253/data/personality0.txt')
        if ret!=0:
            assert False,'Please download http://web.stanford.edu/class/psych253/data/personality0.txt'
print ('File: ',os.path.exists('personality0.txt'))


# In[99]:

with open('personality0.txt') as f:
    headers = [k.strip().replace('"','') for k in f.readline().split(' ')]
print ('Headers: ',headers,len(headers))
data = np.loadtxt('personality0.txt',usecols=range(1,len(headers)+1),skiprows=1)
assert data.shape==(240,len(headers))


# In[100]:

get_ipython().run_cell_magic('latex', '', '$\\textbf{Visualize the Correlations}: $\n$\\text{Cor}(X_i,Y_j) = \\frac{\\text{Cov}(X_i,Y_j)}{\\sigma_{X_i}\\sigma_{Y_j}}$')


# In[101]:

R = np.corrcoef(data.T)
plt.figure(figsize=(10,8))
plt.pcolor(R)
plt.colorbar()
plt.xlim([0,len(headers)])
plt.ylim([0,len(headers)])
plt.xticks(np.arange(32)+0.5,np.array(headers),rotation='vertical')
plt.yticks(np.arange(32)+0.5,np.array(headers))
plt.show()


# In[108]:

#Lets fit both the models using PCA/FA down to two dimensions. 

#construct a function implementing the factor analysis which returns a vector of n_components largest 
# variances and the corresponding components (as column vectors in a matrix). You can
# check your work by using decomposition.FactorAnalysis from sklearn


#### ~THIS FUNCTION IS WAS A STAB, NEW CODE HERE: ###########
def FactorAnalysis(data, n_components) :
    ni=20
    data=data-data.mean(axis=0) 
    var_n=np.ones(data.shape[1])
    var_d = np.var(data, axis=0)
    for _ in range(ni):
        """ in SCIKIT:
    for i in xrange(self.max_iter):
        # SMALL helps numerics
        sqrt_psi = np.sqrt(psi) + SMALL
        s, V, unexp_var = my_svd(X / (sqrt_psi * nsqrt))
        s **= 2
        # Use 'maximum' here to avoid sqrt problems.
        W = np.sqrt(np.maximum(s - 1., 0.))[:, np.newaxis] * V
        del V
        W *= sqrt_psi
    """
        std_deviations = np.sqrt(var_n)
        _, s, v = np.linalg.svd(data / (std_deviations*np.sqrt(data.shape[0])),full_matrices=False)
        components = v[:n_components].T * np.sqrt(s[:n_components] ** 2 - 1.)
        components = (components.T * std_deviations).T
        var_n = var_d - np.sum(components ** 2, axis=1) 
        """ in SCIKIT:
    Wpsi = self.components_ / self.noise_variance_
    cov_z = linalg.inv(Ih + np.dot(Wpsi, self.components_.T))
    tmp = np.dot(X_transformed, Wpsi.T)
    X_transformed = np.dot(tmp, cov_z)

    return X_transformed
    """
    tmp = components.T / var_n
    data_fa = np.dot(np.dot(data, tmp.T),np.linalg.inv(np.eye(n_components) + np.dot(tmp, components)))
        
    return data_fa, var_n, components

n_components = 2
U, s, V = np.linalg.svd(data-data.mean(axis=0))
pca_comp =  V[range(n_components),:].T
data_pca  = U [:,range(n_components)]
data_fa, v, fa_comp = FactorAnalysis(data,n_components)



#fa  = decomposition.FactorAnalysis(n_components=n_components, max_iter=20)
#fa.fit(data)


# In[110]:

#data_fa  = fa.transform(data)
print (data_pca.shape, data_fa.shape)


# In[111]:

#fa_comp     = fa.components_.T
print (pca_comp.shape, fa_comp.shape)


# In[112]:

N = 10
def plot_scatter_annotate(data,labels,title):
    plt.figure(figsize=(10,10))
    assert data.shape[0]==len(labels),'size mismatch'
    plt.subplots_adjust(bottom = 0.1)
    plt.scatter(
        data[:, 0], data[:, 1], marker = 'o', s = 100,
        cmap = plt.get_cmap('Spectral'))
    plt.title(title)
    for label, x, y in zip(labels, data[:, 0], data[:, 1]):
        plt.annotate(
            label, 
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.show()


# In[113]:

np.random.seed(1)
idxlist = np.random.permutation(len(headers))[:15]
dset_pca = pca_comp[idxlist]
dset_fa = fa_comp[idxlist]
hdr_sub = [headers[k] for k in idxlist.tolist()]
plot_scatter_annotate(dset_pca,hdr_sub,'Visualizing Principle Components from PCA')
plot_scatter_annotate(dset_fa,hdr_sub,'Visualizing Factor Loading Matrix from Factor Analysis')


# In[ ]:



