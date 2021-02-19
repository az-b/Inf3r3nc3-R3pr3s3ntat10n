import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from pystruct.models import ChainCRF
from pystruct.learners import OneSlackSSVM
import matplotlib.pyplot as plt
#import sys
#sys.path.append('/Users/BOY/PYTHON/INFREP/HW6')
#import a_sample_code 

def ReadData(whichset):   # because I haven't figured out how to import from another file and use __main__ to avoid running the imported script.      
    if whichset=="train":     
        N = 5000         
    else:                
        N = 1000         
    X, y = [], []
    sentences = []       
    for f in ["/Users/BOY/PYTHON/INFREP/HW6/Data/" + whichset +"-%d.txt" % i for i in range(1, N+1)]:        
        data = pd.read_csv(f, header=None, quoting=3)                       # FROM SAMPLE #Read each training sample file into 'data' variable
        sentences.append(data[0])  
        labels = data[1]                                                    # FROM SAMPLE #Extract 'tag' field into 'labels'
        features = data.values[:, 2:].astype(np.int)                        # FROM SAMPLE #Extract feature fields into 'features'
        for f_idx in range(len(features)):                                  # FROM SAMPLE #Adjust features starting at 1 to start at 0
          f1 = features[f_idx]
          features[f_idx] = [f1[0]-1, f1[1], f1[2], f1[3]-1, f1[4]-1]
        y.append(labels.values - 1)                                         # FROM SAMPLE #Adjust labels to lie in {0,...,9}, and add to 'y'
        X.append(features)                                                  # FROM SAMPLE #Add feature vector to 'X'

    encoder = OneHotEncoder(n_values=[1,2,2,201,201],sparse=False).fit(np.vstack(X))                 
                                                                            # FROM SAMPLE #Represent features using one-of-K scheme: If a feature can take value in 
    X_encoded = [encoder.transform(x) for x in X]                           # FROM SAMPLE #{0,...,K}, then introduce K binary features such that the value of only 
    return X_encoded, y, sentences                                                                   # FROM SAMPLE #the i^th binary feature is non-zero when the feature takes value 'i'.
 

X_train, Y_train, TrainSent = ReadData("train") 
X_test, Y_test, TrainSent = ReadData("test")  
X_val = X_train[-500:]
Y_val = Y_train[-500:]
X_train = X_train[:4500]
Y_train = Y_train[:4500]


crf = ChainCRF(n_states=10,inference_method='max-product',directed=True)
l1 = [10**i for i in range(-4,3,1)]
l1.extend([5*l for l in l1])
Cs = sorted(l1)
error = {}
best_C = {}

Train_Sizes = [100,200,500,1000,4500]
for b in Train_Sizes:
	score = {}
	for C in Cs:
		ssvm = OneSlackSSVM(crf, max_iter=200, C=C)
		ssvm.fit(X_train[:b],Y_train[:b])
		score[C] = ssvm.score(X_val,Y_val)
		print('b = ', b, 'C = ', C, ' : ', score[C])
	best_C[b] = max(score, key=score.get)
	error['train',b] = 1. - score[best_C[b]]

for b in Train_Sizes:
	ssvm = OneSlackSSVM(crf, max_iter=200, C=best_C[b])
	ssvm.fit(X_train[:b],Y_train[:b])
	error['test',b] = 1. - ssvm.score(X_test,Y_test)

plt.xlabel('Size of the training set')
plt.ylabel('Error')
plt.plot(Train_Sizes,[error['train',b] for b in Train_Sizes],label='train')
plt.plot(Train_Sizes,[error['test',b] for b in Train_Sizes],label='test')
plt.legend()
plt.show() #


