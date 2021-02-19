from sklearn.preprocessing import OneHotEncoder
import numpy as np
from pystruct.models import ChainCRF
from pystruct.learners import *
import pandas as pd

def odd(x):
    if (x % 2) == 0:
        s= 0
    else:
        s= 1
    return s

def ReadData(whichset):        
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
                                                                        # FROM SAMPLE #n_values specifies the number of states each feature can take.
                 

def train_one_C(X_train, y_train, X_val, y_val, Cs):
  crf = ChainCRF(n_states=10, inference_method="max-product", directed=True)
  best_model = None
  best_C = None
  smallest_error = None
  for C in Cs:
    print("C =", C, ", training...")
    ssvm = OneSlackSSVM(crf, max_iter=200, C=C)
    ssvm.fit(X_train, y_train)
    error = 1 - ssvm.score(X_val, y_val) # Note: score 1 - error 
    if not smallest_error or error < smallest_error:
      best_model = ssvm
      best_C = C
      smallest_error = error
  print("Completed.")
  return best_model, best_C, smallest_error

def train_all_C(x_Train, y_Train):
  model_tuples = []
  #let's define the range of Cs
  l1 = [10**i for i in range(-4,3,1)]
  l1.extend([5*l for l in l1])
  Cs = sorted(l1)
  model, C, train_error = train_one_C(x_Train[:4500],y_Train[:4500],x_Train[-500:],y_Train[-500:], Cs)
  model_tuples.append((model, C, train_error))
  return model_tuples

X_train, Y_train, TrainSent = ReadData("train")  
X_test, Y_test, TrainSent = ReadData("test")      
                                                     
model, best_C, train_error = train_all_C(X_train, Y_train)[0]
test_error = 1 - model.score(X_test, Y_test)
print("Best C: ", best_C, ", training error: ", train_error, ", validation error: ", test_error)

list = [best_C, train_error]
crf = ChainCRF(n_states=10, inference_method="max-product", directed=True)
ssvm = OneSlackSSVM(crf, max_iter=200, C=best_C)
ssvm.fit(X_train[:4500],Y_train[:4500])
list.append(1. - ssvm.score(X_test,Y_test))
print(list)

ssvm.fit(X_train,Y_train)
error = 1. - ssvm.score(X_test,Y_test)
print('test error: ', error)   


