import random
import numpy as np
from pystruct.models import ChainCRF
from pystruct.learners import OneSlackSSVM
import itertools


def ReadData(whichset):  # because I haven't figured out how to import without executing the file I'm importing      
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

best_C=0.1

crf = ChainCRF(n_states=10, inference_method="max-product", directed=True)
ssvm = OneSlackSSVM(crf, max_iter=200, C=best_C)
ssvm.fit(X_train[:4500],Y_train[:4500])
error = 1 - ssvm.score(X_train[-500:],Y_train[-500:])

tag = np.array(['verb','noun','adjective','adverb','preposition','pronoun',
	'determiner','number','punctuation','other'])
cl = random.sample(range(10),3)
print('Chosen classes: ',tag[cl])

trans_matrix = np.reshape(ssvm.w[-10*10:],(10,10))
pairs=list(itertools.combinations(cl,2))
for pair in pairs:	
	print(tag[pair[0]], "->", tag[pair[1]], trans_matrix[pair[0]][pair[1]])
	print(tag[pair[1]], "->", tag[pair[0]], trans_matrix[pair[1]][pair[0]])


features = ['Bias','Not Initial Capital','Initial Capital',
	'Not All Capitals','All Capitals']
with open('/Users/BOY/PYTHON/INFREP/HW6/prefixes.txt') as _file:
    tmp = _file.readlines()
pre = [x.strip() for x in tmp]
for i in range(len(pre)):
	features.append('Prefix: '+pre[i]) 
with open('/Users/BOY/PYTHON/INFREP/HW6/suffixes.txt') as _file:
    tmp = _file.readlines()
suf = [x.strip() for x in tmp] 
for i in range(len(pre)):
	features.append('Suffix: '+suf[i]) 

# look for most relevant features

feat_array = ssvm.w[:-10*10]
m = int(len(feat_array)/10)
feat = [feat_array[i*m:(i+1)*m] for i in range(10)]
for i in cl:
	tmp = np.abs(np.array(feat[i]))
	tmp = np.argsort(tmp)[-10:][::-1]
	print('Most relevant features for tag ',tag[i],' :')
	print([features[j] for j in tmp]) 


###### PART D ######

predicted = ssvm.predict(X_test)
tagz = np.array(['verb','noun','adjective','adverb','preposition','pronoun',
	'determiner','number','punctuation','other'])
numz = np.sort(np.array(random.sample(range(1000),10)))
for i in numz:
	_file = '/Users/BOY/PYTHON/INFREP/HW6/Data/test-%d.txt'%(i+1)
	tmp = pd.read_csv(_file, header=None, quoting=3)
	sentence = np.array(tmp[0])
	trueTag = np.array(tmp[1])-1
	predTag = predicted[i]
	falseCount = np.sum(trueTag != predTag)
	trueTag = np.array([tagz[j] for j in trueTag])
	predTag = np.array([tagz[j] for j in predTag])
	print('Sentence #',i,':')
	print()
	nwords = sentence.shape[0]
	for c in range(nwords):
		print(sentence[c], "|", trueTag[c], '; Predicted:', predTag[c])
	print()
	print('Error rate for sentence',i,': ', round(falseCount/nwords,3))
	print()
