#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 22:08:24 2017

@author: Маrgаr1tа B0yarskаyа
"""
        
                          
#############  
import operator 
import math  
from matplotlib import pyplot 
import matplotlib.pyplot as plt                    
with open('train.txt', mode='r', encoding='utf-8' ) as f:
    train = f.readlines()

nemails=len(train)

hamcount=0
uniqueWords={} #this will be the vocabulary
uniqueWordsSpam={} #this will be the Spam part of the vocabulary. the values foк each keyword will be the number of occurances of the key word in all SPAM emails
uniqueWordsHam={} #this will be the Hpam part of the vocabulary. ...
spamcount=0
SpamFlag=0

useStop = input('Use stop words? (Y/N):')

if(useStop=="Y"):
    STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours','yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers','herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves','what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are','was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does','did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until','while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into','through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down','in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here','there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more','most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so','than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']


for i in range(0,nemails):
    train[i] = train[i].split()
    
    if(train[i][1] == 'spam'):
        spamcount=spamcount + 1
        SpamFlag=1
    else:
        SpamFlag=0
        
    numpairs=int((0.5)*(len(train[i])-2)) #number of word entries in the line
    for j in range(1,numpairs):#excluding first two entries, let's look at every first entry in a pair
        if(useStop=='Y'):
            if train[i][j*2] in STOP_WORDS:
                continue
        if train[i][j*2] in uniqueWords: 
            uniqueWords[train[i][j*2]]=int(uniqueWords[train[i][j*2]])+int(train[i][j*2+1])
        else :
            uniqueWords[train[i][j*2]]=int(train[i][j*2+1])
        
        if (SpamFlag==1):
            if train[i][j*2] in uniqueWordsSpam:
                uniqueWordsSpam[train[i][j*2]]=int(uniqueWordsSpam[train[i][j*2]])+int(train[i][j*2+1])
            else:
                uniqueWordsSpam[train[i][j*2]]=int(train[i][j*2+1])
        else:
            if train[i][j*2] in uniqueWordsHam:
                uniqueWordsHam[train[i][j*2]]=int(uniqueWordsHam[train[i][j*2]])+int(train[i][j*2+1])
            else:
                uniqueWordsHam[train[i][j*2]]=int(train[i][j*2+1])
pspam=spamcount/nemails
pham=1-pspam
print('------------')
print('p(spam)=',pspam) ### RETURNS ANSWER to part (b)
##################################

nSp=sum(uniqueWordsSpam.values())
nH=sum(uniqueWordsHam.values())
#m=len(uniqueWords)
r=[1,10,100,1000,10000]
mArr=[len(uniqueWords)*v for v in r]

wSp={}
wH={}
acc=[]*5
for i in range(0,len(mArr)):
    m=mArr[i]
    valueSp=math.log(r[i]/(nSp+m))
    valueH=math.log(r[i]/(nH+m))
    
    for word in uniqueWordsSpam:
        wSp[word]=(uniqueWordsSpam[word]+r[i])/(nSp+m)
    
    for word in uniqueWordsHam:
        wH[word]=(uniqueWordsHam[word]+r[i])/(nH+m)
    
    sorted_wSp = sorted(wSp.items(), key=operator.itemgetter(1))
    sorted_wH = sorted(wH.items(), key=operator.itemgetter(1))
    top_wSp=sorted_wSp[len(sorted_wSp)-5:(len(sorted_wSp))]
    top_wSp=list(map(operator.itemgetter(0),top_wSp[0:5]))
    top_wH=sorted_wH[len(sorted_wH)-5:(len(sorted_wH))]
    top_wH=list(map(operator.itemgetter(0),top_wH[0:5]))
    if(i==0):
      print('---------------')
      print('Size of vocabulary |V|=',m)
      print('call "wSp" to see conditional probabilities of voc words given SPAM')
      print('call "wH" to see conditional probabilities of voc words given HAM') 
      print('most likely words given SPAM are:',top_wSp) ### RETURNS ANSWER to part (c)
      print('most likely words given HAM are:',top_wH) ### RETURNS ANSWER to part (c)
###################################

# NOW THE CLASSIFIER
# test data
    accuracy = 0
    count = 0
    with open('test.txt', mode='r', encoding='utf-8' ) as f:
        for line in f.readlines():
            count += 1
            likelhPrSp = 1
            likelhPrH = 1
            id, type, *word_arr = line.split(" ")
            for i, word in enumerate(word_arr):
                if word not in uniqueWords:
                    continue
                if i%2 == 0:
                    if word in wSp.keys():
                        likelhPrSp += math.log(wSp[word])
                    else:
                        likelhPrSp+= valueSp
                    if word in wH.keys():
                        likelhPrH += math.log(wH[word])
                    else:
                        likelhPrH+= valueH
            P_ham = math.log(pham) + likelhPrH
            P_spam = math.log(pspam) + likelhPrSp
            res = "ham" if P_ham > P_spam else "spam"
            if res == type:
                accuracy += 1

    acc.append(accuracy/count*100)
print('---------------')
print("Given the values of m: ", mArr)
print("Accuracies are: ", acc)
plt.plot(mArr,acc,marker='x')
