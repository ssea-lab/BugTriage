#-*- coding:utf-8 -*-
#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import time
import random
import sys
import os
import gc
from numpy.random import permutation
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_predict
from sklearn import svm, neighbors, tree, naive_bayes, linear_model, metrics
from sklearn.linear_model import SGDClassifier
from sklearn.svm import NuSVC
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
root='E:\\HRW\\Eclipse'
dataPath = root + '\\data_whole_fixer.csv'
delfixer = root + '\\deldevs.csv' # developers who never fixed a bug report

if len(sys.argv) < 2:  
    print 'You have to input args'
    sys.exit()

per = float(sys.argv[1])
str = sys.argv[2]

df_data = pd.read_csv(dataPath)
#clean data
rawdeldata = pd.read_csv(delfixer)
deldata = rawdeldata['delfixer'].tolist()

xx=[]
for i in range(len(deldata)):
    rmvdata = deldata[i]   
    idx = df_data[df_data['fixer']==rmvdata].index.tolist()
    xx.extend(idx)

df_data.drop(xx,inplace=True)


from sklearn import preprocessing
le =preprocessing.LabelEncoder()
labelList =df_data['fixer'].tolist()
le.fit(labelList)
labels=le.transform(labelList)   
print len(labels)

del rawdeldata; deldata; labelList
gc.collect()

Ea1 = [27,28] # features- "Project","Component"
Ea2 =[27,28,32] # features- "Project","Component" and "degree"
# other combination of features also can be applied.

featureset = [Ea1,Ea2]

resultStr1 = 'E:\\HRW\\result_new\\'+ str+'\\Eclipse\\Eclipse_SGD_Exp1_GA.csv'
resultStr1s = 'E:\\HRW\\result_new\\'+ str+'\\Eclipse\\Eclipse_SGD_Exp1s.csv'

charfeature = ['Ea1','Ea2']  #'Ea1','Ea2','Eb1',
ii = 0
splitNum=int(len(labels)*per)

for featurelist in featureset:
    print featurelist
    featurelist.extend(list(np.arange(6, 26)))
    df_dataset = df_data[featurelist]
    df_dataset = pd.get_dummies(df_dataset)  
    nd_dataset_org = pd.DataFrame.as_matrix(df_dataset)

    print resultStr1
    clf1 = linear_model.SGDClassifier(loss='log')
    for kk in range(0,10):
        indices = np.random.permutation(labels.shape[0])
        rand_data_x = nd_dataset_org[indices]  
        rand_data_y = labels[indices] 
        nd_labels = rand_data_y
        nd_dataset = rand_data_x

        X_train=nd_dataset[:splitNum,:]
        X_test=nd_dataset[splitNum:,:]
        Y_train=nd_labels[:splitNum]
        Y_test=nd_labels[splitNum:]

        myfile1=open(resultStr1,'a')
        time1 = time.clock()
        clf1.fit(X_train, Y_train)
        predicted_Y = clf1.predict(X_test)
        time2 = time.clock()
        myfile1.write('group:%s   '% charfeature[ii])    
        myfile1.write('run time(minutes):%s   ' % ((time2 - time1) / 60.0))
        pre_topcal.write_file(myfile1,Y_test, predicted_Y,1)

        probs = clf1.predict_proba(X_test)
        best_n = np.argsort(-probs, axis=1)
        topk = 3
        bestn = clf1.classes_[best_n[:,:topk]]
        out_pred = pre_topcal.precal_topK(Y_test,bestn)
        pre_topcal.write_file(myfile1,Y_test, out_pred,topk)

        topk = 5
        bestn = clf1.classes_[best_n[:,:topk]]
        out_pred = pre_topcal.precal_topK(Y_test,bestn)            
        pre_topcal.write_file(myfile1,Y_test, out_pred,topk)
		
        myfile1.write('end   ' +'\n')
        myfile1.close()
    
    del clf1
    gc.collect()
    print resultStr2

    ii = ii + 1
print 'Congratulations,run over!'
