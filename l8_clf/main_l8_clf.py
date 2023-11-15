#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:09:36 2022

@author: jakravit
"""
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from l8_clf.MLPclassifier import MLPclassifier
import matplotlib.pyplot as plt  


refData = pickle.load( open( "/Users/jakravit/Desktop/sensorIDX_ref.p", "rb" ) )
# rrsData = pickle.load(open( "/Users/jakravit/Desktop/sensorIDX_rrs.p", "rb" ) )

#%%
l8data = refData['l8']
l8data.chl = pd.cut(l8data.chl, bins=[0, 5, 10, 30, 60, 100, 500, np.inf], labels=[0,1,2,3,4,5,6])
# l8data.cnap = pd.cut(l8data.chl, bins=[0, 1, 10, 50, 100, np.inf], labels=[0,1,2,3,4])
# l8data.aphy444 = pd.cut(l8data.aphy440, bins=[0, 0.01, 0.1, 0.5, 1, 3, 10, 50, np.inf], labels=[1,2,3,4,5,6,7,8])
# l8data.ag440 = pd.cut(l8data.ag440, bins=[0, .1, 1, 5, 20, np.inf], labels=[0,1,2,3,4])

#% L8

def clean(data):   
    # data.fillna(0,inplace=True)
    # data.replace(np.inf,0)
    data = data.replace([np.inf, -np.inf], np.nan, inplace=False)    
    data = data.dropna(axis=0,how='any',inplace=False)
    return data

feature_names = ['443', '482', '561', '665', '865']
label = "chl"

# get the features and labels into separate variables
X = l8data[feature_names]
y = l8data[label]

# change feature names to l8 bands
X.columns = ['B1','B2','B3','B4','B5']
X = clean(X)
y = y.loc[X.index]   

df = pd.concat([X,y], axis=1)

df2 = df.sample(n=20000)
df2.to_csv('/Users/jakravit/Desktop/l8_chl20k.csv')

#%% s2
s2data = refData['s2']
# s2data.chl = pd.cut(s2data.chl, bins=[0, 5, 10, 30, 60, 100, 500, np.inf], labels=[0,1,2,3,4,5,6])

def clean(data):   
    # data.fillna(0,inplace=True)
    # data.replace(np.inf,0)
    data = data.replace([np.inf, -np.inf], np.nan, inplace=False)    
    data = data.dropna(axis=0,how='any',inplace=False)
    return data

feature_names = ['443', '490', '560', '665', '705', '740','783','842','865']
label = ['chl', 'PC', 'ag440','cnap']

# get the features and labels into separate variables
X = s2data[feature_names]
y = s2data[label]

# change feature names to l8 bands
X.columns = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A']
X = clean(X)
y = y.loc[X.index]   

df = pd.concat([X,y], axis=1)

# df2 = df.sample(n=20000)
df.to_csv('/Users/jakravit/Desktop/s2_MLP_REG_ref_all.csv', index=False)

#%%
y.replace(to_replace=[1,4,10], value='Mild', inplace=True) # Mild
y.replace(to_replace=[0,11], value='NAP', inplace=True) # nap
y.replace(to_replace=[7,12], value='CDOM', inplace=True) # cdom
y.replace(to_replace=[6], value='EUK', inplace=True) # euk
y.replace(to_replace=[5,8], value='CY', inplace=True) # cy
y.replace(to_replace=[2], value='SCUM', inplace=True) # scumd
y.replace(to_replace=[3,9], value='OLIGO', inplace=True) # oligo

encode = {'Mild': 0, 'NAP': 1, 'CDOM': 2, 'EUK': 3, 
          'CY': 4, 'SCUM':5, 'OLIGO':6}

y2 = y.replace(encode)


df = pd.concat([X,y2], axis=1)
df.to_csv('/Users/jakravit/Desktop/s2_owt_all.csv')

df2 = df.sample(n=20000)
df2.to_csv('/Users/jakravit/Desktop/s2_owt20k.csv')

# df2 = df.sample(n=50000)
# df2.to_csv('/Users/jakravit/Desktop/l8_owt50k.csv')

df2 = df.sample(n=100000)
df2.to_csv('/Users/jakravit/Desktop/s2_owt100k.csv')

#%%
# target = ['chl','cnap','ag440']
target = ['chl']

batch_info = {'epochs':10,
              'batch_size':128,
              'lrate':.0001,
              'split':.2,
              'targets': target,
              'cv':2,
              }

model = MLPclassifier(batch_info)
X, y = model.getXY(l8data)

# results = model.prep_results(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=batch_info['split'])
# kfold = KFold(n_splits=batch_info['cv'], shuffle=True)

# for train, test in kfold.split(X_train, y_train):
#     model.build()
#     X_tn, X_tt = X_train.iloc[train,:], X_train.iloc[test,:]
#     y_tn, y_tt = y_train.iloc[train,:], y_train.iloc[test,:] 
#     history = model.fit(X_tn,y_tn)
#     y_pr, y_ht = model.predict(X_tt)
#     # results = model.evaluate(y_pr, y_tt, results,'cv') 

model.build()
history = model.fit(X_train, y_train)
y_proba, y_hat = model.predict(X_test)
y_test_class = np.argmax(y_test.values,axis=1)
results = model.evaluate(y_test, y_proba, y_hat)

#%%
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc
from yellowbrick.classifier import ROCAUC

y_test_class = np.argmax(y_test.values,axis=1)

auc = roc_auc_score(y_test, y_pr, multi_class='ovr', average='macro')
print (auc)

# roc curve for classes
fpr = {}
tpr = {}
thresh ={}

n_class = 5

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_ht, y_pr[:,i], pos_label=i)
    
# plotting    
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--',color='black', label='Class 4 vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
# plt.savefig('Multiclass ROC',dpi=300);    


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test_class, y_ht)

from sklearn.metrics import classification_report
print(classification_report(y_test_class, y_ht))

