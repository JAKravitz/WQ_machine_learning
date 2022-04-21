#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 11:42:15 2021

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
from standard.MLPregressor import MLPregressor
import matplotlib.pyplot as plt

rrsData = pd.read_csv('/Users/jakravit/Desktop/humeshni_dataset.csv',index_col =0)

target = ['chl','size','agd','mineral']

batch_info = {
              'epochs':200,
              'batch_size':32,
              'lrate':.0001,
              'split':.1,
              'targets': target,
              'cv':4,
              'meta': None,
              'Xpca': None, 
              }

for key, value in batch_info.items():
    value = None if value == 'None' else value
    batch_info[key] = value

model = MLPregressor(batch_info)
X, y, y0 = model.getXY(rrsData)
results = model.prep_results(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=batch_info['split'])
kfold = KFold(n_splits=batch_info['cv'], shuffle=True)
for train, test in kfold.split(X_train, y_train):
    model.build()
    X_tn, X_tt = X_train.iloc[train,:], X_train.iloc[test,:]
    y_tn, y_tt = y_train.iloc[train,:], y_train.iloc[test,:] 
    history = model.fit(X_tn,y_tn)
    y_ht = model.predict(X_tt)
    results = model.evaluate(y_ht,y_tt,results,'cv') 

print ('\n## FINAL MODEL ##\n')
history2 = model.fit(X_train,y_train)
results['train_loss'].append(history2.history['loss'])
results['val_loss'].append(history2.history['val_loss'])
y_hat = model.predict(X_test)
results = model.evaluate(y_hat,y_test,results,'final') 
results['batch_info'] = batch_info

# plot
fig, ax = plt.subplots()
ax.scatter(results['chl']['final']['ytest'], results['chl']['final']['yhat'],s=.1,c='k')
            

