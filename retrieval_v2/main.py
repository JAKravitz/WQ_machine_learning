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
try:
    from MLP_retrieval import MLPregressor
except:
    from retrieval_v2.MLP_retrieval import MLPregressor

try:
    rrsData = pickle.load( open( "/content/drive/My Drive/nasa_npp/RT/sensorIDX_rrs.p", "rb" ) )
except:
    rrsData = pickle.load( open( "/Users/jakravit/Desktop/nasa_npp/RT/sensorIDX_rrs.p", "rb" ) )

case = 1
target = ['chl','PC','fl_amp','aphy440','ag440','anap440','bbphy440','bbnap440','cluster']

for n in [None,20,10]:
        
    batch_info = {
                  'sensor':'hico',
                  'epochs':100,
                  'batch_size':32,
                  'lrate':.0001,
                  'split':.1,
                  'targets': target,
                  'cv':3,
                  'meta': None,
                  'Xpca': n, 
                  }
    
    for key, value in batch_info.items():
        value = None if value == 'None' else value
        batch_info[key] = value
    
    print ('\n### CASE {} ###\n'.format(case))

    model = MLPregressor(batch_info)
    X,y = model.getXY(rrsData)
    results = model.prep_results(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=batch_info['split'])
    kfold = KFold(n_splits=batch_info['cv'], shuffle=True)
    count = 0
    for train, test in kfold.split(X_train, y_train):
        print ('\n## FOLD = {}... ##\n'.format(count))
        model.build()
        X_tn, X_tt = X_train.iloc[train,:], X_train.iloc[test,:]
        y_tn, y_tt = y_train.iloc[train,:], y_train.iloc[test,:] 
        history = model.fit(X_tn,y_tn)
        y_ht = model.predict(X_tt)
        results = model.evaluate(y_ht,y_tt,results,'cv') 
        count = count+1
    
    print ('\n## FINAL MODEL ##\n')
    history2 = model.fit(X_train,y_train)
    results['train_loss'].append(history2.history['loss'])
    results['val_loss'].append(history2.history['val_loss'])
    y_hat = model.predict(X_test)
    results = model.evaluate(y_hat,y_test,results,'final') 
    results = model.owt_evaluate(y_hat,y_test,results)
    results['batch_info'] = batch_info
    # save run to disk
    fname = '/content/drive/My Drive/retrieval_results_rrs/case_{}.p'.format(case)
    f = open(fname,'wb')
    pickle.dump(results,f)
    f.close() 
    case = case+1
            
#%%
# import pickle
# import matplotlib.pyplot as plt

# data = pickle.load( open( "/Users/jakravit/Desktop/retrieval_results_rrs/case_1.p", "rb" ) )
# fig, ax = plt.subplots()
# ax.scatter(data['chl']['final']['ytest'], data['chl']['final']['yhat'],s=.1,c='b')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim(.01,2000)
# ax.set_ylim(.01,2000)

