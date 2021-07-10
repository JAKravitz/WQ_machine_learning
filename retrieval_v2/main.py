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
from sklearn.model_selection import KFold
try:
    from MLP_retrieval import MLPregressor
except:
    from retrieval_v2.MLP_retrieval import MLPregressor

try:
    refData = pickle.load( open( "/content/drive/My Drive/nasa_npp/RT/sensorIDX_ref.p", "rb" ) )
except:
    refData = pickle.load( open( "/Users/jakravit/Desktop/nasa_npp/RT/sensorIDX_ref.p", "rb" ) )

case = 1
for meta in [True,False]:
    for n in [None,10,20]:

        batch_info = {
                      'sensor':'hico',
                      'epochs':100,
                      'batch_size':32,
                      'lrate':.0001,
                      'split':.2,
                      'layers':[],
                      'targets': ['chl','PC','fl_amp','aphy440','ag440','anap440','bbphy440','bbnap440'],
                      'cv':5,
                      'meta': meta, #run_info.loc[run,'meta'],
                      'Xpca': n # run_info.loc[run,'Xpca'],}
                      }
        
        for key, value in batch_info.items():
            value = None if value == 'None' else value
            batch_info[key] = value
        
        print ('\n### CASE {} ###\n'.format(case))
    
        model = MLPregressor(batch_info)
        X,y = model.getXY(refData)
        model.build()
        results = model.prep_results(y)
        kfold = KFold(n_splits=batch_info['cv'], shuffle=True)
        count = 0
        for train, test in kfold.split(X, y):
            print ('FOLD = {}...'.format(count))
            X_train, X_test = X.iloc[train,:], X.iloc[test,:]
            y_train, y_test = y.iloc[train,:], y.iloc[test,:] 
            history = model.fit(X_train,y_train)
            results['train_loss'].append(history.history['loss'])
            results['val_loss'].append(history.history['val_loss'])
            y_hat = model.predict(X_test)
            results = model.evaluate(y_hat,np.exp(y_test),results) 
            count = count+1 
        results['batch_info'] = batch_info
        # save run to disk
        fname = '/content/drive/My Drive/retrieval_results_v2/case_{}.p'.format(case)
        f = open(fname,'wb')
        pickle.dump(results,f)
        f.close() 
        case = case+1
        
            