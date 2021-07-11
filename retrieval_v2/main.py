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
    refData = pickle.load( open( "/content/drive/My Drive/nasa_npp/RT/sensorIDX_ref.p", "rb" ) )
except:
    refData = pickle.load( open( "/Users/jakravit/Desktop/nasa_npp/RT/sensorIDX_ref.p", "rb" ) )

case = 1
for meta in [True,False]:
    for n in [None,20]:
        for target in [['PC'],['aphy440'],['bbphy440']]:
        
            batch_info = {
                          'sensor':'hico',
                          'epochs':150,
                          'batch_size':64,
                          'lrate':.0001,
                          'split':.2,
                          'targets': target,
                          'cv':3,
                          'meta': meta, #run_info.loc[run,'meta'],
                          'Xpca': n # run_info.loc[run,'Xpca'],}
                          }
            
            for key, value in batch_info.items():
                value = None if value == 'None' else value
                batch_info[key] = value
            
            print ('\n### CASE {} ###\n'.format(case))
        
            model = MLPregressor(batch_info)
            X,y = model.getXY(refData)
            results = model.prep_results(y)
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
            kfold = KFold(n_splits=batch_info['cv'], shuffle=True)
            count = 0
            for train, test in kfold.split(X_train, y_train):
                print ('FOLD = {}...'.format(count))
                model.build()
                X_tn, X_tt = X_train.iloc[train,:], X_train.iloc[test,:]
                y_tn, y_tt = y_train.iloc[train,:], y_train.iloc[test,:] 
                history = model.fit(X_tn,y_tn)
                results['train_loss'].append(history.history['loss'])
                results['val_loss'].append(history.history['val_loss'])
                y_ht = model.predict(X_tt)
                results = model.evaluate(y_ht,np.exp(y_tt),results,'cv') 
                count = count+1
            y_hat = model.predict(X_test)
            results = model.evaluate(y_hat,np.exp(y_test),results,'final') 
            results['batch_info'] = batch_info
            # save run to disk
            fname = '/content/drive/My Drive/retrieval_results_v2/case_{}.p'.format(case)
            f = open(fname,'wb')
            pickle.dump(results,f)
            f.close() 
            case = case+1
        
#%%
# import pickle
# import matplotlib.pyplot as plt

# data = pickle.load( open( "/Users/jkravz311/GoogleDrive/retrieval_results_v2/case_1.p", "rb" ) )
# fig, ax = plt.subplots()
# ax.scatter(data['chl']['ytest'][0],data['chl']['yhat'][0],s=.1,c='b')
# ax.scatter(data['chl']['ytest'][1],data['chl']['yhat'][1],s=.1,c='r')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim(0,2000)
# ax.set_ylim(0,2000)

