#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 17:28:03 2021

@author: jakravit
"""
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
try:
    from MLP_atcor import MLPregressor
except:
    from atm_cor_v2.MLP_atcor import MLPregressor
from sklearn.model_selection import KFold

# data
try:
    rrsData = pickle.load( open( "/content/drive/My Drive/nasa_npp/RT/sensorIDX_rrs.p", "rb" ) )
    refData = pickle.load( open( "/content/drive/My Drive/nasa_npp/RT/sensorIDX_ref.p", "rb" ) )
    run_info = pd.read_csv('/content/hyperspec_DL/atm_cor_v2/run_info.csv',index_col='batch_id')
except:
    rrsData = pickle.load( open( "/Users/jakravit/Desktop/nasa_npp/RT/sensorIDX_rrs.p", "rb" ) )
    refData = pickle.load( open( "/Users/jakravit/Desktop/nasa_npp/RT/sensorIDX_ref.p", "rb" ) )
    run_info = pd.read_csv('/Users/jakravit/git/hyperspec_DL/atm_cor_v2/run_info.csv', index_col='batch_id')

for run in run_info.index:

    batch_info = {
                  'sensor':'hico',
                  'epochs':100,
                  'batch_size':64,
                  'lrate':.0001,
                  'split':.2,
                  'layers':[100,100,100,100,100],
                  'cv':5,
                  'meta': run_info.loc[run,'meta'],
                  'Xtransform': run_info.loc[run,'Xtransform'],
                  'Xpca': run_info.loc[run,'Xpca'],
                  'ytransform': run_info.loc[run,'ytransform'],
                  'ypca': run_info.loc[run,'ypca']}
    
    for key, value in batch_info.items():
        value = None if value == 'None' else value
        batch_info[key] = value
    
    print ('\n##### {} #####\n##### C:{}/{} #####\n'.format(run_info.loc[run,'name'],run,run_info.shape[0]))
    
    batch = MLPregressor(batch_info)
    X = batch.features(refData)
    y = batch.targets(rrsData)
    model = batch.build()
    results = batch.prep_results()
    kfold = KFold(n_splits=batch_info['cv'], shuffle=True)
    count = 0
    for train, test in kfold.split(X, y):
        print ('FOLD = {}...'.format(count))
        X_train, X_test = X.iloc[train,:], X.iloc[test,:]
        y_train, y_test = y.iloc[train,:], y.iloc[test,:] 
        history = batch.fit(X_train,y_train)
        results['train_loss'].append(history.history['loss'])
        results['val_loss'].append(history.history['val_loss'])
        y_hat = batch.predict(X_test,)
        if batch_info['ypca']:
            # y_hat
            y_hatT = batch.nPCA_revert(y_hat)
            y_hat = batch.transform_inverse(y_hatT)
            # y_test
            y_testT = batch.nPCA_revert(y_test)
            y_test = batch.transform_inverse(y_testT)  
        else:
            y_hat = batch.transform_inverse(y_hat)
            y_test = batch.transform_inverse(y_test)
            
        final = batch.evaluate(y_test,y_hat,results) 
        count = count+1   
    
        # save run to disk
        fname = '/content/drive/My Drive/atm_cor_results_v2/{}.p'.format(run_info.loc[run,'name'])
        f = open(fname,'wb')
        pickle.dump(final,f)
        f.close() 