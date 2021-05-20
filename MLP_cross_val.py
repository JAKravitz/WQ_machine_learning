#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 12:07:14 2021

@author: jakravit
"""
#from MDNregressor import MDNregressor
from MLPregressor import MLPregressor
from sklearn.model_selection import KFold


def reg_prep_results(cols):
    results = {}
    for var in cols:
        if var == 'cluster':
            continue
        results[var] = {'ytest': [],
                        'yhat': [],
                        'R2': [],
                        'RMSE': [],
                        'RMSELE': [],
                        'Bias': [],
                        'MAPE': [],
                        'rRMSE': []}
        results['fit_time'] = []
        results['pred_time'] = []
        results['train_loss'] = []
        results['val_loss'] = []
    
    return results
        

def reg_cross_val(X,y,ti,scoreDict):
    
    results = reg_prep_results(y.columns)
    
    kfold = KFold(n_splits=ti['cv'], shuffle=True)
    count = 0
    for train, test in kfold.split(X, y):
        
        print ('FOLD = {}...'.format(count))
        X_train, X_test = X.iloc[train,:], X.iloc[test,:]
        y_train, y_test = y.iloc[train,:], y.iloc[test,:]   
        
        n_in = X_train.shape[1]
        n_out = y_train.shape[1]
        
        model = MLPregressor(n_in,n_out,ti['epochs'],ti['batch_size'],
                             ti['lrate'],ti['split'])
        
        model.build(ti['layers'])
        
        history = model.fit(X_train,y_train)
        results['train_loss'].append(history.history['loss'])
        results['val_loss'].append(history.history['val_loss'])
        
        y_hat = model.predict(X_test, y_test)
        
        final = model.evaluate(y_test,y_hat,results,scoreDict,log=True) 
        count = count+1
    
    return final
