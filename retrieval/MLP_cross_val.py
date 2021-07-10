#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 12:07:14 2021

@author: jakravit
"""
#from MDNregressor import MDNregressor
from evaluate.MLPregressor import MLPregressor
from sklearn.model_selection import KFold
import evaluate.scorers as sc
import sklearn.metrics as metrics


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
        

def reg_cross_val(X,y,ti,scores):

    scoreDict = {
                'clfScore' : {'Accuracy': metrics.accuracy_score,
                              'ROC_AUC': metrics.roc_auc_score},        
                'regScore' : {'R2': sc.r2,
                              'RMSE': sc.rmse,
                              'RMSELE': sc.rmsele,
                              'Bias': sc.bias,
                              'MAPE': sc.mape,
                              'rRMSE': sc.rrmse,},
                'regLogScore' : {'R2': sc.r2,
                                 'RMSE': sc.log_rmse,
                                 'RMSELE': sc.log_rmsele,
                                 'Bias': sc.log_bias,
                                 'MAPE': sc.log_mape,
                                 'rRMSE': sc.log_rrmse,}
                }
    
    scoreDict = scoreDict[scores]
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
        
        final = model.evaluate(y_test,y_hat,results,scoreDict,scores) 
        count = count+1
    
    return final
