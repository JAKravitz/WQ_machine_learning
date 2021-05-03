#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for training and validating regression 
and classification ML models for water quality retrieval 
using synthetic optical data from Kravitz et al., 2021

@author: jkravitz, May 2021
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np 
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

################################################################################
def clean(data):   
    data.fillna(0,inplace=True)
    data.replace(np.inf,0)
    return data


################################################################################
def evaluate(y_test,y_hat,results,scoreDict,log=False):
    for i,var in enumerate(y_test.columns):
        if var in ['adj','cluster']:
            continue
        print (var)
        y_t = y_test.iloc[:,i].astype(float)
        y_h = y_hat.iloc[:,i].astype(float)
        if log == False:
            true = np.logical_and(y_h > 0, y_t > 0)
            y_tst = y_t[true]
            y_ht = y_h[true]
        else:
            y_tst = y_t
            y_ht = y_h
        results[var]['ytest'].append(y_tst)
        results[var]['yhat'].append(y_ht)
        for stat in scoreDict:
            results[var][stat].append(scoreDict[stat](y_tst,y_ht))
    return results

################################################################################
def results_prep(y):
    results = {}
    for var in y.columns:
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
    return results

################################################################################
def clf_results_prep(y):
    results = {}
    for var in y.columns:
        if var == 'cluster':
            continue
        results[var] = {'fpr': [],
                        'tpr': [],
                        'Acc': [],
                        'AUC': []}
        results['fit_time'] = []
        results['pred_time'] = []
    return results

################################################################################
def clf_owt_results_prep(y):
    results = {}
    for var in y.columns:
        if var == 'cluster':
            continue
        clusters = list(set(y.cluster))
        results[var] = {}
        for k in clusters:
            results[var][k] = {'fpr': [],
                               'tpr': [],
                               'Acc': [],
                               'AUC': []}
    return results

################################################################################
def owt_results_prep(y):
    results = {}
    for var in y.columns:
        if var == 'cluster':
            continue
        clusters = list(set(y.cluster))
        results[var] = {}
        for k in clusters:
            results[var][k] = {'ytest': [],
                                'yhat': [],
                                'R2': [],
                                'RMSE': [],
                                'RMSELE': [],
                                'Bias': [],
                                'MAPE': [],
                                'rRMSE': []}
    return results
  
################################################################################  
def owt_evaluate(y_test,y_hat,results,scoreDict,log=False):
    newcols = []
    for var in y_test.columns:
        if var in ['adj','cluster']:
            continue        
        newvar = var+'2'
        newcols.append(newvar)
        
    datac = pd.concat([y_test,
                        pd.DataFrame(y_hat.iloc[:,:-1].values,index=y_test.index.values,columns=newcols)],
                        axis=1)
    grouped = datac.groupby('cluster') 
    
    for c, group in grouped:
        print ('C',c)
        y_testc = group.loc[:,y_test.columns]
        y_hatc = group.loc[:,newcols]
        
        for i,var in enumerate(y_testc.columns):
            if var in ['adj','cluster']:
                continue
            if var in ['PC'] and c in [3,6,9]:
                continue
            y_t = y_testc.iloc[:,i].astype(float)
            y_h = y_hatc.iloc[:,i].astype(float)
            if log == False:
                true = np.logical_and(y_h > 0, y_t > 0)
                y_tst = y_t[true]
                y_ht = y_h[true]
            else:
                y_tst = y_t
                y_ht = y_h
            
            results[var][c]['ytest'].append(y_tst)
            results[var][c]['yhat'].append(y_ht)
            for stat in scoreDict:
                results[var][c][stat].append(scoreDict[stat](y_tst,y_ht))
    return results    
    
################################################################################
def clf_evaluate(model,X_test,y_test,y_hat,y_proba,results):
    for i,var in enumerate(y_test.columns[:-1]):
        print (var)
        y_t = y_test.iloc[:,i]
        if y_hat is None:
            y_hat = (y_proba > 0.5).astype(int)
            try:
                y_p = y_proba[:,i]
            except:
                y_p = y_proba 
        elif y_proba is None:
            y_proba = model.predict_proba(X_test.values)[i]
            y_p = y_proba[:,1]
        
        try:
            y_h = y_hat[:,i]
        except:
            y_h = y_hat
            
        fpr, tpr, thresh = roc_curve(y_t,y_p)
        AUC = auc(fpr,tpr)
        ACC = np.mean(y_t == y_h)
        results[var]['Acc'].append(ACC)
        results[var]['AUC'].append(AUC)
        results[var]['tpr'].append(tpr)
        results[var]['fpr'].append(fpr)
    return results    
    
 ################################################################################
def clf_owt_evaluate(model,X_test,y_test,y_hat,y_proba,results):
    newcols = []
    newcolsp = []
    probas = pd.DataFrame()
    
    if y_proba is None:
        for i,var in enumerate(y_test.columns[:-1]):
            y_proba = model.predict_proba(X_test.values)[i]
            y_p = y_proba[:,1]
            probas[var] = y_p
            #
            newvar = var+'2'
            newvarp = var+'p'
            newcols.append(newvar)
            newcolsp.append(newvarp)
            
    elif y_hat is None:
        probas = pd.DataFrame(y_proba)
        y_hat = (y_proba > 0.5).astype(int)
        for i,var in enumerate(y_test.columns[:-1]):
            newvar = var+'2'
            newvarp = var+'p'
            newcols.append(newvar)
            newcolsp.append(newvarp)  
        
    datac = pd.concat([y_test,
                       pd.DataFrame(y_hat,index=y_test.index.values,columns=newcols),
                       pd.DataFrame(probas.values,index=y_test.index.values,columns=newcolsp)],
                       axis=1)
    grouped = datac.groupby('cluster')
    
    for c, group in grouped:
        print ('C',c)
        y_testc = group.loc[:,y_test.columns]
        y_hatc = group.loc[:,newcols]
        y_probac = group.loc[:,newcolsp]
        for i,var in enumerate(y_test.columns[:-1]):
            y_t = y_testc.iloc[:,i].astype(float)
            y_h = y_hatc.iloc[:,i].astype(float)
            y_p = y_probac.iloc[:,i].astype(float)
            fpr,tpr,thresh = roc_curve(y_t,y_p)
            AUC = auc(fpr,tpr)
            ACC = np.mean(y_t==y_h)
            results[var][c]['Acc'].append(ACC)
            results[var][c]['AUC'].append(AUC)
            results[var][c]['tpr'].append(tpr)
            results[var][c]['fpr'].append(fpr)
    return results   
 
################################################################################
def getXY(atcor,data,sensor,regex,feats,xlog=False,ylog=False,bandsOnly=False):
    
    
    if bandsOnly == False:
        if sensor in ['s2_10m']:
            X1 = data['s2'].filter(items=regex[sensor])
            X2 = data['s2'].filter(items=[' SZA',' OZA',' SAA',' OAA',])
            X3 = data['s2'].filter(items=feats['s2_10m'])
            data = data['s2']
        elif sensor in ['s2_20m']:
            X1 = data['s2'].filter(items=regex[sensor])
            X2 = data['s2'].filter(items=[' SZA',' OZA',' SAA',' OAA',])
            X3 = data['s2'].filter(items=feats['s2_20m'])
            data = data['s2']
        elif sensor == 's2_60m':
            X1 = data['s2'].filter(regex='^[0-9]')
            X2 = data['s2'].filter(items=[' SZA',' OZA',' SAA',' OAA',])
            X3 = data['s2'].filter(items=feats['s2_20m'])
            data = data['s2']
        elif sensor == 'hicoSVD':
            X1 = data['hico'].filter(regex='^[0-9]')
            X2 = data['hico'].filter(items=[' SZA',' OZA',' SAA',' OAA',])
            X3 = data['hico'].filter(items=feats['hico'])
            X1.dropna(inplace=True)
            u,s,vh = np.linalg.svd(X1,full_matrices=False)
            X1 = pd.DataFrame(u[:,:10],index=X1.index.values)
            data = data['hico']
        else:
            X1 = data[sensor].filter(regex='^[0-9]')
            X2 = data[sensor].filter(items=[' SZA',' OZA',' SAA',' OAA',])
            X3 = data[sensor].filter(items=feats[sensor])
            data = data[sensor]
        X = pd.concat([X1,X3],axis=1) # add/remove X2 geoms
    else:
        if sensor in ['s2_20m','s2_10m']:
            X = data['s2'].filter(items=regex[sensor])
            data = data['s2']
        elif sensor == 's2_60m':
            X = data['s2'].filter(regex='^[0-9]')
            data = data['s2']
        else:
            X = data[sensor].filter(regex='^[0-9]')
            data = data[sensor]
    
    if sensor == 's3':
        X.drop(['761.25','764.375','767.75'], axis=1, inplace=True)
        
    if sensor == 'modis':
        X.drop(['551'],axis=1,inplace=True)
    
    # get targets
    y = data[['chl','PC','cnap','ag440','aphy440','anap440','admix']]
    if atcor == 'ref':
        y['adj'] = data[' adjFactor'].replace(to_replace=(' '),value=0)
        y['adj'] = [float(i) for i in y['adj']]
    #y['cluster'] = data['cluster']
    
    # clean and standardize
    X = clean(X)
    y = clean(y)
    
    if ylog == True:
        y = pd.DataFrame(np.where(y>0,np.log(y),y),index=y.index.values,columns=y.columns)
        y['cluster'] = data['cluster']
    else:
        y['cluster'] = data['cluster']
    
    if xlog == True:
        X = pd.DataFrame(np.where(X>0,np.log(X),X),index=X.index.values,columns=X.columns)
    
    # X = X.iloc[:,:-2]    
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    X = pd.DataFrame(X,index=y.index)
    
    return X, y, scaler    
       
    
    
    
    
