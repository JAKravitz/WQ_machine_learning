#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 11:21:44 2021

@author: jkravz311
"""
import numpy as np
import pandas as pd
import timeit
import keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
#import xgboost as xgb
#from xgboost.sklearn import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from models import CNNreg_rrs, MLPreg_rrs
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
import helpers as hp


def regressor(reg,X,y,scoreDict,epoch,batch,cv=5):
    
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        ]
    
    # get feature/target dims
    N_dims = X.shape
    try:
        N_out = y.shape[1]
    except:
        N_out = 1
        
    # estimators
    funcs = {'RFR': RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1),
             'KNR': KNeighborsRegressor(leaf_size=30, metric='minkowski', n_neighbors=2, p=4, n_jobs=-1),
             #'XGBR': MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror',n_estimators=500,max_depth=6,gamma=.4)),
             #'MLPref': MLPreg_ref(N_dims,N_out),
             'MLPrrs': MLPreg_rrs(N_dims,N_out),
             #'MDNref': MDNr_ref(N_dims,N_out),
             #'MDNrrs': MDNr_rrs(N_dims,N_out)
             }
    # choose estimator
    estimator = funcs[reg]
    # prep results dict
    results = hp.results_prep(y)
    owt_results = hp.owt_results_prep(y)
    
    # Kfold
    count = 0
    kfold = KFold(n_splits=cv,shuffle=True)
    for train, test in kfold.split(X, y):
    #for k in range(cv):
        print ('# Fold {} #'.format(count))
        if reg in ['MLPref','MLPrrs','CNNrrs']:
            model = KerasRegressor(build_fn=estimator, epochs=epoch, 
                                   batch_size=batch, validation_split=0.1,
                                   callbacks=callbacks_list)
        else:
            model = estimator
        X_train, X_test = X.iloc[train,:], X.iloc[test,:]
        y_train, y_test = y.iloc[train,:], y.iloc[test,:]
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15)
        # run the model
        tic=timeit.default_timer()
        model.fit(X_train, y_train)
        toc=timeit.default_timer()
        fittime = toc-tic
        # predict
        tic = timeit.default_timer()
        y_hat =  pd.DataFrame(model.predict(X_test),index=y_test.index.values)
        toc = timeit.default_timer()
        predtime = toc-tic
        # if reg in ['MDNref','MDNrrs']:
        #     y_samples = np.apply_along_axis(mdn.sample_from_output, 1, y_hat, N_out, N_mixes,temp=1.0)
        #     y_hat= pd.DataFrame(y_samples[:,0,:],index=y_test.index.values)         
        # evaluate targets 
        results = hp.evaluate(y_test,y_hat,results,scoreDict,log=True)
        results['fit_time'].append(fittime)
        results['pred_time'].append(predtime)
        # by OWT
        owt_results = hp.owt_evaluate(y_test,y_hat,owt_results,scoreDict,log=True)
        count = count+1
        
    final = {'global':results,
             'OWT': owt_results}
    
    return final

                
                
                
                

                