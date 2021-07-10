#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:45:16 2021

@author: jakravit
"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
import timeit
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np


class MLPregressor(BaseEstimator):
    
    def __init__(self,n_in,n_out,epochs,batch_size,lrate,split):
        self.n_in = n_in
        self.n_out = n_out
        self.epochs = epochs
        self.batch_size = batch_size
        self.lrate = lrate
        self.split = split
        
    def build(self, layers,):
        self.model = Sequential(
                [Dense(layers[0], use_bias=False, input_shape=(self.n_in,)), BatchNormalization(), ReLU(),Dropout(0.1),
                 Dense(layers[1], use_bias=False), BatchNormalization(), ReLU(), Dropout(0.1),
                 Dense(layers[2], use_bias=False), BatchNormalization(), ReLU(), Dropout(0.1), 
                 Dense(layers[3], use_bias=False), BatchNormalization(), ReLU(), Dropout(0.1),
                 Dense(self.n_out)
                 ])
        # compile
        self.model.compile(Adam(lr=self.lrate),loss='mean_absolute_error')
        print (self.model.summary())
        return self.model
    

    def fit(self, X_train, y_train):
        tic=timeit.default_timer()
        history = self.model.fit(X_train, y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_split=self.split,
                                 verbose=1)
        toc=timeit.default_timer()
        self.fit_time = toc-tic
        return history
    
    def predict(self, Xt, yt):
        tic = timeit.default_timer()
        y_hat =  pd.DataFrame(self.model.predict(Xt),index=yt.index.values,columns=yt.columns)
        toc = timeit.default_timer()
        self.pred_time = toc-tic 
        return y_hat
    
    def evaluate(self,y_hat,y_test,results,scoreDict,scores):
        for var in y_test.columns:
            if var in ['adj','cluster','admix']:
                continue
            #print (var)
            y_t = y_test.loc[:,var].astype(float)
            y_h = y_hat.loc[:,var].astype(float)
            
            if scores in ['regScore']:
                true = np.logical_and(y_h > 0, y_t > 0)
                y_tst = y_t[true]
                y_ht = y_h[true]
            else:
                y_tst = y_t
                y_ht = y_h

            for stat in scoreDict:
                results[var][stat].append(scoreDict[stat](y_tst,y_ht))
            
            results[var]['ytest'].append(y_tst)
            results[var]['yhat'].append(y_ht)
            # results['pred_time'].append(self.pred_time)
            # results['fit_time'].append(self.fit_time)            
        return results
    
    