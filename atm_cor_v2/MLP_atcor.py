#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 17:29:42 2021

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
    
    def __init__(self, batch_info):
        self.sensor = batch_info['sensor']
        self.epochs = batch_info['epochs']
        self.batch_size = batch_info['batch_size']
        self.lrate = batch_info['lrate']
        self.split = batch_info['split']
        self.layers = batch_info['layers']
        self.meta = batch_info['meta']
        self.Xtrans = batch_info['Xtransform']
        self.Xpca = batch_info['Xpca']
        self.ytrans = batch_info['ytransform']
        self.ypca = batch_info['ypca']
    
    def clean(self,data):   
        # data.fillna(0,inplace=True)
        # data.replace(np.inf,0)
        data = data.replace([np.inf, -np.inf], np.nan, inplace=False)    
        data = data.dropna(axis=0,how='any',inplace=False)
        return data
    
    def transform(self,data):
        from sklearn.preprocessing import StandardScaler    
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        return data, scaler  
    
    def nPCA(self,data,n):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n)
        pca.fit(data)
        npca = pca.transform(data)
        comp = pca.components_
        var = pca.explained_variance_ratio_
        return npca, comp, var
    
    def nPCA_revert(self,data):
        revert = np.dot(data,self.ycomp)
        return revert
    
    def transform_inverse(self,data):
        inverse = self.yscaler.inverse_transform(data)
        return inverse
    
    def features(self,refData):
        Xraw = refData[self.sensor]
        X = Xraw.filter(regex='^[0-9]')
        if self.meta:
            metacols = [' SZA',' OZA',' SAA',' OAA',' aot550',' astmx',' ssa400',
                        ' ssa675',' ssa875',' altitude',' adjFactor']
            meta = Xraw.loc[:,metacols]
            meta[' adjFactor'] = meta[' adjFactor'].replace(to_replace=(' '),value=0)
            meta[' adjFactor'] = [float(i) for i in meta[' adjFactor']]
            X = pd.concat([X,meta],axis=1)
        if self.Xtrans:
            Xt, self.Xscaler = self.transform(X)
            Xt = pd.DataFrame(X,columns=X.columns) 
            Xt = self.clean(Xt)
            self.keep = Xt.index
        if self.Xpca:
            # requires transform
            Xt, self.Xcomp, self.Xvar = self.nPCA(Xt.values, int(self.Xpca))
            Xt = pd.DataFrame(Xt)
        self.n_in = Xt.shape[1]
        self.X = X
        return Xt
    
    def targets(self,rrsData):
        yraw = rrsData[self.sensor].filter(regex='^[0-9]')
        y = pd.DataFrame(np.repeat(yraw.values,4,axis=0))
        y.columns = yraw.columns
        if self.ytrans:
            yt, self.yscaler = self.transform(y)
            yt = pd.DataFrame(yt, columns=y.columns)
            yt = yt.loc[self.keep,:]
        if self.ypca:
            # requires transfrom
            yt, self.ycomp, self.yvar = self.nPCA(yt.values, int(self.ypca))
            yt = pd.DataFrame(yt)
        self.n_out = yt.shape[1]
        self.y = y
        return yt
                
    def build(self):
        
        self.model = Sequential(
                [Dense(self.layers[0], use_bias=False, input_shape=(self.n_in,)), ReLU(),Dropout(0.1),
                 Dense(self.layers[1], use_bias=False), ReLU(), Dropout(0.1),
                 Dense(self.layers[2], use_bias=False), ReLU(), Dropout(0.1), 
                 Dense(self.layers[3], use_bias=False), ReLU(), Dropout(0.1),
                 Dense(self.layers[4], use_bias=False), ReLU(), Dropout(0.1),
                 Dense(self.layers[5], use_bias=False), ReLU(), Dropout(0.1),
                 Dense(self.layers[6], use_bias=False), ReLU(), Dropout(0.1), 
                 Dense(self.layers[7], use_bias=False), ReLU(), Dropout(0.1),
                 Dense(self.layers[8], use_bias=False), ReLU(), Dropout(0.1),
                 Dense(self.layers[9], use_bias=False), ReLU(), Dropout(0.1),
                 Dense(self.n_out)
                 ])
        # compile
        self.model.compile(Adam(lr=self.lrate),loss='mean_absolute_error')
        print (self.model.summary())
        return self.model

    def prep_results(self):
        results = {}
        for k in self.y.columns:
            results[k] = {'ytest': [],
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

    def predict(self, X_test):
        tic = timeit.default_timer()
        #y_hat =  pd.DataFrame(self.model.predict(Xt),index=yt.index.values,columns=yt.columns)
        y_hat = self.model.predict(X_test)
        toc = timeit.default_timer()
        self.pred_time = toc-tic 
        return y_hat

    def evaluate(self,y_hat,y_test,results):
        import scorers as sc
        scoreDict = {'R2': sc.r2,
                     'RMSE': sc.rmse,
                     'RMSELE': sc.rmsele,
                     'Bias': sc.bias,
                     'MAPE': sc.mape,
                     'rRMSE': sc.rrmse,}
        
        y_hat = pd.DataFrame(y_hat,columns=self.y.columns)
        y_test = pd.DataFrame(y_test,columns=self.y.columns)
        
        for band in y_hat.columns:

            y_t = y_test.loc[:,band].astype(float)
            y_h = y_hat.loc[:,band].astype(float)
            
            # if scores in ['regScore']:
            #     true = np.logical_and(y_h > 0, y_t > 0)
            #     y_tst = y_t[true]
            #     y_ht = y_h[true]
            # else:
            #     y_tst = y_t
            #     y_ht = y_h

            for stat in scoreDict:
                results[band][stat].append(scoreDict[stat](y_t,y_h))
            
            results[band]['ytest'].append(y_t)
            results[band]['yhat'].append(y_h)
            results['pred_time'].append(self.pred_time)
            results['fit_time'].append(self.fit_time)            
        return results
    
    
    
    
    