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
        self.targets = batch_info['targets']
        self.meta = batch_info['meta']
        self.Xpca = batch_info['Xpca']
    
    def clean(self,data):   
        # data.fillna(0,inplace=True)
        # data.replace(np.inf,0)
        data = data.replace([np.inf, -np.inf], np.nan, inplace=False)    
        data = data.dropna(axis=0,how='any',inplace=False)
        return data
    
    def l2norm(self,data):
        from sklearn.preprocessing import Normalizer
        scaler = Normalizer(norm = 'l2')
        data = scaler.fit_transform(data)
        return data, scaler 
    
    def standardScaler(self,data):
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

    def getXY(self,data):
        # get sensor data
        
        sensors = {'s2_60m':['443','490','560','665','705','740','783','842','865'],
                   's2_20m':['490','560','665','705','740','783','842','865'],
                   's2_10m':['490','560','665','842'],
                   's3':'^Oa',
                   'l8':['Aer','Blu','Grn','Red','NIR'],
                   'modis':'^RSR',
                   'meris':'^b',
                   'hico':'^H'} 
            
        if self.sensor in ['s2_20m','s2_10m']:
            X = data['s2'].filter(items=sensors[self.sensor])
            data = data['s2']
        elif self.sensor == 's2_60m':
            X = data['s2'].filter(regex='^[0-9]')
            data = data['s2']
        else:
            X = data[self.sensor].filter(regex='^[0-9]')
            data = data[self.sensor]
        
        # drop o2 bands if s3
        if self.sensor == 's3':
            X.drop(['761.25','764.375','767.75'], axis=1, inplace=True)
        
        # drop if modis
        if self.sensor == 'modis':
            X.drop(['551'],axis=1,inplace=True)

        # get meta columns
        if self.meta:
            metacols = ['cluster',' SZA',' OZA',' SAA',' OAA',' aot550',' astmx',' ssa400',
                        ' ssa675',' ssa875',' altitude',' adjFactor', ]
            meta = data.loc[:,metacols]
            meta[' adjFactor'] = meta[' adjFactor'].replace(to_replace=(' '),value=0)
            meta[' adjFactor'] = [float(i) for i in meta[' adjFactor']]
            X = pd.concat([X,meta],axis=1)

        # get outputs
        # t = [x for x in self.targets if x is not 'cluster']
        y = data[self.targets]
        
        # clean 
        X = self.clean(X)
        y = y.loc[X.index,:]   
        
        # scale/transform X
        Xlog = np.where(X>0,np.log(X),X)
        Xt, self.Xscaler = self.standardScaler(Xlog)
        Xt = pd.DataFrame(Xt,columns=X.columns)
        
        # scale/transform y
        y['cluster'] = y['cluster'] + 1
        y = y + .001
        y['cluster'] = round(y['cluster'])
        #ylog = np.where(y>0,np.log(y),y)
        ylog = np.log(y)
        yt, self.yscaler = self.standardScaler(ylog)
        y2 = pd.DataFrame(yt,columns=y.columns)
        
        # PCA for X
        if self.Xpca:
            # requires transform
            Xt, self.Xcomp, self.Xvar = self.nPCA(Xt.values, int(self.Xpca))
            Xt = pd.DataFrame(Xt)
        self.n_in = Xt.shape[1]
        
        self.n_out = y2.shape[1]
        self.vars = y2.columns.values
        
        return Xt, y2, y
                
    def build(self):
    
        # if self.n_in in [10,20]:
        #     nhid = 10
        # else:
        #     nhid = 60
        
        self.model = Sequential(
                [Dense(500, kernel_initializer='normal', input_shape=(self.n_in,)), ReLU(),
                 Dense(200, kernel_initializer='normal'), ReLU(),
                 Dense(50, kernel_initializer='normal'), ReLU(),
                 Dense(self.n_out)
                 ])
        # compile
        self.model.compile(Adam(lr=self.lrate),loss='mean_absolute_error')
        print (self.model.summary())
        return self.model

    def prep_results(self,y):
        results = {}
        for var in y.columns:
            if var in ['cluster']:
                continue
            results[var] = {'cv' : {'ytest': [],
                                    'yhat': [],
                                    'R2': [],
                                    'RMSE': [],
                                    'RMSELE': [],
                                    'Bias': [],
                                    'MAPE': [],
                                    'rRMSE': []},
                            'final' : {'ytest': [],
                                       'yhat': [],
                                       'R2': [],
                                       'RMSE': [],
                                       'RMSELE': [],
                                       'Bias': [],
                                       'MAPE': [],
                                       'rRMSE': []},
                            'owt' : {'Mild' : [],
                                     'NAP' : [],
                                     'CDOM' : [],
                                     'Euk' : [],
                                     'Cy' : [],
                                     'Scum' : [],
                                     'Oligo': []}
                            }
            
            clusters = ['Mild','NAP','CDOM','Euk','Cy','Scum','Oligo']
            for k in clusters:
                results[var]['owt'][k] = {'ytest': [],
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

    def evaluate(self,y_hat,y_test,results,q):
        import scorers as sc
        scoreDict = {'R2': sc.r2,
                     'RMSE': sc.log_rmse,
                     'RMSELE': sc.log_rmsele,
                     'Bias': sc.log_bias,
                     'MAPE': sc.log_mape,
                     'rRMSE': sc.log_rrmse,}
        
        # revert back to un-transformed data
        y_hat = self.transform_inverse(y_hat)
        y_test = self.transform_inverse(y_test)
        y_hat = pd.DataFrame(y_hat, columns=self.vars)
        #y_test = np.exp(y_test)
        #clus = y_test[:,-1:].astype(int)
        #y_test = np.where(y_test[:,:-1] == 1, 0, y_test[:,:-1])
        #y_test = np.hstack((y_test,clus))
        y_test = pd.DataFrame(y_test, columns=self.vars)
        # clus = np.exp(y_test.cluster)
        # y_test['cluster'] = clus
    
        for band in self.vars:
            if band in ['cluster']:
                continue

            y_t = y_test.loc[:,band].astype(float)
            y_h = y_hat.loc[:,band].astype(float)
            
            # for PC = 0 instances
            # true = np.logical_and(y_h > 0, y_t > 0)
            # true = y_t > 0
            # y_t = y_t[true]
            # y_h = y_h[true]

            for stat in scoreDict:
                results[band][q][stat].append(scoreDict[stat](y_t,y_h))
            
            results[band][q]['ytest'].append(y_t)
            results[band][q]['yhat'].append(y_h)
            results['pred_time'].append(self.pred_time)
            results['fit_time'].append(self.fit_time)            
        return results
    
    def owt_evaluate(self,y_hat,y_test,results):
        import scorers as sc
        scoreDict = {'R2': sc.r2,
                     'RMSE': sc.log_rmse,
                     'RMSELE': sc.log_rmsele,
                     'Bias': sc.log_bias,
                     'MAPE': sc.log_mape,
                     'rRMSE': sc.log_rrmse,}   
        
        # revert back to un-transformed data, add cluster column
        y_hat = self.transform_inverse(y_hat)
        y_test = self.transform_inverse(y_test)
        y_hat = pd.DataFrame(y_hat, columns=self.vars)
        #y_test = np.exp(y_test)
        #clus = y_test[:,-1:].astype(int)
        #y_test = np.where(y_test[:,:-1] == 1, 0, y_test[:,:-1])
        #y_test = np.hstack((y_test,clus))
        y_test = pd.DataFrame(y_test, columns=self.vars)
        clus = np.exp(y_test.cluster).astype(int)
        y_test['cluster'] = clus
        
        # change cluster values to OWT
        clus = clus.replace(to_replace=[2,5,11],value='Mild')
        clus = clus.replace(to_replace=[1,12],value='NAP')
        clus = clus.replace(to_replace=[8,13],value='CDOM')
        clus = clus.replace(to_replace=[7],value='Euk')
        clus = clus.replace(to_replace=[6,9],value='Cy')
        clus = clus.replace(to_replace=[3],value='Scum')
        clus = clus.replace(to_replace=[4,10],value='Oligo')
        y_hat['cluster'] = clus
        y_test['cluster'] = clus
        
        # evaluate by cluster
        grouped = y_test.groupby('cluster')
        for c, testgroup in grouped:
            hatgroup = y_hat.loc[testgroup.index]
            
            for band in self.vars:
                if band in ['cluster']:
                    continue
    
                y_t = testgroup.loc[:,band].astype(float)
                y_h = hatgroup.loc[:,band].astype(float)
                
                # for PC = 0 instances
                # true = np.logical_and(y_h > 0, y_t > 0)
                # true = y_t > 0
                # y_t = y_t[true]
                # y_h = y_h[true]
    
                for stat in scoreDict:
                    results[band]['owt'][c][stat].append(scoreDict[stat](y_t,y_h))
                
                results[band]['owt'][c]['ytest'].append(y_t)
                results[band]['owt'][c]['yhat'].append(y_h)
        return results
    
               
            
        
    
    
    
    
    