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
import l8_clf.scorers as sc
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_curve, roc_auc_score
# from sklearn.metrics import auc
from yellowbrick.classifier import ROCAUC

class MLPclassifier(BaseEstimator):
    
    def __init__(self, batch_info):
        self.epochs = batch_info['epochs']
        self.batch_size = batch_info['batch_size']
        self.lrate = batch_info['lrate']
        self.split = batch_info['split']
        self.targets = batch_info['targets']
        # self.meta = batch_info['meta']
        # self.Xpca = batch_info['Xpca']
    
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

    def transform_inverse(self,data):
        inverse = self.yscaler.inverse_transform(data)
        return inverse

    def getXY(self,data):
        # get inputs
        X  = data.filter(regex='^[0-9]')
        
        # get outputs
        y = data[self.targets]
        
        # clean 
        X = self.clean(X)
        y = y.loc[X.index,:]   
        
        # scale/transform X
        Xlog = np.where(X>0,np.log(X),X)
        Xt, self.Xscaler = self.standardScaler(Xlog)
        Xt = pd.DataFrame(Xt,columns=X.columns)
        
        # y to categorical for encoding
        ycat = to_categorical(y)
        y2 = pd.DataFrame(ycat)
        

        self.n_in = Xt.shape[1]
        self.n_out = y2.shape[1]
        # self.vars = y2.columns.values
        
        return Xt, y2
    
    def build(self):
        self.model = Sequential(
                [Dense(500, kernel_initializer='normal', input_shape=(self.n_in,)), ReLU(),
                 Dense(200, kernel_initializer='normal'), ReLU(),
                 Dense(50, kernel_initializer='normal'), ReLU(),
                 Dense(self.n_out, kernel_initializer='normal', activation='softmax')
                 ])
        # compile
        self.model.compile(Adam(lr=self.lrate),loss='categorical_crossentropy', metrics='accuracy')
        print (self.model.summary())
        return self.model
    
    def prep_results(self, y):
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
        # y_hat = self.model.predict(X_test)
        y_proba = self.model.predict(X_test)
        y_hat = self.model.predict_classes(X_test)
        toc = timeit.default_timer()
        self.pred_time = toc-tic 
        return y_proba, y_hat

    def evaluate(self, y_test, y_proba, y_hat):
        y_test_class = np.argmax(y_test.values, axis=1)
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        # roc curve
        fpr = {}
        tpr = {}
        thresh ={}
        n_class = y_proba.shape[1]
        for i in range(n_class):    
            fpr[i], tpr[i], thresh[i] = roc_curve(y_hat, y_proba[:,i], pos_label=i)
        acc = np.mean(y_test_class == y_hat)
        results = {'AUC':auc,
                   'ACC':acc,
                   'fpr':fpr,
                   'tpr':tpr,
                   'thresh':thresh}
        return results
        
    # def evaluate(self, X_test, y_test, y_proba, results):
    #     for i,var in enumerate(y_test.columns):
    #         print (var)
    #         y_t = y_test.iloc[:,i]
    #         if y_hat is None:
    #             y_hat = (y_proba > 0.5).astype(int)
    #             try:
    #                 y_p = y_proba[:,i]
    #             except:
    #                 y_p = y_proba 
    #         elif y_proba is None:
    #             y_proba = model.predict_proba(X_test.values)[i]
    #             y_p = y_proba[:,1]
            
    #         try:
    #             y_h = y_hat[:,i]
    #         except:
    #             y_h = y_hat
                
    #         fpr, tpr, thresh = roc_curve(y_t,y_p)
    #         AUC = auc(fpr,tpr)
    #         ACC = np.mean(y_t == y_h)
    #         results[var]['Acc'].append(ACC)
    #         results[var]['AUC'].append(AUC)
    #         results[var]['tpr'].append(tpr)
    #         results[var]['fpr'].append(fpr)
    #     return results  
    
    # def evaluate(self,y_hat,y_test,results,q):
    #     scoreDict = {'R2': sc.r2,
    #                  'RMSE': sc.log_rmse,
    #                  'RMSELE': sc.log_rmsele,
    #                  'Bias': sc.log_bias,
    #                  'MAPE': sc.log_mape,
    #                  'rRMSE': sc.log_rrmse,}
        
    #     # revert back to un-transformed data
    #     y_hat = self.transform_inverse(y_hat)
    #     y_test = self.transform_inverse(y_test)
    #     y_hat = pd.DataFrame(np.exp(y_hat), columns=self.vars)
    #     y_test = pd.DataFrame(np.exp(y_test), columns=self.vars)
    
    #     for band in self.vars:
    #         y_t = y_test.loc[:,band].astype(float)
    #         y_h = y_hat.loc[:,band].astype(float)

    #         for stat in scoreDict:
    #             results[band][q][stat].append(scoreDict[stat](y_t,y_h))
            
    #         results[band][q]['ytest'].append(y_t)
    #         results[band][q]['yhat'].append(y_h)
    #         results['pred_time'].append(self.pred_time)
    #         results['fit_time'].append(self.fit_time)            
    #     return results
    
    