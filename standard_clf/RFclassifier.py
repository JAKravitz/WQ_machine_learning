#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 17:29:42 2021

@author: jakravit
"""
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout, Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import roc_curve,  auc, f1_score
import timeit
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import standard.scorers as sc
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV


class RFclassifier(BaseEstimator):
    
    def __init__(self, batch_info):
        self.targets = batch_info['targets']
        self.n_estimators = batch_info['n_estimators']
        self.min_samples_split = batch_info['min_samples_split']
        self.max_depth = batch_info['max_depth']
        self.max_features = batch_info['max_features']
        self.n_jobs = batch_info['n_jobs']
        self.cv = batch_info['cv']
        self.class_weight = batch_info['class_weight']
        self.verbose = batch_info['verbose']
    
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
    
    # def nPCA(self,data,n):
    #     from sklearn.decomposition import PCA
    #     pca = PCA(n_components=n)
    #     pca.fit(data)
    #     npca = pca.transform(data)
    #     comp = pca.components_
    #     var = pca.explained_variance_ratio_
    #     return npca, comp, var
    
    # def nPCA_revert(self,data):
    #     revert = np.dot(data,self.ycomp)
    #     return revert
    
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
        lab = preprocessing.LabelEncoder()
        yt = lab.fit_transform(y)
        yt = pd.DataFrame(yt,index=y.index,columns=['admix'])
        
        # scale/transform y
        # yt, self.yscaler = self.standardScaler(y)
        # y2 = pd.DataFrame(yt, columns=y.columns)
        
        # scale/transform X
        X = X.loc[y.index,:]
        Xlog = np.where(X>0,np.log(X),X)
        Xt, self.Xscaler = self.standardScaler(Xlog)
        Xt = pd.DataFrame(Xt,columns=X.columns)
        
        # PCA for X
        # if self.Xpca:
        #     # requires transform
        #     Xt, self.Xcomp, self.Xvar = self.nPCA(Xt.values, int(self.Xpca))
        #     Xt = pd.DataFrame(Xt)
        # self.n_in = Xt.shape[1]
        
        self.n_out = yt.shape[1]
        self.vars = y.columns.values
        
        return Xt, yt
    
    
    def build(self):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, 
                                            min_samples_split=self.min_samples_split,
                                            max_depth=self.max_depth,
                                            max_features=self.max_features,
                                            n_jobs=self.n_jobs,
                                            class_weight=self.class_weight,
                                            verbose=self.verbose)
        # print (self.model.summary())
        return self.model


    def prep_results(self, y):
        results = {}
        for var in y.columns:
            if var == 'cluster':
                continue
            results[var] = {'cv': {'ytest': [],
                                   'yhat': [],
                                   'yproba': [],
                                   'f1': [],
                                   'fpr': [],
                                   'tpr': [],
                                   'Acc': [],
                                   'AUC': []},
                            'final': {'ytest': [],
                                      'yhat': [],
                                      'yproba': [],
                                      'f1': [],
                                      'fpr': [],
                                      'tpr': [],
                                      'Acc': [],
                                      'AUC': []}
                            }
            results['fit_time'] = []
            results['pred_time'] = []
            results['train_loss'] = []
            results['val_loss'] = []
        return results
    
    def gridsearch(self, X_train, y_train):
        rf = RandomForestClassifier()
        rf_params = {'n_estimators': [50,100,300],
                     'max_depth': [3,5,7],
                     'max_features': [2,4,6,8],
                     'min_samples_split': [2,4,6]
                     }
        rf_cv_model = GridSearchCV(rf, rf_params, cv=5, n_jobs=-1,
                                   verbose=2,).fit(X_train, y_train)
        return rf_cv_model.best_params_

    def fit(self, X_train, y_train):
        tic=timeit.default_timer()
        history = self.model.fit(X_train, y_train)
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        toc=timeit.default_timer()
        self.fit_time = toc-tic
        return importances, std

    def predict(self, X_test):
        tic = timeit.default_timer()
        y_hat = self.model.predict(X_test.values)
        y_proba = self.model.predict_proba(X_test.values)
        y_proba = y_proba[:,1]
        toc = timeit.default_timer()
        self.pred_time = toc-tic 
        return y_hat, y_proba
    

    def evaluate(self, y_test, y_hat, y_proba, results, q):
        dx = y_test.index.values        
        y_hat = pd.DataFrame(y_hat, columns=self.vars, index=dx)
        # y_test = pd.DataFrame(y_test, columns=self.vars)
        y_proba = pd.DataFrame(y_proba, columns=self.vars)
        
        for band in self.vars:
            if band in ['cluster']:
                continue

            y_t = y_test.loc[:,band]#.astype(float)
            y_h = y_hat.loc[:,band]#.astype(float)
            y_p = y_proba.loc[:,band]
            
            fpr, tpr, thresh = roc_curve(y_t, y_p, pos_label=1)
            AUC = auc(fpr,tpr)
            ACC = np.mean(y_t == y_h)
            f1 = f1_score(y_t, y_h, average='macro')
            
            results[band][q]['f1'].append(f1)
            results[band][q]['Acc'].append(ACC)
            results[band][q]['AUC'].append(AUC)
            results[band][q]['tpr'].append(tpr)
            results[band][q]['fpr'].append(fpr)   
            results[band][q]['ytest'].append(y_t)
            results[band][q]['yhat'].append(y_h)            
            results[band][q]['yproba'].append(y_p)            
            results['pred_time'].append(self.pred_time)
            results['fit_time'].append(self.fit_time) 
            
        return results, y_t, y_h
    
    
    
    
    