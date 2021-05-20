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
import mdn
from keras.wrappers.scikit_learn import KerasRegressor
import timeit
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from tensorflow_probability import distributions as tfd
from scipy.stats import norm

class MDNregressor(BaseEstimator):
    
    def __init__(self,n_in,n_out,epochs,batch_size,lrate,split,n_mixes):
        self.n_in = n_in
        self.n_out = n_out
        self.epochs = epochs
        self.batch_size = batch_size
        self.lrate = lrate
        self.split = split
        self.n_mixes = n_mixes
        
    def build(self, layers,):
        self.model = Sequential()
        self.model.add(Dense(layers[0],input_shape=(self.n_in,),activation='relu'))
        #self.model.add(Dropout(0.1))
        self.model.add(Dense(layers[1],activation='relu',))
        #self.model.add(Dropout(0.1))
        self.model.add(Dense(layers[2],activation='relu',))
        #self.model.add(Dropout(0.1))
        self.model.add(Dense(layers[3],activation='relu',))
        #self.model.add(Dropout(0.1))
        self.model.add(mdn.MDN(self.n_out, self.n_mixes))
        # compile
        self.model.compile(loss=mdn.get_mixture_loss_func(self.n_out,self.n_mixes),optimizer=Adam(lr=self.lrate))
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
    
    def predict(self, Xt, yt, ids, results):
        tic = timeit.default_timer()
        y_dist =  pd.DataFrame(self.model.predict(Xt),index=yt.index.values)
        toc = timeit.default_timer()
        pred_time = toc-tic 
        y_samples = np.apply_along_axis(mdn.sample_from_output, 1, y_dist, self.n_out, self.n_mixes,temp=1.0)
        y_hat = pd.DataFrame(y_samples[:,0,:],index=yt.index.values)
        
        # split mixture params
        mus, sigs, pi_logits = mdn.split_mixture_params(y_dist, self.n_out, self.n_mixes)
        pis = mdn.softmax(pi_logits, t=1)
        muc = [list(t) for t in zip(*[iter(mus)]*self.n_mixes)]
        sic = [list(t) for t in zip(*[iter(sigs)]*self.n_mixes)] 
        
        # store results
        for k in y_hat.columns:
            results[ids[k]]['ytest'].append(yt.iloc[:,k])
            results[ids[k]]['yhat'].append(y_hat.iloc[:,k])
            results[ids[k]]['alphas'].append(pis)
            results[ids[k]]['mus'].append(muc[k])
            results[ids[k]]['sigs'].append(sic[k])
            results['pred_time'].append(pred_time)
            results['fit_time'].append(self.fit_time)
        
        return y_hat, results

    def evaluate(self,y_test,y_hat,results,scoreDict,log=False):
        for i,var in enumerate(y_test.columns):
            if var in ['adj','cluster']:
                continue
            y_t = y_test.iloc[:,i].astype(float)
            y_h = y_hat.iloc[:,i].astype(float)
            if log == False:
                true = np.logical_and(y_h > 0, y_t > 0)
                y_tst = y_t[true]
                y_ht = y_h[true]
            else:
                y_tst = y_t
                y_ht = y_h

            for stat in scoreDict:
                results[var][stat].append(scoreDict[stat](y_tst,y_ht))
        return results

    # def cred_interval(self, y_dist, y_test, ids):
        
    #     creds = {'chl': np.zeros(len(y_test)),
    #              'PC': np.zeros(len(y_test)),
    #              'cnap': np.zeros(len(y_test)),
    #              'cdom': np.zeros(len(y_test)),
    #              'admix': np.zeros(len(y_test)),
    #              'aphy440': np.zeros(len(y_test)),
    #              'ag440': np.zeros(len(y_test)),
    #              'anap440': np.zeros(len(y_test)),
    #              'bbphy440': np.zeros(len(y_test)),
    #              'bbnap440': np.zeros(len(y_test)),}
        
    #     for i in range(len(y_test)):
    #         mus, sigs, pi_logits = mdn.split_mixture_params(y_dist.iloc[i,:], self.n_out, self.n_mixes)
    #         pis = mdn.softmax(pi_logits, t=1)
    #         muc = [list(t) for t in zip(*[iter(mus)]*self.n_mixes)]
    #         sic = [list(t) for t in zip(*[iter(sigs)]*self.n_mixes)]
    #         for k in range(y_test.shape[1]):
    #             gm = tfd.MixtureSameFamily(
    #                     mixture_distribution=tfd.Categorical(probs=list(pis)),
    #                     components_distribution=tfd.Normal(
    #                         loc=muc[k],       
    #                         scale=sic[k]))
    #             conf90 = norm.interval(0.9, loc=gm.mean().numpy(), scale=gm.stddev().numpy())
    #             #conf60 = norm.interval(0.60, loc=gm.mean().numpy(), scale=gm.stddev().numpy())
    #             yvar = y_test.iloc[i,k]
    #             if conf90[0] <= yvar and conf90[1] > yvar:
    #                 final[ids[k]][i] = 1        
    #     return final
    
    
        
        
        
        
        