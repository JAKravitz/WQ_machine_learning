#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 12:07:14 2021

@author: jakravit
"""
from MDNregressor import MDNregressor
from sklearn.model_selection import KFold


def reg_prep_results(y_train,ids):
    results = {}
    for var in ids:
        if ids[var] == 'cluster':
            continue
        results[ids[var]] = {'ytest': [],
                             'yhat': [],
                             'alphas': [],
                             'mus' : [],
                             'sigs' : [],
                             'R2': [],
                             'RMSE': [],
                             'RMSELE': [],
                             'Bias': [],
                             'MAPE': [],
                             'rRMSE': [],
                             'CI': []}
    results['fit_time'] = []
    results['pred_time'] = []
    results['history'] = []
    
    return results
        

def reg_cross_val(X,y,ti,scoreDict):
    
    kfold = KFold(n_splits=ti['cv'], shuffle=True)
    count = 0
    for train, test in kfold.split(X, y):
        
        print ('FOLD = {}...'.format(count))
        X_train, X_test = X.iloc[train,:], X.iloc[test,:]
        y_train, y_test = y.iloc[train,:], y.iloc[test,:]   
        
        n_in = X_train.shape[1]
        n_out = y_train.shape[1]
        
        results = reg_prep_results(y_train, ti['ids'])
        
        model = MDNregressor(n_in,n_out,ti['epochs'],ti['batch_size'],
                             ti['lrate'],ti['split'],ti['n_mixes'])
        model.build(ti['layers'])
        results['history'] = model.fit(X_train,y_train)
        y_hat, results = model.predict(X_test, y_test, ti['ids'], results)
        results = model.evaluate(y_test,y_hat,results,scoreDict,log=True) 
        count = count+1
    
    return results
        
        
        
        
        
            # mus, sigs, pi_logits = mdn.split_mixture_params(y_dist.iloc[i,:], self.n_out, self.n_mixes)
            # pis = mdn.softmax(pi_logits, t=1)
            # muc = [list(t) for t in zip(*[iter(mus)]*self.n_mixes)]
            # sic = [list(t) for t in zip(*[iter(sigs)]*self.n_mixes)]
            # for k in range(y_test.shape[1]):
            #     gm = tfd.MixtureSameFamily(
            #             mixture_distribution=tfd.Categorical(probs=list(pis)),
            #             components_distribution=tfd.Normal(
            #                 loc=muc[k],       
            #                 scale=sic[k]))
            #     conf90 = norm.interval(0.9, loc=gm.mean().numpy(), scale=gm.stddev().numpy())
            #     #conf60 = norm.interval(0.60, loc=gm.mean().numpy(), scale=gm.stddev().numpy())
            #     yvar = y_test.iloc[i,k]
            #     if conf90[0] <= yvar and conf90[1] > yvar:
            #         final[ids[k]][i] = 1           
