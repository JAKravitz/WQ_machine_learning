#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance metrics for regression analysis
Metrics for non log scale and log scale data

@author: jkravitz, May 2021
"""
import statsmodels.tools.eval_measures as em
import numpy as np
import sklearn.metrics as metrics

def r2(y_tst,y_hat,multioutput='uniform_average'):
    r2 = metrics.r2_score(y_tst, y_hat, multioutput=multioutput)
    return r2

def rmse(y_tst,y_hat):
    rmse = em.rmse(y_tst, y_hat, axis=0)
    return rmse

def log_rmse(y_tst,y_hat):
    rmse = em.rmse(np.exp(y_tst), np.exp(y_hat), axis=0)
    return rmse

def rmsele(y_tst,y_hat):
    x = np.log(y_tst)
    y = np.log(y_hat)
    rmsele = em.rmse(x,y,axis=0)
    return rmsele

def log_rmsele(y_tst,y_hat):
    rmsele = em.rmse(y_tst,y_hat,axis=0)
    return rmsele

def rrmse(y_tst,y_hat):
    try:
        y_tst = y_tst.values
    except:
        pass
    try:
        y_hat = y_hat.values
    except:
        pass
    y_tst = y_tst
    y_hat = y_hat
    rrmse = em.rmse(y_tst,y_hat,axis=0) / y_tst.mean() * 100
    return rrmse

def log_rrmse(y_tst,y_hat):
    try:
        y_tst = y_tst.values
    except:
        pass
    try:
        y_hat = y_hat.values
    except:
        pass
    y_tst = np.exp(y_tst)
    y_hat = np.exp(y_hat)
    rrmse = em.rmse(y_tst,y_hat,axis=0) / y_tst.mean() * 100
    return rrmse

def bias(y_tst,y_hat):
    try:
        y_tst = y_tst.values
    except:
        pass
    try:
        y_hat = y_hat.values
    except:
        pass
    y_tst = y_tst
    y_hat = y_hat
    y_tst = np.log(y_tst)
    y_hat = np.log(y_hat)
    subs = np.subtract(y_hat,y_tst)
    bias = np.sum(subs,axis=0) / len(y_tst)
    return bias

def log_bias(y_tst,y_hat):
    try:
        y_tst = y_tst.values
    except:
        pass
    try:
        y_hat = y_hat.values
    except:
        pass
    y_tst = y_tst
    y_hat = y_hat
    subs = np.subtract(y_hat,y_tst)
    bias = np.sum(subs,axis=0) / len(y_tst)
    return bias

def mape(y_tst,y_hat):
    try:
        y_tst = y_tst.values
    except:
        pass
    try:
        y_hat = y_hat.values
    except:
        pass
    y_tst = y_tst
    y_hat = y_hat
    subs = np.abs(np.subtract(y_hat,y_tst))
    mae = np.divide(subs, y_tst)
    mape = 100 * np.median(mae,axis=0)
    return mape 

def log_mape(y_tst,y_hat):
    try:
        y_tst = y_tst.values
    except:
        pass
    try:
        y_hat = y_hat.values
    except:
        pass
    y_tst = np.exp(y_tst)
    y_hat = np.exp(y_hat)
    subs = np.abs(np.subtract(y_hat,y_tst))
    mae = np.divide(subs, y_tst)
    mape = 100 * np.median(mae,axis=0)
    return mape 




