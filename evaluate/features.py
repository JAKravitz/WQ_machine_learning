#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:55:55 2021

@author: jakravit
"""
#%%
import numpy as np
import pandas as pd


def clean(data):   
    data.fillna(0,inplace=True)
    data.replace(np.inf,0)
    return data

def getXY(data,sensor,targets):
    # get sensor data
    
    sensors = {'s2_60m':['443','490','560','665','705','740','783','842','865'],
               's2_20m':['490','560','665','705','740','783','842','865'],
               's2_10m':['490','560','665','842'],
               's3':'^Oa',
               'l8':['Aer','Blu','Grn','Red','NIR'],
               'modis':'^RSR',
               'meris':'^b',
               'hico':'^H'} 
        
    if sensor in ['s2_20m','s2_10m']:
        X = data['s2'].filter(items=sensors[sensor])
        data = data['s2']
    elif sensor == 's2_60m':
        X = data['s2'].filter(regex='^[0-9]')
        data = data['s2']
    else:
        X = data[sensor].filter(regex='^[0-9]')
        data = data[sensor]
    
    # drop o2 bands if s3
    if sensor == 's3':
        X.drop(['761.25','764.375','767.75'], axis=1, inplace=True)
    
    # drop if modis
    if sensor == 'modis':
        X.drop(['551'],axis=1,inplace=True)
    
    # get outputs
    y = data[targets]
    
    # clean 
    X = clean(X)
    y = clean(y)   
    y2 = y + .0001
    
    return X, y2

def polyFeatures(data,degree,interaction=False):
    from sklearn.preprocessing import PolynomialFeatures
    # polynomial features transform of dataset
    # input is degree of polynomial transform (generally 2 or 3)
    trans = PolynomialFeatures(degree=degree, interaction_only=interaction)
    data = trans.fit_transform(data)
    return data


def minMaxScale(data,r1,r2):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(r1,r2))
    data = scaler.fit_transform(data)
    return data

def standardScale(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data

def maxAbs(data):
    from sklearn.preprocessing import MaxAbsScaler
    scaler = MaxAbsScaler()
    data = scaler.fit_transform(data)
    return data

def robust(data):
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    data = scaler.fit_transform(data)
    return data

def quantile(data):
    from sklearn.preprocessing import QuantileTransformer
    scaler = QuantileTransformer()
    data = scaler.fit_transform(data)
    return data

def power(data):
    from sklearn.preprocessing import PowerTransformer
    scaler = PowerTransformer(method = 'box-cox')
    data = scaler.fit_transform(data)
    return data

def l2norm(data):
    from sklearn.preprocessing import Normalizer
    scaler = Normalizer(norm = 'l2')
    data = scaler.fit_transform(data)
    return data





