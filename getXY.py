#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:54:39 2021

@author: jkravz311
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean(data):   
    data.fillna(0,inplace=True)
    data.replace(np.inf,0)
    return data

def getXY(atcor,data,sensor,targets,info,xlog=True,ylog=True):
    
    # get sensor data
    if sensor in ['s2_20m','s2_10m']:
        X = data['s2'].filter(items=info['sensorID'][sensor])
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
    if atcor == 'ref':
        y['adj'] = data[' adjFactor'].replace(to_replace=(' '),value=0)
        y['adj'] = [float(i) for i in y['adj']]
    
    # clean and standardize
    X = clean(X)
    y = clean(y)
    
    if ylog == True:
        y = pd.DataFrame(np.where(y>0,np.log(y),y),index=y.index.values,columns=y.columns)
    
    if xlog == True:
        X = pd.DataFrame(np.where(X>0,np.log(X),X),index=X.index.values,columns=X.columns)
    
    # X = X.iloc[:,:-2]    
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    X = pd.DataFrame(X,index=y.index)
    
    return X, y, scaler    
       