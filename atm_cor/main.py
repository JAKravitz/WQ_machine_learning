#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:18:08 2021

@author: jakravit
"""
import pickle
import pandas as pd
import numpy as np
from math import sqrt, cos
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, LSQUnivariateSpline, UnivariateSpline
import warnings
warnings.filterwarnings("ignore")
from atm_cor.MLP_cross_val_atcor import reg_cross_val
from sklearn.model_selection import train_test_split

# data
rrsData = pickle.load( open( "/Users/jakravit/Desktop/nasa_npp/RT/sensorIDX_rrs.p", "rb" ) )
refData = pickle.load( open( "/Users/jakravit/Desktop/nasa_npp/RT/sensorIDX_ref.p", "rb" ) )

# features
sensor = 'hico'
Xraw = refData[sensor]
X = Xraw.filter(regex='^[0-9]')
# meta
metacols = [' SZA',' OZA',' SAA',' OAA',' aot550',' astmx',' ssa400',
            ' ssa675',' ssa875',' altitude',' adjFactor']
meta = Xraw.loc[:,metacols]
meta[' adjFactor'] = meta[' adjFactor'].replace(to_replace=(' '),value=0)
meta[' adjFactor'] = [float(i) for i in meta[' adjFactor']]
Xmeta = pd.concat([X,meta],axis=1)
def transform(data):
    from sklearn.preprocessing import StandardScaler    
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data
Xmeta_T = pd.DataFrame(transform(Xmeta),columns=Xmeta.columns)

# targets
yraw = rrsData[sensor].filter(regex='^[0-9]')
y = pd.DataFrame(np.repeat(yraw.values,4,axis=0))
y.columns = yraw.columns

#%%

kp = [410,420,430,440,460,480,500,515,530,545,560,575,590,600,610,
      620,630,640,650,665,680,695,710,725,740,760,780,800,850]
l = X.columns.values.astype(float)

def bspline(row):
        
    # smooth using bspline
    fit = LSQUnivariateSpline(l, row.values, kp) # spline fit
    kn = fit.get_knots() # knots
    kn = 3*[kn[0]] + list(kn) + 3*[kn[-1]] 
    c = fit.get_coeffs()
    return c

yc = y.apply(bspline,axis=1)
yc = pd.DataFrame(yc.values.tolist(), index=yc.index)

#%%
from atm_cor.MLPregressor_atcor import MLPregressor

ti = {'epochs':100,
      'batch_size':64,
      'lrate':1e-4,
      'split':.2,
      'n_mixes':5,
      'layers':[500,128,64,33],
      'cv':3,
      }

X_train, X_test, y_train, y_test = train_test_split(Xmeta_T, yc, test_size=0.2, random_state=42)
n_in = X_train.shape[1]
n_out = y_train.shape[1]
model = MLPregressor(n_in,n_out,ti['epochs'],ti['batch_size'],
                             ti['lrate'],ti['split'])
model.build(ti['layers'])
history = model.fit(X_train,y_train)
#out = reg_cross_val(Xmeta, yc, ti=train_info, scores='regScore')