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
from MLP_cross_val_atcor import reg_cross_val
from sklearn.model_selection import train_test_split

def clean(data):   
    # data.fillna(0,inplace=True)
    # data.replace(np.inf,0)
    data = data.replace([np.inf, -np.inf], np.nan, inplace=False)    
    data = data.dropna(axis=0,how='any',inplace=False)
    return data

def transform(data):
    from sklearn.preprocessing import StandardScaler    
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, scaler

# data
rrsData = pickle.load( open( "/content/drive/My Drive/nasa_npp/RT/sensorIDX_rrs.p", "rb" ) )
refData = pickle.load( open( "/content/drive/My Drive/nasa_npp/RT/sensorIDX_ref.p", "rb" ) )

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
XmetaT, Xscaler = transform(Xmeta)
XmetaT = pd.DataFrame(XmetaT,columns=Xmeta.columns)
XmetaTClean = clean(XmetaT)

# targets
yraw = rrsData[sensor].filter(regex='^[0-9]')
y = pd.DataFrame(np.repeat(yraw.values,4,axis=0))
y.columns = yraw.columns
yT, yscaler = transform(y)
yT = pd.DataFrame(yT, columns=y.columns)
yTClean = yT.loc[XmetaTClean.index,:]

#%% Spline transform 

# kp = [410,420,430,440,460,480,500,515,530,545,560,575,590,600,610,
#       620,630,640,650,665,680,695,710,725,740,760,780,800,850]
# l = X.columns.values.astype(float)

# def bspline(row):
        
#     # smooth using bspline
#     fit = LSQUnivariateSpline(l, row.values, kp) # spline fit
#     kn = fit.get_knots() # knots
#     kn = 3*[kn[0]] + list(kn) + 3*[kn[-1]] 
#     c = fit.get_coeffs()
#     return c

# yc = y.apply(bspline,axis=1)
# yc = pd.DataFrame(yc.values.tolist(), index=yc.index)

#%% PCA transform
def nPCA(data,n):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n)
    pca.fit(data)
    Xpca = pca.transform(data)
    comp = pca.components_
    var = pca.explained_variance_ratio_
    return Xpca, comp, var

def nPCA_revert(data,comp,scaler):
    XhatT = np.dot(data,comp)
    Xhat = scaler.inverse_transform(XhatT)
    return Xhat

# X
Xpca10, Xcomp10, Xvar10 = nPCA(XmetaTClean,10)
Xpca20, Xcomp20, Xvar20 = nPCA(XmetaTClean,20)
# y
ypca10, ycomp10, yvar10 = nPCA(yTClean,10)
ypca20, ycomp20, yvar20 = nPCA(yTClean,20)

#%%
runs = {}
runs['features'] = {'XmetaT': XmetaTClean.values,
                    'XmetaTpca10': Xpca10,
                    'XmetaTpca20': Xpca20}
runs['targets'] = {'yT': yTClean.values,
                   'yTpca10': ypca10,
                   'ytpca20': ypca20}

#%%
#from atm_cor.MLPregressor_atcor import MLPregressor

ti = {'epochs':25,
      'batch_size':256,
      'lrate':1e-4,
      'split':.2,
      'n_mixes':5,
      'layers':[512,256,128,64],
      'cv':2,
      }

batch = len(runs['features'].keys()) * len(runs['targets'].keys())
count = 0
for k in runs['features']:
    for t in runs['targets']:
        
        name = '_'.join([k,t])
        #name = f + '_' + t
        print ('\n##### {} #####\n##### C:{}/{} #####\n'.format(name,count,batch))
        
        X = runs['features'][k]
        y = runs['targets'][t]
        
        out = reg_cross_val(X, y, ti=ti, scores='regScore')

        count = count+1

        # save run to disk
        fname = '/content/drive/My Drive/atm_cor_results_v1/{}.p'.format(name)
        f = open(fname,'wb')
        pickle.dump(out,f)
        f.close() 

#%%
# X_train, X_test, y_train, y_test = train_test_split(Xpca20, ypca20, test_size=0.2, random_state=42)
# n_in = X_train.shape[1]
# n_out = y_train.shape[1]
# model = MLPregressor(n_in,n_out,ti['epochs'],ti['batch_size'],
#                              ti['lrate'],ti['split'])
# model.build(ti['layers'])
# history = model.fit(X_train,y_train)
# #out = reg_cross_val(Xmeta, yc, ti=train_info, scores='regScore')