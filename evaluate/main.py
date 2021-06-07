#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:22:28 2021

@author: jkravz311
"""
#%%
import pickle
import pandas as pd
import numpy as np
# from classifier_eval import classifier
import warnings
warnings.filterwarnings("ignore")
from MLP_cross_val import reg_cross_val
import features as feats
import matplotlib.pyplot as plt

# data
rrsData = pickle.load( open( "/content/drive/My Drive/sensorIDX_rrs.p", "rb" ) )
#rrsData = pickle.load( open( "/Users/jakravit/Desktop/npp_projects/ML/sensorIDX_rrs.p", "rb" ) )

train_info = {'epochs':10,
              'batch_size':512,
              'lrate':1e-4,
              'split':.2,
              'n_mixes':5,
              'layers':[512,128,64,32],
              'cv':3,
              }

# get features, outputs
atcor = 'rrs'
sensor = 'hico'
targets = ['chl','PC','cnap','cdom','admix','aphy440','ag440','anap440','bbphy440','bbnap440']
X,y_tot = feats.getXY(rrsData,sensor,targets)
targets = ['aphy440','ag440','anap440']
X,y_a = feats.getXY(rrsData,sensor,targets)
targets = ['aphy440']
X,y_phy = feats.getXY(rrsData,sensor,targets)
targets = ['ag440']
X,y_ag = feats.getXY(rrsData,sensor,targets)
targets = ['anap440']
X,y_nap = feats.getXY(rrsData,sensor,targets)

#%%
runs = {}
runs['features'] = {'X': X,
                    'minmax': feats.minMaxScale(X, (X.min()).min(), (X.max()).max()),
                    'standard': feats.standardScale(X),
                    'maxabs': feats.maxAbs(X),
                    'robust': feats.robust(X),
                    'quantile': feats.quantile(X),
                    'log': np.log(X),
                    'power': feats.power(X),
                    'norm': feats.l2norm(X)}
runs['targets'] = {'tot': y_tot,
                   'a': y_a,
                   'phy': y_phy,
                   'ag': y_ag,
                   'nap': y_nap,
                   'tot_log': np.log(y_tot),
                   'a_log': np.log(y_a),
                   'phy_log': np.log(y_phy),
                   'ag_log': np.log(y_ag),
                   'nap_log': np.log(y_nap)}

#%% 
results = {}
for f in runs['features']:
    for t in runs['targets']:
        
        X = runs['features'][f]
        y = runs['targets'][t]
        out = reg_cross_val(X, y, train_info, 'regLogScore')
        name = '_'.join([f,t])
        results[name] = out

# save run to disk
fname = '/content/drive/My Drive/test_results.p'
f = open(fname,'wb')
pickle.dump(results,f)
f.close() 
