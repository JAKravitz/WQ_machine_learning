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
#import matplotlib.pyplot as plt

# data
rrsData = pickle.load( open( "/content/drive/My Drive/nasa_npp/RT/sensorIDX_rrs.p", "rb" ) )
#rrsData = pickle.load( open( "/Users/jakravit/Desktop/npp_projects/ML/sensorIDX_rrs.p", "rb" ) )

train_info = {'epochs':200,
              'batch_size':64,
              'lrate':1e-4,
              'split':.2,
              'n_mixes':5,
              'layers':[512,128,64,32],
              'cv':3,
              }

# get features, outputs
atcor = 'rrs'
sensor = 'hico'

targets = ['chl','PC','cnap','aphy440','ag440','anap440','bbphy440','bbnap440']
X,y_tot = feats.getXY(rrsData,sensor,targets)

targets = ['aphy440','ag440','anap440','anap440','bbphy440','bbnap440']
X,y_iop = feats.getXY(rrsData,sensor,targets)

targets = ['chl','PC','cnap']
X,y_conc = feats.getXY(rrsData,sensor,targets)

targets = ['aphy440']
X,y_aphy = feats.getXY(rrsData,sensor,targets)
targets = ['ag440']
X,y_ag = feats.getXY(rrsData,sensor,targets)
targets = ['anap440']
X,y_anap = feats.getXY(rrsData,sensor,targets)
targets = ['bbphy440']
X,y_bphy = feats.getXY(rrsData,sensor,targets)
targets = ['bbnap440']
X,y_bnap = feats.getXY(rrsData,sensor,targets)
targets = ['chl']
X,y_chl = feats.getXY(rrsData,sensor,targets)
targets = ['PC']
X,y_pc = feats.getXY(rrsData,sensor,targets)
targets = ['cnap']
X,y_cnap = feats.getXY(rrsData,sensor,targets)


#%%
runs = {}
runs['features'] = {'X': X,
                    'minmax': pd.DataFrame(feats.minMaxScale(X, (X.min()).min(), (X.max()).max())),
                    'standard': pd.DataFrame(feats.standardScale(X)),
                    'maxabs': pd.DataFrame(feats.maxAbs(X)),
                    'robust': pd.DataFrame(feats.robust(X)),
                    'quantile': pd.DataFrame(feats.quantile(X)),
                    'log': np.log(X),
                    'power': pd.DataFrame(feats.power(X)),
                    'norm': pd.DataFrame(feats.l2norm(X))}
runs['targets'] = {'tot': y_tot,
                   'iop': y_iop,
                   'conc':y_conc,
                   'aphy': y_aphy,
                   'bphy': y_bphy,
                   'ag': y_ag,
                   'anap': y_anap,
                   'bnap': y_bnap,
                   'tot_log': np.log(y_tot),
                   'iop_log': np.log(y_iop),
                   'conc_log': np.log(y_conc),
                   'aphy_log': np.log(y_aphy),
                   'bphy_log': np.log(y_bphy),
                   'ag_log': np.log(y_ag),
                   'anap_log': np.log(y_anap),
                   'bnap_log': np.log(y_bnap),
                   }

#%% 
logs = ['tot_log','a_log','phy_log','ag_log','nap_log']

batch = len(runs['features'].keys()) * len(runs['targets'].keys())
count = 0
for k in runs['features']:
    for t in runs['targets']:
        
        name = '_'.join([k,t])
        #name = f + '_' + t
        print ('\n##### {} #####\n##### C:{}/{} #####\n'.format(name,count,batch))
        
        X = runs['features'][k]
        y = runs['targets'][t]
        if t in logs:
            out = reg_cross_val(X, y, ti=train_info, scores='regLogScore')
        else:
            out = reg_cross_val(X, y, ti=train_info, scores='regScore')
        
        count = count+1

        # save run to disk
        fname = '/content/drive/My Drive/ML_results/{}.p'.format(name)
        f = open(fname,'wb')
        pickle.dump(out,f)
        f.close() 

#%%
#test = pickle.load( open( "/Users/jakravit/Desktop/test_results.p", "rb" ) )
# yh = test['minmax_ag']['ag440']['yhat'][0]
# yt = test['minmax_ag']['ag440']['ytest'][0]
