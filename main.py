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
import testing.scorers as sc
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings("ignore")
from testing.MLP_cross_val import reg_cross_val
import features as feats

# data
#rrsData = pickle.load( open( "/content/drive/My Drive/sensorIDX_rrs.p", "rb" ) )
#rrsData = pickle.load( open( "/content/drive/MyDrive/sensorIDX_rrs.p", "rb" ) )
rrsData = pickle.load( open( "/Users/jakravit/Desktop/npp_projects/sensorIDX_rrs.p", "rb" ) )

# batch setup
# info = {'sensorID' : {'s2_60m':['443','490','560','665','705','740','783','842','865'],
#                       's2_20m':['490','560','665','705','740','783','842','865'],
#                       's2_10m':['490','560','665','842'],
#                       's3':'^Oa',
#                       'l8':['Aer','Blu','Grn','Red','NIR'],
#                       'modis':'^RSR',
#                       'meris':'^b',
#                       'hico':'^H'},       
#         'clfScore' : {'Accuracy': metrics.accuracy_score,
#                       'ROC_AUC': metrics.roc_auc_score},        
#         'regScore' : {'R2': sc.r2,
#                       'RMSE': sc.rmse,
#                       'RMSELE': sc.rmsele,
#                       'Bias': sc.bias,
#                       'MAPE': sc.mape,
#                       'rRMSE': sc.rrmse,},
#         'regLogScore' : {'R2': sc.r2,
#                          'RMSE': sc.log_rmse,
#                          'RMSELE': sc.log_rmsele,
#                          'Bias': sc.log_bias,
#                          'MAPE': sc.log_mape,
#                          'rRMSE': sc.log_rrmse,}
#         }

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
X,y = feats.getXY(rrsData,sensor,targets)

#### feature interactions ####
# polynomial features
X_poly2 = feats.polyFeatures(X, 2)

### Scaling ###
# minmax scaling
X_minmax = feats.minMaxScale(X, (X.min()).min(), (X.max()).max())
# standard scaler
X_standard = feats.standardScale(X)
# maxAbs scaler
X_maxabs = feats.maxAbs(X)
# robust scaler
X_robust = feats.robust(X)
# quantile transformer  scaler 
X_quantile = feats.quantile(X)
# log transform
X_log = np.log(X)
# power transform
X_power = feats.power(X)
# l2 normalization
X_norm = feats.l2norm(X)


# # results
# final = reg_cross_val(X, y, train_info, info['regLogScore'])

# # save run to disk
# fname = '/content/drive/My Drive/test_results.p'
# f = open(fname,'wb')
# pickle.dump(final,f)
# f.close() 



