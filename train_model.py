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
import helpers as hp
from getXY import getXY
# from classifier_eval import classifier
import scorers as sc
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from MLP_cross_val import reg_cross_val

# data
rrsData = pickle.load( open( "/content/drive/My Drive/sensorIDX_rrs.p", "rb" ) )
#rrsData = pickle.load( open( "/content/drive/MyDrive/sensorIDX_rrs.p", "rb" ) )
#rrsData = pickle.load( open( "/Users/jakravit/Desktop/npp_projects/sensorIDX_rrs.p", "rb" ) )

# batch setup
info = {'sensorID' : {'s2_60m':['443','490','560','665','705','740','783','842','865'],
                      's2_20m':['490','560','665','705','740','783','842','865'],
                      's2_10m':['490','560','665','842'],
                      's3':'^Oa',
                      'l8':['Aer','Blu','Grn','Red','NIR'],
                      'modis':'^RSR',
                      'meris':'^b',
                      'hico':'^H'},       
        'clfScore' : {'Accuracy': metrics.accuracy_score,
                      'ROC_AUC': metrics.roc_auc_score},        
        'regScore' : {'R2': sc.r2,
                      'RMSE': sc.rmse,
                      'RMSELE': sc.rmsele,
                      'Bias': sc.bias,
                      'MAPE': sc.mape,
                      'rRMSE': sc.rrmse,},
        'regLogScore' : {'R2': sc.r2,
                         'RMSE': sc.log_rmse,
                         'RMSELE': sc.log_rmsele,
                         'Bias': sc.log_bias,
                         'MAPE': sc.log_mape,
                         'rRMSE': sc.log_rrmse,}
        }


ids = {0:'chl',
       1:'PC',
       2:'cnap',
       3:'cdom',
       4:'admix',
       5:'aphy440',
       6:'ag440',
       7:'anap440',
       8:'bbphy440',
       9:'bbnap440'
       }

CI = {'chl': [],
      'PC': [],
      'cnap': [],
      'cdom': [],
      'admix': [],
      'aphy440': [],
      'ag440': [],
      'anap440': [],
      'bbphy440': [],
      'bbnap440': [],
      }

train_info = {'epochs':10,
              'batch_size':512,
              'lrate':1e-4,
              'split':.2,
              'n_mixes':5,
              'layers':[512,128,64,32],
              'cv':3,
              'ids':ids,
              'CI':CI
              }

# get features, outputs
atcor = 'rrs'
sensor = 's3'
targets = ['chl','PC','cnap','cdom','admix','aphy440','ag440','anap440','bbphy440','bbnap440']
X,y, scaler = getXY(atcor,rrsData,sensor,targets,info,xlog=True,ylog=True)

# results
final = reg_cross_val(X, y, train_info, info['regLogScore'])

# save run to disk
fname = '/content/drive/My Drive/test_results.p'
f = open(fname,'wb')
pickle.dump(final,f)
f.close() 


