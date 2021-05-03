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
from regressor import regressor
# from classifier_eval import classifier
import scorers as sc
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings("ignore")

# data
rrsData = pickle.load( open( "/Users/jkravz311/GoogleDrive/sensorIDX_rrs.p", "rb" ) )
refData = pickle.load( open( "/Users/jkravz311/GoogleDrive/sensorIDX_ref.p", "rb" ) )

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

# get features, outputs
atcor = 'rrs'
sensor = 'hico'
targets = ['chl','PC','cnap','cdom','admix','aphy440','ag440','anap440','bbphy440','bbnap440','cluster']
X,y, scaler = getXY(atcor,rrsData,sensor,targets,info,xlog=True,ylog=True)

# Train and get results
result = regressor('MLPrrs',X,y,info['regLogScore'],cv=2,epoch=100)





