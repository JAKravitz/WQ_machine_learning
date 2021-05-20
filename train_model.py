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
#rrsData = pickle.load( open( "/content/drive/My Drive/sensorIDX_rrs.p", "rb" ) )
#rrsData = pickle.load( open( "/content/drive/MyDrive/sensorIDX_rrs.p", "rb" ) )
rrsData = pickle.load( open( "/Users/jakravit/Desktop/npp_projects/sensorIDX_rrs.p", "rb" ) )

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





#%% train
from MDNregressor import MDNregressor
import mdn

kfold = KFold(n_splits=train_info['cv'],shuffle=True)
for train, test in kfold.split(X, y):
    X_train, X_test = X.iloc[train,:], X.iloc[test,:]
    y_train, y_test = y.iloc[train,:], y.iloc[test,:]   
    #history, model, model_params = fit_BNN(X_train,y_train,epoch=20,batch=256,lrate=1e-4,split=.2)
    n_in = X_train.shape[1]
    n_out = y_train.shape[1]
    model = MDNregressor(n_in,n_out,epochs,batch_size,lrate,split,n_mixes)
    model.build(layers)
    history = model.fit(X_train,y_train)
    y_dist, y_samples, y_hat = model.predict(X_test, y_test)

#%%
import matplotlib.pyplot as plt
from tensorflow_probability import distributions as tfd
from scipy.stats import nbinom, norm
import tensorflow as tf


#tf.compat.v1.enable_eager_execution()
mus, sigs, pi_logits = mdn.split_mixture_params(y_dist.iloc[1,:], n_out, n_mixes)
pis = mdn.softmax(pi_logits, t=1)   

gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=list(pis)),
        components_distribution=tfd.Normal(
            loc=list(mus),       
            scale=list(sigs)))
x = np.linspace(-3,10,int(1e3))
pyx = gm.prob(x)


ids = {0:'chl',
       1:'PC',
       2:'cnap',
       3:'cdom',
       4:'admix',
       5:'aphy440',
       6:'ag440',
       7:'anap440',
       8:'bbphy440',
       9:'bbnap440'}

creds = {'chl': np.zeros(len(y_test)),
         'PC': np.zeros(len(y_test)),
         'cnap': np.zeros(len(y_test)),
         'cdom': np.zeros(len(y_test)),
         'admix': np.zeros(len(y_test)),
         'aphy440': np.zeros(len(y_test)),
         'ag440': np.zeros(len(y_test)),
         'anap440': np.zeros(len(y_test)),
         'bbphy440': np.zeros(len(y_test)),
         'bbnap440': np.zeros(len(y_test)),}

for i in range(len(y_test)):
    mus, sigs, pi_logits = mdn.split_mixture_params(y_dist.iloc[i,:], n_out, n_mixes)
    pis = mdn.softmax(pi_logits, t=1)
    muc = [list(t) for t in zip(*[iter(mus)]*n_mixes)]
    sic = [list(t) for t in zip(*[iter(sigs)]*n_mixes)]
    for k in range(y_test.shape[1]):
        gm = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=list(pis)),
                components_distribution=tfd.Normal(
                    loc=muc[k],       
                    scale=sic[k]))
        conf90 = norm.interval(0.9, loc=gm.mean().numpy(), scale=gm.stddev().numpy())
        conf60 = norm.interval(0.60, loc=gm.mean().numpy(), scale=gm.stddev().numpy())
        yvar = y_test.iloc[i,k]
        if conf90[0] <= yvar and conf90[1] > yvar:
            creds[ids[k]][i] = 1
        


fig,ax = plt.subplots() 
ax.plot(x,pyx)
for k in range(5):
    ax.plot(x,norm.pdf(x,list(mus)[k],list(sigs)[k]))
    
#%%

conf = norm.interval(0.95, loc=gm.mean().numpy(), scale=gm.stddev().numpy())
print (conf)

#%%
### Create a mixture of two scalar Gaussians:

gm = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=[0.3, 0.7]),
    components_distribution=tfd.Normal(
      loc=[-1., 1],       # One for each component.
      scale=[0.1, 0.5]))  # And same here.

# gm.mean()
# # ==> 0.4

# gm.variance()
# # ==> 1.018

# Plot PDF.
x = np.linspace(-2., 3., int(1e4), dtype=np.float32)

import matplotlib.pyplot as plt
with tf.compat.v1.Session() as sess:
    plt.plot(x, gm.prob(x).eval());

#%%
import matplotlib.pyplot as plt

# validation error over training
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.show()

yhat = model()
