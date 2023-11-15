#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:19:54 2022

@author: jakravit
"""
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from standard.MLPregressor import MLPregressor
import matplotlib.pyplot as plt

def clean(data):   
    # data.fillna(0,inplace=True)
    # data.replace(np.inf,0)
    data = data.replace([np.inf, -np.inf], np.nan, inplace=False)    
    data = data.dropna(axis=0,how='any',inplace=False)
    return data

refData = pd.read_csv('/Users/jakravit/Desktop/ee_datasets/s2_MLP_REG_ref_all.csv',index_col =None)
# refData.chl = pd.cut(refData.chl, bins=[0, 2, 5, 10, 30, 60, 100, 500, np.inf], labels=[0,1,2,3,4,5,6,7])
# refData.to_csv('/Users/jakravit/Desktop/s2_ref_chl_clf_all.csv',index=False)

X = refData.filter(regex='^B')
y = refData.loc[:,'chl']

X = clean(X)
y = y.loc[X.index]  

out = pd.concat([X,y],axis=1).astype(np.float32)
out.to_csv('/Users/jakravit/Desktop/s2_chl_ref_v2.csv',index=False)


#%%
y2 = pd.cut(y, bins=[0, 5, 10, 30, 60, 100, 500, np.inf], labels=[0,1,2,3,4,5,6])


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 256 for patching
model = Sequential(
        [Dense(500, kernel_initializer='normal', input_shape=(1,X.shape[1],)), ReLU(),
         Dense(200, kernel_initializer='normal'), ReLU(),
         Dense(50, kernel_initializer='normal'), ReLU(),
         Dense(y.shape[1])
         ])
# compile
model.compile(Adam(lr=.0001),loss='mean_absolute_error')
print (model.summary())

model.fit(X, y,
          batch_size=16,
          epochs=20,
          validation_split=.1,
          verbose=1)

save_model_path = '/Users/jakravit/Desktop/basic_mlp_reg2'
model.save(save_model_path)
