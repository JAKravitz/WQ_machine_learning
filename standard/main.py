#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 11:42:15 2021

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
from MLPregressor import MLPregressor
import matplotlib.pyplot as plt
import os

data = pd.read_csv('/Users/jakravit/data/PRISM/prismDataset_synthetic_case1.csv',index_col=0)
# data = pd.read_csv('/Users/jakravit/data/cyanosat/Synthetic_rrs_1nm.csv',index_col=0)
# data = pd.read_csv('/Users/jakravit/data/cyanosat/cyanosat_12nm_resolved_synthetic_brr_3nm.csv',index_col=0)
case = 'prism_v1'

# obs = pd.read_csv('/Users/jakravit/Desktop/JPL_hyper_ML_project/Rrs_chl_matchups.csv')
# X_obs = obs.filter(regex='^[0-9]')
# y_obs = obs.chl

targets = ['totChl', 'nap_Cmin', 'cdom_slope_ratio', # concentrations & slopes
           #'cdom_a_tot_440', 'phy_a_tot_440', 'nap_amin440', 'nap_adet440',# component a @ 440
           'tota440', 'totbb440',
           'phy_bb_tot_440', 'nap_bbmin440', # component bb @ 440
           # 'tota660','totb660','totbb660', # for CORAL
           'SG_amp', # sunglint amplitutde
           'benthic_coral_gfx', 'benthic_algae_gfx', 'benthic_sediment_gfx', 'benthic_seagrass_gfx', 'benthic_bleachedCoral_gfx'] 

targets = ['benthic_coral_gfx', 'benthic_algae_gfx', 'benthic_sediment_gfx', 'benthic_seagrass_gfx', 'benthic_bleachedCoral_gfx']

batch_info = {
              'epochs':200,
              'batch_size':16,
              'lrate':.0001,
              'split':.1,
              'targets': targets,
              'cv':0,
              'meta': None,
              'Xpca': None, 
              }

for key, value in batch_info.items():
    value = None if value == 'None' else value
    batch_info[key] = value

model = MLPregressor(batch_info)


X, y, y0, Xscaler, Yscaler = model.getXY(data)
results = model.prep_results(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=batch_info['split'])
# kfold = KFold(n_splits=batch_info['cv'], shuffle=True)
# for train, test in kfold.split(X_train, y_train):
#     model.build()
#     X_tn, X_tt = X_train.iloc[train,:], X_train.iloc[test,:]
#     y_tn, y_tt = y_train.iloc[train,:], y_train.iloc[test,:] 
#     history = model.fit(X_tn,y_tn)
#     y_ht = model.predict(X_tt)
#     results = model.evaluate(y_ht,y_tt,results,'cv') 

print ('\n## FINAL MODEL ##\n')
model.build()
history2 = model.fit(X_train,y_train)
results['train_loss'].append(history2.history['loss'])
results['val_loss'].append(history2.history['val_loss'])

# synthetic predict
y_hat = model.predict(X_test)
results = model.evaluate(y_hat,y_test,results,'final') 
results['batch_info'] = batch_info

# observed predict
# X_obs, y_obs, y_obs0 = model.getXY_obs(obs)
# y_obs_hat = model.predict(X_obs)
# results_obs = model.evaluate(y_obs_hat, y_obs, results, 'final')

# save run to disk

# os.mkdir(f'/Users/jakravit/data/wq_models/{case}/')
fname = f'/Users/jakravit/data/wq_models/{case}/results_{case}.p'
xScalerFname = f'/Users/jakravit/data/wq_models/{case}/Xscaler_{case}.p'
yScalerFname = f'/Users/jakravit/data/wq_models/{case}/Yscaler_{case}.p'
f = open(fname,'wb')
pickle.dump(results,f)
pickle.dump(Xscaler, open(xScalerFname,'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(Yscaler, open(yScalerFname,'wb'), protocol=pickle.HIGHEST_PROTOCOL)
f.close() 


#%% plot
from matplotlib.colors import LogNorm
fig, ax = plt.subplots()
# ax.hist2d(results['benthic_algae_gfx']['final']['ytest'][0], results['benthic_algae_gfx']['final']['yhat'][0], 
#           norm=LogNorm(), bins=50, cmap='rainbow')
ax.scatter(np.exp(results['tota440']['final']['ytest'][0]), 
           np.exp(results['tota440']['final']['yhat'][0]), 
          s=1)
# ax.set_xlim(0,1)
# ax.set_ylim(-2,6)

# #%%
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve,  auc, f1_score
# from sklearn import preprocessing
# import seaborn as sns

# x = results['admix']['final']['ytest'][0]
# y = results['admix']['final']['yhat'][0]

# y2 = round(y,1)
# y2 = np.where(y2>1, 1, y2)
# y2 = np.where(y2<0, 0, y2)
# y2 = abs(y2)


# lab = preprocessing.LabelEncoder()
# x = lab.fit_transform(x)
# y2 = lab.fit_transform(y2)
# acc = accuracy_score(x,y2)
# f1 = f1_score(x, y2, average='macro')

# # Get and reshape confusion matrix data
# matrix = confusion_matrix(x, y2)
# matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# # Build the plot
# plt.figure(figsize=(12,10))
# sns.set(font_scale=1.4)
# sns.heatmap(matrix, annot=True, annot_kws={'size':10},
#             cmap=plt.cm.Greens, linewidths=0.2)

# # Add labels to the plot
# class_names = ['0','1','2','3','4','5','6','7','8','9','10']
# tick_marks = np.arange(len(class_names))
# tick_marks2 = tick_marks + 0.5
# plt.xticks(tick_marks, class_names, rotation=25)
# plt.yticks(tick_marks2, class_names, rotation=0)
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.title('Confusion Matrix for Random Forest Model')
# plt.show()