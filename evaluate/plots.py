#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:24:43 2021

@author: jakravit
"""
#%%
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/jakravit/Desktop/ML_results_v1.p'
data = pickle.load(open(path,'rb'))

phydict = {'X':pd.DataFrame(),
           'log':pd.DataFrame(),
           'maxabs':pd.DataFrame(),
           'minmax':pd.DataFrame(),
           'standard':pd.DataFrame(),
           'robust':pd.DataFrame(),
           'quantile':pd.DataFrame(),
           'power':pd.DataFrame(),
           'norm':pd.DataFrame()
           }

abv = {'phy':'aphy440',
       'ag':'ag440',
       'nap':'anap440'}

p = 'phy'
for k in data:
    results = data[k]
    run = k.split('_')
    transform = run[0]
    target = run[1:]
    
    if len(target) > 1:
        target = '_'.join(target)
    else:
        target = target[0]
        
    if target in ['a','tot','a_log','tot_log']:
        phydict[transform][target] = results[abv[p]]['MAPE']
    elif target in [p,'{}_log'.format(p)]:
        phydict[transform][target] = results[abv[p]]['MAPE']
    

#%%
fig, axs = plt.subplots(3,3,figsize=(9,7),sharey=True, sharex=True)
axs = axs.ravel()
count = 0
for transform in phydict:
    phydict[transform].mean().plot.bar(ax=axs[count],yerr=phydict[transform].std(),legend=False, width=.7)
    axs[count].set_ylim(0,60)
    axs[count].set_title(transform)
    count = count+1
axs[3].set_ylabel('MAPE (%)')
plt.subplots_adjust(wspace=.05)
fig.savefig('/Users/jakravit/Desktop/ML_results_v1_plot.png',bbox_inches='tight',dpi=300)
