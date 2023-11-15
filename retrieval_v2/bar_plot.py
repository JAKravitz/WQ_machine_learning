#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 21:53:32 2021

@author: jakravit
"""
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm

case1 = pickle.load( open( "/Users/jakravit/Desktop/retrieval_results_rrs_v2/case_1.p", "rb" ) )
case2 = pickle.load( open( "/Users/jakravit/Desktop/retrieval_results_rrs_v2/case_2.p", "rb" ) )
case3 = pickle.load( open( "/Users/jakravit/Desktop/retrieval_results_rrs_v2/case_3.p", "rb" ) )
targets = ['chl','PC','fl_amp','dinoD','aphy440','ag440','anap440','bbphy440','bbnap440','astarD440','astarCy440',]
lims = {'chl': [-2,9],
        'PC': [-5,10],
        'fl_amp': [-6,4],
        'dinoD': [2,4],
        'aphy440': [-5,6],
        'ag440': [-4,3],
        'anap440': [-6,4],
        'bbphy440': [-7,2],
        'bbnap440': [-7,3],
        'astarD440': [-6,-3],
        'astarCy440': [-6,-3]}


import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
          'axes.labelsize': 22,
          'axes.titlesize': 24,
          'xtick.labelsize': 13,
          'ytick.labelsize': 14
          }
pylab.rcParams.update(params)

fig, axs = plt.subplots(3,4, figsize=(15,10), sharex=True, sharey=True)
axs = axs.ravel()

for i,k in enumerate(targets):
    dat = [case1[k]['final']['MAPE'][0],case2[k]['final']['MAPE'][0],case3[k]['final']['MAPE'][0]] 
    dat = pd.DataFrame(dat,index=['Hyper','PCA20','PCA10'])
    dat = dat.T

    if i != 0:
        dat.plot.bar(color={'Hyper':'red','PCA20':'blue','PCA10':'green'},ax=axs[i],legend=False)
    else:
        dat.plot.bar(color={'Hyper':'red','PCA20':'blue','PCA10':'green'},ax=axs[i],legend=True)
        axs[i].legend(loc='upper left')
    
    if i == 6:
        loc=4
    else:
        loc=1
    ax1 = inset_axes(axs[i],width='50%',height='50%',loc=loc)
    #ax1.hist2d(case2[k]['final']['ytest'][0], case2[k]['final']['yhat'][0], bins=300, norm=LogNorm(),cmap='rainbow')
    ax1.scatter(case2[k]['final']['ytest'][0], case2[k]['final']['yhat'][0],c='k',s=.01)
    ax1.plot(lims[k],lims[k],c='k',ls='--',lw=2)
    ax1.set_xlim(lims[k][0],lims[k][1])
    ax1.set_ylim(lims[k][0],lims[k][1])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    
    axs[i].set_title(k)
    axs[i].set_ylim(0,55)
    axs[i].grid()
    axs[i].set_xticklabels([])
    axs[11].axis('off')
    axs[4].set_ylabel('MAPE (%)')
    
fig.savefig('/Users/jakravit/Desktop/hyper_vs_pca.png',bbox_inches='tight',dpi=300)


#%%



