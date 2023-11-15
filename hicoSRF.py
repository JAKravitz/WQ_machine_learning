#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:02:41 2022

@author: jakravit
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import netCDF4

path = '/Users/jakravit/Desktop/clusterDict.p'

with open(path, 'rb')as fp:
    data = pickle.load(fp)


# clusters = data['smooth_clusters']
# rrsData = data['dataset_clean_nonOut']
# rrsData.to_csv('/Users/jakravit/Desktop/Synthetic_rrs_1nm.csv')

#%%

f = netCDF4.Dataset('/Users/jakravit/Desktop/iss_hico_RSR.nc')
print(f.variables.keys())

l = f['wavelength'][:]
# b = f['bands'][:]
r = f['RSR'][:]

hicoL = pd.read_csv('/Users/jakravit/Desktop/hico_lambda.csv')

srf = pd.DataFrame(r, index=hicoL.values[:,0], columns=l.astype(np.float32))

hicoSRF = srf.T
hicoSRF = hicoSRF.loc[:,404.08:896.688]


#%%

solar = pd.read_csv('/Users/jakravit/Desktop/thuillier.csv',index_col='wl')
# rrsData = data['rrs_smooth_clean']
# spec = rrsData.iloc[0,:-1]
# name = spec.name

def hico_srf(spec, solar, srf):
    name = spec.name
    merged = pd.merge(pd.merge(srf, solar,right_index=True, left_index=True), spec, right_index=True, left_index=True)
    rrs_res = []
    for l in hicoSRF.columns.values:        
        merged['numerator'] = merged.apply(lambda row: (row[l]*row[name]*row['Ed']), axis=1)
        merged['denominator'] = merged.apply(lambda row: (row[l]*row['Ed']), axis=1)
        rrs_res.append(merged.numerator.sum()/merged.denominator.sum())
    return rrs_res

# rrs_res = hico_srf(spec, solar, hicoSRF)


# fig, ax = plt.subplots()
# ax.plot(spec.index,spec.values, '-ro')
# ax.plot(hicoSRF.columns, rrs_res, '-bo')


#%%

clusters = data['smooth_clusters']
rrsData = data['dataset_clean_nonOut']
rrs = rrsData.filter(regex='^\d').astype(float)
rrs.columns = rrs.columns.astype(float)
meta = rrsData.filter(regex='^[a-zA-Z]')

new_rrs = rrs.T.apply(lambda row: hico_srf(row,solar,hicoSRF))
new_rrs = new_rrs.T
new_rrs.columns = hicoSRF.columns
final = pd.concat([meta, new_rrs], axis=1)
final.to_csv('/Users/jakravit/Desktop/Rrs_dataset_v2.csv')

# new_rrs = pd.DataFrame()
# for i,row in rrs.iterrows():
#     rrs_res = hico_srf(row, solar, hicoSRF)
#     new_rrs[i] = rrs_res
# new_rrs = new_rrs.T

#%%
data = pd.read_csv('/Users/jakravit/Desktop/kuts_final.csv')

rrs = data.filter(regex='^\d').astype(float)
rrs.columns = rrs.columns.astype(float)
meta = data.filter(regex='^[a-zA-Z]')

new_rrs = rrs.T.apply(lambda row: hico_srf(row,solar,hicoSRF))
new_rrs = new_rrs.T
new_rrs.columns=hicoSRF.columns

#%%
kuts = pd.concat([meta,new_rrs], axis=1)


#%%
final = pd.concat([kravz, kuts, cast], axis=0)

final.to_csv('/Users/jakravit/Desktop/Rrs_chl_matchups.csv')




























