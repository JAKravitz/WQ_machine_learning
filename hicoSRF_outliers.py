#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:31:42 2022

@author: jakravit
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import netCDF4

path = '/Volumes/SECUREDRIVE/313469/Partition_1/Phd/HE_model_train/clusterDict.p'

with open(path, 'rb')as fp:
    data = pickle.load(fp)

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

solar = pd.read_csv('/Users/jakravit/Desktop/thuillier.csv',index_col='wl')

#%%
def hico_srf(spec, solar, srf):
    name = spec.name
    merged = pd.merge(pd.merge(srf, solar,right_index=True, left_index=True), spec, right_index=True, left_index=True)
    rrs_res = []
    for l in hicoSRF.columns.values:        
        merged['numerator'] = merged.apply(lambda row: (row[l]*row[name]*row['Ed']), axis=1)
        merged['denominator'] = merged.apply(lambda row: (row[l]*row['Ed']), axis=1)
        rrs_res.append(merged.numerator.sum()/merged.denominator.sum())
    return rrs_res


#%% 

rrs = data['smooth_clusters']
optics = data['optics_clusters']

newdict = {}
for cluster in rrs:
    print (cluster)
    out = rrs[cluster]['outliers']
    out2 = out.T.apply(lambda x: hico_srf(x,solar,hicoSRF))
    out2 = out2.T
    out2.columns = hicoSRF.columns
    # matching biogeo data for outliers
    dAll = optics[cluster]['dataAll']
    dClean = optics[cluster]['dataNonOut']
    dif = dAll.index.difference(dClean.index)
    outOpt = dAll.loc[dif,:]
    final = pd.concat([out2,outOpt], axis=1)
    newdict[cluster] = final

with open('/Users/jakravit/Desktop/outliers.p','wb') as fp:
    pickle.dump(newdict,fp)

#%%
out = rrs[0]['outliers']
out2 = out.T.apply(lambda x: hico_srf(x,solar,hicoSRF))
out2 = out2.T
out2.columns = hicoSRF.columns
    
dAll = optics[0]['dataAll']
dclean = optics[0]['dataNonOut']
dif = dAll.index.difference(dclean.index)
outOpt = dAll.loc[dif,:]

final = pd.concat([out2,outOpt], axis=1)

#%% Mark data

data = pd.read_csv('/Users/jakravit/git/JPL_WQ_ML/field_data/Rrs_field_matchups_MM.csv')
rrs = data.filter(regex='^[0-9]')
meta = data.filter(regex='^[a-zA-Z]')
rrs.columns = rrs.columns.astype(float)
new_rrs = rrs.T.apply(lambda x: hico_srf(x,solar,hicoSRF))
new_rrs = new_rrs.T
new_rrs.columns = hicoSRF.columns
final = pd.concat([new_rrs, meta], axis=1)
final.to_csv('/Users/jakravit/Desktop/Rrs_field_matchups_MM.csv')
