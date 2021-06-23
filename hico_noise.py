#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:20:21 2021

@author: jakravit
"""
#%%
import pickle
import pandas as pd
import numpy as np
from math import sqrt, cos
import matplotlib.pyplot as plt

# data
rrsData = pickle.load( open( "/Users/jakravit/Desktop/nasa_npp/RT/sensorIDX_rrs.p", "rb" ) )
refData = pickle.load( open( "/Users/jakravit/Desktop/nasa_npp/RT/sensorIDX_ref.p", "rb" ) )
meta = refData['hico']
hicoref = refData['hico'].filter(regex='^[0-9]')
hicorrs = rrsData['hico'].filter(regex='^[0-9]')
hicowl = hicorrs.columns.values.astype(float) / 1000

# quantum efficiency
with open('/Users/jakravit/Desktop/moses_noise/HICO_Det_Quantum_Efficiency.txt') as f:
    qeraw = f.readlines()
qe = pd.Series()
for l in qeraw[1:]:
    qe[l.strip().split('\t')[0]] = l.strip().split('\t')[1]
qel = qe.index.values.astype(float)
qeval = qe.values.astype(float)
qeint = np.interp(hicowl,qel,qeval)
    
# FWHM
with open('/Users/jakravit/Desktop/moses_noise/HICO_wl_fwhm_in_um_with_fwhm_5p7_nm.txt') as f:
    fwhmraw = f.readlines()
fwhm = pd.Series()
for l in fwhmraw[1:]:
    fwhm[l.strip().split('\t')[0]] = l.strip().split('\t')[1]
    
# dark noise
with open('/Users/jakravit/Desktop/moses_noise/HICO_Dark_StDev.txt') as f:
    darkraw = f.readlines()
dark = pd.Series()
for l in darkraw[1:]:
    dark[l.strip().split('\t')[0]] = l.strip().split('\t')[1]
dkl = dark.index.values.astype(float)
dkval = dark.values.astype(float)
dkint = np.interp(hicowl,dkl,dkval)

# F0
with open('/Users/jakravit/git/MODTRAN/data/SUNnmCEOSThuillier2005.dat') as f:
    F0raw = f.readlines()
F0 = pd.Series()
for l in F0raw[1:]:
    F0[l.split('    ')[0]] = l.split('    ')[1].strip()
F0l = F0.index.values.astype(float) / 1000
F0val = F0.values.astype(float)
F0int = np.interp(hicowl,F0l,F0val)
    
#%% shot noise

def shot_noise(hicowl,reflectance,F0int):
    # variables
    h = 6.63e-34 # planck constant
    c = 3e8 # speed of light
    fg = 0.92 # fraction of grating groove at blaze angle
    ng0 = 0.8 # grating efficiency at blaze wl
    nop = 0.8 # optical transmissive efficiency
    l0 = 400.001 # blaze wl
    p = 16e-6 # spatial width of detector pixel (m)
    D = 0.019 # diameter of aperture (m)
    f = 0.067 # focal length
    pu = 16 # detector pixel width (um)
    salt = 400 # sensor altitude (km)
    Hs = 400 * 1000 # sensor altitude (m)
    re = 6378000 # earth radius
    g = 9.8 # gravitational acceleration (ms^-2) 
    ng = ng0 * (np.sinc(fg * (1 - (l0/hicowl)))**2) # grating efficiency 
    nsys = nop * qeint * ng # overal system efficiency
    GSD = (pu * salt) / f # ground sampling distance
    V = (re**2 / (re + Hs)) * sqrt(g / (re + Hs)) # sensor ground velocity
    T = GSD / V # exposure time
    # toa calc
    theta = reflectance.loc[' SZA']
    refl = reflectance.filter(regex='\d')
    toa = (F0int * refl * cos(theta)) / np.pi()
    # shot noise calc
    shot = sqrt( (hicowl/(h*c)) * toa * (np.pi()/4) * (D**2/f**2) * p**2 * T * nsys )
    return toa,shot

#%%





#%%    
fig, ax = plt.subplots()
ax.plot(dkl,dkval,c='b')
ax.plot(hicowl,dkint,c='r')
