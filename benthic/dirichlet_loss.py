#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 20:22:15 2023

@author: jakravit
"""
#%%
import pandas as pd
import datatable as dt

#%%
inputs1 = pd.read_csv('/Users/jakravit/Desktop/case1v0_1/BOA_inputs_case1V0.csv')
inputs2 = pd.read_csv('/Users/jakravit/Desktop/case1v0_2/BOA_inputs_case1V0.csv')
outputs1 = pd.read_csv('/Users/jakravit/Desktop/case1v0_1/BOA_outputs_case1V0.csv')
outputs2 = pd.read_csv('/Users/jakravit/Desktop/case1v0_2/BOA_outputs_case1V0.csv')

#%%
inputs3 = pd.concat([inputs1, inputs2], axis=0)
outputs3 = pd.concat([outputs1, outputs2], axis=0)

inputs = inputs3[inputs3.modCase == 1]
outputs = outputs3[outputs3.modCase == 1]

#%%
targets = ['benthic_Coral_cfx', 'benthic_Algae_cfx', 'benthic_Substrate_cfx', 
           'benthic_Seagrass/weed_cfx', 'benthic_Bleached coral_cfx',
           'totChl', 'nap_Cmin',]
y = inputs[targets]
X = outputs.filter(regex='^rrs')

#%% for chatgpt

X2 = X.iloc[:100,:]
y2 = y.iloc[:100,:]

X2.to_csv('/Users/jakravit/Desktop/Xsub.csv')
y2.to_csv('/Users/jakravit/Desktop/ysub.csv')

#%% dirichlet loss function

import numpy as np
from scipy.special import gamma

def dirichlet_loss(alpha, x):
    """
    Compute the negative log likelihood of the Dirichlet distribution.
    
    Parameters:
    - alpha: Vector of Dirichlet parameters (shape: [batch_size, num_classes]).
    - x: Vector of observed proportions (shape: [batch_size, num_classes]).
    
    Returns:
    - Negative log likelihood for each sample in the batch.
    """
    
    # Compute the log of the gamma function for each alpha
    log_gamma_alpha = np.log(gamma(alpha))
    
    # Compute the negative log likelihood based on the Dirichlet formula
    nll = (np.log(gamma(np.sum(alpha, axis=1))) 
           - np.sum(log_gamma_alpha, axis=1) 
           + np.sum((alpha - 1) * np.log(x), axis=1))
    
    return nll

#%%
import tensorflow as tf
from tensorflow.keras import layers, Model

# Number of features in Xsub
input_dim = Xsub.shape[1] - 1  # Excluding the unnamed index column

# Model Architecture
input_layer = layers.Input(shape=(input_dim,))
x = input_layer
for _ in range(5):
    x = layers.Dense(100, activation='relu')(x)
alpha_output = layers.Dense(5, activation='softplus')(x)
benthic_output = ... # Use alpha_output to compute Dirichlet likelihood or sampling
concentration_output = layers.Dense(2, activation='linear')(x)
model = Model(inputs=input_layer, outputs=[benthic_output, concentration_output])

# Display the model summary
model.summary()

#%% 
# Model Compilation
model.compile(optimizer='adam',
              loss={'benthic_output': dirichlet_loss,  # Custom loss for Dirichlet
                    'concentration_output': 'mean_squared_error'},
              metrics={'benthic_output': 'accuracy', 
                       'concentration_output': 'mse'})

model.fit(Xsub.drop(columns='Unnamed: 0'), 
          {'benthic_output': ysub[['benthic_Coral_cfx', 'benthic_Algae_cfx', 'benthic_Substrate_cfx', 'benthic_Seagrass/weed_cfx', 'benthic_Bleached coral_cfx']],
           'concentration_output': ysub[['totChl', 'nap_Cmin']]},
          epochs=50,
          batch_size=32)
