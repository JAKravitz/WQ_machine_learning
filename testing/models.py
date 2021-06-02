#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:20:29 2021

@author: jkravz311
"""

# Multilayer Perceptron
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability.python.math import random_rademacher
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Define MLP w/ dropout
def MLPreg(Din,Dout,lrate):
    def create():
        # create model
        model = Sequential(
            [Dense(512, use_bias=False, input_shape=(Din,)), BatchNormalization(), ReLU(),Dropout(0.1),
             Dense(128, use_bias=False), BatchNormalization(), ReLU(), Dropout(0.1),
             Dense(64, use_bias=False), BatchNormalization(), ReLU(), Dropout(0.1), 
             Dense(32, use_bias=False), BatchNormalization(), ReLU(), Dropout(0.1),
             Dense(Dout)
             ])
        # compile model
        model.compile(tf.keras.optimizers.Adam(lr=lrate),
              loss='mean_absolute_error')
        
        return model
    return create
     
        
# define BNN model
def BNNreg(X,Din,Dout,lrate):
    def create():
        
        # loss
        def NLL(y, distr): 
            return -distr.log_prob(y) 

        def normal_sp(params): 
          return tfd.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2]))# both parameters are learnable

        # create model
        kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (X.shape[0] * 1.0)
        bias_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (X.shape[0] * 1.0)
        
        inputs = Input(shape=(Din,))
        hidden = tfp.layers.DenseFlipout(512,
                                         kernel_divergence_fn=kernel_divergence_fn,
                                         bias_divergence_fn=bias_divergence_fn,
                                         bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                         bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                         activation='relu')(inputs)                                       
        hidden = tfp.layers.DenseFlipout(128, 
                                         kernel_divergence_fn=kernel_divergence_fn,
                                         bias_divergence_fn=bias_divergence_fn,
                                         bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                         bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                         activation='relu')(hidden) 
        hidden = tfp.layers.DenseFlipout(64, 
                                         kernel_divergence_fn=kernel_divergence_fn,
                                         bias_divergence_fn=bias_divergence_fn,
                                         bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                         bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                         activation='relu')(hidden) 
        hidden = tfp.layers.DenseFlipout(32,
                                         kernel_divergence_fn=kernel_divergence_fn,
                                         bias_divergence_fn=bias_divergence_fn,
                                         bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                         bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                         activation='relu')(hidden)
        params = tfp.layers.DenseFlipout(Dout, 
                                         kernel_divergence_fn=kernel_divergence_fn,
                                         bias_divergence_fn=bias_divergence_fn,
                                         bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                         bias_prior_fn=tfp.layers.default_multivariate_normal_fn)(hidden) 
        dist = tfp.layers.DistributionLambda(normal_sp)(params)
        model = Model(inputs=inputs, outputs=dist)
        
        # compile
        model.compile(Adam(learning_rate=lrate), loss=NLL)
        model_params = Model(inputs=inputs, outputs=params)
        
        return model
    return create

        

            
            
            
            
        
