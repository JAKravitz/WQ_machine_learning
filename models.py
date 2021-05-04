#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:20:29 2021

@author: jkravz311
"""

# Multilayer Perceptron
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D

def MLPreg_rrs(N_dims,N_out):
    def create():
        from keras import backend as K
        K.clear_session()
        # create model
        model = Sequential()
        model.add(Dense(500, input_dim=N_dims[1], kernel_initializer='normal', activation='relu'))
        model.add(Dense(200, kernel_initializer='normal', activation='relu'))
        model.add(Dense(50, kernel_initializer='normal', activation='relu'))
        model.add(Dense(N_out, kernel_initializer='normal', activation='linear'))
        # compile model
        model.compile(loss='mean_absolute_error', optimizer='adam')
        print (model.summary())
        plot_model(model)
        return model
    return create


def CNNreg_rrs(N_dims,N_out):
    def create():
        from keras import backend as K
        K.clear_session()
        # model
        model = Sequential()
        model.add(Conv1D(100, 10, activation='relu', input_shape=(N_dims[0],N_dims[1],1)))
        model.add(Conv1D(100, 10, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(160, 10, activation='relu'))
        model.add(Conv1D(160, 10, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(N_out, activation='relu'))
        # compile
        model.compile(loss='mean_absolute_error', optimizer='adam')
        print (model.summary())
        plot_model(model)
        return model
    return create
        
        
        
        



