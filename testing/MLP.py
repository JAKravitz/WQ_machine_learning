#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras Multi-layer perceptron models

@author: jkravitz, may 2021
"""
from keras.models import Sequential
from keras.layers import Dense

# Define keras regression model
def MLPr_rrs(N_dims,N_out):
    def create():
        from keras import backend as K
        K.clear_session()
        # create model
        model = Sequential()
        model.add(Dense(500, input_dim=N_dims, kernel_initializer='normal', activation='relu'))
        model.add(Dense(200, kernel_initializer='normal', activation='relu'))
        model.add(Dense(50, kernel_initializer='normal', activation='relu'))
        model.add(Dense(N_out, kernel_initializer='normal', activation='linear'))
        # compile model
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model
    return create

def MLPr_ref(N_dims,N_out):
    def create():
        from keras import backend as K
        K.clear_session()
        # create model
        model = Sequential()
        model.add(Dense(500, input_dim=N_dims, kernel_initializer='normal', activation='relu'))
        model.add(Dense(300, kernel_initializer='normal', activation='relu'))
        model.add(Dense(200, kernel_initializer='normal', activation='relu'))
        model.add(Dense(100, kernel_initializer='normal', activation='relu'))
        model.add(Dense(N_out, kernel_initializer='normal', activation='linear'))
        # compile model
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model
    return create

def MLPc(N_dims,N_out):
    def create():
        from keras import backend as K
        K.clear_session()
        # create model
        model = Sequential()
        model.add(Dense(200, input_dim=N_dims, kernel_initializer='normal', activation='relu'))
        model.add(Dense(200, kernel_initializer='normal', activation='relu'))
        model.add(Dense(200, kernel_initializer='normal', activation='relu'))
        model.add(Dense(200, kernel_initializer='normal', activation='relu'))
        model.add(Dense(N_out, kernel_initializer='normal', activation='sigmoid'))
        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        return model
    return create


def MLPcc(N_dims,N_out):
    def create():
        from keras import backend as K
        K.clear_session()
        # create model
        model = Sequential()
        model.add(Dense(500, input_dim=N_dims, kernel_initializer='normal', activation='relu'))
        model.add(Dense(200, kernel_initializer='normal', activation='relu'))
        model.add(Dense(50, kernel_initializer='normal', activation='relu'))
        #model.add(Dense(200, kernel_initializer='normal', activation='relu'))
        model.add(Dense(N_out, kernel_initializer='normal', activation='linear'))
        # compile model
        model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
        return model
    return create