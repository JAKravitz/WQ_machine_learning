#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian Density Network
https://www.kaggle.com/brendanhasz/bayesian-density-network/notebook#Data

@author: jkravz311
"""
#%% PACKAGES

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import vaex

from sklearn.dummy import DummyRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability.python.math import random_rademacher
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout
from tensorflow.keras.optimizers import Adam

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

import pickle
from getXY import getXY
#from regressor import regressor
import scorers as sc
import sklearn.metrics as metrics

# Settings
sns.set()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
np.random.seed(12345)
tf.random.set_seed(12345)

#%% DATA (already pre-processed)

# data
#rrsData = pickle.load( open( "/content/drive/My Drive/sensorIDX_rrs.p", "rb" ) )
rrsData = pickle.load( open( "/Users/jkravz311/GoogleDrive/sensorIDX_rrs.p", "rb" ) )
#refData = pickle.load( open( "/Users/jkravz311/GoogleDrive/sensorIDX_ref.p", "rb" ) )

# batch setup
info = {'sensorID' : {'s2_60m':['443','490','560','665','705','740','783','842','865'],
                      's2_20m':['490','560','665','705','740','783','842','865'],
                      's2_10m':['490','560','665','842'],
                      's3':'^Oa',
                      'l8':['Aer','Blu','Grn','Red','NIR'],
                      'modis':'^RSR',
                      'meris':'^b',
                      'hico':'^H'},       
        'clfScore' : {'Accuracy': metrics.accuracy_score,
                      'ROC_AUC': metrics.roc_auc_score},        
        'regScore' : {'R2': sc.r2,
                      'RMSE': sc.rmse,
                      'RMSELE': sc.rmsele,
                      'Bias': sc.bias,
                      'MAPE': sc.mape,
                      'rRMSE': sc.rrmse,},
        'regLogScore' : {'R2': sc.r2,
                         'RMSE': sc.log_rmse,
                         'RMSELE': sc.log_rmsele,
                         'Bias': sc.log_bias,
                         'MAPE': sc.log_mape,
                         'rRMSE': sc.log_rrmse,}
        }

atcor = 'rrs'
sensor = 's3'
targets = ['chl','PC','cnap','cdom','admix','aphy440','ag440','anap440','bbphy440','bbnap440','cluster']
X,y, scaler = getXY(atcor,rrsData,sensor,targets,info,xlog=True,ylog=True)

#%% Baseline Multilayer dense neural network

# Batch size
BATCH_SIZE = 128
# Number of training epochs
EPOCHS = 100
# Learning rate
L_RATE = 1e-4
# Proportion of samples to hold out
VAL_SPLIT = 0.2

# build model
Din = X.shape[1]
Dout = y.shape[1]
model = Sequential(
    [Dense(512, use_bias=False, input_shape=(Din,)), BatchNormalization(), ReLU(),Dropout(0.1),
     Dense(128, use_bias=False), BatchNormalization(), ReLU(), Dropout(0.1),
     Dense(64, use_bias=False), BatchNormalization(), ReLU(), Dropout(0.1), 
     Dense(32, use_bias=False), BatchNormalization(), ReLU(), Dropout(0.1),
     Dense(Dout)
     ])

# compile model w/ MAE loss, adam optimizer
model.compile(tf.keras.optimizers.Adam(lr=L_RATE),
              loss='mean_absolute_error')

# Fit the model
history = model.fit(X, y,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=VAL_SPLIT,
                    verbose=1)

# validation error over training
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.show()

# predict
y_hat = pd.DataFrame(model.predict(X),index=y.index.values)


#%% simple BNN
''' 
Keras difficult for BNN b/c loss isnt a simple function of true vs predicted 
target values. For BNN, will use variational inference which depends on: true
target value, predictive distribution, and kullback-leibler divergence between
parameter's variational posteriors and priors. Must be written in "raw" TF.
'''

# Split data randomly into training + validation
tr_ind = np.random.choice([False, True],
                          size=X.shape[0],
                          p=[VAL_SPLIT, 1.0-VAL_SPLIT])
x_train = X[tr_ind].values
y_train = y[tr_ind].values
x_val = X[~tr_ind].values
y_val = y[~tr_ind].values
N_train = x_train.shape[0]
N_val = x_val.shape[0]

# Make a TensorFlow Dataset from training data
data_train = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)

# Make a TensorFlow Dataset from validation data
data_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(N_val)

# Xavier initializer to initialize the weights of the network
def xavier(shape):
    return tf.random.truncated_normal(
        shape, 
        mean=0.0,
        stddev=np.sqrt(2/sum(shape)))

#%% Manual definition of bayesian dense network

class BayesianDenseLayer(tf.keras.Model):
    """A fully-connected Bayesian neural network layer
    
    Parameters
    ----------
    d_in : int
        Dimensionality of the input (# input features)
    d_out : int
        Output dimensionality (# units in the layer)
    name : str
        Name for the layer
        
    Attributes
    ----------
    losses : tensorflow.Tensor
        Sum of the Kullback–Leibler divergences between
        the posterior distributions and their priors
        
    Methods
    -------
    call : tensorflow.Tensor
        Perform the forward pass of the data through
        the layer
    """
    
    def __init__(self, d_in, d_out, name=None):
        
        super(BayesianDenseLayer, self).__init__(name=name)
        self.d_in = d_in
        self.d_out = d_out
        
        self.w_loc = tf.Variable(xavier([d_in, d_out]), name='w_loc')
        self.w_std = tf.Variable(xavier([d_in, d_out])-6.0, name='w_std')
        self.b_loc = tf.Variable(xavier([1, d_out]), name='b_loc')
        self.b_std = tf.Variable(xavier([1, d_out])-6.0, name='b_std')
    
    
    def call(self, x, sampling=True):
        """Perform the forward pass"""
        
        if sampling:
        
            # Flipout-estimated weight samples
            s = random_rademacher(tf.shape(x))
            r = random_rademacher([x.shape[0], self.d_out])
            w_samples = tf.nn.softplus(self.w_std)*tf.random.normal([self.d_in, self.d_out])
            w_perturbations = r*tf.matmul(x*s, w_samples)
            w_outputs = tf.matmul(x, self.w_loc) + w_perturbations
            
            # Flipout-estimated bias samples
            r = random_rademacher([x.shape[0], self.d_out])
            b_samples = tf.nn.softplus(self.b_std)*tf.random.normal([self.d_out])
            b_outputs = self.b_loc + r*b_samples
            
            return w_outputs + b_outputs
        
        else:
            return x @ self.w_loc + self.b_loc
    
    
    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""
        weight = tfd.Normal(self.w_loc, tf.nn.softplus(self.w_std))
        bias = tfd.Normal(self.b_loc, tf.nn.softplus(self.b_std))
        prior = tfd.Normal(0, 1)
        return (tf.reduce_sum(tfd.kl_divergence(weight, prior)) +
                tf.reduce_sum(tfd.kl_divergence(bias, prior)))


class BayesianDenseNetwork(tf.keras.Model):
    """A multilayer fully-connected Bayesian neural network
    
    Parameters
    ----------
    dims : List[int]
        List of units in each layer
    name : str
        Name for the network
        
    Attributes
    ----------
    losses : tensorflow.Tensor
        Sum of the Kullback–Leibler divergences between
        the posterior distributions and their priors, 
        over all layers in the network
        
    Methods
    -------
    call : tensorflow.Tensor
        Perform the forward pass of the data through
        the network
    """
    
    def __init__(self, dims, name=None):
        
        super(BayesianDenseNetwork, self).__init__(name=name)
        
        self.steps = []
        self.acts = []
        for i in range(len(dims)-1):
            self.steps += [BayesianDenseLayer(dims[i], dims[i+1])]
            self.acts += [tf.nn.relu]
            
        self.acts[-1] = lambda x: x
        
    
    def call(self, x, sampling=True):
        """Perform the forward pass"""

        for i in range(len(self.steps)):
            x = self.steps[i](x, sampling=sampling)
            x = self.acts[i](x)
            
        return x
    
    
    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""
        return tf.reduce_sum([s.losses for s in self.steps])


class BayesianDenseRegression(tf.keras.Model):
    """A multilayer fully-connected Bayesian neural network regression
    
    Parameters
    ----------
    dims : List[int]
        List of units in each layer
    name : str
        Name for the network
        
    Attributes
    ----------
    losses : tensorflow.Tensor
        Sum of the Kullback–Leibler divergences between
        the posterior distributions and their priors, 
        over all layers in the network
        
    Methods
    -------
    call : tensorflow.Tensor
        Perform the forward pass of the data through
        the network, predicting both means and stds
    log_likelihood : tensorflow.Tensor
        Compute the log likelihood of y given x
    samples : tensorflow.Tensor
        Draw multiple samples from the predictive distribution
    """    
    
    
    def __init__(self, dims, name=None):
        
        super(BayesianDenseRegression, self).__init__(name=name)
        
        # Multilayer fully-connected neural network to predict mean
        self.loc_net = BayesianDenseNetwork(dims)
        
        # Variational distribution variables for observation error
        self.std_alpha = tf.Variable([10.0], name='std_alpha')
        self.std_beta = tf.Variable([10.0], name='std_beta')

    
    def call(self, x, sampling=True):
        """Perform the forward pass, predicting both means and stds"""
        
        # Predict means
        loc_preds = self.loc_net(x, sampling=sampling)
    
        # Predict std deviation
        posterior = tfd.Gamma(self.std_alpha, self.std_beta)
        transform = lambda x: tf.sqrt(tf.math.reciprocal(x))
        N = x.shape[0]
        if sampling:
            std_preds = transform(posterior.sample([N]))
        else:
            std_preds = tf.ones([N, 1])*transform(posterior.mean())
    
        # Return mean and std predictions
        return tf.concat([loc_preds, std_preds], 1)
    
    
    def log_likelihood(self, x, y, sampling=True):
        """Compute the log likelihood of y given x"""
        
        # Compute mean and std predictions
        preds = self.call(x, sampling=sampling)
        
        # Return log likelihood of true data given predictions
        return tfd.Normal(preds[:,0], preds[:,1]).log_prob(y[:,0])
    
    
    @tf.function
    def sample(self, x):
        """Draw one sample from the predictive distribution"""
        preds = self.call(x)
        return tfd.Normal(preds[:,0], preds[:,1]).sample()
    
    
    def samples(self, x, n_samples=1):
        """Draw multiple samples from the predictive distribution"""
        samples = np.zeros((x.shape[0], n_samples))
        for i in range(n_samples):
            samples[:,i] = self.sample(x)
        return samples
    
    
    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""
                
        # Loss due to network weights
        net_loss = self.loc_net.losses

        # Loss due to std deviation parameter
        posterior = tfd.Gamma(self.std_alpha, self.std_beta)
        prior = tfd.Gamma(10.0, 10.0)
        std_loss = tfd.kl_divergence(posterior, prior)

        # Return the sum of both
        return net_loss + std_loss


#%% Function to create single training step
'''
computes the log likelihood of traget values according to the model given the 
predictors, and also the kullback-leibler divergence between the parameter's
variational posteriors and their priors.
Uses TF's GradientTape which allows to backpropogate the loss gradients to our 
variables. Functin then passes those gradients to the optimizer, which 
updates the variables controling the network weights' variational posteriors. 
'''

# instantiate fully-connected BNN with however many layers/units
model1 = BayesianDenseRegression([x_train.shape[1], 256, 128, 64, 32, y_train.shape[1]])

# adam optimizer
optimizer = tf.keras.optimizers.Adam(lr=L_RATE)

N = x_train.shape[0]

@tf.function
def train_step(x_data, y_data):
    with tf.GradientTape() as tape:
        log_likelihoods = model1.log_likelihood(x_data, y_data)
        kl_loss = model1.losses
        elbo_loss = kl_loss/N - tf.reduce_mean(log_likelihoods)
    gradients = tape.gradient(elbo_loss, model1.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model1.trainable_variables))
    return elbo_loss

#%% fit standard BNN

elbo1 = np.zeros(EPOCHS)
mae1 = np.zeros(EPOCHS)
for epoch in range(EPOCHS):
    print ('EPOCH = {}'.format(epoch))
    
    # Update weights each batch
    for x_data, y_data in data_train:
        elbo1[epoch] += train_step(x_data, y_data)
        
    # Evaluate performance on validation data
    for x_data, y_data in data_val:
        y_pred = model1(x_data, sampling=False)[:, 0]
        mae1[epoch] = mean_absolute_error(y_pred, y_data)

#%% Dual-headed BDN
'''
Predicts both the target value and the uncertainty to that value. One head
will predict the mean of the distribution, the other will predict the st.dev.
'''

class BayesianDensityNetwork(tf.keras.Model):
    """Multilayer fully-connected Bayesian neural network, with
    two heads to predict both the mean and the standard deviation.
    
    Parameters
    ----------
    units : List[int]
        Number of output dimensions for each layer
        in the core network.
    units : List[int]
        Number of output dimensions for each layer
        in the head networks.
    name : None or str
        Name for the layer
    """
    
    
    def __init__(self, units, head_units, name=None):
        
        # Initialize
        super(BayesianDensityNetwork, self).__init__(name=name)
        
        # Create sub-networks
        self.core_net = BayesianDenseNetwork(units)
        self.loc_net = BayesianDenseNetwork([units[-1]]+head_units)
        self.std_net = BayesianDenseNetwork([units[-1]]+head_units)

    
    def call(self, x, sampling=True):
        """Pass data through the model
        
        Parameters
        ----------
        x : tf.Tensor
            Input data
        sampling : bool
            Whether to sample parameter values from their variational
            distributions (if True, the default), or just use the
            Maximum a Posteriori parameter value estimates (if False).
            
        Returns
        -------
        preds : tf.Tensor of shape (Nsamples, 2)
            Output of this model, the predictions.  First column is
            the mean predictions, and second column is the standard
            deviation predictions.
        """
        
        # Pass data through core network
        x = self.core_net(x, sampling=sampling)
        x = tf.nn.relu(x)
        
        # Make predictions with each head network
        loc_preds = self.loc_net(x, sampling=sampling)
        std_preds = self.std_net(x, sampling=sampling)
        std_preds = tf.nn.softplus(std_preds)
        
        # Return mean and std predictions
        return tf.concat([loc_preds, std_preds], 1)
    
    
    def log_likelihood(self, x, y, sampling=True):
        """Compute the log likelihood of y given x"""
        
        # Compute mean and std predictions
        preds = self.call(x, sampling=sampling)
        
        # Return log likelihood of true data given predictions
        return tfd.Normal(preds[:,0], preds[:,1]).log_prob(y[:,0])
        
        
    @tf.function
    def sample(self, x):
        """Draw one sample from the predictive distribution"""
        preds = self.call(x)
        return tfd.Normal(preds[:,0], preds[:,1]).sample()
    
    
    def samples(self, x, n_samples=1):
        """Draw multiple samples from the predictive distribution"""
        samples = np.zeros((x.shape[0], n_samples))
        for i in range(n_samples):
            samples[:,i] = self.sample(x)
        return samples
    
    
    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""
        return (self.core_net.losses +
                self.loc_net.losses +
                self.std_net.losses)

#%% instantiate dual head model

model2 = BayesianDensityNetwork([x_train.shape[0], 256, 128], [64, 32, x_train.shape[0]])
# Use the Adam optimizer
optimizer = tf.keras.optimizers.Adam(lr=L_RATE)

N = x_train.shape[0]

@tf.function
def train_step(x_data, y_data):
    with tf.GradientTape() as tape:
        log_likelihoods = model2.log_likelihood(x_data, y_data)
        kl_loss = model2.losses
        elbo_loss = kl_loss/N - tf.reduce_mean(log_likelihoods)
    gradients = tape.gradient(elbo_loss, model2.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model2.trainable_variables))
    return elbo_loss

#%%

