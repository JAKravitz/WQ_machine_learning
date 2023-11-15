#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 20:32:25 2023

@author: jakravit
"""
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

xdata = pd.read_csv('/Users/jakravit/Desktop/SWIPE_test_copy/emit_rrs.csv',index_col=0)
ydata = pd.read_csv('/Users/jakravit/Desktop/SWIPE_test_copy/BOA_inputs_case2n500.csv',index_col=0)
ydata2 = pd.read_csv('/Users/jakravit/Desktop/SWIPE_test_copy/targets_rrs.csv',index_col=0)
targets = ['Chlorophytes','Cryptophytes','Diatoms (centric)','Diatoms (pennate)','Dinoflagellates',
              'Prasinophytes','Pelagophytes','Haptophytes (Pavlovaceae)','Haptophytes (Prymnesiaceae)','Raphidophytes',
              'Eustigmatophytes','Cyanobacteria blue','Rhodophytes',]

t2 = [f'PFT_{c}_class_chl' for c in targets]
y = ydata[t2]

# Convert the class contributions to class labels (for classification)
# data["class"] = data[targets].idxmax(axis=1)


#%%

# Split the data into features (X) and targets (y)
X = data.drop(targets + ["class"], axis=1)
y = data["class"]

# label encoder
lab = LabelEncoder()
yt = lab.fit_transform(y)
      
# Normalize input data
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, yt, test_size=0.2, random_state=42)


#%%
model = RandomForestClassifier(n_estimators=300, 
                               min_samples_split=4,
                               max_depth=7,
                               max_features=4,
                               n_jobs=-1,
                               class_weight='balanced_subsample',
                               verbose=1)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# You can also get the probability of each class
probabilities = model.predict_proba(X_test)

# class names
predicted_classes= lab.inverse_transform(predictions)
