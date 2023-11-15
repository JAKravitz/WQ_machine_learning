#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 11:42:15 2021

@author: jakravit
"""
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from standard_clf.RFclassifier import RFclassifier
# from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('/Users/jakravit/data/cyanosat/cyanosat_12nm_resolved_synthetic_rrs_3nm.csv',index_col=0)
case = 'admix_full'

target = ['admix']

batch_info = {'targets': target,
              'n_estimators': 300,
              'min_samples_split': 4,
              'max_features': 4,
              'max_depth': 7,
              'n_jobs': -1,
              'cv': 0,
              'split': .1,
              'class_weight': 'balanced_subsample',
              'verbose':1}

model = RFclassifier(batch_info)
X, y = model.getXY(data)
results = model.prep_results(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=batch_info['split'])
# best_params = model.gridsearch(X_train, y_train)

print ('\n## FINAL MODEL ##\n')
model.build()
imp, std = model.fit(X_train, y_train)

# synthetic predict
y_hat, y_proba = model.predict(X_test)
results, yt, yh = model.evaluate(y_test,y_hat,y_proba,results,'final') 
results['batch_info'] = batch_info


# save run to disk
fname = '/Users/jakravit/data/cyanosat/cases/12_3nm/results_{}.p'.format(case)
f = open(fname,'wb')
pickle.dump(results,f)
f.close() 
#%% plot
import seaborn as sns

# View accuracy score
acc = accuracy_score(yt, yh)

# Get and reshape confusion matrix data
matrix = confusion_matrix(yt, yh)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['0','1','2','3','4','5','6','7','8','9','10']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()
            
#%%
imp = pd.Series(imp, )

fig, ax = plt.subplots()
imp.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
