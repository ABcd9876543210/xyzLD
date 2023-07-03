# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:31:39 2023

@author: Aditi
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

df =pd.read_csv('D:/abhishek/model college/sem4/DL/housing.csv', delim_whitespace=True, header=None)

x = df.drop('MEDV', axis=1)
y = df['MEDV']

#print(x)
print(y)
def wider_model():
    model = Sequential()
    model.add(Dense(15,input_dim = 13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer ='adam')
    return model

estimator =[]
estimator.append(('standardize', StandardScaler()))
estimator.append(('mlp', KerasRegressor(build_fn=wider_model, epochs =100, batch_size=5)))
pipeline = Pipeline(estimator)
Kfold =KFold(n_splits=10)
results = cross_val_score(pipeline, x, y, cv=Kfold)
print("wider:%.2f(%.2f)MSE"%(results.mean(), results.std()))