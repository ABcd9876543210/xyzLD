# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:29:23 2023

@author: Aditi
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('D:/abhishek/model college/sem4/DL/flowers.csv')
print(df)
x = df.iloc[:,0:4].astype(float)
y = df.iloc[:,4]
encoder = LabelEncoder()
encoder.fit(y)
encoder_y = encoder.transform(y)
print(encoder_y)
dummy = np_utils.to_categorical(encoder_y)
print(dummy)

def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model
estimator = baseline_model()
estimator.fit(x, dummy, epochs=100, shuffle=True)
action = estimator.predict(x)
for i in range(25):
    print(dummy[i])
print('^^^^^^^^^^^^^^^^^^')

for i in range(25):
    print(action[i])