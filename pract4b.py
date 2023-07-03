# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 09:38:54 2023

@author: Aditi
"""

from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
x,y = make_blobs(n_samples=100, n_features=2, random_state=1)

scalar = MinMaxScaler()
scalar.fit(x)
x = scalar.transform(x)

model = Sequential()
model.add(Dense(4, activation='relu', input_dim=2))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer ='adam')
model.fit(x,y, epochs=500)
xnew, yreal = make_blobs(n_samples=3,n_features=2, centers=2, random_state=1)

xnew = scalar.transform(xnew)

yclass = model.predict(xnew)
ynew= model.predict_on_batch(xnew)

for i in range(len(xnew)):
    print("x=%s.predicted_probablity=%s, predicted_class%s"%(xnew[i],ynew[i],yclass[i]))