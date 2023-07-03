# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:20:49 2023

@author: Aditi
"""

from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

x,y = make_regression(n_samples=100, n_features=2,noise=0.1,random_state=1)
scalarx, scalary = MinMaxScaler(), MinMaxScaler()

scalarx.fit(x)
scalary.fit(y.reshape(100, 1))
x = scalarx.transform(x)
y = scalary.transform(y.reshape(100, 1))

model = Sequential()
model.add(Dense(4, input_dim =2, activation ='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, verbose=0)

xnew, a = make_regression(n_samples=3, n_features=2,noise=0.1,random_state=1)
xnew = scalarx.transform(xnew)

ynew = model.predict(xnew)

for i in range(len(xnew)):
    print("x =%s, Predicted=%s"%(xnew[i], ynew[i]))
