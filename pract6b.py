# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:54:16 2023

@author: Aditi
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
x,y = make_moons(n_samples=100, noise=0.2, random_state=1)
n_train = 30
x_train, x_test = x[:n_train,:],x[n_train:]
y_train, y_test = y[:n_train],y[n_train:]

model = Sequential()
model.add(Dense(500,input_dim=2,activation='relu', kernel_regularizer=l2(0.0001)))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=4000)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'],label='test')
plt.legend()
plt.show()