# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:31:09 2023

@author: Aditi
"""

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt

(x_train, y_train),(x_test,y_test)= mnist.load_data()
plt.imshow(x_train[0])
plt.show()
print(x_train[0].shape)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train[0]
print(y_train[0])
model = Sequential()
model.add(Conv2D(64, kernel_size=3 , activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics =['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)
print(model.predict(x_test[:4]))
print(y_test[:4])