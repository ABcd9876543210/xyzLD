# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 08:45:05 2023

@author: Aditi
"""

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

data = loadtxt('pima-indians-diabetes.csv',delimiter=',')
data
x = data[:,0:8]
y = data[:,8]
x
y
model = Sequential()
model.add(Dense(12, input_dim=8,activation ='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x,y, epochs=150,batch_size=10)
_,accuracy = model.evaluate(x,y)
print('Acurracy of model is',(accuracy*100))

prediction = model.predict(x)

exec("for i in range(5):print(x[i].tolist(),prediction[i],y[i])")