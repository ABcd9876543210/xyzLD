# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:15:30 2023

@author: Aditi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('D:/abhishek/model college/sem4/DL/Google_Stock_Price_Train.csv')
train_set = df.iloc[:,1:2].values
sc = MinMaxScaler(feature_range=(0,1))
train_set_scal = sc.fit_transform(train_set)
x_train =[]
y_train =[]
for i in range(60, 1258):
    x_train.append(train_set_scal[i - 60:i,0])
    y_train.append(train_set_scal[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train)
print('*************************')
print(y_train)
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))
print('*************************')
print(x_train)
regrssor = Sequential()
regrssor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
regrssor.add(Dropout(0.2))
regrssor.add(LSTM(units=50, return_sequences=True))
regrssor.add(Dropout(0.2))
regrssor.add(LSTM(units=50, return_sequences=True))
regrssor.add(Dropout(0.2))
regrssor.add(LSTM(units=50))
regrssor.add(Dropout(0.2))
regrssor.add(Dense(units=1))
regrssor.compile(optimizer='adam', loss='mean_squared_error')
regrssor.fit(x_train, y_train, epochs=100, batch_size=32)

dataset_test = pd.read_csv('D:/abhishek/model college/sem4/DL/Google_Stock_Price_Train.csv')
real_stock_price = dataset_test.iloc[:,1:2].values
dataset_total = pd.concat((df['Open'],dataset_test['Open']), axis=0)
iputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
iputs = iputs.reshape(-1,1)
iputs= sc.transform(iputs)
x_test =[]
for i in range(60,80):
    x_test.append(iputs[i - 60 :i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1],1))
predicted_stock_price = regrssor.predict(x_test) 
predicted_stock_price= sc.inverse_transform(predicted_stock_price)
plt.plot(real_stock_price, color='red',label='real google stock price')
plt.plot(predicted_stock_price, color='blue', label='predicted stock price')
plt.xlabel('time')
plt.ylabel('google stock price')
plt.legend()
plt.show()   

