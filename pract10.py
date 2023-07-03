# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:25:06 2023

@author: Aditi
"""

import keras
from keras.datasets import mnist
from keras import layers
import numpy as np
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

(x_train,_),(x_test,_)= mnist.load_data()
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = np.reshape(x_train, (len(x_train),28,28,1))
x_test = np.reshape(x_test, (len(x_test),28,28,1))

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy,0.,1.)
x_test_noisy = np.clip(x_test_noisy,0.,1.)
n = 10
plt.figure(figsize=(20,2))
for i in range(1, n +1):
    ax = plt.subplot(1,n,i)
    plt.imshow(x_test_noisy[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_im = keras.Input(shape=(28,28,1))
x = layers.Conv2D(33,(3,3), activation='relu', padding='same')(input_im)
x = layers.MaxPooling2D((2,2), padding='same')(x)
x = layers.Conv2D(32,(3,3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2,2), padding='same')(x)
x = layers.Conv2D(32,(3,3), activation='relu',padding='same')(encoded)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(32,(3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
decoded = layers.Conv2D(1,(3,3), activation='sigmoid', padding='same')(x)
autoencoder = keras.Model(input_im, decoded)
autoencoder.compile(optimizer ='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train, epochs=3, batch_size = 128, shuffle=True,validation_data=(x_test_noisy,x_test), callbacks=[TensorBoard(log_dir='/tmo/tb', histogram_freq=0, write_graph=False)])
predictions = autoencoder.predict(x_test_noisy)
m = 10
plt.figure(figsize=(20,2))
for i in range(1, m+1):
    ax = plt.subplot(1, m ,i)
    plt.imshow(predictions[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()