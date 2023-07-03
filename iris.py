# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 18:32:08 2023

@author: Aditi
"""

import pandas as pd
import numpy as np

df = pd.read_csv('D:/abhishek/model college/sem4/DL/flowers.csv')
print(df.head(50))
print(df.info())
df['species'] = df['species'].map({'setosa':0,'versicolor':1,'virginica':2})

x = df.iloc[:,:-1]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)

y_prd = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_prd)
print(score)

import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()
