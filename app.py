# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 19:25:21 2023

@author: Aditi
"""
import pickle
import pandas as pd
import numpy as np
import streamlit as st

pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

def welcome():
    return 'welcome all'

def prediction(sepal_length, sepal_width, petal_length, petal_width):
    prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    print(prediction)
    
    return prediction

def main():
    st.title("iris flower prediction")
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML App </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    sepal_length = st.text_input("sepal length", "type Here")
    sepal_width = st.text_input("sepal width","type here")
    petal_length = st.text_input("petal length","type here")
    petal_width = st.text_input("petal width","type here")
    result = ""
    
    if st.button("predict"):
        result = prediction(sepal_length, sepal_width, petal_length, petal_width)
    st.success('the output is {}'.format(result))
    
if __name__=="__main__":
    main()
