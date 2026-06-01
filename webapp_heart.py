# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:59:06 2023

@author: Subhasree
"""

import numpy as np
import pickle
import streamlit as st
loaded_model=pickle.load(open('C:/Users/Subhasree/Downloads/trained_model.sav','rb'))
def heart_predict(input_data):

    input_data_as_numpy_array=np.array(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    if(prediction[0]==0):
        return("healthy")
    else:
        return("heart disease")
def main():
    st.title("heart disease prediction")
    Age=st.text_input("age of person")
    sex=st.text_input("sex of person")
    cp=st.text_input("cp score")
    trsbs=st.text_input("trscp")
    chol=st.text_input("cholestrol value person")
    fbs=st.text_input("fbs")
    ecg=st.text_input("ecg measure")
    thalach=st.text_input("thal value")
    exang=st.text_input("exang")
    oldpeak=st.text_input("oldpeak of person")
    slope=st.text_input("slope test")
    ca=st.text_input("ca of person")
    thal=st.text_input("thal of person")
    
diagnosis=" "
if st.button('testresult'):
    diagnosis = heart_predict([Age,sex,cp,trsbs,chol,fbs,ecg,thalach,exang,oldpeak,slope,ca,thal])
st.success(diagnosis)
if __name__ =='__main__':
    
    main()