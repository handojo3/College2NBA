import streamlit as st
import pickle
import numpy as np

def load_model():
    

def show_predict_page():
    st.title("College2NBA")
    st.subheader("Predictor")

    with st.form(key = 'form1'):
        playername = st.text_input("Player Name:")
        years = st.slider("Year", 2008, 2021)
        submit_button = st.form_submit_button(label = 'Calculate')

    if submit_button:
