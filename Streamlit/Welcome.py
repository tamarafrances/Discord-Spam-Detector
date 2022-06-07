import streamlit as st
st.set_page_config(layout="wide")
st.title('Discord Spam Detector')
st.subheader('Tamara Frances')
import numpy as np
import time
import pickle

with open("saved_model.pkl", "rb") as f:
    trained_classifier = pickle.load(f)

import numpy as np


def predict_spam(trained_classifier, text_input):
    return trained_classifier.predict([text_input])

st.header('Test Your Message')

with st.form(key='my_form'):
    text_input = st.text_input(label='Enter your message below')
    submit_button = st.form_submit_button(label='Predict')
    if text_input != '':
        with st.spinner("Predicting..."):
            time.sleep(2)
            prediction = predict_spam(trained_classifier, text_input)
            if prediction == 'N':
                st.write('Your message was not flagged as spam!')
                st.balloons()
            else:
                st.subheader('FLAGGED AS SPAM')