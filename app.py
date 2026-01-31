"""Streamlit Web Application for Phishing Website Detection.

This application provides a user-friendly interface for detecting phishing websites
using various machine learning models. It extracts features from a given URL's HTML
content and predicts whether the website is legitimate or potentially phishing.

Features:
    - Multiple ML model selection (Naive Bayes, SVM, Decision Tree, Random Forest, etc.)
    - Real-time URL analysis
    - Visual feedback with balloons/snow animations
    - Model performance metrics display
    - Dataset distribution visualization

Author: Andoni Gonzalez
Date: 2026
"""

import streamlit as st
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as re
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

# Set up the main title and description
st.title('Phishing Website Detection using Machine Learning')
st.write('This ML-based app is developed for educational purposes.')

# --- LOADING MODELS AND DATA ---

# Function to load models with caching (to avoid reloading frequently)
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Gaussian Naive Bayes': 'NB_model.pkl',
        'Support Vector Machine': 'SVM_model.pkl',
        'Decision Tree': 'DT_model.pkl',
        'Random Forest': 'RF_model.pkl',
        'AdaBoost': 'AB_model.pkl',
        'Neural Network': 'NN_model.pkl',
        'K-Neighbours': 'KN_model.pkl'
    }
    
    # Directory where we saved the models in the previous step
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    for name, filename in model_files.items():
        try:
            path = os.path.join(models_dir, filename)
            with open(path, 'rb') as file:
                models[name] = pickle.load(file)
        except FileNotFoundError:
            st.error(f"Error: The file {filename} was not found. Please run machine_learning.py first.")
            return None
    return models

# Load the models
loaded_models = load_models()
models_ready = loaded_models is not None

# Try to load the metrics (df_results) from the CSV
try:
    df_results = pd.read_csv('metrics.csv', index_col=0)
    df_results_available = True
except FileNotFoundError:
    df_results_available = False

# --- GRAPHICAL INTERFACE ---

with st.expander("PROJECT DETAILS"):
    st.subheader('Approach')
    st.write('I used supervised learning to classify phishing and legitimate websites using content-based features.')
    
    st.subheader('Data set')
    st.write('Sources: "phishtank.org" & "tranco-list.eu"')
    
    # Pie Chart
    # Load only what is necessary for the chart
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_legit = os.path.join(current_dir, 'datasets', 'structured_data_legitimate.csv')
        path_phish = os.path.join(current_dir, 'datasets', 'structured_data_phishing.csv')
        
        # Read only if files exist to count rows
        if os.path.exists(path_legit) and os.path.exists(path_phish):
            # Use chunksize or nrows if very large, but here we read directly
            legit_len = len(pd.read_csv(path_legit, usecols=[0]))
            phishing_len = len(pd.read_csv(path_phish, usecols=[0]))
            
            labels = 'Phishing', 'Legitimate'
            sizes = [phishing_len, legit_len]
            explode = (0.1, 0)
            fig, ax = plt.subplots()
            ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.write("Dataset files not found for visualization.")
    except Exception as e:
        st.write(f"Could not load dataset visualization: {e}")

    st.subheader('Results')
    if df_results_available:
        st.table(df_results)
    else:
        st.info("Results table unavailable: metrics.csv not found.")

# Model selector
choice = st.selectbox("Please select your machine learning model",
    ['Gaussian Naive Bayes', 'Support Vector Machine', 'Decision Tree', 'Random Forest',
     'AdaBoost', 'Neural Network', 'K-Neighbours']
)

# Select the loaded model from the dictionary
model = None
if models_ready:
    model = loaded_models.get(choice)

st.write(f'{choice} model is selected!')
if not models_ready:
    st.warning("Models are not available. Please ensure .pkl files are in the 'models/' directory.")

# URL input and prediction
url = st.text_input('Enter the URL to check')

if st.button('Check!', disabled=not models_ready):
    try:
        response = re.get(url, verify=False, timeout=4)
        
        if response.status_code != 200:
            st.warning("HTTP connection was not successful.")
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            vector = [fe.create_vector(soup)]
            
            # Prediction
            result = model.predict(vector)
            
            if result[0] == 0:
                st.success("This web page seems Legitimate!")
                st.balloons()
            else:
                st.warning("Attention! This web page is potential PHISHING!")
                st.snow()
                
    except re.exceptions.RequestException as e:
        st.error(f"Error: {e}")