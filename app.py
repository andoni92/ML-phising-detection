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
import machine_learning as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as re
import matplotlib.pyplot as plt

# Set up the main title and description
st.title('Phishing Website Detection using Machine Learning')
st.write('This ML-based app is developed for educational purposes.')

# Check if all required machine learning models have been trained and are available
# These models should be created by running machine_learning.py before starting the app
model_attrs = [
    'nb_model',   # Gaussian Naive Bayes
    'svm_model',  # Support Vector Machine
    'dt_model',   # Decision Tree
    'rf_model',   # Random Forest
    'ab_model',   # AdaBoost
    'nn_model',   # Neural Network
    'kn_model'    # K-Nearest Neighbors
]

# Verify that all models are accessible from the machine_learning module
models_ready = all(hasattr(ml, attr) for attr in model_attrs)

# Check if performance results DataFrame is available for display
df_results_available = hasattr(ml, 'df_results')

# Create an expandable section containing project information and results
with st.expander("PROJECT DETAILS"):
    st.subheader('Approach')
    st.write('I used supervised learning to classify phishing and legitimate websites using content-based features.')
    
    st.subheader('Data set')
    st.write('Sources: "phishtank.org" & "tranco-list.eu"')
    
    # Display a pie chart showing the distribution of phishing vs legitimate websites in the dataset
    try:
        labels = 'Phishing', 'Legitimate'
        phishing_len = ml.phishing_df.shape[0]
        legit_len = ml.legitimate_df.shape[0]
        sizes = [phishing_len, legit_len]
        explode = (0.1, 0)
        fig, ax = plt.subplots()
        ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    except:
        st.write("Dataframes not loaded correctly for visualization.")

    st.subheader('Results')
    if df_results_available:
        st.table(ml.df_results)
    else:
        st.info("Results table unavailable: training artifacts not found. Run machine_learning.py to generate metrics and models.")

# Allow the user to select which machine learning model to use for prediction
choice = st.selectbox("Please select your machine learning model",
    ['Gaussian Naive Bayes', 'Support Vector Machine', 'Decision Tree', 'Random Forest',
     'AdaBoost', 'Neural Network', 'K-Neighbours']
)

# Map the user's selection to the corresponding trained model object
model = None
if models_ready:
    if choice == 'Gaussian Naive Bayes': 
        model = ml.nb_model
    elif choice == 'Support Vector Machine': 
        model = ml.svm_model
    elif choice == 'Decision Tree': 
        model = ml.dt_model
    elif choice == 'Random Forest': 
        model = ml.rf_model
    elif choice == 'AdaBoost': 
        model = ml.ab_model
    elif choice == 'Neural Network': 
        model = ml.nn_model
    elif choice == 'K-Neighbours': 
        model = ml.kn_model

st.write(f'{choice} model is selected!')
if not models_ready:
    st.warning("Models are not available. Please run machine_learning.py to train and expose models.")

# Create text input for URL and prediction button
url = st.text_input('Enter the URL to check')

# When the user clicks the "Check!" button (disabled if models aren't ready)
if st.button('Check!', disabled=not models_ready):
    try:
        # Fetch the website content with SSL verification disabled and a 4-second timeout
        response = re.get(url, verify=False, timeout=4)
        
        # Check if the HTTP request was successful
        if response.status_code != 200:
            st.warning("HTTP connection was not successful.")
        else:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Extract feature vector from the parsed HTML (must be a 2D array for prediction)
            vector = [fe.create_vector(soup)]
            
            # Use the selected model to predict if the website is phishing or legitimate
            result = model.predict(vector)
            
            # Display the prediction result with visual feedback
            if result[0] == 0:
                # Label 0 indicates a legitimate website
                st.success("This web page seems Legitimate!")
                st.balloons()
            else:
                # Label 1 indicates a potential phishing website
                st.warning("Attention! This web page is potential PHISHING!")
                st.snow()
                
    except re.exceptions.RequestException as e:
        # Handle any errors that occur during the HTTP request
        st.error(f"Error: {e}")