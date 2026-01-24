import streamlit as st
import machine_learning as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as re
import matplotlib.pyplot as plt

st.title('Phishing Website Detection using Machine Learning')
st.write('This ML-based app is developed for educational purposes.')

# Determine availability of training artifacts
model_attrs = [
    'nb_model', 'svm_model', 'dt_model', 'rf_model', 'ab_model', 'nn_model', 'kn_model'
]
models_ready = all(hasattr(ml, attr) for attr in model_attrs)
df_results_available = hasattr(ml, 'df_results')

with st.expander("PROJECT DETAILS"):
    st.subheader('Approach')
    st.write('I used supervised learning to classify phishing and legitimate websites using content-based features.')
    
    st.subheader('Data set')
    st.write('Sources: "phishtank.org" & "tranco-list.eu"')
    
    # Gráfico de pastel de distribución de datos
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

# Selección del modelo
choice = st.selectbox("Please select your machine learning model",
    ['Gaussian Naive Bayes', 'Support Vector Machine', 'Decision Tree', 'Random Forest',
     'AdaBoost', 'Neural Network', 'K-Neighbours']
)

model = None
if models_ready:
    if choice == 'Gaussian Naive Bayes': model = ml.nb_model
    elif choice == 'Support Vector Machine': model = ml.svm_model
    elif choice == 'Decision Tree': model = ml.dt_model
    elif choice == 'Random Forest': model = ml.rf_model
    elif choice == 'AdaBoost': model = ml.ab_model
    elif choice == 'Neural Network': model = ml.nn_model
    elif choice == 'K-Neighbours': model = ml.kn_model

st.write(f'{choice} model is selected!')
if not models_ready:
    st.warning("Models are not available. Please run machine_learning.py to train and expose models.")

# Input y Predicción
url = st.text_input('Enter the URL to check')

if st.button('Check!', disabled=not models_ready):
    try:
        response = re.get(url, verify=False, timeout=4)
        if response.status_code != 200:
            st.warning("HTTP connection was not successful.")
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            vector = [fe.create_vector(soup)] # Debe ser 2d array
            result = model.predict(vector)
            
            if result[0] == 0:
                st.success("This web page seems Legitimate!")
                st.balloons()
            else:
                st.warning("Attention! This web page is potential PHISHING!")
                st.snow()
                
    except re.exceptions.RequestException as e:
        st.error(f"Error: {e}")