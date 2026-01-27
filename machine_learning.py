"""Machine Learning Model Training and Evaluation for Phishing Detection.

This module trains and evaluates multiple machine learning models for phishing website
detection. It implements K-Fold cross-validation to assess model performance and
exposes trained models for use in the Streamlit web application.

Models Implemented:
    - Random Forest (RF)
    - Decision Tree (DT)
    - Support Vector Machine (SVM)
    - AdaBoost (AB)
    - Gaussian Naive Bayes (NB)
    - Neural Network/Multi-layer Perceptron (NN)
    - K-Nearest Neighbors (KN)

The script performs the following steps:
    1. Load and combine legitimate and phishing datasets
    2. Preprocess data (shuffle, clean, split features/labels)
    3. Train all models on initial train/test split
    4. Perform manual K-Fold cross-validation
    5. Calculate and display accuracy, precision, and recall metrics
    6. Generate performance comparison visualizations

Output:
    - Trained model objects (accessible as module attributes)
    - Performance metrics DataFrame (df_results)
    - Optional bar chart visualization when run as main script

Usage:
    Run directly: python machine_learning.py
    Import in other scripts: import machine_learning as ml
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

# Get the absolute path of the current script's directory
# This ensures correct file paths regardless of where the script is run from
current_dir = os.path.dirname(os.path.abspath(__file__))

# ========== 1. DATA LOADING AND PREPROCESSING ==========
try:
    # Construct paths to the structured datasets
    path_legit = os.path.join(current_dir, 'datasets\structured_data_legitimate.csv')
    path_phish = os.path.join(current_dir, 'datasets\structured_data_phishing.csv')
    
    # Load the datasets into pandas DataFrames
    legitimate_df = pd.read_csv(path_legit)
    phishing_df = pd.read_csv(path_phish)
    
    # Combine both datasets and shuffle to avoid any ordering bias
    df = pd.concat([legitimate_df, phishing_df], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)  # frac=1 means shuffle 100% of data
    
    # Clean the dataset by removing URL column and duplicate entries
    df = df.drop('URL', axis=1)  # URL is not a useful feature for ML
    df = df.drop_duplicates()     # Remove any duplicate records
    
    # Separate features (X) from labels (Y)
    X = df.drop('label', axis=1)  # Features: all columns except 'label'
    Y = df['label']                # Labels: 0 = legitimate, 1 = phishing
    
    # Split data into training and testing sets (80% train, 20% test)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
    
    # ========== 2. MODEL DEFINITION ==========
    # Initialize all machine learning models with appropriate hyperparameters
    
    svm_model = svm.LinearSVC()  # Linear Support Vector Classifier
    rf_model = RandomForestClassifier(n_estimators=60)  # 60 decision trees
    dt_model = tree.DecisionTreeClassifier()  # Single decision tree
    ab_model = AdaBoostClassifier()  # Adaptive Boosting ensemble
    nb_model = GaussianNB()  # Gaussian Naive Bayes (assumes normal distribution)
    nn_model = MLPClassifier(alpha=1, max_iter=1000)  # Neural Network with L2 regularization
    kn_model = KNeighborsClassifier()  # K-Nearest Neighbors (default k=5)

    # Train all models on the initial train/test split
    # These trained models will be accessible to the Streamlit app
    svm_model.fit(x_train, y_train)
    rf_model.fit(x_train, y_train)
    dt_model.fit(x_train, y_train)
    ab_model.fit(x_train, y_train)
    nb_model.fit(x_train, y_train)
    nn_model.fit(x_train, y_train)
    kn_model.fit(x_train, y_train)

    # ========== 3. K-FOLD CROSS-VALIDATION ==========
    # Implement manual K-Fold cross-validation to evaluate model performance
    
    def calculate_measures(TN, TP, FN, FP):
        """Calculate performance metrics from confusion matrix values.
        
        Args:
            TN (int): True Negatives - correctly classified as legitimate
            TP (int): True Positives - correctly classified as phishing
            FN (int): False Negatives - phishing incorrectly classified as legitimate
            FP (int): False Positives - legitimate incorrectly classified as phishing
        
        Returns:
            tuple: (accuracy, precision, recall) where:
                - accuracy: Overall correctness (TP + TN) / total
                - precision: Of predicted phishing, how many are actually phishing TP / (TP + FP)
                - recall: Of actual phishing, how many were detected TP / (TP + FN)
        
        Note:
            Returns 0 for precision/recall if denominator would be zero.
        """
        accuracy = (TP + TN) / (TP + TN + FN + FP)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        return accuracy, precision, recall

    # Configure K-Fold parameters
    K = 5  # Number of folds for cross-validation
    total = X.shape[0]  # Total number of samples
    index = int(total / K)  # Size of each fold
    
    # Lists to store the K train/test splits
    X_train_list, X_test_list, Y_train_list, Y_test_list = [], [], [], []
    
    # Create K different train/test splits by rotating which portion is used for testing
    for i in range(K):
        # Calculate the start and end indices for the test fold
        start = index * i
        end = index * (i + 1)
        
        # For the last fold, ensure we include all remaining samples
        if i == K - 1: 
            end = total
        
        # Extract test fold (one K-th portion of the data)
        X_test_fold = X.iloc[start:end]
        Y_test_fold = Y.iloc[start:end]
        
        # Extract train fold (all data except the test portion)
        # np.r_ concatenates slices: [0:start] and [end:total]
        X_train_fold = X.iloc[np.r_[:start, end:]]
        Y_train_fold = Y.iloc[np.r_[:start, end:]]
        
        # Store this fold's train/test split
        X_train_list.append(X_train_fold)
        X_test_list.append(X_test_fold)
        Y_train_list.append(Y_train_fold)
        Y_test_list.append(Y_test_fold)

    # Prepare lists for storing results
    models = [rf_model, dt_model, svm_model, ab_model, nb_model, nn_model, kn_model]
    model_names = ['RF', 'DT', 'SVM', 'AB', 'NB', 'NN', 'KN']
    
    # Dictionary to store metrics for each model across all K folds
    results = {name: {'acc': [], 'pre': [], 'rec': []} for name in model_names}

    # ========== 4. TRAINING AND EVALUATION LOOP ==========
    # Train each model on each of the K folds and calculate metrics
    print("Starting Cross-Validation...")
    
    for i in range(K):
        for model, name in zip(models, model_names):
            # Train the model on this fold's training data
            model.fit(X_train_list[i], Y_train_list[i])
            
            # Make predictions on this fold's test data
            preds = model.predict(X_test_list[i])
            
            # Calculate confusion matrix values
            # labels=[0, 1] ensures: TN, FP, FN, TP order for legitimate=0, phishing=1
            tn, fp, fn, tp = confusion_matrix(Y_test_list[i], preds, labels=[0, 1]).ravel()
            # Calculate metrics from confusion matrix
            acc, pre, rec = calculate_measures(tn, tp, fn, fp)
            
            # Store metrics for this fold
            results[name]['acc'].append(acc)
            results[name]['pre'].append(pre)
            results[name]['rec'].append(rec)

    # ========== 5. AGGREGATE RESULTS AND CREATE DATAFRAME ==========
    # Calculate average metrics across all K folds for each model
    final_data = {'accuracy': [], 'precision': [], 'recall': []}
    
    for name in model_names:
        # Average each metric across all K folds
        final_data['accuracy'].append(sum(results[name]['acc'])/K)
        final_data['precision'].append(sum(results[name]['pre'])/K)
        final_data['recall'].append(sum(results[name]['rec'])/K)

    # Create a DataFrame with models as rows and metrics as columns
    df_results = pd.DataFrame(data=final_data, index=model_names)
    print(df_results)
    
    # ========== 6. VISUALIZATION (optional - only when run directly) ==========
    # Display a bar chart comparing model performance
    if __name__ == "__main__":
        df_results.plot.bar(rot=0)
        plt.show()

except FileNotFoundError:
    print("Structured CSV files not found. Please run data_collector_legit.py first.")