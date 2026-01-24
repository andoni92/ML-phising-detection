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

# OBTENER RUTA ABSOLUTA DEL DIRECTORIO ACTUAL
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Cargar Datos
try:
    path_legit = os.path.join(current_dir, 'datasets\structured_data_legitimate.csv')
    path_phish = os.path.join(current_dir, 'datasets\structured_data_phishing.csv')
    
    legitimate_df = pd.read_csv(path_legit)
    phishing_df = pd.read_csv(path_phish)
    
    # Combinar y mezclar
    df = pd.concat([legitimate_df, phishing_df], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Limpiar
    df = df.drop('URL', axis=1)
    df = df.drop_duplicates()
    
    X = df.drop('label', axis=1)
    Y = df['label']
    
    # Split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
    
    # 2. Definir Modelos
    svm_model = svm.LinearSVC()
    rf_model = RandomForestClassifier(n_estimators=60)
    dt_model = tree.DecisionTreeClassifier()
    ab_model = AdaBoostClassifier()
    nb_model = GaussianNB()
    nn_model = MLPClassifier(alpha=1, max_iter=1000)
    kn_model = KNeighborsClassifier()

    # Entrenar modelos iniciales (para exportar a App)
    svm_model.fit(x_train, y_train)
    rf_model.fit(x_train, y_train)
    dt_model.fit(x_train, y_train)
    ab_model.fit(x_train, y_train)
    nb_model.fit(x_train, y_train)
    nn_model.fit(x_train, y_train)
    kn_model.fit(x_train, y_train)

    # 3. Validación Cruzada Manual (K-Fold)
    def calculate_measures(TN, TP, FN, FP):
        accuracy = (TP + TN) / (TP + TN + FN + FP)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        return accuracy, precision, recall

    K = 5
    total = X.shape[0]
    index = int(total / K)
    
    # Listas para guardar splits
    X_train_list, X_test_list, Y_train_list, Y_test_list = [], [], [], []
    
    for i in range(K):
        # Lógica de slicing manual del PDF para K-Fold
        start = index * i
        end = index * (i + 1)
        if i == K - 1: end = total # Asegurar coger el resto
        
        X_test_fold = X.iloc[start:end]
        Y_test_fold = Y.iloc[start:end]
        
        X_train_fold = X.iloc[np.r_[:start, end:]]
        Y_train_fold = Y.iloc[np.r_[:start, end:]]
        
        X_train_list.append(X_train_fold)
        X_test_list.append(X_test_fold)
        Y_train_list.append(Y_train_fold)
        Y_test_list.append(Y_test_fold)

    # Listas de resultados
    models = [rf_model, dt_model, svm_model, ab_model, nb_model, nn_model, kn_model]
    model_names = ['RF', 'DT', 'SVM', 'AB', 'NB', 'NN', 'KN']
    results = {name: {'acc': [], 'pre': [], 'rec': []} for name in model_names}

    # Loop de entrenamiento K-Fold
    print("Iniciando Validación Cruzada...")
    for i in range(K):
        for model, name in zip(models, model_names):
            model.fit(X_train_list[i], Y_train_list[i])
            preds = model.predict(X_test_list[i])
            tn, fp, fn, tp = confusion_matrix(Y_test_list[i], preds, labels=[0, 1]).ravel()
            acc, pre, rec = calculate_measures(tn, tp, fn, fp)
            results[name]['acc'].append(acc)
            results[name]['pre'].append(pre)
            results[name]['rec'].append(rec)

    # Promedios y DataFrame final
    final_data = {'accuracy': [], 'precision': [], 'recall': []}
    for name in model_names:
        final_data['accuracy'].append(sum(results[name]['acc'])/K)
        final_data['precision'].append(sum(results[name]['pre'])/K)
        final_data['recall'].append(sum(results[name]['rec'])/K)

    df_results = pd.DataFrame(data=final_data, index=model_names)
    print(df_results)
    
    # Gráfica (opcional si se corre directo)
    if __name__ == "__main__":
        df_results.plot.bar(rot=0)
        plt.show()

except FileNotFoundError:
    print("No se encontraron los archivos CSV estructurados. Ejecuta data_collector.py primero.")