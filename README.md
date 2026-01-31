# Machine Learning for Phishing Website Detection

An educational project that detects potential phishing websites using supervised machine learning and content-based HTML features. It includes:
- A data collection script to build structured datasets from raw URL lists
- Multiple ML models trained on extracted HTML features
- A Streamlit app to test URLs interactively

> Disclaimer: This project is for educational purposes. Do not rely on it as a security product. Visiting unknown URLs can be risky; proceed with caution.

Accesible at: [ml-phising-detection.streamlit.app](https://ml-phising-detection.streamlit.app/)

---

## 🫣 Overview
- **Approach:** Supervised classification of phishing vs. legitimate websites using features extracted from HTML via BeautifulSoup.
- **Models:** Linear SVM, Random Forest, Decision Tree, AdaBoost, Gaussian Naive Bayes, MLP (Neural Network), K-Nearest Neighbors.
- **Sources:** Phishtank (verified phishing) and Tranco (top websites).

---

## 📂 Project Structure
```
ML-phising-detection/
├─ app.py                      # Streamlit app for interactive predictions
├─ machine_learning.py         # Training, manual K-Fold CV, model export
├─ feature_extraction.py       # Convert BeautifulSoup to feature vector
├─ features.py                 # Individual feature functions
├─ data_collector_legit.py     # Dataset builder (legit or phishing)
├─ environment.yml             # Conda environment with dependencies
├─ metrics.csv                 # Metrics of the trained models
├─ README.md                   # This file
├─ models/					   # Trained models
│    ├─ AB_model.pkl
│	 └─ ...             
└─ datasets/
	 ├─ structured_data_legitimate.csv
	 ├─ structured_data_phishing.csv
	 ├─ top-1m.csv               # Tranco list (legitimate candidates)
	 └─ verified_online.csv      # Phishtank verified (phishing candidates)
```

---

## 🖥️ Setup (Windows + Conda)
Prerequisites:
- Miniconda/Anaconda installed
- Internet access

Create and activate the environment:
```powershell
conda env create -f environment.yml
conda activate ML-phising-detection
```

Verify versions (optional):
```powershell
python --version
python -c "import sklearn, bs4, pandas, streamlit; print('OK')"
```

---

## Data Collection
The training scripts expect two structured CSVs built from raw URL lists.

1) Place raw datasets under `datasets/`:
- `top-1m.csv` (Tranco list): https://tranco-list.eu/
- `verified_online.csv` (Phishtank): https://phishtank.org/

2) Configure `data_collector_legit.py` to choose which dataset to build:
- For legitimate data (default):
	```python
	url_filename = r"datasets\top-1m.csv"
	output_filename = r"structured_data_legitimate.csv"
	label_value = 0
	```
- For phishing data (uncomment the phishing block and comment the legit block):
	```python
	# url_filename = r"datasets\verified_online.csv"
	# output_filename = r"structured_data_phishing.csv"
	# label_value = 1
	```

3) Run the collector (it processes ~1,000 URLs by default):
```powershell
python data_collector_legit.py
```

Notes:
- The script uses `requests.get(..., verify=False, timeout=4)` to be resilient; TLS verification is disabled for collection convenience.
- Errors and timeouts are handled per-URL to avoid stopping the run.

---

## 🦾 Training & Evaluation
Train models and compute manual K-Fold metrics:
```powershell
python machine_learning.py
```
This script:
- Loads `structured_data_legitimate.csv` and `structured_data_phishing.csv`
- Builds features and labels, removes `URL`, deduplicates
- Trains 7 models and performs manual K-Fold ($K=5$) cross-validation
- Prints a summary DataFrame (`accuracy`, `precision`, `recall`) per model
- Optionally shows a bar chart if run directly

If the structured CSV files are missing, it will print:
```
No se encontraron los archivos CSV estructurados. Ejecuta data_collector.py primero.
```
Generate the data as described above before training or running the app.

---

## 👑 Streamlit App (Interactive)
Launch the UI:
```powershell
streamlit run app.py
```
Features:
- Displays dataset distribution and evaluation table (if training data is present)
- Lets you select a model (SVM, RF, DT, AdaBoost, NB, NN, KNN)
- Accepts a URL, fetches the page, extracts features, and predicts

Behaviors:
- Uses `requests` with `verify=False` and a short timeout to fetch pages
- Shows success/warning feedback based on prediction
- If dataframes/models aren’t loaded (missing CSVs), the app will warn and may not allow predictions

---

## ⛲ Feature Set
From `features.py` (binary and quantitative):
- Binary: `has_title`, `has_input`, `has_button`, `has_image`, `has_submit`, `has_link`, `has_password`, `has_email_input`, `has_hidden_element`, `has_audio`, `has_video`
- Counts: `number_of_inputs`, `number_of_buttons`, `number_of_images` (including meta image), `number_of_option`, `number_of_list`, `number_of_th`, `number_of_tr`, `number_of_href`, `number_of_paragraph`, `number_of_script`, `length_of_title`

Vectors are built in `feature_extraction.py` and consumed by both the training script and the app.

---

## 🐞 Troubleshooting
- Missing structured CSVs: Build datasets via `data_collector_legit.py` first.
- Request errors/timeouts: The collector/app log per-URL issues; continue with available pages.
- Streamlit shows "Dataframes not loaded correctly": Train first (`machine_learning.py`) so `ml.df_results` exists.
- SSL warnings: Disabled in collection for convenience; re-enable for production use.

---

## References
- Phishtank (verified phishing feed): https://phishtank.org/
- Tranco list (top websites): https://tranco-list.eu/
