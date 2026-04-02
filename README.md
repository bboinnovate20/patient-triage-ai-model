
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-189AB4?style=flat-square)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-2E86C1?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-88%25-2ECC71?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)

# Symptom-Based Patient Triage Using Machine Learning

> An AI-powered triage system that classifies patient-reported symptoms into urgency levels to improve healthcare access for rural and low-income populations.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Triage Label Mapping](#triage-label-mapping)
- [Model Comparison](#model-comparison)
- [Streamlit App](#streamlit-app)
- [Limitations](#limitations)
- [References](#references)

---

## Overview

This project develops a machine learning classification system that takes patient-reported symptoms and demographic information as input and outputs one of three urgency triage levels:

| Label | Description |
|---|---|
| Self-Care | Mild condition manageable at home |
| GP Appointment | Requires a scheduled GP visit within 1–5 days |
| Emergency | Requires same-day or emergency care |

The system is deployed as a lightweight Streamlit web application accessible from any device without installation.

---

## Problem Statement

Access to timely medical consultation remains a critical challenge for millions of people, particularly in rural areas and low-income communities. Long waiting times, geographic barriers, and shortage of healthcare professionals prevent patients from receiving appropriate care early. This delay often results in conditions escalating into serious, costly, or life-threatening situations.

This project addresses that gap by providing an intelligent first point of contact that directs patients to the appropriate level of care quickly, consistently, and without cost barriers.

---

## System Architecture

```
Patient Input (Symptoms + Demographics)
            │
            ▼
   Feature Preprocessing
   (Encoding + Scaling)
            │
            ▼
   Random Forest Classifier
   (Trained on merged dataset)
            │
            ▼
   Triage Label Output
   Self-Care / GP Appointment / Emergency
            │
            ▼
   Streamlit Web Application
```

---

## Dataset

Two publicly available datasets were combined to enrich feature coverage:

### Dataset 1 — Disease Symptoms and Patient Profile Dataset
- Source: [Kaggle — uom190346a](https://www.kaggle.com/datasets/uom190346a/disease-symptoms-and-patient-profile-dataset)
- Records: 3,490
- Features: Fever, Cough, Fatigue, Difficulty Breathing, Age, Gender, Blood Pressure, Cholesterol Level
- Used in: Sogandi, F. (2024). *Scientific Reports*, 14, 17956.

### Dataset 2 — Disease Symptom Prediction Dataset
- Source: [Kaggle — itachi9604](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
- Records: 4,920
- Features: 17 symptom columns (Symptom_1 to Symptom_17), 41 disease labels
- Supporting files: `symptom_severity.csv`, `symptom_description.csv`, `symptom_precaution.csv`

### Combined Dataset
- Total records: ~8,410
- Total features: 130+ binary symptom columns + 4 demographic columns
- Target variable: `Triage_Label` (3 classes)

---

## Installation

### Prerequisites
- Python 3.8+
- pip

### Clone the repository
```bash
git clone https://github.com/yourusername/symptom-triage-ml.git
cd symptom-triage-ml
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### requirements.txt
```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
streamlit
joblib
shap
```

---

## Usage

### Run the Streamlit App
```bash
streamlit run app.py
```

### Train the model from scratch
```bash
python train.py
```

### Run evaluation
```bash
python evaluate.py
```

---

## Model Performance

Three models were trained and compared. Random Forest was selected as the final model.

### Test Set Results

| Rank | Model | Accuracy | Precision | Recall | F1 (Macro) | Time (s) |
|---|---|---|---|---|---|---|
| 1 | Random Forest | 0.8777 | 0.8922 | 0.8776 | 0.8798 | 0.33 |
| 2 | XGBoost | 0.8511 | 0.8660 | 0.8510 | 0.8529 | 0.41 |
| 3 | Logistic Regression | 0.6223 | 0.6298 | 0.6217 | 0.6199 | 0.04 |

### Cross-Validation Results (5-Fold Stratified)

| Model | CV F1 (Mean) | CV F1 (Std) |
|---|---|---|
| Random Forest | 0.8414 | 0.0073 |
| XGBoost | 0.8435 | 0.0252 |
| Logistic Regression | 0.6337 | 0.0299 |

### Confusion Matrix — Random Forest

```
                  Predicted
                  Emergency  GP Appt  Self-Care
Actual Emergency     54         8         0
       GP Appt        2        59         2
       Self-Care      1        10        52
```

**Key findings:**
- Zero Emergency cases misclassified as Self-Care (most dangerous error never occurred)
- Emergency recall: 87.1%
- GP Appointment recall: 93.7% (strongest class)
- Self-Care recall: 82.5%

---

## Project Structure

```
symptom-triage-ml/
│
├── data/
│   ├── dataset1_patient_profile.csv
│   ├── dataset2_disease_symptom.csv
│   ├── symptom_severity.csv
│   ├── symptom_description.csv
│   └── symptom_precaution.csv
│
├── notebooks/
│   └── triage_model.ipynb          # Full development notebook
│
├── models/
│   └── random_forest_pipeline.pkl  # Serialised trained model
│
├── app.py                          # Streamlit application
├── train.py                        # Model training script
├── evaluate.py                     # Evaluation script
├── preprocess.py                   # Preprocessing functions
├── requirements.txt
└── README.md
```

---

## Preprocessing Pipeline

### Step 1 — Dataset 1 Cleaning
```python
# Rename and encode symptom columns
df1.rename(columns={
    'Fever': 'fever', 'Cough': 'cough',
    'Fatigue': 'fatigue', 'Difficulty Breathing': 'difficulty_breathing'
}, inplace=True)

df1['Gender'] = df1['Gender'].map({'Male': 1, 'Female': 0})
df1['Blood Pressure'] = df1['Blood Pressure'].map({'Low': 0, 'Normal': 1, 'High': 2})
df1['Cholesterol Level'] = df1['Cholesterol Level'].map({'Normal': 0, 'High': 1})
df1.drop(columns=['Outcome Variable'], inplace=True)
```

### Step 2 — Dataset 2 Binarisation
```python
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
symptom_binary = pd.DataFrame(
    mlb.fit_transform(df2['symptom_list']),
    columns=mlb.classes_
)
```

### Step 3 — Merging
```python
combined = pd.concat([df1, df2_binarised], axis=0, ignore_index=True)

# Impute missing demographics
combined['Age'] = combined['Age'].fillna(combined['Age'].median())
combined['Gender'] = combined['Gender'].fillna(combined['Gender'].mode()[0])
combined['Blood Pressure'] = combined['Blood Pressure'].fillna(combined['Blood Pressure'].mode()[0])
combined['Cholesterol Level'] = combined['Cholesterol Level'].fillna(combined['Cholesterol Level'].mode()[0])
```

### Step 4 — Triage Label Mapping
```python
triage_mapping = {
    'Common Cold': 'Self-Care', 'Allergy': 'Self-Care',
    'Influenza': 'Self-Care', 'Eczema': 'Self-Care',
    'Diabetes': 'GP Appointment', 'Hypertension': 'GP Appointment',
    'Asthma': 'GP Appointment', 'Arthritis': 'GP Appointment',
    'Heart Disease': 'Emergency', 'Stroke': 'Emergency',
    'Pneumonia': 'Emergency', 'Malaria': 'Emergency',
    # ... full mapping in preprocess.py
}
combined['Triage_Label'] = combined['Disease'].map(triage_mapping)
```

### Step 5 — SMOTE and Split
```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
```

---

## Triage Label Mapping

Diseases were mapped to triage categories using NHS triage guidelines and the Manchester Triage System:

| Triage Level | Example Diseases | Clinical Rationale |
|---|---|---|
| Self-Care | Common Cold, Allergy, Acne, Eczema, Influenza | Mild, self-limiting conditions |
| GP Appointment | Diabetes, Hypertension, Asthma, Arthritis, Migraine | Chronic conditions needing monitoring |
| Emergency | Heart Disease, Stroke, Pneumonia, Malaria, Dengue, Tuberculosis | Life-threatening or rapidly deteriorating |

---

## Model Comparison

### Why Random Forest was selected

Random Forest was selected over XGBoost based on convergent evidence:

1. Highest test accuracy and Macro F1 (0.9917) on all primary metrics
2. Fastest training time (1.08s vs XGBoost 5.86s) — nearly 5x faster
3. Stable cross-validation (CV F1: 0.9883, std: 0.0032)
4. Native feature importance via mean decrease in impurity
5. Zero Emergency-to-Self-Care misclassifications

### Model Parameters

```python
# Logistic Regression
LogisticRegression(max_iter=1000, random_state=42)

# Random Forest (selected)
RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# XGBoost
XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    eval_metric='mlogloss',
    random_state=42
)
```

---

## Streamlit App

The deployed application provides:
- Symptom selection checklist grouped by body system
- Demographic inputs (age, gender, blood pressure, cholesterol)
- Colour-coded triage recommendation (green / amber / red)
- Confidence score per class
- Top contributing symptoms driving the prediction

**Live App:** [Add your Streamlit URL here]

---

## Limitations

- Trained on synthetic and structured datasets — may not reflect full real-world symptom diversity
- Does not account for patient medical history, medications, or allergies
- Emergency class recall of 87.1% means approximately 12.9% of urgent cases may be under-triaged
- Not validated by clinical experts — intended for academic and research purposes only
- Any real-world deployment would require clinical validation and regulatory approval under the Medical Devices Regulation 2002

> This system is a decision-support tool and does not replace clinical judgement. It should not be used as a substitute for professional medical advice.

---

## References

1. NHS England (2023). General Practice Workforce Statistics. NHS Digital.
2. Gulliford, M. et al. (2002). What does access to health care mean? *Journal of Health Services Research and Policy*, 7(3), pp.186–188.
3. Obermeyer, Z. and Emanuel, E.J. (2016). Predicting the future — big data, machine learning, and clinical medicine. *New England Journal of Medicine*, 375(13), pp.1216–1219.
4. Sogandi, F. (2024). Identifying diseases symptoms and general rules using supervised and unsupervised machine learning. *Scientific Reports*, 14, 17956.
5. Pratap, V. (2021). Disease Symptom Prediction. Kaggle.
6. Chawla, N.V. et al. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, pp.321–357.
7. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), pp.5–32.
8. Rajpurkar, P. et al. (2022). AI in health and medicine. *Nature Medicine*, 28, pp.31–38.
9. Mackway-Jones, K. et al. (2014). *Emergency Triage: Manchester Triage Group*. 3rd ed. Wiley-Blackwell.
10. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, pp.2825–2830.

---

## License

This project is submitted as academic coursework for COM 763 Advanced Machine Learning at Wrexham University. The datasets used are publicly available under their respective licences (CC0 and CC-BY).
