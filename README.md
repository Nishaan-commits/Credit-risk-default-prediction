# German Credit Risk Prediction & Deployment

# Final Year Machine Learning Project
## By: Mohd. Nishaan

### 📌 Project Overview

This project focuses on predicting credit risk — whether a loan applicant is likely to be a good or bad credit risk — using the German Credit Dataset.

The primary objective was to build a clean, end-to-end machine learning pipeline that follows proper ML practices:

- structured preprocessing
- feature engineering
- model selection
- evaluation using appropriate metrics for imbalanced data
- deployment through a simple, usable application

**This project serves both as my final year project and a portfolio project to demonstrate my readiness for Machine Learning Intern / Entry-Level roles.**

### 🧠 Problem Statement

Banks face significant losses when high-risk customers are incorrectly classified as low-risk.
Therefore, this problem is treated as a cost-sensitive classification task, where false negatives (bad credit predicted as good) must be minimized.

### 📊 Dataset

Dataset: German Credit Dataset (UCI Machine Learning Repository)

Size: 1000 records, 20 features

Target:

1 → Good Credit Risk

0 → Bad Credit Risk

Dataset Link:
https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data

The dataset is imbalanced, which influenced both metric selection and threshold tuning.

### 🔧 Machine Learning Pipeline

The project follows a clear ML pipeline:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Encoding & Preprocessing
- Model Comparison
- Hyperparameter Tuning
- Threshold Optimization
- Pipeline Serialization

Model Deployment using Streamlit

### 🛠 Feature Engineering

In addition to the original dataset, I engineered three new features to better represent financial behavior:

- Monthly Payment
- Installment Rate Percentage
- Income Proxy (to approximate repayment capability)

These features improved the model’s ability to capture financial stability patterns.

### 🔄 Encoding & Preprocessing

Ordinal Encoding applied to 4 ordered categorical features

One-Hot Encoding for nominal features

No feature removal
→ Empirically validated that keeping all features resulted in better ROC-AUC

Data Resampling (SMOTEENN) was tested but did not provide gains due to the small dataset size

All preprocessing and feature engineering steps are included inside a single reusable pipeline.

### 🤖 Model Selection

Multiple baseline and advanced models were evaluated, including:

- Logistic Regression
- Decision Tree
- Random Forest
- LightGBM
- XGBoost (Final Model)

### ✅ Why XGBoost?

XGBoost was selected as the final model because it offered:

- Strong ROC-AUC performance
- Good balance between bias and variance
- High tuning flexibility
- Better interpretability compared to deep models

### 📈 Evaluation Strategy

Because this is an imbalanced classification problem, accuracy alone was avoided.

Metrics Used:

- ROC-AUC → for model & hyperparameter selection
- F2-Score → for threshold tuning
- Penalizes false negatives more heavily
- Aligns with real-world banking risk

Validation:
Stratified 5-Fold Cross Validation

Threshold tuning performed post-training to optimize business impact

### 🚀 Deployment (Streamlit App)

A local Streamlit application was built to demonstrate real-world usability.

App Capabilities:

- User fills a form with raw customer attributes

Pipeline handles:

- Encoding
- Feature engineering
- Scaling

Model outputs:

- Predicted Label (Good / Bad Credit Risk)

- Prediction Probability

- Decision Threshold

This simulates how a non-technical user (e.g., loan officer) might interact with an ML system.

The complete Streamlit app is available inside the
german_credit_streamlit_app/ folder.

### 🧪 Tech Stack

Python: 3.12

Libraries:

- pandas, numpy
- scikit-learn
- xgboost, lightgbm
- imbalanced-learn
- matplotlib, seaborn, plotly
- streamlit
- joblib

- Model Storage: Fully serialized end-to-end pipeline
