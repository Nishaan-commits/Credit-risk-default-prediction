Credit Risk Prediction using XGBoost

A machine learning project to predict whether a loan applicant is a good or bad credit risk using the UCI German Credit Dataset.
This project focuses on building strong baseline models, performing feature engineering, and developing a tuned XGBoost final model.

📌 Project Overview

The goal of this project is to identify high-risk applicants based on demographic, financial, and behavioral attributes.
Accurate credit risk prediction helps banks reduce loan defaults and optimize lending decisions.

🧱 Workflow
1. Data Preparation

Loaded and explored the UCI German Credit dataset

Checked distributions, datatypes, and outliers

No missing values found

Separated numeric, ordinal, and nominal features

Applied Ordinal Encoding (for ordered categories)

Applied One-Hot Encoding (for nominal categories)

2. Baseline Models

Trained and evaluated the following models:

Logistic Regression

Decision Tree

Random Forest

XGBoost (Advanced Baseline)

LightGBM (Advanced Baseline)

Recorded metrics:

Accuracy

Precision

Recall

F1-Score

ROC-AUC

3. Feature Engineering

Created multiple domain-inspired features such as:

Monthly Payment

Installment Rate Percentage

Income Proxy

… and tested each feature individually

Accepted features only if they improved generalization

Maintained a comparison table for before vs after performance

Rejected features that did not improve metrics

4. Final Model

Used XGBoost as the final model:

Tuned key hyperparameters

Trained on the final feature set

Evaluated on the test split

Achieved strong generalization with improved Recall and ROC-AUC

📊 Results Summary

Significant improvements using domain-driven features

XGBoost outperformed all baseline models

Achieved a balanced precision–recall profile suitable for risk prediction

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-Learn

XGBoost

LightGBM

Matplotlib / Seaborn
