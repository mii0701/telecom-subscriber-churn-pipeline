# Telecom Subscriber Churn Prediction Pipeline

## Overview & Purpose
An end-to-end Python machine learning pipeline that predicts subscriber churn probability in telecommunications networks. It automates feature scaling, categorical encoding, and classification scoring to help retention teams proactively address customer attrition.

## Problem Addressed
In telecom, acquiring new customers is significantly more expensive than retaining existing ones. This pipeline flags high-risk accounts prior to service cancellation using usage patterns, billing metrics, and contract indicators.

## Tech Stack
- Python 3
- Pandas, NumPy
- Scikit-learn (Pipeline, ColumnTransformer, RandomForestClassifier)

## Core Architecture
- **Data Preprocessing:** Standard scaling for tenure and monthly billing numerical features.
- **Categorical Encoding:** One-hot encoding for contract parameters and support plan features.
- **Model Training:** Random Forest Classifier with class-weight balancing.
- **Inference Engine:** Automated probability scoring for incoming subscriber profile data.

## How to Run
1. Clone the repository: `git clone https://github.com/[your-username]/telecom-subscriber-churn-pipeline.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Execute the script: `python churn_predictor.py`

## Key Business Outcomes
- Evaluates models using **Precision, Recall, and ROC-AUC** metrics tailored for imbalanced retention datasets.
- Generates automated risk percentages for customer relationship management (CRM) workflows.


*Project developed for Data Science Portfolio & CWIE Personal Branding.*
