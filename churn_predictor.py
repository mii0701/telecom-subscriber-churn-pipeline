import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Generate Synthetic Telecom Subscriber Dataset
np.random.seed(42)
n_samples = 500

data = {
    'tenure_months': np.random.randint(1, 72, n_samples),
    'monthly_charges': np.random.uniform(20.0, 120.0, n_samples),
    'contract_type': np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], n_samples, p=[0.5, 0.3, 0.2]),
    'tech_support': np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6]),
    'churn': np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
}

df = pd.DataFrame(data)

# 2. Separate Features and Target
X = df.drop('churn', axis=1)
y = df['churn']

# 3. Build Preprocessing & Feature Encoding Pipeline
numeric_features = ['tenure_months', 'monthly_charges']
categorical_features = ['contract_type', 'tech_support']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# 4. Integrate Model into Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# 5. Train-Test Split and Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)

# 6. Evaluation
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("=== Telecom Subscriber Churn Prediction Pipeline ===")
print("\n[Classification Report]")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
print(f"ROC-AUC Performance Score: {roc_auc_score(y_test, y_proba):.4f}")

# 7. Sample Prediction Scoring
sample_subscribers = pd.DataFrame({
    'tenure_months': [2, 36],
    'monthly_charges': [105.00, 45.00],
    'contract_type': ['Month-to-Month', 'Two-Year'],
    'tech_support': ['No', 'Yes']
})

sample_probs = pipeline.predict_proba(sample_subscribers)[:, 1]
print("\n[Sample High-Risk Subscriber Scoring]")
for i, prob in enumerate(sample_probs):
    print(f"Subscriber {i+1} Churn Risk Probability: {prob:.2%}")