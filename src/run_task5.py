# src/run_task5.py
"""
Task 5 - Model Training and Tracking
Author: Nigus Dibekulu
Date: 2025-12-25
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import your feature engineering pipeline from Task 3
from src.data_processing import full_feature_engineering

# -----------------------------
# 1. Load processed data
# -----------------------------
data_file = "data/processed/processed_data.csv"
df = pd.read_csv(data_file)

# -----------------------------
# 2. Split data into features & target
# -----------------------------
# Assuming you already added 'is_high_risk' in Task 4
X = df.drop(columns=['CustomerId', 'is_high_risk'])
y = df['is_high_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 3. Train models
# -----------------------------
models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

# Print results
for model_name, metrics in results.items():
    print(f"\n{model_name} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# -----------------------------
# 4. Hyperparameter tuning (Random Forest example)
# -----------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=3, scoring='roc_auc')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("\nBest hyperparameters:", grid_search.best_params_)

# -----------------------------
# 5. MLflow Experiment Tracking
# -----------------------------
mlflow.set_experiment("credit_risk_modeling")

with mlflow.start_run(run_name="random_forest_gridsearch"):
    # Log model parameters
    mlflow.log_params(grid_search.best_params_)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    
    # Log metrics
    mlflow.log_metrics(metrics)
    
    
    # Log model
    mlflow.sklearn.log_model(best_model, "random_forest_model")
    
    print("\nMLflow logging complete. Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
