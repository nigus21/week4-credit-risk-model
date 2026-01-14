# End-to-End Credit Risk Analytics & Predictive Modeling (CrediTrust Financial)

## 1. Project Overview

This project implements a **full end-to-end credit risk analytics and predictive modeling system** for **CrediTrust Financial**, a digital-first financial services company. The objective is to predict the probability of customer default and demonstrate how machine learning systems are built, evaluated, deployed, and maintained in a real-world financial setting.

The project follows industry best practices across the entire ML lifecycle, including:

* Data cleaning and feature engineering
* Target engineering using credit risk techniques
* Model training and evaluation
* Experiment tracking and model registry with MLflow
* API-based model deployment with FastAPI
* Containerization using Docker
* Automated testing and CI/CD with GitHub Actions

---

## 2. Business Objective

Credit risk assessment is a core function for financial institutions. Poor risk evaluation can result in:

* High default rates
* Revenue loss
* Regulatory and compliance risks

**Primary Business Goal:**
Develop a robust, reproducible, and explainable machine learning pipeline that predicts customer default risk and supports data-driven lending decisions.

**Key Questions Addressed:**

* Which customer attributes are most predictive of default risk?
* How do different models compare in predicting credit risk?
* How can experiments be tracked and models deployed reliably?

---

## 3. Dataset Description

The dataset consists of historical customer and loan-related information, including:

* Demographic features
* Financial indicators
* Loan characteristics
* Credit behavior
* Target variable indicating default / non-default

Data is processed in multiple stages to ensure quality, consistency, and suitability for modeling.

---

## 4. Project Structure

```
week4-credit-risk-model/
│
├── data/
│   ├── raw/                     # Original datasets
│   ├── interim/                 # Intermediate cleaned data
│   └── processed/               # Final datasets ready for modeling
│
├── src/
│   ├── data_processing.py       # Data cleaning and preprocessing logic
│   ├── feature_engineering.py   # Feature creation and transformations
│   ├── target_engineering.py    # Target variable definition (credit risk)
│   ├── model_training.py        # Model training and evaluation
│   ├── api/
│   │   ├── main.py               # FastAPI application
│   │   └── pydantic_models.py   # Request/response schemas
│   │
│   └── utils.py                 # Shared helper functions
│
├── tests/
│   ├── test_data_processing.py  # Unit tests for data functions
│   └── test_features.py         # Unit tests for feature engineering
│
├── models/                      # Saved models (local)
├── mlruns/                      # MLflow experiment tracking
│
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Service orchestration
├── requirements.txt             # Project dependencies
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI pipeline
│
├── README.md                    # Project documentation
└── .gitignore
```

---

## 5. Data Processing and Feature Engineering

### Data Processing

* Handle missing values
* Encode categorical variables
* Normalize / scale numerical features
* Ensure schema consistency

### Target Engineering

* Define default risk target variable
* Apply domain-specific transformations
* Ensure class balance awareness

### Feature Engineering

* Create derived financial ratios
* Encode risk-related attributes
* Ensure reproducibility of transformations

---

## 6. Model Training and Experiment Tracking

### Models Trained

At least two models are trained and compared:

* Logistic Regression (baseline, interpretable)
* Decision Tree / Random Forest
* Gradient Boosting (where applicable)

### Hyperparameter Tuning

* Grid Search
* Random Search

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Experiment Tracking

* MLflow is used to log:

  * Model parameters
  * Metrics
  * Artifacts
* Best-performing model is registered in the **MLflow Model Registry**

---

## 7. Model Deployment (FastAPI)

The best model is deployed as a REST API using **FastAPI**.

### API Features

* Loads model directly from MLflow registry
* `/predict` endpoint for real-time inference
* Input validation using Pydantic schemas
* Returns probability of default

### Example API Flow

1. Client sends customer data as JSON
2. API validates input
3. Model generates risk probability
4. API returns prediction response

---

## 8. Containerization with Docker

The application is fully containerized for portability and scalability.

### Components

* **Dockerfile**: Defines the runtime environment
* **docker-compose.yml**: Builds and runs the API service

### Run Locally

```bash
docker-compose up --build
```

---

## 9. CI/CD Pipeline (GitHub Actions)

A CI pipeline is configured to run on every push to the `main` branch.

### Pipeline Steps

* Install dependencies
* Run code linting (flake8 or black)
* Execute unit tests using pytest
* Fail build if linting or tests fail

This ensures:

* Code quality
* Reliability
* Early bug detection

---

## 10. Testing Strategy

Unit tests are written using **pytest** to validate:

* Data processing functions
* Feature engineering outputs
* Schema consistency

Example tests include:

* Verifying expected columns
* Checking handling of missing values

---

## 11. Key Results and Insights

* Machine learning models outperform basic rule-based approaches
* Certain financial indicators are strong predictors of default
* Experiment tracking significantly improves model comparison and reproducibility

---

## 12. Limitations and Future Work

### Limitations

* Limited dataset size
* Potential class imbalance
* No real-time data ingestion

### Future Improvements

* Add explainability tools (SHAP, LIME)
* Integrate real-time streaming data
* Improve fairness and bias evaluation
* Deploy on cloud infrastructure

---

## 13. Tools and Technologies

* Python
* Pandas, NumPy, Scikit-learn
* MLflow
* FastAPI, Pydantic
* Docker, Docker Compose
* Pytest
* Git & GitHub Actions

---

## 14. Author

**Nigus Dibekulu**
Artificial Intelligence & Machine Learning Engineer

---
