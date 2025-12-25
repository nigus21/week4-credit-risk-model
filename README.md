 task-1-credit-risk-understanding
# Credit Risk Probability Model for Alternative Data

## 10 Academy – Artificial Intelligence Mastery

**Week 4 Challenge: Credit Risk Probability Model**

**Project Duration:** 10 Dec – 16 Dec 2025
**Role:** Analytics Engineer – Bati Bank

---

## 📌 Project Overview

Bati Bank is partnering with an eCommerce platform to launch a **Buy-Now-Pay-Later (BNPL)** service.
The goal of this project is to develop a **credit risk scoring system** using alternative behavioral data to:

* Estimate the probability that a customer is high-risk
* Support loan approval and pricing decisions
* Enable responsible credit allocation in compliance with regulatory standards

The core innovation is transforming **transaction behavior** into a **predictive risk signal** using data-driven methods.

---

## 🧠 Task 1: Credit Scoring Business Understanding

### 1. Basel II Accord and Model Interpretability

The Basel II Capital Accord emphasizes **accurate risk measurement, transparency, and accountability** in credit decision-making. Financial institutions must not only predict risk but also **justify how risk estimates are produced**.

In this project, Basel II principles influence the design in the following ways:

* Preference for **interpretable and auditable models**
* Clear documentation of assumptions and feature transformations
* Reproducible pipelines for data processing and model training
* Traceability from raw data to final predictions

This ensures that the resulting credit risk model can be reviewed by regulators, auditors, and internal risk committees.

---

### 2. Proxy Target Variable for Credit Risk

The dataset does not contain a direct label indicating whether a customer defaulted on a loan. Since supervised learning requires labeled outcomes, a **proxy credit risk variable** must be created.

Customer transaction behavior—specifically **Recency, Frequency, and Monetary (RFM)** patterns—is used to identify disengaged customers. These customers are assumed to have a higher likelihood of default and are labeled as **high-risk**, while more active customers are labeled as **low-risk**.

**Business risks of using a proxy variable include:**

* The proxy may not perfectly represent true default behavior
* Label noise may lead to misclassification
* Creditworthy customers may be rejected or risky customers approved

To mitigate these risks, the proxy definition is grounded in behavioral finance principles, statistically validated through clustering, and clearly documented.

---

### 3. Trade-offs Between Interpretable and High-Performance Models

In regulated financial environments, there is a fundamental trade-off between **interpretability** and **predictive performance**.

**Interpretable models (e.g., Logistic Regression with WoE):**

* Transparent and explainable
* Easier regulatory approval
* Clear feature-to-risk relationships
* Limited ability to capture non-linear patterns

**Complex models (e.g., Gradient Boosting):**

* Higher predictive accuracy
* Capture complex feature interactions
* Harder to explain and audit
* Higher regulatory and operational risk

This project adopts a **comparative modeling approach**, evaluating both model types and selecting the final model based on a balance of **performance, explainability, and compliance**.


### task 2. Data-Driven Insights Informing Credit Risk Modeling

Exploratory Data Analysis revealed several key patterns that influence credit risk modeling decisions:

* Customer transaction behavior is highly heterogeneous, requiring **customer-level aggregation** for stability.
* Numerical transaction features exhibit strong skewness and outliers, necessitating **robust scaling or transformation**.
* Categorical features are heavily imbalanced, motivating **category grouping or WoE encoding**.
* Certain variables (e.g., `CountryCode`) have zero variance and are excluded.
* The dataset contains no missing values, enabling reliable feature engineering without imputation bias.

These insights ensure that the resulting credit scoring model is both **statistically sound** and **business-aligned**.


