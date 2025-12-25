## 📌 Credit Scoring Business Understanding

### 1. Basel II, Risk Measurement, and Model Interpretability

The Basel II Capital Accord places strong emphasis on **accurate risk measurement, transparency, and regulatory accountability**. Financial institutions are required not only to estimate credit risk reliably but also to **justify and explain** how risk estimates are produced.

In this project, this requirement directly influences our modeling approach in several ways:

* Models must be **interpretable**, auditable, and reproducible.
* Feature engineering and data transformations must be **well-documented** and traceable.
* The rationale behind variable selection, target construction, and model choice must be clearly explained.

As a result, we prioritize structured pipelines, documented assumptions, and models whose predictions can be explained to risk managers, auditors, and regulators.

---

### 2. Need for a Proxy Default Variable and Associated Business Risks

The dataset does not contain an explicit loan default or repayment outcome. Since supervised machine learning models require labeled targets, it is necessary to **engineer a proxy variable** that represents credit risk.

In this project, customer behavioral data—specifically **Recency, Frequency, and Monetary (RFM) patterns**—is used to identify disengaged customers who are more likely to default. These customers are labeled as **high-risk**, while more active customers are labeled as **low-risk**.

However, relying on a proxy introduces important business risks:

* The proxy may not perfectly reflect true default behavior, leading to **label noise**.
* Misclassification may cause the bank to **reject creditworthy customers** or **approve risky borrowers**.
* Model predictions must therefore be interpreted as **risk indicators**, not absolute default guarantees.

To mitigate these risks, the proxy definition is grounded in business logic, statistically validated through clustering, and clearly documented.

---

### 3. Trade-offs Between Interpretable and High-Performance Models

In a regulated financial context, there is a critical trade-off between **model interpretability** and **predictive performance**.

**Simple, interpretable models (e.g., Logistic Regression with WoE):**

* Advantages:

  * High transparency and explainability
  * Easy regulatory approval
  * Clear linkage between features and risk
* Limitations:

  * Limited ability to capture non-linear relationships
  * Potentially lower predictive power

**Complex, high-performance models (e.g., Gradient Boosting):**

* Advantages:

  * Strong predictive accuracy
  * Ability to capture complex interactions
* Limitations:

  * Reduced interpretability
  * Higher regulatory and operational risk
  * Increased difficulty in explaining decisions to stakeholders

Given these trade-offs, this project adopts a **comparative approach**, evaluating both interpretable and complex models. Final model selection balances **performance, explainability, and regulatory compliance**, consistent with Basel II principles.

---

### 4. Data-Driven Insights Informing Credit Risk Modeling

Exploratory Data Analysis revealed several key patterns that influence credit risk modeling decisions:

* Customer transaction behavior is highly heterogeneous, requiring **customer-level aggregation** for stability.
* Numerical transaction features exhibit strong skewness and outliers, necessitating **robust scaling or transformation**.
* Categorical features are heavily imbalanced, motivating **category grouping or WoE encoding**.
* Certain variables (e.g., `CountryCode`) have zero variance and are excluded.
* The dataset contains no missing values, enabling reliable feature engineering without imputation bias.

These insights ensure that the resulting credit scoring model is both **statistically sound** and **business-aligned**.

--