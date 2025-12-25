"""
data_processing.py

Modular functions for data exploration, preprocessing, and aggregation
for the Credit Risk Modeling project.

Author: Nigus Dibekulu
Date: 2025-12-25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict
# from xverse.transformers import WOEEncoder

use_woe=False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)

# -----------------------------
# 1. Data Loading & Validation
# -----------------------------
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path}, shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def validate_columns(df: pd.DataFrame, required_cols: List[str]):
    """
    Ensure required columns exist in the dataframe.

    Args:
        df (pd.DataFrame)
        required_cols (List[str])
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logging.error(f"Missing required columns: {missing}")
        raise ValueError(f"Missing required columns: {missing}")
    logging.info("All required columns are present.")

# -----------------------------
# 2. Overview & Summary
# -----------------------------
def data_overview(df: pd.DataFrame) -> Dict:
    """
    Return basic overview of the dataframe.

    Args:
        df (pd.DataFrame)

    Returns:
        dict: Shape, dtypes, unique customers/accounts
    """
    overview = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "num_customers": df['CustomerId'].nunique() if 'CustomerId' in df.columns else None,
        "num_accounts": df['AccountId'].nunique() if 'AccountId' in df.columns else None
    }
    logging.info(f"Data overview: {overview}")
    return overview

def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive statistics for numerical columns.
    """
    stats = df.describe()
    logging.info("Computed summary statistics for numerical columns.")
    return stats

def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return missing value counts and percentages for each column.
    """
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent
    })
    logging.info("Missing value summary computed.")
    return missing_df

# -----------------------------
# 3. Visualization Functions
# -----------------------------
def plot_numerical_distribution(df: pd.DataFrame, numerical_cols: List[str]):
    """
    Plot histogram and boxplot for each numerical column.
    """
    for col in numerical_cols:
        if col not in df.columns:
            logging.warning(f"{col} not found in dataframe, skipping.")
            continue

        plt.figure()
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()
        logging.info(f"Plotted distributions for {col}")

def plot_categorical_distribution(df: pd.DataFrame, categorical_cols: List[str]):
    """
    Plot countplots for categorical columns.
    """
    for col in categorical_cols:
        if col not in df.columns:
            logging.warning(f"{col} not found in dataframe, skipping.")
            continue

        plt.figure(figsize=(12,5))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.show()
        logging.info(f"Plotted categorical distribution for {col}")

def correlation_matrix(df: pd.DataFrame, numerical_cols: List[str]):
    """
    Plot correlation heatmap for numerical features.
    """
    available_cols = [c for c in numerical_cols if c in df.columns]
    if not available_cols:
        logging.warning("No numerical columns found for correlation plot.")
        return

    corr = df[available_cols].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()
    logging.info("Plotted correlation matrix.")

# -----------------------------
# 4. Aggregation Helpers
# -----------------------------
def aggregate_transactions(df: pd.DataFrame, group_by_col: str = 'CustomerId') -> pd.DataFrame:
    """
    Aggregate transaction-level data to customer-level.

    Args:
        df (pd.DataFrame)
        group_by_col (str): Column to group by (default: 'CustomerId')

    Returns:
        pd.DataFrame: Aggregated features
    """
    required_cols = ['Amount', 'Value']
    for col in required_cols:
        if col not in df.columns:
            logging.error(f"Column {col} not found for aggregation.")
            raise ValueError(f"Column {col} not found for aggregation.")

    agg_df = df.groupby(group_by_col).agg({
        'Amount': ['sum', 'mean', 'std', 'count'],
        'Value': ['sum', 'mean', 'std']
    })
    agg_df.columns = ['_'.join(col) for col in agg_df.columns]
    logging.info(f"Aggregated transactions by {group_by_col}, resulting shape: {agg_df.shape}")
    return agg_df.reset_index()

# -----------------------------
# 5. Save EDA Summary
# -----------------------------
def save_eda_summary(summary_dict: dict, file_path: str):
    """
    Save EDA summary dictionary as JSON.

    Args:
        summary_dict (dict)
        file_path (str)
    """
    import json
    import json
import pandas as pd
import numpy as np
import logging

def save_eda_summary(summary_dict, file_path):
    import json

    """
    Saves the EDA summary dictionary as a JSON file.
    Converts any pandas or numpy objects to JSON-serializable types.
    """
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        return obj

    try:
        with open(file_path, 'w') as f:
            json.dump(summary_dict, f, indent=4, default=convert)
        logging.info(f"EDA summary saved to {file_path}")
        print(f"EDA summary saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save EDA summary: {e}")
        print(f"Error saving EDA summary: {e}")
    """
    Saves the EDA summary dictionary as a JSON file.
    Converts any pandas or numpy objects to JSON-serializable types.
    """
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        return obj

    try:
        with open(file_path, 'w') as f:
            json.dump(summary_dict, f, indent=4, default=convert)
        logging.info(f"EDA summary saved to {file_path}")
        print(f"EDA summary saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save EDA summary: {e}")
        print(f"Error saving EDA summary: {e}")

    try:
        with open(file_path, 'w') as f:
            json.dump(summary_dict, f, indent=4)
        logging.info(f"EDA summary saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save EDA summary: {e}")
        raise





# src/data_processing.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Optional: WoE encoding


# ------------------------------
# Step 1: Aggregate Features
# ------------------------------
def aggregate_transactions(df, customer_col='CustomerId', amount_col='Amount', value_col='Value'):
    """
    Aggregate transaction data per customer.
    Returns customer-level summary features.
    """
    df_agg = df.groupby(customer_col).agg(
        Amount_sum=(amount_col, 'sum'),
        Amount_mean=(amount_col, 'mean'),
        Amount_std=(amount_col, 'std'),
        Amount_count=(amount_col, 'count'),
        Value_sum=(value_col, 'sum'),
        Value_mean=(value_col, 'mean'),
        Value_std=(value_col, 'std')
    ).reset_index()

    df_agg['Amount_std'] = df_agg['Amount_std'].fillna(0)
    df_agg['Value_std'] = df_agg['Value_std'].fillna(0)
    return df_agg


# ------------------------------
# Step 2: Extract Datetime Features
# ------------------------------
def extract_datetime_features(df, datetime_col='TransactionStartTime'):
    """
    Extract hour, day, month, year from transaction timestamp
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df['Transaction_Hour'] = df[datetime_col].dt.hour
    df['Transaction_Day'] = df[datetime_col].dt.day
    df['Transaction_Month'] = df[datetime_col].dt.month
    df['Transaction_Year'] = df[datetime_col].dt.year
    return df


# ------------------------------
# Step 3,4,5: Preprocessing Pipeline (Encode, Impute, Scale)
# ------------------------------
def create_preprocessing_pipeline(categorical_cols, numerical_cols, normalize=False):
    """
    Build full preprocessing pipeline:
    - Impute missing values
    - Encode categorical variables
    - Scale/normalize numerical features
    """
    # Categorical pipeline
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # Numerical pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler() if normalize else StandardScaler())
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, categorical_cols),
        ('num', num_pipeline, numerical_cols)
    ])

    return preprocessor





# ------------------------------
# Full Feature Engineering Pipeline
# ------------------------------
def full_feature_engineering(df, categorical_cols, numerical_cols, customer_col='CustomerId',
                            amount_col='Amount', value_col='Value', datetime_col='TransactionStartTime',
                            target_col='is_high_risk', normalize=False, use_woe=False):
    """
    Complete feature engineering: aggregation, datetime, preprocessing, optional WoE
    Returns final feature matrix and optional WOE encoder
    """
    # Step 1
    df_agg = aggregate_transactions(df, customer_col, amount_col, value_col)

    # Step 2
    df_feat = extract_datetime_features(df_agg, datetime_col)

    # Step 3+4+5: pipeline
    pipeline = create_preprocessing_pipeline(categorical_cols, numerical_cols, normalize=normalize)
    X_ready = pipeline.fit_transform(df_feat)

   

    return X_ready, pipeline