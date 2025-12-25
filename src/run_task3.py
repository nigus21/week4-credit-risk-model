# src/run_task3.py
import pandas as pd
from src.data_processing import full_feature_engineering

# Load your raw data
df = pd.read_csv("data/raw/data.csv")

# Define categorical and numerical columns
categorical_cols = ['ProductCategory', 'ChannelId', 'ProviderId', 'CountryCode']
numerical_cols = ['Amount_sum', 'Amount_mean', 'Amount_std', 'Amount_count', 
                  'Value_sum', 'Value_mean', 'Value_std']

# Run full feature engineering
X_ready, pipeline, woe_enc = full_feature_engineering(
    df,
    categorical_cols=categorical_cols,
    numerical_cols=numerical_cols,
    use_woe=False  # Set True only if xverse is installed
)

print("Feature engineering completed. Shape:", X_ready.shape)
