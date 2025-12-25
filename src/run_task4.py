# run_task4.py

import pandas as pd

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.task_4_target_engineering import (
    calculate_rfm, scale_rfm, cluster_customers,
    assign_high_risk_label, merge_high_risk_target
)

# Load raw data
df = pd.read_csv("data/raw/data.csv")

# Step 1: Calculate RFM metrics
rfm = calculate_rfm(df)

# Step 2: Scale RFM
rfm_scaled, scaler = scale_rfm(rfm)

# Step 3: Cluster customers
rfm_clustered, kmeans_model = cluster_customers(rfm_scaled)

# Step 4: Assign high-risk label
rfm_clustered, high_risk_cluster = assign_high_risk_label(rfm_clustered)

# Step 5: Merge target variable into main dataset
df_final = merge_high_risk_target(df, rfm_clustered)

# Save the processed dataset
df_final.to_csv("data/processed/df_final_with_target.csv", index=False)

print("Task 4 completed! Dataset saved to data/processed/df_final_with_target.csv")
