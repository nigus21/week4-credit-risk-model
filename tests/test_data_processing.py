import pytest
import pandas as pd
from src.data_processing import aggregate_transactions

def test_aggregate_transactions():
    df = pd.DataFrame({
        "CustomerId": ["C1", "C1", "C2"],
        "Amount": [100, 200, 300],
        "Value": [10, 20, 30]
    })
    
    df_agg = aggregate_transactions(df)
    
    assert df_agg.shape[0] == 2
    assert "Amount_sum" in df_agg.columns
    assert "Value_mean" in df_agg.columns
