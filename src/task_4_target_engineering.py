import pandas as pd

def calculate_rfm(df, customer_col='CustomerId', datetime_col='TransactionStartTime', value_col='Amount', snapshot_date=None):
    """
    Calculate RFM metrics per customer.
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Define snapshot date (default: max date in dataset + 1 day)
    if snapshot_date is None:
        snapshot_date = df[datetime_col].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby(customer_col).agg(
        Recency = lambda x: (snapshot_date - x[datetime_col].max()).days,
        Frequency = (datetime_col, 'count'),
        Monetary = (value_col, 'sum')
    ).reset_index()
    
    return rfm


from sklearn.preprocessing import StandardScaler

def scale_rfm(rfm_df, features=['Recency', 'Frequency', 'Monetary']):
    """
    Standardize RFM features for clustering
    """
    scaler = StandardScaler()
    rfm_scaled = rfm_df.copy()
    rfm_scaled[features] = scaler.fit_transform(rfm_scaled[features])
    return rfm_scaled, scaler


from sklearn.cluster import KMeans

def cluster_customers(rfm_scaled, features=['Recency','Frequency','Monetary'], n_clusters=3, random_state=42):
    """
    Cluster customers into high/mid/low engagement groups
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_scaled['Cluster'] = kmeans.fit_predict(rfm_scaled[features])
    return rfm_scaled, kmeans




def assign_high_risk_label(rfm_clustered):
    """
    Identify the least engaged cluster as high-risk
    """
    # Compute mean RFM per cluster
    cluster_summary = rfm_clustered.groupby('Cluster').agg({'Recency':'mean','Frequency':'mean','Monetary':'mean'})
    
    # High-risk cluster: high Recency, low Frequency & Monetary
    high_risk_cluster = cluster_summary.sort_values(['Recency','Frequency','Monetary'], ascending=[False, True, True]).index[0]
    
    rfm_clustered['is_high_risk'] = rfm_clustered['Cluster'].apply(lambda x: 1 if x==high_risk_cluster else 0)
    
    return rfm_clustered, high_risk_cluster


def merge_high_risk_target(df, rfm_clustered, customer_col='CustomerId'):
    """
    Merge high-risk binary label into main dataset
    """
    df_final = df.merge(rfm_clustered[[customer_col,'is_high_risk']], on=customer_col, how='left')
    return df_final


