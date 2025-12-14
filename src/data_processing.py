import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from datetime import datetime
# Install xverse
# pip install xverse
from xverse.transformer import WOE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class RFMAnalysis:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        
    def calculate_rfm(self, df, snapshot_date=None):
        """Calculate RFM metrics for each customer"""
        if snapshot_date is None:
            snapshot_date = df['TransactionStartTime'].max()
        
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Calculate RFM
        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
            'TransactionId': 'count',  # Frequency
            'Amount': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
        
        return rfm
    
    def cluster_customers(self, rfm_df, n_clusters=3, random_state=42):
        """Perform K-Means clustering on RFM data"""
        # Select RFM columns
        rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
        
        # Handle any missing values
        rfm_features = rfm_features.fillna(0)
        
        # Scale features
        rfm_scaled = self.scaler.fit_transform(rfm_features)
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        rfm_df['Cluster'] = self.kmeans.fit_predict(rfm_scaled)
        
        return rfm_df
    
    def identify_high_risk_cluster(self, rfm_df):
        """Identify the high-risk cluster (lowest engagement)"""
        # Analyze cluster characteristics
        cluster_summary = rfm_df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        })
        
        print("Cluster Summary:")
        print(cluster_summary)
        
        # High risk: High Recency (inactive), Low Frequency, Low Monetary
        # Calculate risk score for each cluster
        cluster_summary['risk_score'] = (
            cluster_summary['Recency'] / cluster_summary['Recency'].max() +
            (1 - cluster_summary['Frequency'] / cluster_summary['Frequency'].max()) +
            (1 - cluster_summary['Monetary'] / cluster_summary['Monetary'].max())
        )
        
        high_risk_cluster = cluster_summary['risk_score'].idxmax()
        print(f"\nHigh Risk Cluster: {high_risk_cluster}")
        
        return high_risk_cluster
    
    def create_target_variable(self, df, rfm_df, high_risk_cluster):
        """Create binary target variable"""
        rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
        
        # Merge with main dataframe
        df = df.merge(rfm_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
        
        print(f"\nTarget Variable Distribution:")
        print(df['is_high_risk'].value_counts())
        
        return df

def create_target_variable_pipeline():
    """Complete pipeline for creating target variable"""
    # Load processed data
    df = pd.read_csv('../data/processed/processed_data.csv')
    
    # Initialize RFM analysis
    rfm_analyzer = RFMAnalysis()
    
    # Calculate RFM
    rfm_df = rfm_analyzer.calculate_rfm(df)
    
    # Cluster customers
    rfm_df = rfm_analyzer.cluster_customers(rfm_df, n_clusters=3)
    
    # Identify high-risk cluster
    high_risk_cluster = rfm_analyzer.identify_high_risk_cluster(rfm_df)
    
    # Create target variable
    df_with_target = rfm_analyzer.create_target_variable(df, rfm_df, high_risk_cluster)
    
    # Save final dataset
    df_with_target.to_csv('../data/processed/data_with_target.csv', index=False)
    print("\nTarget variable created and saved!")
    
    return df_with_target

class FeatureEngineering:
    def __init__(self):
        self.pipeline = None
        
    def create_aggregate_features(self, df):
        """Create customer-level aggregate features"""
        # Group by CustomerId
        agg_features = df.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'std', 'count', 'min', 'max'],
            'Value': ['sum', 'mean', 'std'],
            'TransactionId': 'count'
        }).reset_index()
        
        # Flatten column names
        agg_features.columns = ['_'.join(col).strip('_') 
                                for col in agg_features.columns.values]
        
        # Rename for clarity
        agg_features.rename(columns={
            'Amount_sum': 'total_transaction_amount',
            'Amount_mean': 'avg_transaction_amount',
            'Amount_std': 'std_transaction_amount',
            'Amount_count': 'transaction_count',
            'TransactionId_count': 'transaction_frequency'
        }, inplace=True)
        
        return agg_features
    
    def extract_datetime_features(self, df):
        """Extract time-based features"""
        df = df.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        df['transaction_hour'] = df['TransactionStartTime'].dt.hour
        df['transaction_day'] = df['TransactionStartTime'].dt.day
        df['transaction_month'] = df['TransactionStartTime'].dt.month
        df['transaction_year'] = df['TransactionStartTime'].dt.year
        df['transaction_dayofweek'] = df['TransactionStartTime'].dt.dayofweek
        df['transaction_weekend'] = (df['transaction_dayofweek'] >= 5).astype(int)
        
        return df
    
    def handle_missing_values(self, df, numerical_cols, categorical_cols):
        """Handle missing values with imputation"""
        df = df.copy()
        
        # Numerical imputation (median)
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical imputation (mode)
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def build_preprocessing_pipeline(self, numerical_features, categorical_features):
        """Build sklearn pipeline for preprocessing"""
        
        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
        
        self.pipeline = preprocessor
        return preprocessor
    
    def process_data(self, df):
        """Main processing function"""
        # Extract datetime features
        df = self.extract_datetime_features(df)
        
        # Create aggregate features
        agg_features = self.create_aggregate_features(df)
        
        # Merge back
        df = df.merge(agg_features, on='CustomerId', how='left')
        
        return df

    def apply_woe_transformation(df, target_col, feature_cols):
        """Apply Weight of Evidence transformation"""
        woe = WOE()
        
        # Fit and transform
        df_woe = woe.fit_transform(df[feature_cols], df[target_col])
        
        return df_woe, woe

def main():
    # Load raw data
    df = pd.read_csv('../data/raw/data.csv')
    
    # Initialize feature engineering
    fe = FeatureEngineering()
    
    # Process data
    df_processed = fe.process_data(df)
    
    # Save processed data
    df_processed.to_csv('../data/processed/processed_data.csv', index=False)
    print("Data processing complete!")

if __name__ == "__main__":
    # Run feature engineering
    main()
    
    # Create target variable
    create_target_variable_pipeline()