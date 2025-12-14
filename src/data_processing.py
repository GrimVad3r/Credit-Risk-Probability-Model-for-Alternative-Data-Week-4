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
    main()