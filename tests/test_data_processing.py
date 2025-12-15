import pytest
import pandas as pd
import numpy as np
import sys
import os # Import the os module

# --- START: Path Setup (MUST come before the import of data_processing) ---

# Get the absolute path to the directory containing this test file
test_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up one level ('..') and then into 'src'
scripts_path = os.path.join(test_dir, '..', 'src')

# Ensure the 'src' directory is in the system path
if os.path.isdir(scripts_path):
    # Use insert(0, ...) to ensure it's checked first
    sys.path.insert(0, scripts_path) 
    
# --- END: Path Setup ---

# --- FIX: The import now works because 'scripts_path' (the 'src' directory) 
# is now in sys.path.
try:
    from data_processing import FeatureEngineering, RFMAnalysis
except ImportError as e:
    print(f"Import Error: {e}. Check if data_processing.py is directly inside the 'src' folder.")
    raise

class TestFeatureEngineering:
    def test_create_aggregate_features(self):
        """Test aggregate feature creation"""
        # Create sample data
        df = pd.DataFrame({
            'CustomerId': [1, 1, 2, 2, 3],
            'Amount': [100, 200, 150, 250, 300],
            'Value': [100, 200, 150, 250, 300],
            'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5']
        })
        
        fe = FeatureEngineering()
        agg_features = fe.create_aggregate_features(df)
        
        # Assertions
        assert 'total_transaction_amount' in agg_features.columns
        assert 'avg_transaction_amount' in agg_features.columns
        assert len(agg_features) == 3  # 3 unique customers
        assert agg_features['total_transaction_amount'].iloc[0] == 300  # Customer 1
    
    def test_extract_datetime_features(self):
        """Test datetime feature extraction"""
        df = pd.DataFrame({
            'TransactionStartTime': ['2023-01-15 14:30:00', '2023-02-20 09:15:00']
        })
        
        fe = FeatureEngineering()
        df_with_features = fe.extract_datetime_features(df)
        
        # Assertions
        assert 'transaction_hour' in df_with_features.columns
        assert 'transaction_day' in df_with_features.columns
        assert df_with_features['transaction_hour'].iloc[0] == 14
        assert df_with_features['transaction_day'].iloc[0] == 15

class TestRFMAnalysis:
    def test_calculate_rfm(self):
        """Test RFM calculation"""
        df = pd.DataFrame({
            'CustomerId': [1, 1, 2, 2],
            'TransactionStartTime': pd.to_datetime([
                '2023-01-01', '2023-01-15', 
                '2023-02-01', '2023-02-10'
            ]),
            'TransactionId': ['T1', 'T2', 'T3', 'T4'],
            'Amount': [100, 200, 150, 250]
        })
        
        rfm_analyzer = RFMAnalysis()
        snapshot_date = pd.to_datetime('2023-03-01')
        rfm = rfm_analyzer.calculate_rfm(df, snapshot_date)
        
        # Assertions
        assert len(rfm) == 2  # 2 unique customers
        assert 'Recency' in rfm.columns
        assert 'Frequency' in rfm.columns
        assert 'Monetary' in rfm.columns

if __name__ == "__main__":
    pytest.main([__file__, '-v'])