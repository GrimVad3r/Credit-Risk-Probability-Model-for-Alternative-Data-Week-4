import pandas as pd
import numpy as np

def inspect_dataset_columns(csv_path='../data/processed/data_with_target.csv'):
    """
    Detailed column inspection to identify all potential leakage sources
    """
    print("="*70)
    print("🔍 COMPREHENSIVE COLUMN INSPECTION")
    print("="*70)
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"\nDataset shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")
    
    # 1. List all columns
    print("\n" + "="*70)
    print("📋 ALL COLUMNS IN DATASET")
    print("="*70)
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        nunique = df[col].nunique()
        print(f"{i:3d}. {col:40s} | Type: {str(dtype):10s} | Unique: {nunique:6d}")
    
    # 2. Check for suspicious column names
    print("\n" + "="*70)
    print("⚠️  SUSPICIOUS COLUMN NAMES")
    print("="*70)
    
    suspicious_keywords = [
        'risk', 'fraud', 'flag', 'score', 'label', 'target', 
        'prediction', 'class', 'default', 'churn', 'result',
        'status', 'outcome', 'decision', 'verdict'
    ]
    
    suspicious_cols = []
    for col in df.columns:
        col_lower = col.lower()
        for keyword in suspicious_keywords:
            if keyword in col_lower:
                suspicious_cols.append((col, keyword))
                print(f"   🚨 '{col}' contains keyword '{keyword}'")
                break
    
    if not suspicious_cols:
        print("   ✅ No obviously suspicious column names")
    
    # 3. Check columns with high cardinality (potential IDs)
    print("\n" + "="*70)
    print("🆔 HIGH CARDINALITY COLUMNS (Potential IDs)")
    print("="*70)
    
    high_card_threshold = len(df) * 0.9  # 90% unique values
    for col in df.columns:
        nunique = df[col].nunique()
        if nunique > high_card_threshold:
            print(f"   ⚠️  '{col}': {nunique}/{len(df)} unique values ({nunique/len(df)*100:.1f}%)")
    
    # 4. Analyze target variable
    print("\n" + "="*70)
    print("🎯 TARGET VARIABLE ANALYSIS")
    print("="*70)
    
    target_col = 'is_high_risk'
    if target_col in df.columns:
        print(f"\nTarget: {target_col}")
        print(f"Distribution:")
        print(df[target_col].value_counts())
        print(f"\nNormalized:")
        print(df[target_col].value_counts(normalize=True))
    else:
        print(f"❌ Target column '{target_col}' not found!")
    
    # 5. Check for perfect correlations with target
    print("\n" + "="*70)
    print("🔗 CORRELATIONS WITH TARGET")
    print("="*70)
    
    if target_col in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        correlations = []
        for col in numeric_cols:
            try:
                corr = df[col].corr(df[target_col])
                if not np.isnan(corr):
                    correlations.append((col, corr))
            except:
                pass
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("\nTop 15 correlations with target:")
        for col, corr in correlations[:15]:
            emoji = "🚨" if abs(corr) > 0.99 else "⚠️" if abs(corr) > 0.8 else "  "
            print(f"{emoji} {col:40s} {corr:+.4f}")
    
    # 6. Sample data preview
    print("\n" + "="*70)
    print("📊 SAMPLE DATA (First 3 rows)")
    print("="*70)
    print(df.head(3).to_string())
    
    # 7. Check for columns that perfectly separate classes
    print("\n" + "="*70)
    print("🎯 PERFECT CLASS SEPARATION CHECK")
    print("="*70)
    
    if target_col in df.columns:
        perfect_separators = []
        for col in df.columns[:30]:  # Check first 30 columns
            if col != target_col:
                # Check if each target class has unique values in this column
                try:
                    grouped = df.groupby(target_col)[col].apply(lambda x: x.nunique())
                    if (grouped == 1).all() and len(grouped) > 1:
                        perfect_separators.append(col)
                        print(f"   🚨 '{col}' perfectly separates target classes!")
                        # Show the unique values per class
                        for target_val in df[target_col].unique():
                            unique_vals = df[df[target_col] == target_val][col].unique()
                            print(f"      Target={target_val}: {unique_vals[:5]}")
                except:
                    pass
        
        if not perfect_separators:
            print("   ✅ No perfect separators found in first 30 columns")
    
    # 8. Summary recommendations
    print("\n" + "="*70)
    print("📝 SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    print("\nColumns to DEFINITELY exclude:")
    exclude_list = ['TransactionId', 'BatchId', 'CustomerId', 'SubscriptionId', 
                   'TransactionStartTime', target_col]
    
    # Add suspicious columns
    for col, keyword in suspicious_cols:
        if col not in exclude_list:
            exclude_list.append(col)
    
    # Add perfect separators
    if 'perfect_separators' in locals():
        for col in perfect_separators:
            if col not in exclude_list:
                exclude_list.append(col)
    
    print("\nExclude these columns in your model:")
    print("leakage_features = [")
    for col in exclude_list:
        if col != target_col:
            print(f"    '{col}',")
    print("]")
    
    return df

if __name__ == "__main__":
    df = inspect_dataset_columns()
    print("\n✅ Inspection complete!")