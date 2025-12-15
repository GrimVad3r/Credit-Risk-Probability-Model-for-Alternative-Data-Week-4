import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def diagnose_data_leakage(df, target_col='is_high_risk'):
    """
    Comprehensive data leakage diagnostic tool
    """
    print("="*70)
    print("🔍 DATA LEAKAGE DIAGNOSTIC REPORT")
    print("="*70)
    
    # 1. Check for duplicate or near-duplicate rows
    print("\n1️⃣  DUPLICATE ROWS CHECK")
    print("-" * 70)
    duplicates = df.duplicated().sum()
    print(f"   Total duplicate rows: {duplicates}")
    if duplicates > 0:
        print(f"   ⚠️  WARNING: {duplicates} duplicate rows found!")
        print(f"   Duplicates as % of data: {duplicates/len(df)*100:.2f}%")
    
    # 2. Check for features perfectly correlated with target
    print("\n2️⃣  PERFECT CORRELATION CHECK")
    print("-" * 70)
    
    exclude_cols = ['TransactionId', 'BatchId', 'CustomerId', 'SubscriptionId', 
                    'TransactionStartTime', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Only check numeric columns
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    
    perfect_correlations = []
    for col in numeric_cols:
        if col in df.columns:
            corr = df[col].corr(df[target_col])
            if abs(corr) > 0.99:  # Nearly perfect correlation
                perfect_correlations.append((col, corr))
                print(f"   🚨 LEAKAGE ALERT: '{col}' has correlation {corr:.4f} with target!")
    
    if not perfect_correlations:
        print("   ✅ No perfect correlations found")
    
    # 3. Check feature importance (top features that might be leaking)
    print("\n3️⃣  FEATURE IMPORTANCE CHECK")
    print("-" * 70)
    
    # Prepare data
    X = df[feature_cols].copy()
    y = df[target_col]
    
    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    X.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X.columns]
    
    # Fill NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Quick Random Forest to check feature importance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n   Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:40s} {row['importance']:.4f}")
        if row['importance'] > 0.5:
            print(f"      ⚠️  Suspiciously high importance (>0.5) - check for leakage!")
    
    # 4. Check train/test performance gap
    print("\n4️⃣  TRAIN/TEST PERFORMANCE GAP")
    print("-" * 70)
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    print(f"   Training accuracy: {train_score:.4f}")
    print(f"   Test accuracy:     {test_score:.4f}")
    print(f"   Gap:               {abs(train_score - test_score):.4f}")
    
    if test_score > 0.99:
        print(f"   🚨 LEAKAGE ALERT: Test accuracy is {test_score:.4f} - this is too good to be true!")
    elif train_score - test_score < 0.01:
        print(f"   ⚠️  WARNING: Very small train-test gap might indicate leakage")
    
    # 5. Check for target-derived features
    print("\n5️⃣  SUSPICIOUS FEATURE NAMES")
    print("-" * 70)
    suspicious_keywords = ['risk', 'fraud', 'flag', 'score', 'label', 'target', 
                          'prediction', 'class', 'default', 'churn']
    
    suspicious_features = []
    for col in feature_cols:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in suspicious_keywords):
            suspicious_features.append(col)
            print(f"   ⚠️  Suspicious: '{col}' contains risk-related keyword")
    
    if not suspicious_features:
        print("   ✅ No suspicious feature names found")
    
    # 6. Check for features with single value per target class
    print("\n6️⃣  PERFECT SEPARATION CHECK")
    print("-" * 70)
    perfect_separators = []
    
    for col in numeric_cols[:20]:  # Check first 20 numeric columns
        if col in df.columns:
            # Check if feature perfectly separates classes
            grouped = df.groupby(target_col)[col].apply(lambda x: x.nunique())
            if (grouped == 1).all():
                perfect_separators.append(col)
                print(f"   🚨 LEAKAGE ALERT: '{col}' perfectly separates target classes!")
    
    if not perfect_separators:
        print("   ✅ No perfect separators found in first 20 features")
    
    # 7. Summary and recommendations
    print("\n" + "="*70)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    issues_found = []
    if duplicates > 0:
        issues_found.append("duplicate rows")
    if perfect_correlations:
        issues_found.append("perfect correlations")
    if test_score > 0.99:
        issues_found.append("unrealistic test performance")
    if suspicious_features:
        issues_found.append("suspicious feature names")
    if perfect_separators:
        issues_found.append("perfect separator features")
    
    if issues_found:
        print(f"\n❌ ISSUES DETECTED: {', '.join(issues_found)}")
        print("\n🔧 RECOMMENDED ACTIONS:")
        print("   1. Review feature engineering pipeline")
        print("   2. Remove features that are derived from the target")
        print("   3. Check if target was accidentally included in features")
        print("   4. Verify time-based features don't leak future information")
        print("   5. Remove the following suspected leakage features:")
        
        all_suspected = list(set([col for col, _ in perfect_correlations] + 
                                suspicious_features + perfect_separators))
        for feat in all_suspected[:10]:  # Show top 10
            print(f"      - {feat}")
    else:
        print("\n✅ No obvious data leakage detected")
        print("\n🔍 If you're still getting perfect scores, check:")
        print("   1. Is your dataset too small or too simple?")
        print("   2. Are there hidden relationships in the data?")
        print("   3. Is the target distribution extremely imbalanced?")
    
    return {
        'perfect_correlations': perfect_correlations,
        'suspicious_features': suspicious_features,
        'perfect_separators': perfect_separators,
        'feature_importance': feature_importance,
        'test_score': test_score
    }

def main():
    # Load data
    try:
        df = pd.read_csv('../data/processed/data_with_target.csv')
        print(f"✅ Loaded data: {df.shape}")
        print(f"\nColumns in dataset: {df.columns.tolist()}")
        
        # Run diagnostic
        results = diagnose_data_leakage(df)
        
        # Optional: Show first few rows
        print("\n" + "="*70)
        print("📊 DATASET PREVIEW")
        print("="*70)
        print(df.head())
        
    except FileNotFoundError:
        print("❌ Error: Could not find '../data/processed/data_with_target.csv'")
        return

if __name__ == "__main__":
    main()