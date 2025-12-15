import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import joblib
import warnings
from sklearn.model_selection import GroupShuffleSplit
warnings.filterwarnings('ignore')

class ModelTrainer:

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None

    def prepare_data(self, df, target_col='is_high_risk', test_size=0.2):
        """
        Leakage-safe data preparation:
        - Removes all RFM / aggregate-derived features
        - Uses customer-level split
        """

        # =========================================================
        # 1. HARD EXCLUSIONS (NON-NEGOTIABLE)
        # =========================================================
        base_exclude_cols = [
            'TransactionId',
            'BatchId',
            'CustomerId',
            'SubscriptionId',
            'TransactionStartTime',
            target_col
        ]

        # Explicit leakage features
        leakage_features = [
            'FraudResult',
            'CountryCode',  # correlated proxy
            'Cluster',
            'Recency',
            'Frequency',
            'Monetary'
        ]

        # Aggregate / RFM-derived features (pattern-based)
        aggregate_prefixes = (
            'total_',
            'avg_',
            'std_',
            'Amount_',
            'Value_',
            'transaction_count',
            'transaction_frequency'
        )

        exclude_cols = set(base_exclude_cols + leakage_features)

        for col in df.columns:
            if col.startswith(aggregate_prefixes):
                exclude_cols.add(col)

        feature_cols = [c for c in df.columns if c not in exclude_cols]

        print(f"\n📋 Feature Selection Summary")
        print(f"   ▶ Total columns: {df.shape[1]}")
        print(f"   ▶ Excluded columns: {len(exclude_cols)}")
        print(f"   ▶ Final features: {len(feature_cols)}")

        # =========================================================
        # 2. FEATURE MATRIX
        # =========================================================
        X = df[feature_cols].copy()
        y = df[target_col]
        groups = df['CustomerId']

        # One-hot encode categoricals
        X = pd.get_dummies(X, drop_first=True)

        # Clean column names (XGBoost safe)
        X.columns = ["".join(c if c.isalnum() else "_" for c in col) for col in X.columns]

        # Handle NaN / inf
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))

        # =========================================================
        # 3. CUSTOMER-LEVEL SPLIT (CRITICAL)
        # =========================================================
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=self.random_state
        )

        train_idx, test_idx = next(splitter.split(X, y, groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(f"\n📊 Split Summary")
        print(f"   ▶ Train customers: {groups.iloc[train_idx].nunique()}")
        print(f"   ▶ Test customers:  {groups.iloc[test_idx].nunique()}")
        print(f"   ▶ Train rows: {X_train.shape[0]}")
        print(f"   ▶ Test rows:  {X_test.shape[0]}")

        # =========================================================
        # 4. FINAL LEAKAGE SANITY CHECK
        # =========================================================
        print("\n🔍 Leakage Sanity Check (Post-Split)")
        suspicious = []

        for col in X_train.select_dtypes(include=[np.number]).columns:
            corr = X_train[col].corr(y_train)
            if abs(corr) > 0.95:
                suspicious.append((col, corr))

        if suspicious:
            print(f"⚠️  Found {len(suspicious)} suspiciously correlated features:")
            for col, corr in suspicious[:10]:
                print(f"   - {col}: corr={corr:.4f}")
        else:
            print("✅ No high-correlation features detected")

        print("\n📊 Target Distribution")
        print(y_train.value_counts(normalize=True))

        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Handle edge cases
        if len(np.unique(y_test)) < 2:
            print("⚠️  Warning: Test set has only one class")
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
        }
        
        return metrics
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression with hyperparameter tuning"""
        with mlflow.start_run(run_name="Logistic_Regression"):
            # Simplified parameter grid to avoid solver-penalty conflicts
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],  # L2 works with all solvers
                'solver': ['lbfgs', 'liblinear'],
                'max_iter': [1000]
            }
            
            # Grid search with error_score to debug
            lr = LogisticRegression(random_state=self.random_state, class_weight='balanced')
            grid_search = GridSearchCV(
                lr, param_grid, cv=3, scoring='roc_auc', 
                n_jobs=-1, error_score='raise', verbose=1
            )
            
            try:
                grid_search.fit(X_train, y_train)
            except Exception as e:
                print(f"❌ Logistic Regression failed: {str(e)}")
                raise
            
            # Best model
            best_model = grid_search.best_estimator_
            
            # Evaluate
            metrics = self.evaluate_model(best_model, X_test, y_test)
            
            # Log to MLflow
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model, "model")
            
            print("\n✅ Logistic Regression Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            self.models['LogisticRegression'] = best_model
            return best_model, metrics
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest with hyperparameter tuning"""
        with mlflow.start_run(run_name="Random_Forest"):
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            # Random search for efficiency
            rf = RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced'
            )
            random_search = RandomizedSearchCV(
                rf, param_grid, n_iter=10, cv=3, scoring='roc_auc', 
                random_state=self.random_state, n_jobs=-1,
                error_score='raise', verbose=1
            )
            
            try:
                random_search.fit(X_train, y_train)
            except Exception as e:
                print(f"❌ Random Forest failed: {str(e)}")
                raise
            
            # Best model
            best_model = random_search.best_estimator_
            
            # Evaluate
            metrics = self.evaluate_model(best_model, X_test, y_test)
            
            # Log to MLflow
            mlflow.log_params(random_search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model, "model")
            
            print("\n✅ Random Forest Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            self.models['RandomForest'] = best_model
            return best_model, metrics
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Train Gradient Boosting (XGBoost)"""
        with mlflow.start_run(run_name="XGBoost"):
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
            
            # Random search
            xgb_model = xgb.XGBClassifier(
                random_state=self.random_state, 
                eval_metric='logloss', 
                use_label_encoder=False,
                scale_pos_weight=scale_pos_weight
            )
            random_search = RandomizedSearchCV(
                xgb_model, param_grid, n_iter=10, cv=3, scoring='roc_auc',
                random_state=self.random_state, n_jobs=-1,
                error_score='raise', verbose=1
            )
            
            try:
                random_search.fit(X_train, y_train)
            except Exception as e:
                print(f"❌ XGBoost failed: {str(e)}")
                raise
            
            # Best model
            best_model = random_search.best_estimator_
            
            # Evaluate
            metrics = self.evaluate_model(best_model, X_test, y_test)
            
            # Log to MLflow
            mlflow.log_params(random_search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model, "model")
            
            print("\n✅ XGBoost Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            self.models['XGBoost'] = best_model
            return best_model, metrics
    
    def select_best_model(self):
        """Select the best model based on ROC-AUC"""
        print("\n📊 Compare models in MLflow UI to select the best one")
        print("   Run: mlflow ui")
    
    def save_model(self, model, filename='best_model.pkl'):
        """Save the best model"""
        import os
        os.makedirs('../models', exist_ok=True)
        
        joblib.dump(model, f'../models/{filename}')
        print(f"\n💾 Model saved to ../models/{filename}")

def main():
    # Set MLflow tracking URI
    import os
    os.makedirs('./mlruns', exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Credit_Risk_Modeling")
    
    # Load data
    try:
        df = pd.read_csv('../data/processed/data_with_target.csv')
        print(f"✅ Loaded data: {df.shape}")
    except FileNotFoundError:
        print("❌ Error: Could not find '../data/processed/data_with_target.csv'")
        print("   Ensure the file exists and the path is correct.")
        return
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data
    print("\n" + "="*50)
    print("📋 Preparing Data")
    print("="*50)
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    print(f"\n✅ Training set: {X_train.shape}")
    print(f"✅ Test set: {X_test.shape}")
    
    # Train models
    print("\n" + "="*50)
    print("🤖 Training Logistic Regression")
    print("="*50)
    lr_model, lr_metrics = trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*50)
    print("🌲 Training Random Forest")
    print("="*50)
    rf_model, rf_metrics = trainer.train_random_forest(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*50)
    print("🚀 Training XGBoost")
    print("="*50)
    xgb_model, xgb_metrics = trainer.train_gradient_boosting(X_train, y_train, X_test, y_test)
    
    # Compare models
    print("\n" + "="*50)
    print("🏆 Model Comparison (ROC-AUC)")
    print("="*50)
    print(f"Logistic Regression: {lr_metrics['roc_auc']:.4f}")
    print(f"Random Forest:       {rf_metrics['roc_auc']:.4f}")
    print(f"XGBoost:             {xgb_metrics['roc_auc']:.4f}")
    
    # Select best model
    best_model_name, best_model = max(
        [('Logistic Regression', lr_model, lr_metrics['roc_auc']), 
         ('Random Forest', rf_model, rf_metrics['roc_auc']), 
         ('XGBoost', xgb_model, xgb_metrics['roc_auc'])],
        key=lambda x: x[2]
    )[:2]
    
    print(f"\n🎯 Best Model: {best_model_name}")
    
    # Save best model
    trainer.save_model(best_model)
    
    print("\n✅ Training complete! View results in MLflow UI:")
    print("   Run: mlflow ui")

if __name__ == "__main__":
    main()