'''
Fraud Detection Model Training Pipeline
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

from utils import (
    load_data, plot_class_distribution, 
    evaluate_model, plot_roc_pr_curves, explain_predictions
)

def create_features(df):
    '''Feature engineering - add time-based features'''
    df = df.copy()
    
    # Time features (Time is in seconds from first transaction)
    df['Hour'] = (df['Time'] / 3600) % 24
    df['Day'] = (df['Time'] / 86400)
    
    # Amount features
    df['Amount_log'] = np.log1p(df['Amount'])
    
    return df

def main():
    print("="*60)
    print("FRAUD DETECTION MODEL TRAINING PIPELINE")
    print("="*60)
    
    # 1. Load Data
    print("[1/7] Loading data...")
    df = load_data('data/creditcard.csv')
    
    # 2. Feature Engineering
    print("[2/7] Engineering features...")
    df = create_features(df)
    
    # Prepare features and target
    X = df.drop(['Class', 'Time'], axis=1)  # Drop Time as we engineered features from it
    y = df['Class']
    
    feature_names = X.columns.tolist()
    print(f"Total features: {len(feature_names)}")
    
    # 3. Train-Test Split
    print("[3/7] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 4. Handle Class Imbalance with SMOTE
    print("[4/7] Applying SMOTE for class balance...")
    smote = SMOTE(random_state=42, sampling_strategy=0.5)  # Make fraud 50% of majority
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_resampled)} samples")
    print(f"Fraud ratio: {y_train_resampled.mean()*100:.2f}%")
    
    # 5. Train Models
    print("[5/7] Training models...")
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=267,
            max_depth=11,
            min_samples_split=9,
            min_samples_leaf=3,
            max_features='sqrt',
            class_weight='balanced', 
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=400,          
            max_depth=6,               
            learning_rate=0.05,       
            subsample=0.8,             
            colsample_bytree=0.8,     
            scale_pos_weight=ratio,   
            tree_method='hist',        
            eval_metric='auc',
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"Training {name}...")
        if name=="XGBoost":
            model.fit(X_train, y_train)
        else:
            model.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate
        eval_results = evaluate_model(model, X_test, y_test, name)
        results[name] = eval_results
        
        # Track best model by f1 score
        if eval_results['f1'] > best_score:
            best_score = eval_results['f1']
            best_model = (name, model)
    
    # 6. Visualizations for best model
    print(f"[6/7] Creating visualizations for best model: {best_model[0]}...")
    plot_roc_pr_curves(y_test, results[best_model[0]]['probabilities'])
    
    # 7. SHAP Explanations
    print(f"[7/7] Generating SHAP explanations...")
    if isinstance(best_model[1], (RandomForestClassifier, XGBClassifier)):
        explain_predictions(best_model[1], X_test, feature_names)
    
    # Saving best model
    print(f"{'='*60}")
    print(f"Saving best model: {best_model[0]}")
    joblib.dump(best_model[1], 'models/fraud_detection_model.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print(f"Model saved to: models/fraud_detection_model.pkl")
    print(f"{'='*60}")
    
    return best_model, results

if __name__ == "__main__":
    import os
    os.makedirs('models', exist_ok=True)
    best_model, results = main()