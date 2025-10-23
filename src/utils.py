import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score,f1_score
)
import shap

def load_data(filepath='data/creditcard.csv'):
    '''Load and return the fraud detection dataset'''
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    return df

def plot_class_distribution(y):
    '''Visualize class imbalance'''
    plt.figure(figsize=(8, 5))
    pd.Series(y).value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title('Class Distribution (0: Legitimate, 1: Fraud)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()

def evaluate_model(model, X_test, y_test, model_name="Model"):
    '''Comprehensive model evaluation'''
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"{'='*60}")
    print(f"{model_name} Evaluation")
    print(f"{'='*60}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(cm)
    
    # Classification Report
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Average Precision (better for imbalanced data)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    print(f"Average Precision Score: {avg_precision:.4f}")

    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.4f}")
    
    # Calculate cost (example: FP costs $10, FN costs $100)
    tn, fp, fn, tp = cm.ravel()
    cost = (fp * 10) + (fn * 100)
    print(f"Business Cost Analysis:")
    print(f"False Positives (legitimate flagged as fraud): {fp} → Cost: ${fp * 10}")
    print(f"False Negatives (fraud missed): {fn} → Cost: ${fn * 100}")
    print(f"Total Cost: ${cost}")
    
    return {
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def plot_roc_pr_curves(y_test, y_pred_proba):
    '''Plot ROC and Precision-Recall curves'''
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    axes[0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    axes[1].plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})', linewidth=2)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def explain_predictions(model, X_test, feature_names, num_samples=100):
    '''Generate SHAP explanations for model predictions'''
    print("Generating SHAP explanations (this may take a minute)...")
    
    # Use a sample for speed
    X_sample = X_test.sample(min(num_samples, len(X_test)), random_state=42)
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.show()
    
    return explainer, shap_values