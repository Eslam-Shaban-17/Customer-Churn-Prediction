"""
Model Training Module
Trains and compares multiple machine learning models
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and return results
    
    Parameters:
    -----------
    X_train, X_test : np.ndarray
        Scaled training and test features
    y_train, y_test : np.ndarray
        Training and test labels
        
    Returns:
    --------
    tuple
        (models_dict, results_dict)
    """
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            solver='lbfgs'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10,
            min_samples_split=20,
            n_jobs=-1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_scores': cv_scores
        }
        
        print(f"  ✓ Accuracy: {accuracy:.4f}")
        print(f"  ✓ ROC-AUC: {roc_auc:.4f}")
        print(f"  ✓ CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return models, results

def save_models(models):
    """
    Save trained models to disk
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    """
    for name, model in models.items():
        filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, filename)
        print(f"  ✓ {name} saved to {filename}")