"""
Feature Engineering Module
Creates new features and prepares data for modeling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def engineer_features(df):
    """
    Engineer features and prepare data for modeling
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe
        
    Returns:
    --------
    tuple
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    """
    # Create copy for processing
    df_processed = df.copy()
    
    # Drop customerID (not useful for prediction)
    df_processed = df_processed.drop('customerID', axis=1)
    
    # Create new features
    df_processed = _create_new_features(df_processed)
    
    # Encode categorical variables
    df_processed, label_encoders = _encode_categorical(df_processed)
    
    # Separate features and target
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save artifacts
    df_processed.to_csv('data/processed/processed_churn_data.csv', index=False)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    print(f"✓ Feature engineering complete")
    print(f"  - New features created: 3")
    print(f"  - Total features: {len(feature_names)}")
    print(f"  - Categorical features encoded: {len(label_encoders)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

def _create_new_features(df):
    """Create new features from existing ones"""
    
    # Tenure groups
    df['TenureGroup'] = pd.cut(df['tenure'], 
                               bins=[0, 12, 24, 48, 72], 
                               labels=[0, 1, 2, 3])
    df['TenureGroup'] = df['TenureGroup'].astype(int)
    
    # Average charges per month
    df['ChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # Total number of services
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['TotalServices'] = (df[service_cols] != 'No').sum(axis=1)
    
    return df

def _encode_categorical(df):
    """Encode categorical variables"""
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target if present
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')
    
    # Encode each categorical column
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    return df, label_encoders