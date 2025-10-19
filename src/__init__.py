"""
Customer Churn Prediction Package
Modular ML pipeline for predicting customer churn
"""

__version__ = '1.0.0'
__author__ = 'Eslam Shaban'

from .data_preprocessing import load_and_clean_data
from .feature_engineering import engineer_features
from .model_training import train_models, save_models
from .model_evaluation import evaluate_models, generate_reports

__all__ = [
    'load_and_clean_data',
    'engineer_features',
    'train_models',
    'save_models',
    'evaluate_models',
    'generate_reports'
]