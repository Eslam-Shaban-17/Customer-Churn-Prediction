"""
Customer Churn Prediction - Main Pipeline
Author: [Eslam Shaban]
Description: Orchestrates the complete ML pipeline for churn prediction
"""

import os
import sys
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import engineer_features
from src.model_training import train_models, save_models
from src.model_evaluation import evaluate_models, generate_reports

def main():
    """Execute the complete churn prediction pipeline"""
    
    print("="*80)
    print("CUSTOMER CHURN PREDICTION - COMPLETE PIPELINE")
    print("="*80)
    
    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)
    
    try:
        # Step 1: Load and clean data
        print("\n[1/5] Loading and cleaning data...")
        df_clean = load_and_clean_data('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        print(f"✓ Data loaded: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        
        # Step 2: Feature engineering
        print("\n[2/5] Engineering features...")
        X_train, X_test, y_train, y_test, feature_names = engineer_features(df_clean)
        print(f"✓ Features prepared: {X_train.shape[1]} features")
        print(f"✓ Train set: {X_train.shape[0]} samples")
        print(f"✓ Test set: {X_test.shape[0]} samples")
        
        # Step 3: Train models
        print("\n[3/5] Training models...")
        models, results = train_models(X_train, X_test, y_train, y_test)
        print("✓ Models trained successfully")
        
        # Step 4: Save models
        print("\n[4/5] Saving models...")
        save_models(models)
        print("✓ Models saved")
        
        # Step 5: Evaluate and generate reports
        print("\n[5/5] Evaluating models and generating reports...")
        evaluate_models(results, y_test, feature_names, X_train.shape[1])
        generate_reports(results, y_test)
        print("✓ Evaluation complete")
        
        # Summary
        print("\n" + "="*80)
        print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\n📊 Model Performance Summary:")
        print("-" * 60)
        for name, result in results.items():
            print(f"{name}:")
            print(f"  • Accuracy: {result['accuracy']:.4f}")
            print(f"  • ROC-AUC: {result['roc_auc']:.4f}")
            print(f"  • CV Score: {result['cv_scores'].mean():.4f} ± {result['cv_scores'].std():.4f}")
        
        print("\n📁 Generated Files:")
        print("-" * 60)
        print("✓ data/processed/processed_churn_data.csv")
        print("✓ models/logistic_regression_model.pkl")
        print("✓ models/random_forest_model.pkl")
        print("✓ models/scaler.pkl")
        print("✓ models/label_encoders.pkl")
        print("✓ outputs/figures/eda_analysis.png")
        print("✓ outputs/figures/model_comparison.png")
        print("✓ outputs/reports/classification_reports.txt")
        
        print("\n" + "="*80)
        print("Ready for deployment! 🚀")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure the dataset is placed at:")
        print("  data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        print("\nDownload from:")
        print("  https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()