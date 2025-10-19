"""
Data Preprocessing Module
Handles data loading, cleaning, and initial transformations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(file_path):
    """
    Load and clean the customer churn dataset
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Handle missing values in TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Convert target variable to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Save basic statistics
    print(f"\nDataset Statistics:")
    print(f"  - Total Customers: {len(df)}")
    print(f"  - Churned: {df['Churn'].sum()} ({df['Churn'].mean()*100:.1f}%)")
    print(f"  - Retained: {(df['Churn']==0).sum()} ({(1-df['Churn'].mean())*100:.1f}%)")
    print(f"  - Missing values handled: {df.isnull().sum().sum()}")
    
    # Generate EDA visualizations
    _generate_eda_plots(df)
    
    return df

def _generate_eda_plots(df):
    """Generate exploratory data analysis plots"""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Customer Churn Analysis - Exploratory Data Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. Churn Distribution
    ax1 = axes[0, 0]
    churn_counts = df['Churn'].value_counts()
    ax1.pie(churn_counts, labels=['Retained', 'Churned'], autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'], startangle=90)
    ax1.set_title('Overall Churn Distribution')
    
    # 2. Churn by Contract
    ax2 = axes[0, 1]
    contract_churn = df.groupby('Contract')['Churn'].mean() * 100
    contract_churn.plot(kind='bar', ax=ax2, color='steelblue')
    ax2.set_title('Churn Rate by Contract Type')
    ax2.set_ylabel('Churn Rate (%)')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    # 3. Tenure Distribution
    ax3 = axes[0, 2]
    df[df['Churn']==0]['tenure'].hist(bins=30, alpha=0.7, label='Retained', 
                                       color='green', ax=ax3)
    df[df['Churn']==1]['tenure'].hist(bins=30, alpha=0.7, label='Churned', 
                                       color='red', ax=ax3)
    ax3.set_title('Tenure Distribution')
    ax3.set_xlabel('Tenure (months)')
    ax3.legend()
    
    # 4. Monthly Charges
    ax4 = axes[1, 0]
    df.boxplot(column='MonthlyCharges', by='Churn', ax=ax4)
    ax4.set_title('Monthly Charges by Churn Status')
    ax4.set_xlabel('Churn (0=No, 1=Yes)')
    plt.sca(ax4)
    plt.xticks([1, 2], ['Retained', 'Churned'])
    
    # 5. Internet Service
    ax5 = axes[1, 1]
    internet_churn = df.groupby('InternetService')['Churn'].mean() * 100
    internet_churn.plot(kind='bar', ax=ax5, color='coral')
    ax5.set_title('Churn Rate by Internet Service')
    ax5.set_ylabel('Churn Rate (%)')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
    
    # 6. Payment Method
    ax6 = axes[1, 2]
    payment_churn = df.groupby('PaymentMethod')['Churn'].mean() * 100
    payment_churn.plot(kind='bar', ax=ax6, color='mediumpurple')
    ax6.set_title('Churn Rate by Payment Method')
    ax6.set_ylabel('Churn Rate (%)')
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
    
    # 7. Gender Distribution
    ax7 = axes[2, 0]
    gender_churn = df.groupby('gender')['Churn'].mean() * 100
    gender_churn.plot(kind='bar', ax=ax7, color='teal')
    ax7.set_title('Churn Rate by Gender')
    ax7.set_ylabel('Churn Rate (%)')
    
    # 8. Senior Citizen
    ax8 = axes[2, 1]
    senior_churn = df.groupby('SeniorCitizen')['Churn'].mean() * 100
    senior_churn.plot(kind='bar', ax=ax8, color='orange')
    ax8.set_title('Churn Rate: Senior vs Non-Senior')
    ax8.set_ylabel('Churn Rate (%)')
    ax8.set_xticklabels(['Non-Senior', 'Senior'], rotation=0)
    
    # 9. Correlation Heatmap
    ax9 = axes[2, 2]
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax9, cbar_kws={'shrink': 0.8})
    ax9.set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/eda_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ EDA visualizations saved to outputs/figures/eda_analysis.png")