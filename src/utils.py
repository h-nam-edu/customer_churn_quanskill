import pandas as pd
import numpy as np

def get_data():
    path = '../data/raw/telco_customer_churn.csv'
    df = pd.read_csv(path)
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    
    X_num = df[['tenure', 'MonthlyCharges', 'TotalCharges']].values
    return df, X_num