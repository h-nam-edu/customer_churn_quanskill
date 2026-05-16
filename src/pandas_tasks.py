import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('../outputs/figures', exist_ok=True)
os.makedirs('../outputs/tables', exist_ok=True)

df = pd.read_csv('../data/raw/telco_customer_churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
df['churn_num'] = (df['Churn'] == 'Yes').astype(int)

df['AvgCharge'] = df['TotalCharges'] / df['tenure'].replace(0, 1)

df['tenure_grp'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 100], labels=['0-12', '12-24', '24+'])
df['charge_bin'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 200], labels=['Low', 'Med', 'High'])

contract_stats = df.groupby('Contract').agg({'Churn': 'count', 'churn_num': 'mean'})
contract_stats.to_csv('../outputs/tables/contract_stats.csv')

pivot = df.pivot_table(index='Contract', columns='Churn', values='customerID', aggfunc='count')
pivot.to_csv('../outputs/tables/contract_churn_pivot.csv')

tenure_stats = df.groupby('tenure_grp', observed=False)['churn_num'].mean().to_frame()
charge_stats = df.groupby('charge_bin', observed=False)['churn_num'].mean().to_frame()
tenure_stats.to_csv('../outputs/tables/tenure_churn_stats.csv')
charge_stats.to_csv('../outputs/tables/charge_churn_stats.csv')

plt.figure()
plt.hist(df['MonthlyCharges'], bins=30)
plt.title('Monthly Charges')
plt.savefig('../outputs/figures/monthly_charges_hist.png')
plt.close()

plt.figure()
sns.countplot(x=df['Churn'])
plt.title('Overall Churn')
plt.savefig('../outputs/figures/churn_count.png')
plt.close()

plt.figure()
sns.heatmap(df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr(), annot=True)
plt.title('Correlation')
plt.savefig('../outputs/figures/correlation_heatmap.png')
plt.close()

plt.figure()
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Charges by Churn')
plt.savefig('../outputs/figures/monthly_charges_boxplot.png')
plt.close()

cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
            'InternetService', 'Contract', 'PaperlessBilling', 'PaymentMethod']
df_final = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df_final = df_final.drop(columns=['customerID', 'Churn', 'tenure_grp', 'charge_bin'])
df_final.to_csv('../data/processed/clean_telco_data.csv', index=False)

with open('../outputs/results.txt', 'w') as f:
    f.write(f"Total customers evaluated: {len(df)}\n")
    f.write(f"Overall churn rate: {df['churn_num'].mean():.4f}\n")