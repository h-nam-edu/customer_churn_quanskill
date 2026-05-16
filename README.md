# Telco Customer Churn Pipeline

An end-to-end pipeline analyzing telecom customer data, building mathematical insights, and processing features to support a machine learning pipeline. This project was completed as part of the AI Engineer Foundation.

## Project Structure

```text
telco-churn-project/
├── data/
│   ├── raw/
│   │   └── telco_customer_churn.csv
│   └── processed/
│       └── clean_data.csv
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── numpy_tasks.py
│   ├── math_tasks.py
│   ├── pandas_tasks.py
│   └── utils.py
├── outputs/
│   └── visuals.png
├── requirements.txt
└── README.md
```

## Setup and Installation

1. Clone or navigate to the project root directory.
2. Install dependencies:
```bash
pip install -r requirements.txt

```


3. Place the raw dataset in `data/raw/telco_customer_churn.csv`.

## Execution

You can run the modules locally via scripts or execute everything step-by-step using the Google Colab/Jupyter notebook located in `notebooks/analysis.ipynb`.

To run modules individually:

```bash
python src/utils.py
python src/numpy_tasks.py
python src/math_tasks.py
python src/pandas_tasks.py

```

## Key Findings

* **Contract Duration:** Month-to-month contracts exhibit drastically higher churn rates compared to one or two-year commitments.
* **Tenure Risk:** Churn probability drops significantly after a customer crosses their first 12 months of service.
* **Financial Catalyst:** Customers who churned consistently possessed higher median monthly charges than retained users, indicating high price sensitivity.

## Outputs

* `clean_data.csv`: Missing values handled, categorical columns one-hot encoded, and fully ready for ML models.
* `visuals.png`: Correlation matrix, feature distributions, and bivariate analysis plots.

```

```
