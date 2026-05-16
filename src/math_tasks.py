import numpy as np
from sklearn.linear_model import Ridge, Lasso
from utils import get_data

df, X_num = get_data()

# Q1
rank = np.linalg.matrix_rank(X_num)

# Q2
y = (df['Churn'] == 'Yes').astype(int).values
X_lr = df[['tenure', 'MonthlyCharges']].values
X_bias = np.c_[np.ones(X_lr.shape[0]), X_lr]
w = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y

# Q3
p_churn = np.mean(y)
p_churn_mtm = np.mean(y[df['Contract'] == 'Month-to-month'])

# Q4
n = len(y)
preds = X_bias @ w
grad = (-2/n) * X_bias.T @ (y - preds)

# Q5
var_monthly = np.var(X_num[:, 1], ddof=1)
cov_mt = np.cov(X_num[:, 1], X_num[:, 0])[0, 1]

# Q6
cov_matrix = np.cov(X_num.T)
evals, evecs = np.linalg.eig(cov_matrix)

# Q7
x_min = np.min(X_num, axis=0)
x_max = np.max(X_num, axis=0)
x_norm = (X_num - x_min) / (x_max - x_min)
U, S, VT = np.linalg.svd(x_norm, full_matrices=False)

# Q8
w_gd = np.zeros(X_bias.shape[1])
lr = 1e-5
for _ in range(100):
    pred_gd = X_bias @ w_gd
    w_gd -= lr * ((-2/n) * X_bias.T @ (y - pred_gd))

# Q9
high_charge = df['MonthlyCharges'] > df['MonthlyCharges'].median()
p_high = np.mean(high_charge)
p_high_churn = np.mean(high_charge[y == 1])
p_churn_high = (p_high_churn * p_churn) / p_high

# Q10
lasso = Lasso(alpha=0.1).fit(X_lr, y)
ridge = Ridge(alpha=1.0).fit(X_lr, y)