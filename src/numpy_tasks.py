import numpy as np
from utils import get_data

df, X_num = get_data()

# Q1
print(X_num.shape, X_num.dtype)

# Q2
means = np.mean(X_num, axis=0)
medians = np.median(X_num, axis=0)
stds = np.std(X_num, axis=0)

# Q3
x_min = np.min(X_num, axis=0)
x_max = np.max(X_num, axis=0)
x_norm = (X_num - x_min) / (x_max - x_min)

# Q4
filtered = X_num[(X_num[:, 0] > 12) & (X_num[:, 1] > means[1])]

# Q5
corr_matrix = np.corrcoef(X_num.T)

# Q6
X_sub = X_num[:1000] 
sq = np.sum(X_sub**2, axis=1)
dists = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * (X_sub @ X_sub.T), 0))

# Q7
x_zscore = (X_num - means) / stds

# Q8
v1, v2 = X_num[0], X_num[1]
cos_sim = (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Q9
cov_mat = np.cov(x_norm.T)
evals, evecs = np.linalg.eigh(cov_mat)
evecs_sorted = evecs[:, np.argsort(evals)[::-1]]
x_pca = x_norm @ evecs_sorted[:, :2]

# Q10
batches = np.array_split(X_num, 5)
batch_means = [np.mean(b, axis=0) for b in batches]