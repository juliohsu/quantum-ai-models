import torch
import torch.nn as nn
import torch.optim as optimm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# planting random seed
torch.manual_seed(42)
np.random.seed(42)

# create main dataset
X = torch.linspace(0, 10, 100).reshape(-1, 1)
y = X * 2.5 + 7.0 + torch.rand_like(X) * 2
print(X)
# create noisy features for the dataset
X_noise = torch.randn(100, 2) * 5
X_unnecessary = torch.linspace(10, 20, 100).reshape(-1, 1) * torch.randn(100, 1)
X_combined = torch.cat([X, X_noise, X_unnecessary], dim=1)
print(X_combined)
# create one more feature polynomial functions, which can be insightful
X_final = torch.cat([X_combined, X ** 2 + X * 3], dim=1)
print(X_final)
# create outlier for 10th output sequences to mimic real world incertain
y[::10] += torch.randn(10, 1) * 20

# exporting the dataset (cat of features and target) into csv file
file = torch.cat([X_final, y], dim=1)
file2 = torch.cat([X, y], dim=1)

file = pd.DataFrame(file)
file2 = pd.DataFrame(file2)

file.to_csv("dataset_noisy", index=False)
file2.to_csv("dataset_clean", index=False)

# visualize the result dataset
plt.scatter(X_final[:, 0].numpy(), y.numpy(), label="Noisy dataset")
plt.title("Synthetic data with noises and outliers")
plt.legend()
plt.show()