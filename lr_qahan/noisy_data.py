import torch
import torch.nn as nn
import torch.optim as optimm
import matplotlib.pyplot as plt
import numpy as np

# planting random seed
torch.manual_seed(42)
np.random.seed(42)

# create main dataset
X = torch.linspace(0, 10, 100).reshape(-1, 1)
y = X * 2.5 + 7.0 + torch.rand_like(X) * 2

# create noise for the input dataset
X_noise = torch.randn(100, 2) * 5
X_unnecessary = torch.linspace(10, 20, 100).reshape(-1, 1) * torch.randn(100, 1)

X_combined = torch.cat([X, X_noise, X_unnecessary], dim=1)
X_interaction = X ** 2 + X * 3

# create outlier for the outpt
y[::10] += torch.randn(10, 1) * 20

# combine two dataset
X_final = torch.cat([X_combined, X_interaction], dim=1)

# visualize the result dataset
plt.scatter(X_final[:, 0].numpy(), y.numpy(), label="Noisy dataset")
plt.title("Synthetic data with noises and outliers")
plt.legend()
plt.show()