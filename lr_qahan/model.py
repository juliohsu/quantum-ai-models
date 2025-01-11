import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

clean_dataset = pd.read_csv("dataset_clean")
noisy_dataset = pd.read_csv("dataset_noisy")

# active and use gpu resource
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# plant data seed format
torch.manual_seed(42)

# clean and noisy dataset
X_clean = torch.tensor(clean_dataset["0"], dtype=torch.float32, device=device).reshape(-1, 1)
y_clean = torch.tensor(clean_dataset["1"], dtype=torch.float32, device=device).reshape(-1, 1)

X_noisy = torch.tensor(noisy_dataset[["0", "1", "2", "3", "4"]].values, dtype=torch.float32, device=device).reshape(-1, 5)
y_noisy = torch.tensor(noisy_dataset["5"], dtype=torch.float32, device=device).reshape(-1, 1)

# linear regression with qahan technique
class HardAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HardAttention, self).__init__()
        self.attention_weights = nn.Parameter(torch.rand(input_dim, output_dim))
    def forward(self, x):
        attention_scores = torch.matmul(x, self.attention_weights)
        hard_mask = (attention_scores > torch.median(self.attention_weights, dim=0, keepdim=True).values).float()
        return x * hard_mask

class LRQAHAN(nn.Module):
    def __init__(self):
        super(LRQAHAN, self).__init__()
        self.hard_attention = HardAttention(5, 1)
        self.linear = nn.Linear(5, 1)
    def forward(self, x):
        x = self.hard_attention(x)
        return self.linear(x)

def quantum_annealing_update(param, grad, lr=0.01, gamma=0.9):
    if grad is not None:
        return grad * lr * torch.exp(-gamma * torch.abs(param))
    else:
        return torch.zeros_like(param)

# linear regression
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

# define model hyperparamters
model = LRQAHAN().to(device)
model2 = LR().to(device)

criterion = nn.MSELoss()
criterion2 = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer2 = optim.SGD(model2.parameters(), lr=0.01)

# train the model
epochs = 500
loss = 0.0
for epoch in range(epochs):

    pred = model(X_noisy)
    loss = criterion(pred, y_noisy)

    if (torch.isnan(loss) or torch.isinf(loss)):
        print("the model has broken!!!!!!")
        break

    optimizer.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        for param in model.parameters():
            param -= quantum_annealing_update(param, param.grad, lr=0.01)

    print(f"epoch {epoch}, loss {loss.item()}")

# visualize the predictions and its losses
if not (torch.isnan(loss) or torch.isinf(loss)):
    plt.figure(figsize=(10, 6))
    prediction = model(X_noisy).detach()
    for i in range(X_noisy.shape[1]):
        plt.scatter(X_noisy[:, i].to("cpu"), y_noisy.to("cpu"), label=f"Feature {i}")
    plt.plot(X_noisy.to("cpu")[:, 0], prediction.to("cpu"), color="red", label="Fitted Line")
    plt.legend()
    plt.title("Linear Regression with QAHA")
    plt.show()