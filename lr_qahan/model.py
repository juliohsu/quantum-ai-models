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

# splitting clean and noisy dataset
X_clean = torch.tensor(clean_dataset["0"], dtype=torch.float32, device=device).reshape(-1, 1)
y_clean = torch.tensor(clean_dataset["1"], dtype=torch.float32, device=device).reshape(-1, 1)
X_noisy = torch.tensor(noisy_dataset[["0", "1", "2", "3", "4"]], dtype=torch.float32, device=device)
y_noisy = torch.tensor(noisy_dataset["5"], dtype=torch.float32, device=device)

# create the model class
class linear_reg_qaha(nn.Module):
    def __init__(self):
        super(linear_reg_qaha, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

# define model hyperparamters
model = linear_reg_qaha().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# train the model
epochs = 500
for epoch in range(epochs):
    pred = model(X_clean)
    loss = criterion(pred, y_clean)
    optimizer.zero_grad()
    loss.backward()
    # IMPLEMENT QAHAN
    optimizer.step()
    print(f"epoch {epoch}, loss {loss}")
    
# visualize the predictions and its losses
prediction = model(X_clean).detach()
plt.scatter(X_clean.to("cpu"), y_clean.to("cpu"), label="Original Data")
plt.plot(X_clean.to("cpu"), prediction.to("cpu"), color="red", label="Fitted Line")
plt.legend()
plt.title("Linear Regression with QAHA")
plt.show()