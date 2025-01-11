import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# active and use gpu resource
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a random dataset
torch.manual_seed(42)
X = torch.linspace(0, 10, 100).reshape(-1, 1).to(device)
y = (X * 2.5 + 7.0 + torch.rand_like(X) * 2).to(device)

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
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    # IMPLEMENT QAHAN
    optimizer.step()
    print(f"epoch {epoch}, loss {loss}")
    
# visualize the predictions and its losses
preds = model(X).detach()
plt.scatter(X, y, label="Original Data")
plt.plot(X, preds, color="red", label="Fitted Line")
plt.legend()
plt.title("Linear Regression with QAHA")
plt.show()