import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def target_function(x):
    """Target function with multiple oscillations."""
    return torch.sin(10 * np.pi * x)


class DeepNetwork(nn.Module):
    def __init__(self):
        super(DeepNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 20), nn.ReLU(),
            nn.Linear(20, 20), nn.ReLU(),
            nn.Linear(20, 20), nn.ReLU(),
            nn.Linear(20, 20), nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.layers(x)


class ShallowNetwork(nn.Module):
    def __init__(self):
        super(ShallowNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 1000), nn.ReLU(),
            nn.Linear(1000, 1)
        )

    def forward(self, x):
        return self.layers(x)


def train_model(model, optimizer, criterion, x_train, y_train, epochs=1000):
    """Train the model on the given data."""
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(x_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
    return model


# Generate dataset
np.random.seed(42)
torch.manual_seed(42)
x_train = torch.linspace(0, 1, 1000).unsqueeze(1)
y_train = target_function(x_train)

# Initialize models, optimizers, and loss function
deep_model = DeepNetwork()
shallow_model = ShallowNetwork()
deep_optimizer = optim.Adam(deep_model.parameters(), lr=0.01)
shallow_optimizer = optim.Adam(shallow_model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Train models
deep_model = train_model(deep_model, deep_optimizer,
                         criterion, x_train, y_train)
shallow_model = train_model(
    shallow_model, shallow_optimizer, criterion, x_train, y_train)

# Evaluate models
x_test = torch.linspace(0, 1, 500).unsqueeze(1)
y_test = target_function(x_test)
deep_predictions = deep_model(x_test).detach()
shallow_predictions = shallow_model(x_test).detach()

deep_mse = criterion(deep_predictions, y_test).item()
shallow_mse = criterion(shallow_predictions, y_test).item()
print(f"Deep Network MSE: {deep_mse:.4f}")
print(f"Shallow Network MSE: {shallow_mse:.4f}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(x_test, y_test, label='Target Function', color='black')
plt.plot(x_test, deep_predictions,
         label=f'Deep Network (MSE: {deep_mse:.4f})', color='blue')
plt.plot(x_test, shallow_predictions,
         label=f'Shallow Network (MSE: {shallow_mse:.4f})', color='red')
plt.legend()
plt.title('Comparison of Deep and Shallow Networks on Noisy Data')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.show()
