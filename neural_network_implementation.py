import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

# Define the AND gate dataset
X_and = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_and = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

# Define the OR gate dataset
X_or = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_or = torch.tensor([[0], [1], [1], [1]], dtype=torch.float32)

# Define the XOR gate dataset
X_xor = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_xor = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define the neural network model for AND gate
class ANDModel(nn.Module):
    def __init__(self):
        super(ANDModel, self).__init__()
        self.fc = nn.Linear(2, 1)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        return x

# Define the neural network model for OR gate
class ORModel(nn.Module):
    def __init__(self):
        super(ORModel, self).__init__()
        self.fc = nn.Linear(2, 1)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        return x

# Define the neural network model for XOR gate
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model_and = ANDModel()
model_or = ORModel()
model_xor = XORModel()

# Define loss function and optimizer for each gate
criterion_and = nn.BCELoss()
optimizer_and = optim.SGD(model_and.parameters(), lr=0.1)

criterion_or = nn.BCELoss()
optimizer_or = optim.SGD(model_or.parameters(), lr=0.1)

criterion_xor = nn.BCELoss()
optimizer_xor = optim.SGD(model_xor.parameters(), lr=0.1)


print('----------------Train Models----------------')

# Train the models
num_epochs = 1000
for epoch in range(num_epochs):
    # Train AND gate model
    optimizer_and.zero_grad()
    outputs_and = model_and(X_and)
    loss_and = criterion_and(outputs_and, y_and)
    loss_and.backward()
    optimizer_and.step()

    # Train OR gate model
    optimizer_or.zero_grad()
    outputs_or = model_or(X_or)
    loss_or = criterion_or(outputs_or, y_or)
    loss_or.backward()
    optimizer_or.step()

    # Train XOR gate model
    optimizer_xor.zero_grad()
    outputs_xor = model_xor(X_xor)
    loss_xor = criterion_xor(outputs_xor, y_xor)
    loss_xor.backward()
    optimizer_xor.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'AND Loss: {loss_and.item():.4f}, OR Loss: {loss_or.item():.4f}, XOR Loss: {loss_xor.item():.4f}')


print('----------------Test Models----------------')

# Test the models
with torch.no_grad():
    test_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    predictions_and = model_and(test_inputs)
    predictions_or = model_or(test_inputs)
    predictions_xor = model_xor(test_inputs)

    print("\nPredictions for AND gate:")
    print(predictions_and.round())
    print("\nPredictions for OR gate:")
    print(predictions_or.round())
    print("\nPredictions for XOR gate:")
    print(predictions_xor.round())
    
    
print('----------------Evalute Models----------------')
def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        outputs = model(X_test)
        predictions = outputs.round()
        accuracy = (predictions == y_test).sum().item() / len(y_test)
        return accuracy

# Evaluate AND gate
accuracy_and = evaluate_model(model_and, X_and, y_and)
print(f"Accuracy for AND gate: {accuracy_and:.2f}")

# Evaluate OR gate
accuracy_or = evaluate_model(model_or, X_or, y_or)
print(f"Accuracy for OR gate: {accuracy_or:.2f}")

# Evaluate XOR gate
accuracy_xor = evaluate_model(model_xor, X_xor, y_xor)
print(f"Accuracy for XOR gate: {accuracy_xor:.2f}")


print('----------------Visualization of logic gates----------------')

def visualize_logic_gate(model, X, y, gate_name):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    
    Z = model(torch.tensor(X_grid, dtype=torch.float32)).detach().numpy().reshape(xx.shape)
    
    # Plot the boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title(f'Decision Boundary for {gate_name} Gate')

    # training data points
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap=plt.cm.RdBu, edgecolors='k')
    plt.colorbar(label='Output')
    plt.show()

# AND gate decision boundary and predictions
visualize_logic_gate(model_and, X_and, y_and, "AND")

# OR gate decision boundary and predictions
visualize_logic_gate(model_or, X_or, y_or, "OR")

# XOR gate decision boundary and predictions
visualize_logic_gate(model_xor, X_xor, y_xor, "XOR")
