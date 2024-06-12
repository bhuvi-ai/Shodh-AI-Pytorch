import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

# Define the neural network model for MNIST
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 output classes for digits 0-9
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load and preprocess dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define loss function and optimizer
model = MNISTModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 10
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_losses[-1]:.4f}")

    # Evaluate the model
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}, Test Accuracy: {100. * correct / total:.2f}%")

# Plot training and test losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Losses')
plt.legend()
plt.show()

# Visualize the weights of the first layer
weights = model.fc1.weight.detach().numpy()
plt.figure(figsize=(10, 6))
plt.imshow(weights, cmap='gray')
plt.title('Weights of the First Layer')
plt.xlabel('Neuron Index')
plt.ylabel('Input Pixel Index')
plt.colorbar()
plt.show()

# Visualize the decision boundary for a given input image
def visualize_decision_boundary(image):
    with torch.no_grad():
        output = model(torch.tensor(image).unsqueeze(0))
    probabilities = nn.functional.softmax(output, dim=1).numpy()[0]
    plt.figure(figsize=(8, 6))
    plt.bar(range(10), probabilities, color='blue')
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.title('Model Predictions')
    plt.xticks(range(10), [str(i) for i in range(10)])
    plt.show()



def predict_digit(image):
    image = torch.tensor(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

def preprocess_image(image):
    image = np.array(image)
    image = image.mean(axis=2)  # Convert to grayscale
    image = 255 - image  # Invert colors
    image = image / 255.0  # Normalize
    return image

input_interface = gr.inputs.Image(preprocessing_fn=preprocess_image, shape=(28, 28), label="Input Digit Image")

output_interface = gr.outputs.Label(num_top_classes=1, label="Predicted Digit")

gr.Interface(fn=predict_digit, inputs=input_interface, outputs=output_interface, title="MNIST Digit Recognizer").launch()