import torch
from torch.utils.data import DataLoader, TensorDataset

from consts import num_classes
from model import NeuralNet
from train import X_test, y_test

data = torch.load('trained_model.pth')

input_size = data["input_size"]
hidden_size = data["hidden_size"]
words = data['words']
tags = data['tags']
model_state = data["model_state"]


model = NeuralNet(input_size, hidden_size, num_classes)
model.load_state_dict(model_state)
model.eval()  # Переведіть модель у режим оцінки

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in DataLoader(TensorDataset(X_test, y_test)):
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test data: {accuracy}%')

