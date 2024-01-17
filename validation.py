import torch
from torch.utils.data import DataLoader, TensorDataset

from model import NeuralNet

data = torch.load('trained_model.pth')

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
test_data = data["test_data"]


model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()  # Переведіть модель у режим оцінки

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in DataLoader(test_data):
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test data: {accuracy}%')

