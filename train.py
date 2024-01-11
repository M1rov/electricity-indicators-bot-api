import json

import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from consts import hidden_size, num_classes, num_epochs
from model import NeuralNet
from utils import tokenize, lemmatize_token, vectorize, encode_tags

# Завантаження даних з JSON файлу
with open('dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Обробка даних
words = []
tags = []
for intent in data:
    tag = intent['tag']
    for pattern in intent['patterns']:
        tokens = tokenize(pattern)
        lemmatized_pattern = ' '.join([lemmatize_token(token) for token in tokens])
        words.append(lemmatized_pattern)
        tags.append(tag)

# Векторизація тексту
X = vectorize(words)

# # Кодування міток
y = encode_tags(tags)


# Розділення на навчальні та тестові набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Конвертація даних у тензори PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Створення DataLoader для навчальних даних
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


input_size = len(X_train[0])
model = NeuralNet(input_size, hidden_size, num_classes)
# Втрати та оптимізатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Навчання моделі
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # Передача даних через модель
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Оптимізація
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Виведення інформації про прогрес
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Збереження навченої моделі
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "words": words,
    "tags": tags
}

torch.save(data, 'trained_model.pth')


