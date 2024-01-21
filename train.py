import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nltk_utils import vectorize, tokenize, lemmatize_token
from model import NeuralNet, ChatDataset

with open('./dataset.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        # токенизіруємо
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# ігноруємо пунктуаційні знаки та лематизуємо кожне слово
ignore_words = ['?', '.', '!']
all_words = [lemmatize_token(w) for w in all_words if w not in ignore_words]

# видаляємо дублікати та сортуємо
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Ініціалізуємо навчальні дані
X_test = []
y_test = []
for (pattern_sentence, tag) in xy:
    # Векторизуємо кожен патерн та зберігаємо у X_test
    vector = vectorize(pattern_sentence, all_words)
    X_test.append(vector)
    # в y_test зберігаємо індекс тега
    label = tags.index(tag)
    y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Гіпер-параметри
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
hidden_size = 8
input_size = len(all_words)
num_classes = len(tags)

dataset = ChatDataset(X_test, y_test)
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Ініціалізація функцій втрати та оптимізатора
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Тренування моделі
for epoch in range(num_epochs):
    for (words, labels) in loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        # Оптимізація
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": num_classes,
    "all_words": all_words,
    "tags": tags,
}

FILE = "trained_model.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
