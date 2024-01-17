import json
import random

import torch
from model import NeuralNet
from utils import tokenize, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('dataset.json', 'r', encoding='utf-8') as json_file:
    intents = json.load(json_file)

data = torch.load('trained_model.pth')

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(prob.item())
    if prob.item() > 0.75:
        for intent in intents:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    raise Exception("Нажаль, я поки що не можу відповісти на ваше питання. Проте, ваше повідомлення було відправлено у " \
           "підтримку і буде оброблено у найближчий час.")

