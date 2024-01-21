import json
import random

import torch

from .message_handlers import message_handlers
from model import NeuralNet
from nltk_utils import tokenize, vectorize

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


def send_response(message, bot):
    sentence = tokenize(message.text)
    X = vectorize(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(prob.item())
    if prob.item() < 0.75:
        raise Exception(
            "Нажаль, я поки що не можу відповісти на ваше питання. Проте, ваше повідомлення було відправлено у " \
            "підтримку і буде оброблено у найближчий час.")

    for intent in intents:
        if tag == intent["tag"]:
            if 'handler' in intent:
                return message_handlers[intent['handler']](bot).start(message)

            response = random.choice(intent['responses'])
            bot.send_message(message.chat.id, response)
