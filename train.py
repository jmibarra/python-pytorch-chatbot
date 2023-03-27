import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json', 'r') as file:
    intents = json.load(file)

# Inicializo mis listas
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        # No uso append por que es un array y no quiero armar una array de arrays
        all_words.extend(w)
        xy.append((w, tag))

# quiero excluir los puntos de exclamación y stemmear las palabras
ignore_words = ['?', '¿', '.', ',', '!', '¡']

all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentense, tag) in xy:
    bag = bag_of_words(pattern_sentense, all_words,)
    X_train.append(bag)

    label = tags.index(tag)  # CrossEntropyLoss
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

        # dataset[idx]
        def __getItem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples


# Hyperparametros
batch_size = 8


dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
