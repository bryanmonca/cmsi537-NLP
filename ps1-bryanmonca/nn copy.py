import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


## Data
df = pd.read_pickle("data/labeled_data.p")
df = df.reset_index()
x = df.iloc[:, -1]  # tweets
y = df.iloc[:, -2]  # class


# Split data into Train 80%, Val 10%, Test 10%
idx = np.random.RandomState(seed=42).permutation(len(df)) # Creates permutations of all indices
train_split_idx = int(0.80 * len(df))
val_split_idx = int(0.90 * len(df))

train_idx = idx[:train_split_idx]
val_idx = idx[train_split_idx:val_split_idx]
test_idx = idx[val_split_idx:]

x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]
x_test, y_test = x[test_idx], y[test_idx]

# Data in a list of tuples
train_data = list(zip(x_train, y_train)) #zip tweet, class
val_data = list(zip(x_val, y_val))
test_data = list(zip(x_test, y_test))

## Tokenizer
def tokenize(text):
    return text.split()

## Vocabulary
def load_vocab(text):
    word_to_ix = {}
    for sent, label in text:
        for word in tokenize(sent): # tokenize
            word_to_ix.setdefault(word, len(word_to_ix))
    return word_to_ix

# Load Vocab
tok_to_ix = load_vocab(train_data)


## Sets to tensors - Vectorizing
def vectorize_data(data, train=False):
    vect_data = []
    if train:
        for text, label in data:
            x = [tok_to_ix[token] for token in tokenize(text)]
            x_train_tensor = torch.LongTensor(x)
            y_train_tensor = torch.tensor([label])
            vect_data.append([x_train_tensor, y_train_tensor])
    elif train==False:
        for text, label in data:
            x = [tok_to_ix[tok] for tok in tokenize(text) if tok in tok_to_ix]
            x_train_tensor = torch.LongTensor(x)
            y_train_tensor = torch.tensor([label])
            vect_data.append([x_train_tensor, y_train_tensor])
    return vect_data

# Data ready to use
train_data = vectorize_data(train_data, True)
val_data = vectorize_data(val_data)
test_data = vectorize_data(test_data)

## Training
def train(epochs, train_data, model, optimizer, loss_fn):
    for epoch in range(epochs):
        for text, label in train_data:
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(text)
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
        # Print statistics
        print("\nEpoch:", epoch)
        print("Training loss:", loss.item())
        

## Calculate accuracy
def accuracy(data, model, y_len):
    correct = 0
    with torch.no_grad():
        for text, label in data:
            outputs = model(text)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == label).sum().item()
    return correct/y_len


## GRU
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes, hidden_dim):
        super(GRUClassifier, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        embeds = torch.mean(self.emb(inputs), dim=0)
        gru_out, _ = self.gru(embeds.view(1, 1, len(embeds)))
        out = self.fc1(gru_out)
        return out.view(1, -1)

## New model with optimzer and loss function
num_classes = len(np.unique(y))
emb_dim = 6
learning_rate = 0.01
hidden_dim = 6

modelGRU = GRUClassifier(len(tok_to_ix), emb_dim, num_classes, hidden_dim)
optimizer = optim.Adam(modelGRU.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
epochs = 10

# Train
train(epochs, train_data, modelGRU, optimizer, loss_fn)
        
# Train accuracy
train_accuracy = accuracy(train_data, modelGRU, len(y_train))
print("Train set mean accuracy:", train_accuracy)

# Validation accuracy
val_accuracy = accuracy(val_data, modelGRU, len(y_val))
print("Val set mean accuracy:", val_accuracy)

