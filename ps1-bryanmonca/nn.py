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


## Network
class Classifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes):
        super(Classifier, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(emb_dim, num_classes)

    def forward(self, inputs):
        embeds = torch.mean(self.emb(inputs), dim=0)
        out = self.fc1(embeds)
        return out.view(1, -1)

## New model with optimzer and loss function
num_classes = len(np.unique(y))
emb_dim = 10
learning_rate = 0.01

model = Classifier(len(tok_to_ix), emb_dim, num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


## Training
epochs = 10
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

## Train accuracy
train_accuracy = accuracy(train_data, model, len(y_train))
print("Train set mean accuracy:", train_accuracy)

# Validation accuracy
val_accuracy = accuracy(val_data, model, len(y_val))
print("Val set mean accuracy:", val_accuracy)


## LSTM 
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes, hidden_dim):
        super(LSTMClassifier, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        embeds = torch.mean(self.emb(inputs), dim=0)
        #print(embeds.size())
        lstm_out, _ = self.lstm(embeds.view(1, 1, len(embeds)))
        #print(lstm_out.size())
        out = self.fc1(lstm_out)
        #print(out.size())
        return out.view(1, -1)

## New model with optimzer and loss function
num_classes = len(np.unique(y))
emb_dim = 6
learning_rate = 0.01
hidden_dim = 6

model = LSTMClassifier(len(tok_to_ix), emb_dim, num_classes, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


## Training
epochs = 10
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
        
# Train accuracy
train_accuracy = accuracy(train_data, model, len(y_train))
print("Train set mean accuracy:", train_accuracy)

# Validation accuracy
val_accuracy = accuracy(val_data, model, len(y_val))
print("Val set mean accuracy:", val_accuracy)


## GRU
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes, hidden_dim):
        super(GRUClassifier, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        embeds = torch.mean(self.emb(inputs), dim=0)
        #print(embeds.size())
        gru_out, _ = self.gru(embeds.view(1, 1, len(embeds)))
        #print(lstm_out.size())
        out = self.fc1(gru_out)
        #print(out.size())
        return out.view(1, -1)

## New model with optimzer and loss function
num_classes = len(np.unique(y))
emb_dim = 6
learning_rate = 0.01
hidden_dim = 6

modelGRU = GRUClassifier(len(tok_to_ix), emb_dim, num_classes, hidden_dim)
optimizer = optim.Adam(modelGRU.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


## Training
epochs = 10
for epoch in range(epochs):
    for text, label in train_data:
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = modelGRU(text)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()

    # Print statistics
    print("\nEpoch:", epoch)
    print("Training loss:", loss.item())
        
# Train accuracy
train_accuracy = accuracy(train_data, modelGRU, len(y_train))
print("Train set mean accuracy:", train_accuracy)

# Validation accuracy
val_accuracy = accuracy(val_data, modelGRU, len(y_val))
print("Val set mean accuracy:", val_accuracy)


## BiLSTM 
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes, hidden_dim):
        super(BiLSTMClassifier, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, inputs):
        embeds = torch.mean(self.emb(inputs), dim=0)
        #print(embeds.size())
        lstm_out, _ = self.lstm(embeds.view(1, 1, len(embeds)))
        #print(lstm_out.size())
        out = self.fc1(lstm_out)
        #print(out.size())
        return out.view(1, -1)

## New model with optimzer and loss function
num_classes = len(np.unique(y))
emb_dim = 6
learning_rate = 0.01
hidden_dim = 6

model = BiLSTMClassifier(len(tok_to_ix), emb_dim, num_classes, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


## Training
epochs = 10
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
        
# Train accuracy
train_accuracy = accuracy(train_data, model, len(y_train))
print("Train set mean accuracy:", train_accuracy)

# Validation accuracy
val_accuracy = accuracy(val_data, model, len(y_val))
print("Val set mean accuracy:", val_accuracy)



## Word Embeddings
# Glove apply to our best model - GRU
# code based on https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

## Import Glove 50-dim
embeddings_index = {}
f = open('/Users/bryanmonca/Workspace/cmsi-537/word_emb/glove.6B/glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    coefs = torch.from_numpy(coefs)
    embeddings_index[word] = coefs
f.close()

# Create Weights Matrix
words = list(tok_to_ix.keys())
num_words = len(words)
weights_matrix = np.zeros((num_words, 50))
words_found = 0
for i, word in enumerate(words):
    try:
        weights_matrix[i] = embeddings_index[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(50, ))
        
weights_matrix = torch.from_numpy(weights_matrix)

# Create embedding layer
def create_emb_layer(weights_matrix, non_trainable=False):
    vocab_size, emb_dim = weights_matrix.size()
    emb_layer = nn.Embedding(vocab_size, emb_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, emb_dim

# GRU
class GloveClassifier(nn.Module):
    def __init__(self, num_classes, hidden_dim, weights_matrix):
        super(GloveClassifier, self).__init__()
        self.emb, emb_dim = create_emb_layer(weights_matrix, True)
        self.gru = nn.GRU(emb_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        embeds = torch.mean(self.emb(inputs), dim=0)
        #print(embeds.size())
        gru_out, _ = self.gru(embeds.view(1, 1, len(embeds)))
        #print(lstm_out.size())
        out = self.fc1(gru_out)
        #print(out.size())
        return out.view(1, -1)

## New model with optimzer and loss function
num_classes = len(np.unique(y))
learning_rate = 0.001
hidden_dim = 50

model = GloveClassifier(num_classes, hidden_dim, weights_matrix)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

## Training
epochs = 10
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
        
# Train accuracy
train_accuracy = accuracy(train_data, model, len(y_train))
print("Train set mean accuracy:", train_accuracy)

# Validation accuracy
val_accuracy = accuracy(val_data, model, len(y_val))
print("Val set mean accuracy:", val_accuracy)


## Test set on best model 
test_accuracy = accuracy(test_data, modelGRU, len(y_test))
print("Test set mean accuracy:", test_accuracy)

'''
Results:

Neural Network
Batch size=1, Epochs=10, lr=0.01, emb_dim=10
Train set mean accuracy: 0.9492081105618885
Val set mean accuracy: 0.8260694108151735

LSTM
Batch size=1, Epochs=10, lr=0.01, emb_dim=6, hidden_dim=6
Train set mean accuracy: 0.9792192071017856
Val set mean accuracy: 0.8506860371267151

GRU
Batch size=1, Epochs=10, lr=0.01, emb_dim=6, hidden_dim=6
Train set mean accuracy: 0.9858266922223343
Val set mean accuracy: 0.8595641646489104

Bi-LSTM
Batch size=1, Epochs=10, lr=0.01, emb_dim=6, hidden_dim=6
Train set mean accuracy: 0.9849692323211944
Val set mean accuracy: 0.8297013720742534

Glove 50dim 
GRU
Batch size=1, Epochs=10, lr=0.001, hidden_dim=50
Train set mean accuracy: 0.841218601835973
Val set mean accuracy: 0.8220338983050848

Test using GRU Model
Test set mean accuracy: 0.8559903186768858
'''

## Word2Vec (I worked on it, but it takes too long to load)
#embeds = KeyedVectors.load_word2vec_format("/Users/bryanmonca/Downloads/vectors-negative300.bin.gz", binary=True).wv.vectors
#
### Network
#class Classifier(nn.Module):
#    def __init__(self, vocab_size, emb_dim, num_classes, embeds):
#        super(Classifier, self).__init__()
#        self.emb = nn.Embedding(vocab_size, emb_dim)
#        self.emb.weight = nn.Parameter(torch.Tensor(embeds))
#        self.fc1 = nn.Linear(emb_dim, num_classes)
#
#    def forward(self, inputs):
#        embeds = torch.mean(self.emb(inputs), dim=0)
#        out = self.fc1(embeds)
#        return out.view(1, -1)
#
### New model with optimzer and loss function
#num_classes = len(np.unique(y))
#emb_dim = 300
#learning_rate = 0.01
#
#model = Classifier(len(tok_to_ix), emb_dim, num_classes, embeds)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#loss_fn = nn.CrossEntropyLoss()
#
#
### Training
#epochs = 5
#for epoch in range(epochs):
#    for text, label in train_data:
#        # forward + backward + optimize
#        optimizer.zero_grad()
#        outputs = model(text)
#        loss = loss_fn(outputs, label)
#        loss.backward()
#        optimizer.step()
#
#    # Print statistics
#    print("\nEpoch:", epoch)
#    print("Training loss:", loss.item())
#        
#
## Train accuracy
#train_accuracy = accuracy(train_data, model, len(y_train))
#print("Train set mean accuracy:", train_accuracy)
#
## Validation accuracy
#val_accuracy = accuracy(val_data, model, len(y_val))
#print("Val set mean accuracy:", val_accuracy)