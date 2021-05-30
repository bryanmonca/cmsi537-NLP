import loader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load the training and testing data.
train_sentences = loader.load_sentences('data/eng.train', 'lower')
train_corpus, dics = loader.prepare_dataset(train_sentences, mode='train', 
                                            lower='lower')

test_sentences = loader.load_sentences('data/eng.val', 'lower')
test_corpus = loader.prepare_dataset(test_sentences, mode='test', 
                                     lower='lower', word_to_id=dics['word_to_id'], 
                                     tag_to_id=dics['tag_to_id'])

# Number of labels
def labels_len(corpus):
    total_len = 0
    for sent in corpus:
        total_len += len(sent['tags'])
    return total_len

train_labels_len = labels_len(train_corpus)
test_labels_len = labels_len(test_corpus)


# Feed Forward Network
class FFNet(nn.Module):
  def __init__(self, num_words, emb_dim, num_y):
    super().__init__()
    self.emb = nn.Embedding(num_words, emb_dim)
    self.linear = nn.Linear(emb_dim, num_y)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, text):
    embeds = self.emb(text)
    return self.softmax(self.linear(embeds))

emb_dim = 10
learning_rate = 0.01
model = FFNet(len(dics['word_to_id']), emb_dim, len(dics['tag_to_id']))
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

n_epochs = 10

def train(n_epochs, model, data, optimizer, loss_fn):
    for epoch in range(n_epochs):
        model.train()
        for sentence in data:
            text = torch.LongTensor(sentence['words'])
            label = torch.LongTensor(sentence['tags'])
            
            pred_y = model(text)
            loss = loss_fn(pred_y, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("\nEpoch:", epoch)
        print("Training loss:", loss.item())

def accuracy(data, model, y_len):
    correct = 0
    with torch.no_grad():
        for sentence in data:
            text = torch.LongTensor(sentence['words'])
            label = torch.LongTensor(sentence['tags'])

            outputs = model(text)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == label).sum().item()
    return correct/y_len

## Training
train(n_epochs, model, train_corpus, optimizer, loss_fn)

## Train accuracy
train_accuracy = accuracy(train_corpus, model, train_labels_len)
print("Train set mean accuracy:", train_accuracy)

# Validation accuracy
val_accuracy = accuracy(test_corpus, model, test_labels_len)
print("Val set mean accuracy:", val_accuracy)


## LSTM
class LSTMtagger(nn.Module):
  def __init__(self, num_words, emb_dim, num_y, hidden_dim):
    super().__init__()
    self.emb = nn.Embedding(num_words, emb_dim)
    self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1, 
                        bidirectional=False, batch_first=True)
    self.linear = nn.Linear(hidden_dim, num_y)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, text):
    embeds = self.emb(text)
    lstm_out, _ = self.lstm(embeds.view(len(text), 1, -1))
    tag_space = self.linear(lstm_out.view(len(text), -1))
    return self.softmax(tag_space)

emb_dim = 10
hidden_dim = 10
learning_rate = 0.01
LSTMmodel = LSTMtagger(len(dics['word_to_id']), emb_dim, len(dics['tag_to_id']), hidden_dim)
optimizer = optim.RMSprop(LSTMmodel.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

n_epochs = 10

## Training
train(n_epochs, LSTMmodel, train_corpus, optimizer, loss_fn)

## Train accuracy
train_accuracy = accuracy(train_corpus, LSTMmodel, train_labels_len)
print("Train set mean accuracy:", train_accuracy)

# Validation accuracy
val_accuracy = accuracy(test_corpus, LSTMmodel, test_labels_len)
print("Val set mean accuracy:", val_accuracy)


'''
Results:

Neural Network 
Train set mean accuracy: 0.9135747295220041
Val set mean accuracy: 0.8787026850833232

LSTM
Train set mean accuracy: 0.9160008054179087
Val set mean accuracy: 0.8808621602901031
'''