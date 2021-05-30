import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim


## Data
# Importing
data = pd.read_csv('data/trainDataWithPOS.csv')

# Rename first column name
data = data.rename({"Sentence #": "Sent"}, axis=1)
# Editing first column from "Sentence: x" to "x"
data.iloc[:,0] = data.iloc[:,0].str.replace("Sentence: ", "").astype(int) - 1  # sequence range [0, 2399]
# all words in lower case
data['Word'] = data['Word'].str.lower()

# Split data into Train 80%, Val 20%
len_data = len(data.iloc[:,0].unique())
idx = np.random.RandomState(seed=42).permutation(range(len_data)) # Permutations of all sentences
train_split_idx = int(0.80 * len_data)

train_idx = idx[:train_split_idx]
val_idx = idx[train_split_idx:]

train = data[data.iloc[:,0].isin(train_idx)]
train = train.reset_index().iloc[:, 1:]
train_len = len(train)

val = data[data.iloc[:,0].isin(val_idx)]
val = val.reset_index().iloc[:, 1:]


### Features
## Build features for words and pos in a window size of 3
#Features for window size 3 (prev word, current word, next word)
def features_builder(data, vocab, column):
    features = np.zeros([len(data), len(vocab)])
    sentences_numbers = list(data.iloc[:,0].unique())
    word_index = 0
    for number in sentences_numbers:
        sent = data[data['Sent']==number][column]
        sent_len = len(sent)
        for i in range(sent_len):
            curr_word = sent.iloc[i]
            if curr_word not in vocab:
                curr_word_index = vocab['UNK']
                features[word_index][curr_word_index] = 1 
                word_index += 1
                continue     
            curr_word_index = vocab[curr_word]
            features[word_index][curr_word_index] = 1
            # prev and next elem
            if i > 0:
                prev_word = sent.iloc[i-1]
                if prev_word in vocab:
                    prev_word_index = vocab[prev_word]
                    features[word_index][prev_word_index] = 1
                elif prev_word not in vocab:
                    prev_word_index = vocab['UNK']
                    features[word_index][prev_word_index] = 1  
            if i < sent_len-1:
                next_word = sent.iloc[i+1]
                if next_word in vocab:
                    next_word_index = vocab[next_word]
                    features[word_index][next_word_index] = 1
                elif next_word not in vocab:
                    next_word_index = vocab['UNK']
                    features[word_index][next_word_index] = 1
            word_index += 1
    return features

## Accespts a series and returns an elem_to_idx dict.
def load_vocab(word_series):
    word_to_ix = {}
    for word in word_series:
        word_to_ix.setdefault(word, len(word_to_ix))
    return word_to_ix

## Unigrams
words = train['Word']
vocab_to_idx = load_vocab(words)
# add UNK
vocab_to_idx['UNK'] = len(vocab_to_idx)

## Tags
tags_train = train['Tag']
tag_to_idx = load_vocab(tags_train)


## Train data
y_train = np.array([tag_to_idx[tag] for tag in tags_train])
X_train = features_builder(train, vocab_to_idx, 'Word')


## Val
tags_val = val['Tag']

y_val = np.array([tag_to_idx[tag] for tag in tags_val])
X_val = features_builder(val, vocab_to_idx, 'Word')


## Test
test = pd.read_csv('data/testDataWithPOS.csv')
test = test.rename({"Sentence #": "Sent"}, axis=1)
test.iloc[:,0] = test.iloc[:,0].str.replace("Sentence: ", "").astype(int) - 1 
test['Word'] = test['Word'].str.lower()

tags_test = test['Tag']

y_test = np.array([tag_to_idx[tag] for tag in tags_test])
X_test = features_builder(test, vocab_to_idx, 'Word')


## Multinomial Naive Bayes classifier
print("Naive Bayes - Test set results")
model_nb  = MultinomialNB()
model_nb.fit(X_train, y_train)

scores_train = model_nb.score(X_train, y_train)
scores_val = model_nb.score(X_val, y_val)
scores_test = model_nb.score(X_test, y_test)

print("Train Accuracy: ", scores_train)
print("Val Accuracy: ", scores_val)
print("Test Accuracy: ", scores_test)
y_test_pred = model_nb.predict(X_test)

report = classification_report(y_test, y_test_pred)
print(report)

## Logistic Regression
print("\nLogistic Regression - Test set results")
model_lr  = LogisticRegression(solver='liblinear')
model_lr.fit(X_train, y_train)

scores_test = model_lr.score(X_train, y_train)
scores_test = model_lr.score(X_val, y_val)
scores_test = model_lr.score(X_test, y_test)

print("Train Accuracy: ", scores_train)
print("Val Accuracy: ", scores_val)
print("Test Accuracy: ", scores_test)
y_test_pred = model_lr.predict(X_test)

report = classification_report(y_test, y_test_pred)
print(report)

'''
Naive Bayes - Test set results
Train Accuracy:  0.8120166114222578
Val Accuracy:  0.7904098994586234
Test Accuracy:  0.7908986175115207
              precision    recall  f1-score   support

           0       0.86      0.90      0.88      6722
           1       0.49      0.27      0.34       615
           2       0.49      0.55      0.51       831
           3       0.44      0.29      0.35       259
           4       0.65      0.39      0.49       253

    accuracy                           0.79      8680
   macro avg       0.58      0.48      0.52      8680
weighted avg       0.78      0.79      0.78      8680


Logistic Regression - Test set results
Train Accuracy:  0.8120166114222578
Val Accuracy:  0.7904098994586234
Test Accuracy:  0.8126728110599079
              precision    recall  f1-score   support

           0       0.83      0.96      0.89      6722
           1       0.60      0.23      0.33       615
           2       0.61      0.33      0.43       831
           3       0.72      0.24      0.36       259
           4       0.83      0.40      0.53       253

    accuracy                           0.81      8680
   macro avg       0.72      0.43      0.51      8680
weighted avg       0.79      0.81      0.78      8680

'''

### Neural Network Models ###
def vectorize_data(data):
    data_sentences = data.groupby('Sent')['Word'].apply(list)
    data_sentences = list(data_sentences)
    data_tags = data.groupby('Sent')['Tag'].apply(list)
    data_tags = list(data_tags)
    # zip data
    data = zip(data_sentences, data_tags)
    vect_data = []
    for text, tags in data:
        x = [vocab_to_idx['UNK'] if tok not in vocab_to_idx else vocab_to_idx[tok] for tok in text]
        y = [tag_to_idx[tag] for tag in tags]
        x_train_tensor = torch.LongTensor(x)
        y_train_tensor = torch.LongTensor(y)
        vect_data.append([x_train_tensor, y_train_tensor])
    return vect_data

## Data ready to use as input
train_data = vectorize_data(train)
val_data = vectorize_data(val)
test_data = vectorize_data(test)

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


emb_dim = 100
learning_rate = 0.1
model = FFNet(len(vocab_to_idx), emb_dim, len(tag_to_idx))
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

n_epochs = 100

def train(n_epochs, model, data, optimizer, loss_fn):
    for epoch in range(n_epochs):
        model.train()
        for text, label in train_data:
            pred_y = model(text)
            loss = loss_fn(pred_y, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 0:
            print("\nEpoch:", epoch)
            print("Training loss:", loss.item())

def accuracy(data, model, y_len):
    correct = 0
    with torch.no_grad():
        for text, label in data:
            outputs = model(text)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == label).sum().item()
    return correct/y_len

## Training
train(n_epochs, model, train_data, optimizer, loss_fn)

## Train accuracy
train_accuracy = accuracy(train_data, model, len(y_train))
print("Train set mean accuracy:", train_accuracy)

# Validation accuracy
val_accuracy = accuracy(val_data, model, len(y_val))
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


emb_dim = 100
hidden_dim = 32
learning_rate = 0.1
LSTMmodel = LSTMtagger(len(vocab_to_idx), emb_dim, len(tag_to_idx), hidden_dim)
optimizer = optim.SGD(LSTMmodel.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

n_epochs = 100

## Training
train(n_epochs, LSTMmodel, train_data, optimizer, loss_fn)

## Train accuracy
train_accuracy = accuracy(train_data, LSTMmodel, len(y_train))
print("Train set mean accuracy:", train_accuracy)

# Validation accuracy
val_accuracy = accuracy(val_data, LSTMmodel, len(y_val))
print("Val set mean accuracy:", val_accuracy)


'''
Results for different models. Other models are in other_nn_tried.py
FF
Train set mean accuracy: 0.9185049719967996
Val set mean accuracy: 0.8991492652745553
LSTM
Train set mean accuracy: 0.927801272526384
Val set mean accuracy: 0.908739365815932
GRU
Train set mean accuracy: 0.9274202766030404
Val set mean accuracy: 0.8946635730858469
Bi-LSTM
Train set mean accuracy: 0.9243342096239571
Val set mean accuracy: 0.9031709203402939
Glove300 LSTM
Train set mean accuracy: 0.9266963843486875
Val set mean accuracy: 0.908584686774942
'''

## The models above had pretty similar results.
## However, LSTM did a little bit better.
# Test accuracy
test_accuracy = accuracy(test_data, LSTMmodel, len(y_test))
print("Test set mean accuracy for best model:", val_accuracy)

'''
Test set mean accuracy for best model: 0.908739365815932
'''