### Other neural networks tried for literal c) of programming question of midterm
## In tagger.py file, I am only showing 2 models FeedForward and LSTM

## GRU
class GRUtagger(nn.Module):
  def __init__(self, num_words, emb_dim, num_y, hidden_dim):
    super().__init__()
    self.emb = nn.Embedding(num_words, emb_dim)
    self.gru = nn.GRU(emb_dim, hidden_dim, num_layers=1, 
                        bidirectional=False, batch_first=True)
    self.linear = nn.Linear(hidden_dim, num_y)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, text):
    embeds = self.emb(text)
    gru_out, _ = self.gru(embeds.view(len(text), 1, -1))
    tag_space = self.linear(gru_out.view(len(text), -1))
    return self.softmax(tag_space)


emb_dim = 100
hidden_dim = 32
learning_rate = 0.1
GRUmodel = GRUtagger(len(vocab_to_idx), emb_dim, len(tag_to_idx), hidden_dim)
optimizer = optim.SGD(GRUmodel.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

n_epochs = 100

## Training
train(n_epochs, GRUmodel, train_data, optimizer, loss_fn)

## Train accuracy
train_accuracy = accuracy(train_data, GRUmodel, len(y_train))
print("Train set mean accuracy:", train_accuracy)

# Validation accuracy
val_accuracy = accuracy(val_data, GRUmodel, len(y_val))
print("Val set mean accuracy:", val_accuracy)


## Bi-LSTM
class BiLSTMtagger(nn.Module):
  def __init__(self, num_words, emb_dim, num_y, hidden_dim):
    super().__init__()
    self.emb = nn.Embedding(num_words, emb_dim)
    self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1, 
                        bidirectional=True, batch_first=True)
    self.linear = nn.Linear(hidden_dim * 2, num_y)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, text):
    embeds = self.emb(text)
    lstm_out, _ = self.lstm(embeds.view(len(text), 1, -1))
    tag_space = self.linear(lstm_out.view(len(text), -1))
    return self.softmax(tag_space)


emb_dim = 100
hidden_dim = 32
learning_rate = 0.1
BiLSTMmodel = BiLSTMtagger(len(vocab_to_idx), emb_dim, len(tag_to_idx), hidden_dim)
optimizer = optim.SGD(BiLSTMmodel.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

n_epochs = 100

## Training
train(n_epochs, BiLSTMmodel, train_data, optimizer, loss_fn)

## Train accuracy
train_accuracy = accuracy(train_data, BiLSTMmodel, len(y_train))
print("Train set mean accuracy:", train_accuracy)

# Validation accuracy
val_accuracy = accuracy(val_data, BiLSTMmodel, len(y_val))
print("Val set mean accuracy:", val_accuracy)


## GloVe - pre-trained

## Import Glove 100-dim
embeddings_index = {}
f = open('data/glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    coefs = torch.from_numpy(coefs)
    embeddings_index[word] = coefs
f.close()

# Create Weights Matrix
words = list(vocab_to_idx.keys())
num_words = len(words)
weights_matrix = np.zeros((num_words, 300))
words_found = 0
for i, word in enumerate(words):
    try:
        weights_matrix[i] = embeddings_index[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))
        
weights_matrix = torch.from_numpy(weights_matrix)

# Create embedding layer
def create_emb_layer(weights_matrix, non_trainable=False):
    vocab_size, emb_dim = weights_matrix.size()
    emb_layer = nn.Embedding(vocab_size, emb_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, emb_dim

# pretrained LSTM
class GloveLSTMtagger(nn.Module):
  def __init__(self, num_y, hidden_dim, weights_matrix):
    super().__init__()
    self.emb, emb_dim = create_emb_layer(weights_matrix, True)
    self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1, 
                        bidirectional=False, batch_first=True)
    self.linear = nn.Linear(hidden_dim, num_y)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, text):
    embeds = self.emb(text)
    lstm_out, _ = self.lstm(embeds.view(len(text), 1, -1))
    tag_space = self.linear(lstm_out.view(len(text), -1))
    return self.softmax(tag_space)

hidden_dim = 32
learning_rate = 0.1
GloveLSTMmodel = GloveLSTMtagger(len(tag_to_idx), hidden_dim, weights_matrix)
optimizer = optim.SGD(GloveLSTMmodel.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

## Training
train(n_epochs, GloveLSTMmodel, train_data, optimizer, loss_fn)

## Train accuracy
train_accuracy = accuracy(train_data, GloveLSTMmodel, len(y_train))
print("Train set mean accuracy:", train_accuracy)

# Validation accuracy
val_accuracy = accuracy(val_data, GloveLSTMmodel, len(y_val))
print("Val set mean accuracy:", val_accuracy)