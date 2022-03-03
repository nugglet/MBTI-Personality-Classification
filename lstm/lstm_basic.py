import numpy as np
import pandas as pd
import torch, re, string
import spacy, nltk

from collections import Counter
from nltk.stem import WordNetLemmatizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch import nn

nltk.download('wordnet')

data = pd.read_csv('./dataset/mbti_1.csv', converters={'type': str.strip, 'posts': str.strip})

tok = spacy.load('en_core_web_sm')

def tokenize(text):
  lemmatizer = WordNetLemmatizer()
  # replace url
  text = re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', text)
  text = re.sub(r"[^\x00-\x7F]+", " ", text)
  # remove punctuation and numbers
  regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
  nopunct = regex.sub(" ", text.lower())
  return [lemmatizer.lemmatize(token.text) for token in tok.tokenizer(nopunct) if not token.text.isspace()]

counts = Counter()
max_length = -1
for idx, row in data.iterrows():
  tokenized = tokenize(row.posts)
  if len(tokenized) > max_length: max_length = len(tokenized)
  counts.update(tokenized)

#deleting infrequent words
print("num_words before:", len(counts.keys()))
for word in list(counts):
    if counts[word] < 3:
        del counts[word]
print("num_words after:", len(counts.keys()))

# create vocab
word2idx = {"":0, "<UNK>":1}
words = ["", "<UNK>"]
for word in counts:
  word2idx[word] = len(words)
  words.append(word)

def encode_sentence(text, word2idx, N):
  tokenized = tokenize(text)
  encoded = np.zeros(N, dtype=int)
  enc1 = np.array([word2idx.get(word, word2idx["<UNK>"]) for word in tokenized])
  length = min(N, len(enc1))
  encoded[:length] = enc1[:length]
  return encoded, length

data['encoded'] = data.posts.apply(lambda x: np.array(encode_sentence(x, word2idx, max_length)))
category = dict(zip(data.type.unique(), np.arange(16)))
data['numtype'] = data.type.apply(lambda x: category[x])

# train valid test split
class MBTIDataset(Dataset):
  def __init__(self, X, Y):
    self.X = X
    self.y = Y
  
  def __len__(self):
    return len(self.y)
  
  def __getitem__(self, idx):
    return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]

X = list(data.encoded)
y = list(data.numtype)

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.7, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, stratify=y_rem)

train_ds = MBTIDataset(X_train, y_train)
valid_ds = MBTIDataset(X_valid, y_valid)
test_ds = MBTIDataset(X_test, y_test)

def train(model, epochs=10, lr=0.001):
  parameters = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = torch.optim.Adam(parameters, lr=lr)
  for i in range(epochs):
    model.train()
    sum_loss = 0.0
    total = 0.0
    for x, y, l in train_dl:
      x = x.long()
      y = y.long()
      y_pred = model(x, l)
      optimizer.zero_grad()
      loss = F.cross_entropy(y_pred, y)
      loss.backward()
      optimizer.step()
      sum_loss += loss.item() * y.shape[0]
      total += y.shape[0]
    val_loss, val_acc = evaluate(model, valid_dl)

    if i % 5 == 0:
      print("Train loss: {} | val loss: {} | val accu: {}".format(sum_loss/total, val_loss, val_acc))

def evaluate(model, valid_dl):
  model.eval()
  correct = 0
  total = 0
  sum_loss = 0.0
  for x, y, l in valid_dl:
    x = x.long()
    y = y.long()
    y_hat = model(x,l)
    loss = F.cross_entropy(y_hat, y)
    pred = torch.max(y_hat, 1)[1]
    correct += (pred == y).float().sum()
    total += y.shape[0]
    sum_loss += loss.item() * y.shape[0]
  return sum_loss/total, correct/total

batch_size = 5000
vocab_size = len(words)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)
test_dl = DataLoader(test_ds, batch_size=batch_size)

class LSTM_fixed_len(torch.nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim):
    super().__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    self.linear = nn.Linear(hidden_dim, 16)
    self.dropout = nn.Dropout(0.3)
  
  def forward(self, x, l):
    x = self.embeddings(x)
    x = self.dropout(x)
    lstm_out, (ht, ct) = self.lstm(x)
    return self.linear(ht[-1])

lstm_fixed = LSTM_fixed_len(vocab_size, 200, 200)

train(lstm_fixed, epochs=20, lr=0.001)
