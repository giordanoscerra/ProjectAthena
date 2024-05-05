import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from scoring import SCHOOLS, scorePhilosophy
from utilities import getData
import numpy as np


tr, vl, _ = getData(min_chars=300, max_chars=500)
texts = tr['sentence_str']
labels = tr['school']
texts_vl = vl['sentence_str']
labels_vl = vl['school']

# bert-location
model_path = "../../../opt/models/bert-base-cased"

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
device = torch.device("cpu")
print('Device:', device)

# Step 2: Tokenize the data
tokenizer = BertTokenizer.from_pretrained(model_path)
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True, padding='max_length', max_length = 512) for text in texts]

# Step 3: Fine-tune BERT
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(SCHOOLS))
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
model.to(device)

# Step 4: Train the classifier

# Convert tokenized_texts and labels to tensors
input_ids = torch.tensor(tokenized_texts)
#assign each school a number
y_train = labels.apply(lambda x: SCHOOLS.index(x))
# Convert labels to numpy array
labels = np.array(labels)

# Convert y_train to one-hot encoded numpy array
num_classes = len(SCHOOLS)
y_train = np.eye(num_classes)[y_train]
labels = torch.tensor(y_train)
# Create TensorDataset and DataLoader
dataset = TensorDataset(input_ids, labels)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
print('Training...')
# Define training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Loss:  {loss.item():.4f}          ', end='\r')
    print()

# Step 5: Evaluate the model
model.eval()
