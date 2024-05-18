from datetime import datetime
import gc
import os
import sys
import time
import torch
import torch.nn as nn
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from scoring import SCHOOLS
from utilities import Logger, getData

# Load data
tr, vl, _ = getData(min_chars=200, max_chars=1700)
batchSize = 30
num_epochs = 3
learning_rate = 2e-5
print(f'training quantity: {len(tr)}', f'validation quantity: {len(vl)}')
texts_tr = tr['sentence_str']
labels_tr = tr['school']
texts_vl = vl['sentence_str']
labels_vl = vl['school']

labels_tr = torch.tensor([SCHOOLS.index(label) for label in labels_tr])
labels_vl = torch.tensor([SCHOOLS.index(label) for label in labels_vl])

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(SCHOOLS))
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

print('tokenizing training...')
texts_tr = texts_tr.tolist()
tok = tokenizer(texts_tr, add_special_tokens=True, padding='max_length', max_length=280, truncation=True)
tokenized_texts = tok['input_ids']
attention_texts = tok['attention_mask']
input_ids = torch.tensor(tokenized_texts)
attention_texts = torch.tensor(attention_texts)
dataset = TensorDataset(input_ids, attention_texts, labels_tr)
dataloader_tr = DataLoader(dataset, batch_size=batchSize, shuffle=True)

print('tokenizing validation...')
texts_vl = texts_vl.tolist()
tok = tokenizer(texts_vl, add_special_tokens=True, padding='max_length', max_length=280, truncation=True)
tokenized_texts_vl = tok['input_ids']
attention_texts_vl = tok['attention_mask']
input_ids_vl = torch.tensor(tokenized_texts_vl)
attention_texts_vl = torch.tensor(attention_texts_vl)
dataset_vl = TensorDataset(input_ids_vl, attention_texts_vl, labels_vl)
dataloader_vl = DataLoader(dataset_vl, batch_size=batchSize, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print('device is:',device)
model.to(device)

def compute_epoch(model:DistilBertForSequenceClassification, dataloader, optimizer, criterion=nn.functional.cross_entropy, backpropagate=True, epoch=0, device=None) -> tuple[float, float]:
    batchIndex = 0
    total_loss = 0
    total_accuracy = 0
    begin = time.time()
    for batch in dataloader:
        batchIndex += 1
        input_ids, attention_mask, labels = batch
        if device is not None:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
        outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        if backpropagate:    
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        optimizer.zero_grad()
        total_accuracy += (logits.argmax(1) == labels).float().mean()
        now = time.time()
        eta = (now - begin) * (len(dataloader) - batchIndex) / batchIndex
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, accuracy: {total_accuracy/batchIndex:.2f}     eta = {int(eta//60):03d}m {int(eta%60):02d}s      ', end='\r')
        del input_ids, labels, outputs, logits, loss
        gc.collect()
        torch.cuda.empty_cache()
    return total_loss/batchIndex, total_accuracy/batchIndex

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = datetime.now()
print('Training and Validation -> Start Time:', start_time.strftime("H%M%S"))
log_path = os.path.join(sys.path[0], 'log_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.txt')
logger = Logger(log_path)
logger.add("Training and Validation -> Start Time: " + start_time.strftime("H%M%S"))

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    logger.add(f'Epoch {epoch+1}/{num_epochs}')
    model.train()
    loss,acc = compute_epoch(model, dataloader_tr, optimizer, epoch=epoch, device=device)
    logger.add(f'Epoch: {epoch}, TR Loss: {loss:.4f}, TR accuracy: {acc:.2f}')
    model.eval()
    loss,acc = compute_epoch(model, dataloader_vl, optimizer, backpropagate=False, epoch=epoch, device=device)
    logger.add(f'Epoch: {epoch}, VL Loss: {loss:.4f}, VL accuracy: {acc:.2f}')