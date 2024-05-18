import time
import torch
import torch.nn as nn
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import os
import sys
import gc
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from scoring import SCHOOLS  # Assuming SCHOOLS is a list of school names
from hyperparameters import *

'''
tr, vl, _ = getData(min_chars=20, max_chars=1700)
batchSize = 38
num_epochs = 3
print(f'training quantity: {len(tr)}', f'validation quantity: {len(vl)}')
texts_tr = tr['sentence_str']
labels_tr = tr['school']
texts_vl = vl['sentence_str']
labels_vl = vl['school']'''

def getModel(model_path='distilbert-base-uncased'):
    return DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=len(SCHOOLS))

def getTokenizer(model_path='distilbert-base-uncased'):
    return DistilBertTokenizerFast.from_pretrained(model_path)

def getDevice():
    return torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

class PhilosophyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts:pd.DataFrame = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts.iloc[idx], max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return inputs['input_ids'], inputs['attention_mask'], self.labels[idx]

def getDataloader(texts, labels, tokenizer, max_length=512, batch_size=32):
    data = PhilosophyDataset(texts, labels, tokenizer, max_length)
    return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)

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