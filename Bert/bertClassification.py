import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import gc
import time
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from scoring import SCHOOLS  # Assuming SCHOOLS is a list of school names
from utilities import getData, Logger
from datetime import datetime
import numpy as np

def compute_epoch(model:BertForSequenceClassification, dataloader, optimizer, criterion=nn.functional.cross_entropy, backpropagate=True):
    batchIndex = 0
    total_loss = 0
    total_accuracy = 0
    begin = time.time()
    for batch in dataloader:
        batchIndex += 1
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
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

# Load data
tr, vl, _ = getData(min_chars=20, max_chars=1700)
batchSize = 38
num_epochs = 3
print(len(tr), len(vl))
texts = tr['sentence_str']
labels = tr['school']
texts_vl = vl['sentence_str']
labels_vl = vl['school']

# BERT model path
model_path = "../../../opt/models/bert-base-cased"

# log path
log_path = os.path.join(sys.path[0], 'log_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.txt')
logger = Logger(log_path)

# Device assignment
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
#device = torch.device("cpu")
# Show all available GPUs
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs available.")
logger.add(f"Device: {device}")
print('Device:', device)
# Tokenize the data
tokenizer = BertTokenizer.from_pretrained(model_path)
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True, padding='max_length', max_length=512) for text in texts]
attention_texts = [[float(i > 0) for i in text] for text in tokenized_texts]
# Fine-tune BERT
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(SCHOOLS))
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
model.to(device)

# Convert labels to indices
label_to_index = {label: index for index, label in enumerate(SCHOOLS)}
labels = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)

# Create TensorDataset and DataLoader
input_ids = torch.tensor(tokenized_texts)
attention_texts = torch.tensor(attention_texts)
dataset = TensorDataset(input_ids, attention_texts, labels)
dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

# Loss function
criterion = nn.CrossEntropyLoss()

tokenized_texts_vl = [tokenizer.encode(text, add_special_tokens=True, padding='max_length', max_length=512) for text in texts_vl]
attention_texts_vl = [[float(i > 0) for i in text] for text in tokenized_texts_vl]
input_ids_vl = torch.tensor(tokenized_texts_vl)
attention_texts_vl = torch.tensor(attention_texts_vl)
labels_vl = torch.tensor([label_to_index[label] for label in labels_vl], dtype=torch.long)
dataset_vl = TensorDataset(input_ids_vl, attention_texts_vl, labels_vl)
dataloader_vl = DataLoader(dataset_vl, batch_size=batchSize, shuffle=True)
start_time = datetime.now()
logger.add("Training and Validation -> Start Time: " + start_time.strftime("H%M%S"))
print('Training + Validation...')
# Training AND Validation loop
batchIndex = 0
for epoch in range(num_epochs):
    model.train()
    batchIndex = 0
    loss, acc = compute_epoch(model, dataloader, optimizer, criterion, backpropagate=True)
    val_loss, val_acc = compute_epoch(model, dataloader_vl, optimizer, criterion, backpropagate=False)
    logger.add(f'Epoch: {epoch}, TR Loss: {loss:.4f}, TR accuracy: {acc:.2f}')
    logger.add(f'Epoch: {epoch}, VL Loss: {val_loss:.4f}, VL accuracy: {val_acc:.2f}')
    model.save_pretrained(os.path.join(sys.path[0], 'bert_model_', datetime.now().strftime("%d%H%M")))
    tokenizer.save_pretrained(os.path.join(sys.path[0], 'bert_tokenizer_', datetime.now().strftime("%d%H%M")))
    print()

end_time = datetime.now()
tokenizer.save_pretrained(os.path.join(sys.path[0], 'bert_tokenizer_', datetime.now().strftime("%d%H%M")))
model.save_pretrained(os.path.join(sys.path[0], 'bert_model_', datetime.now().strftime("%d%H%M")))

logger.add("Training and Validation -> End Time: " + end_time.strftime("H%M%S"))
logger.add("Training and Validation -> Duration: " + str(end_time - start_time))



