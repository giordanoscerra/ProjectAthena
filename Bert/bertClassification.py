import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import gc
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from scoring import SCHOOLS  # Assuming SCHOOLS is a list of school names
from utilities import getData, Logger
from datetime import datetime
import numpy as np

# Load data
tr, vl, _ = getData(min_chars=400, max_chars=500)
print(len(tr), len(vl))
texts = tr['sentence_str']
labels = tr['school']
texts_vl = vl['sentence_str']
labels_vl = vl['school']

# BERT model path
model_path = "../../../opt/models/bert-base-cased"

# log path
log_path = os.path.join(sys.path[0], 'log_', datetime.now().strftime("%Y%m%d%H%M%S"), '.txt')
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

# Fine-tune BERT
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(SCHOOLS))
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
model.to(device)

# Convert labels to indices
label_to_index = {label: index for index, label in enumerate(SCHOOLS)}
labels = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)

# Create TensorDataset and DataLoader
input_ids = torch.tensor(tokenized_texts)
dataset = TensorDataset(input_ids, labels)
dataloader = DataLoader(dataset, batch_size=40, shuffle=True)

# Loss function
criterion = nn.CrossEntropyLoss()
total_batches = len(dataloader)
logger.add("Training. Time: " + datetime.now().strftime("H%M%S"))
print('Training...')
# Training loop
num_epochs = 3
batchIndex = 0
for epoch in range(num_epochs):
    model.train()
    batchIndex = 0
    for batch in dataloader:
        batchIndex += 1
        input_ids, labels = batch
        input_ids, labels = input_ids.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        accuracy = (logits.argmax(1) == labels).float().mean()
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, accuracy: {accuracy:.2f}     completion: {(batchIndex/total_batches)*100:.1f}%      ', end='\r')
        del input_ids, labels, outputs, logits, loss
        gc.collect()
        torch.cuda.empty_cache()
    print()

logger.add("Training finished at: " + datetime.now().strftime("H%M%S"))

# Evaluation
model.eval()
# You can evaluate the model on the validation set here
logger.add("Evaluation...")
print('Evaluation...')
tokenized_texts_vl = [tokenizer.encode(text, add_special_tokens=True, padding='max_length', max_length=512) for text in texts_vl]
input_ids_vl = torch.tensor(tokenized_texts_vl)
labels_vl = torch.tensor([label_to_index[label] for label in labels_vl], dtype=torch.long)
dataset_vl = TensorDataset(input_ids_vl, labels_vl)
dataloader_vl = DataLoader(dataset_vl, batch_size=2, shuffle=True)
total_batches_vl = len(dataloader_vl)
batchIndex = 0
total_accuracy = 0
for batch in dataloader_vl:
    batchIndex += 1
    input_ids, labels = batch
    input_ids, labels = input_ids.to(device), labels.to(device)
    outputs = model(input_ids, labels=labels)
    logits = outputs.logits
    accuracy = (logits.argmax(1) == labels).float().mean()
    total_accuracy += accuracy
    print(f'Accuracy: {accuracy:.2f}     completion: {(batchIndex/total_batches_vl)*100:.1f}%      ', end='\r')
    del input_ids, labels, outputs, logits
    gc.collect()
    torch.cuda.empty_cache()
logger.add("Evaluation finished at. Time: " + datetime.now().strftime("H%M%S"))
logger.add(f'Validation accuracy: {total_accuracy/total_batches_vl:.2f}')
print(f'Validation accuracy: {total_accuracy/total_batches_vl:.2f}')

tokenizer.save_pretrained(os.path.join(sys.path[0], 'bert_tokenizer_', datetime.now().strftime("%d%H%M")))
model.save_pretrained(os.path.join(sys.path[0], 'bert_model_', datetime.now().strftime("%d%H%M")))