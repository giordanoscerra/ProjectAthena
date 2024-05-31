from datetime import datetime
import gc
import os
import sys
import time
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from scoring import SCHOOLS
from utilities import Logger, getData
from sklearn.metrics import classification_report

min_chars = None # or could be None
save_string = 'min_chars_'+ str(min_chars) if min_chars is not None else 'full_dataset'

# Load data
#tr, vl, _ = getData(min_chars=200, max_chars=1700)
tr, vl, _ = getData(min_chars=min_chars)
batchSize = 50
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
tok = tokenizer(texts_tr, add_special_tokens=True, padding='max_length', max_length=512, truncation=True)
tokenized_texts = tok['input_ids']
attention_texts = tok['attention_mask']
input_ids = torch.tensor(tokenized_texts)
attention_texts = torch.tensor(attention_texts)
dataset = TensorDataset(input_ids, attention_texts, labels_tr)
dataloader_tr = DataLoader(dataset, batch_size=batchSize, shuffle=True)

print('tokenizing validation...')
texts_vl = texts_vl.tolist()
tok = tokenizer(texts_vl, add_special_tokens=True, padding='max_length', max_length=512, truncation=True)
tokenized_texts_vl = tok['input_ids']
attention_texts_vl = tok['attention_mask']
input_ids_vl = torch.tensor(tokenized_texts_vl)
attention_texts_vl = torch.tensor(attention_texts_vl)
dataset_vl = TensorDataset(input_ids_vl, attention_texts_vl, labels_vl)
dataloader_vl = DataLoader(dataset_vl, batch_size=batchSize, shuffle=True)

device = torch.device("cuda:1")
print('device is:',device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.to(device)

def compute_epoch(model:DistilBertForSequenceClassification, dataloader, optimizer, criterion=nn.functional.cross_entropy, backpropagate=True, epoch=0, device=None) -> tuple[float, float, str]:
    batchIndex = 0
    total_loss = 0
    total_accuracy = 0
    all_predictions = []
    all_labels = []
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
        batch_accuracy = (logits.argmax(1) == labels).float().mean()
        total_accuracy += batch_accuracy
        all_predictions.extend(logits.argmax(1).tolist())
        all_labels.extend(labels.tolist())
        now = time.time()
        eta = (now - begin) * (len(dataloader) - batchIndex) / batchIndex
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, accuracy: {batch_accuracy:.2f}     eta = {int(eta//60):03d}m {int(eta%60):02d}s      ', end='\r')
        
    report = classification_report(all_labels, all_predictions)
    return total_loss/batchIndex, total_accuracy/batchIndex, report

start_time = datetime.now()
print('Training and Validation -> Start Time:', start_time.strftime("H%M%S"))
log_path = os.path.join(sys.path[0], 'log_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.txt')
logger = Logger(log_path)
logger.add("Training and Validation -> Start Time: " + start_time.strftime("H%M%S"))

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    logger.add(f'Epoch {epoch+1}/{num_epochs}')
    model.train()
    loss,acc,_ = compute_epoch(model, dataloader_tr, optimizer, epoch=epoch, device=device)
    logger.add(f'Epoch: {epoch}, TR Loss: {loss:.4f}, TR accuracy: {acc:.2f}')
    #model.eval()
    #loss,acc = compute_epoch(model, dataloader_vl, optimizer, backpropagate=False, epoch=epoch, device=device)
    #logger.add(f'Epoch: {epoch}, VL Loss: {loss:.4f}, VL accuracy: {acc:.2f}')
    model.save_pretrained(os.path.join(sys.path[0], 'tuned',f'bert_model_{epoch}_{save_string}'))
    tokenizer.save_pretrained(os.path.join(sys.path[0], 'tuned',f'bert_tokenizer_{epoch}_{save_string}'))

end_time = datetime.now()

model.save_pretrained(os.path.join(sys.path[0], 'tuned',f'final_bert_model_{datetime.now().strftime("%d%H%M")}_{save_string}'))
tokenizer.save_pretrained(os.path.join(sys.path[0], 'tuned',f'final_bert_tokenizer_{datetime.now().strftime("%d%H%M")}_{save_string}'))

model.eval()
loss,acc,report = compute_epoch(model, dataloader_vl, optimizer, backpropagate=False, epoch=epoch, device=device)
logger.add(f'Epoch: {epoch}, VL Loss: {loss:.4f}, VL accuracy: {acc:.2f}')
logger.add(report)
print('\nTraining and Validation -> End Time:', end_time.strftime("H%M%S"))
logger.add("Training and Validation -> End Time: " + end_time.strftime("H%M%S"))
logger.add("Duration: " + str(end_time - start_time))