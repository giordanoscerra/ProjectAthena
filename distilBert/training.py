from datetime import datetime
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

min_chars = None # could be 84 or None
save_string = 'min_chars_'+ str(min_chars) if min_chars is not None else 'full_dataset' # to save the model with a meaningful name

# Load data
tr, vl, _ = getData(min_chars=min_chars)
batchSize = 50
num_epochs = 3
learning_rate = 2e-5
print(f'training quantity: {len(tr)}', f'validation quantity: {len(vl)}')
texts_tr = tr['sentence_str']
labels_tr = tr['school']
texts_vl = vl['sentence_str']
labels_vl = vl['school']
# first fancy thing: we need to convert the strings (school names) to numbers (school indexes), and yes this is a tensor.
labels_tr = torch.tensor([SCHOOLS.index(label) for label in labels_tr])
labels_vl = torch.tensor([SCHOOLS.index(label) for label in labels_vl])

# Load DistilBERT model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(SCHOOLS))
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize the training data.
print('tokenizing training...')
# this gotta be a list
texts_tr = texts_tr.tolist()
# tok is a dictionary. input_ids is the tokenized text, attention_mask is the mask to tell the model which tokens are real and which are padding
# padding='max_length' is to pad the text to the max length of the model, which is 512
# truncation=True is to truncate the text to the max length of the model, which is 512. I think we truncate just one sentence.
tok = tokenizer(texts_tr, add_special_tokens=True, padding='max_length', max_length=512, truncation=True)
# here we just retrieve results from the dictionary and tensorize them
tokenized_texts = tok['input_ids']
attention_texts = tok['attention_mask']
input_ids = torch.tensor(tokenized_texts)
attention_texts = torch.tensor(attention_texts)
# here we create a dataset from the tensors using inputs, attention masks and labels
dataset = TensorDataset(input_ids, attention_texts, labels_tr)
# here we create a dataloader from the dataset, with a batch size of batchSize and shuffle=True
dataloader_tr = DataLoader(dataset, batch_size=batchSize, shuffle=True)

# Tokenize the validation data. it's the same as above, but with the validation data.
print('tokenizing validation...')
texts_vl = texts_vl.tolist()
tok = tokenizer(texts_vl, add_special_tokens=True, padding='max_length', max_length=512, truncation=True)
tokenized_texts_vl = tok['input_ids']
attention_texts_vl = tok['attention_mask']
input_ids_vl = torch.tensor(tokenized_texts_vl)
attention_texts_vl = torch.tensor(attention_texts_vl)
dataset_vl = TensorDataset(input_ids_vl, attention_texts_vl, labels_vl)
dataloader_vl = DataLoader(dataset_vl, batch_size=batchSize, shuffle=True)

# set device. if cuda1 is available, use it. if not, use mps if available. if not, use cpu.
device = torch.device("cuda:1" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print('device is:',device)
# our optimizer is Adam, which is a fancy version of gradient descent. lr is the learning rate.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.to(device)

# Training epoch
def compute_epoch(model:DistilBertForSequenceClassification, dataloader, optimizer, criterion=nn.functional.cross_entropy, backpropagate=True, epoch=0, device=None) -> tuple[float, float, str]:
    batchIndex = 0
    total_loss = 0
    total_accuracy = 0
    all_predictions = []
    all_labels = []
    begin = time.time()
    # extract a batch from the dataloader
    for batch in dataloader:
        batchIndex += 1
        # retrieve the input_ids, attention_mask and labels from the batch
        input_ids, attention_mask, labels = batch
        # move them tensors to the device
        if device is not None:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        # forward pass. the model returns the logits, which are the outputs of the model before the softmax
        outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        # give me them logits !!!
        logits = outputs.logits
        # compute loss. for training and validation until here is the same.
        loss = criterion(logits, labels)
        # if we're training, let model learn something.
        if backpropagate:
            # learn model, learn !!!    
            loss.backward()
            optimizer.step()
        # accumulate total loss
        total_loss += loss.item()
        # important ! zero the gradients. if you don't do this, the gradients will accumulate and the model will learn garbage.
        optimizer.zero_grad()
        # compute accuracy. we compare the argmax of the logits with the labels. if they're the same, we have a hit.
        batch_accuracy = (logits.argmax(1) == labels).float().mean()
        # accumulate total accuracy
        total_accuracy += batch_accuracy
        # this is for computing total score. 
        # add softmax winner to the predictions list
        all_predictions.extend(logits.argmax(1).tolist())
        # this is for computing total score. 
        # add the labels to the labels list
        all_labels.extend(labels.tolist())
        now = time.time()
        eta = (now - begin) * (len(dataloader) - batchIndex) / batchIndex
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, accuracy: {batch_accuracy:.2f}     eta = {int(eta//60):03d}m {int(eta%60):02d}s      ', end='\r')
    
    # compute classification report
    report = classification_report(all_labels, all_predictions)
    # return normalized loss and accuracy, and report.
    return total_loss/batchIndex, total_accuracy/batchIndex, report

# Logger. just to log stuff.
start_time = datetime.now()
print('Training and Validation -> Start Time:', start_time.strftime("H%M%S"))
log_path = os.path.join(sys.path[0], 'log_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.txt')
logger = Logger(log_path)
logger.add("Training and Validation -> Start Time: " + start_time.strftime("H%M%S"))
# Training loop.
# for each epoch
for epoch in range(num_epochs):
    # log stuff
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    logger.add(f'Epoch {epoch+1}/{num_epochs}')
    # model is in train mode. used to set the module and its child modules in training mode. 
    # this is necessary because some modules behave differently in training and evaluation mode.
    model.train()
    # compute epoch. this is the function we defined above.
    loss,acc,_ = compute_epoch(model, dataloader_tr, optimizer, epoch=epoch, device=device)
    # log stuff
    logger.add(f'Epoch: {epoch}, TR Loss: {loss:.4f}, TR accuracy: {acc:.2f}')
    # save intermediate model and tokenizer
    model.save_pretrained(os.path.join(sys.path[0], 'tuned',f'bert_model_{epoch}_{save_string}'))
    tokenizer.save_pretrained(os.path.join(sys.path[0], 'tuned',f'bert_tokenizer_{epoch}_{save_string}'))
end_time = datetime.now()

# model is in evaluation mode. used to set the module and its child modules in evaluation mode.
model.eval()
# compute epoch but without learning. this is the function we defined above.
loss,acc,report = compute_epoch(model, dataloader_vl, optimizer, backpropagate=False, epoch=epoch, device=device)
# log stuff, print stuff.
logger.add(f'Epoch: {epoch}, VL Loss: {loss:.4f}, VL accuracy: {acc:.2f}')
logger.add(report)
print('\nTraining and Validation -> End Time:', end_time.strftime("H%M%S"))
logger.add("Training and Validation -> End Time: " + end_time.strftime("H%M%S"))
logger.add("Duration: " + str(end_time - start_time))