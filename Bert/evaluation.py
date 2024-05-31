import time
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from scoring import SCHOOLS
from utilities import getData
import numpy as np
from scoring import scorePhilosophy

#pick between 20 and 84
train_len: int = 84
#pick between 20 and 84
val_char: int = 84
# Load the fine-tuned DistilBERT model
model = BertForSequenceClassification.from_pretrained(f'./Bert/Bert_{train_len}/model', local_files_only=True)
#model = BertForSequenceClassification.from_pretrained(f'./Bert/1_bert_model_162356', local_files_only=True)
# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(f'./Bert/Bert_{train_len}/tokenizer', local_files_only=True)
#tokenizer = BertTokenizer.from_pretrained(f'./Bert/1_bert_tokenizer_162356', local_files_only=True)

# Load data
_, vl, _ = getData(min_chars=val_char, max_chars=1700)
texts_vl = vl['sentence_str']
texts_vl = texts_vl.tolist()
encoded_inputs_vl = tokenizer(texts_vl, padding=True, truncation=True, return_tensors='pt', max_length=360)
labels_vl = vl['school']
labels_vl = [SCHOOLS.index(label) for label in labels_vl]

validation_dataset = TensorDataset(encoded_inputs_vl['input_ids'], encoded_inputs_vl['attention_mask'], torch.tensor(labels_vl))
validation_dataloader = DataLoader(validation_dataset, batch_size=100, shuffle=False)

def compute_accuracy(model, dataloader)->tuple[float, list[int], list[int]]:
    device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
    print('device is:',device)
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    predictions = []
    labels_original = []
    with torch.no_grad():
        begin = time.time()
        iteration = 0
        for inputs, attention_mask, labels in dataloader:
            iteration += 1
            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            predicted_labels = outputs.logits.argmax(dim=1)
            predictions.extend(predicted_labels.cpu().numpy())
            labels_original.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()
            part = time.time() - begin
            eta = part/iteration*(len(dataloader)-iteration)
            print(f'eta: {eta//60//60}h {eta//60%60}m {eta%60:.2f}s', end='\r')
    return correct / total, labels_original, predictions

# Compute the accuracy on the validation set
validation_accuracy, labels, predictions = compute_accuracy(model, validation_dataloader)
print(f'Validation accuracy: {validation_accuracy}')
# Save labels and predictions in a file
#np.savetxt(f'./Bert/Bert_{train_len}/results_{val_char}.txt', np.column_stack((labels, predictions)), fmt='%d')
# Compute the confusion matrix
labels = [SCHOOLS[label-1] for label in labels]
predictions = [SCHOOLS[label-1] for label in predictions]
print(f'results for training set of {train_len} and validation set of {val_char}')
scorePhilosophy(predictions, labels, 
                modelName=f'BERT_{train_len}', 
                subtitle='Validation set', 
                saveName=f'Bert/Bert_{train_len}/confusion_matrix_No1_{val_char}.png', 
                showConfusionMatrix=True, saveFolder='.')