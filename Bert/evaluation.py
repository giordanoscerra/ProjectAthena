import time
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from scoring import SCHOOLS
from utilities import getData
from scoring import scorePhilosophy

#pick between 20 and 84
train_len: int = 84
#pick between 20 and 84
val_char: int = 84


if train_len != 84 and train_len != 20:
    print('Invalid train_len, pick between 20 and 84')
    exit(1)
if val_char != 84 and val_char != 20:
    print('Invalid val_char, pick between 20 and 84')
    exit(1)

try:
    # Load the fine-tuned DistilBERT model
    model = BertForSequenceClassification.from_pretrained(f'./Bert/Bert_{train_len}/model', local_files_only=True)
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(f'./Bert/Bert_{train_len}/tokenizer', local_files_only=True)
except:
    print('Model not found')
    print('Check the path to the model and tokenizer')
    print(f'Correct paths are: ./Bert/Bert_{train_len}/model and ./Bert/Bert_{train_len}/tokenizer')
    exit(1)

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
    '''
    Function to compute the accuracy of the model, returns the accuracy, the original labels and the predictions
    '''
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
# Compute the confusion matrix
labels = [SCHOOLS[label-1] for label in labels]
predictions = [SCHOOLS[label-1] for label in predictions]
print(f'results for training set of {train_len} and validation set of {val_char}')
scorePhilosophy(predictions, labels, 
                modelName=f'BERT_{train_len}', 
                subtitle='Validation set', 
                saveName=f'Bert/Bert_{train_len}/confusion_matrix_No1_{val_char}.png', 
                showConfusionMatrix=True, saveFolder='.')