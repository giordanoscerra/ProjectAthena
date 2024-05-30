import time
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from scoring import SCHOOLS
from utilities import getData
import numpy as np
from scoring import scorePhilosophy

# Load the fine-tuned DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained('./distilBert/tuned/bert_model_2/', local_files_only=True)
# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('./distilBert/tuned/bert_tokenizer_2/', local_files_only=True)


device = torch.device("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu")
print('device is:',device)
model.to(device)
model.eval()
userPhrase = input('Enter a phrase: ')
while userPhrase != 'exit': 
    with torch.no_grad():
        begin = time.time()
        encoded_inputs = tokenizer(userPhrase, padding=True, truncation=True, return_tensors='pt', max_length=280)
        inputs = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        outputs = model(input_ids=inputs, attention_mask=attention_mask)
        predicted_labels = outputs.logits.argmax(dim=1)
        part = time.time() - begin
        print(f'Predicted label: {SCHOOLS[predicted_labels]} in {part:.0f}s')
        userPhrase = input('Enter a phrase: ')