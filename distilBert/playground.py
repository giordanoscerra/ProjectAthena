import time
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from scoring import SCHOOLS

###################################################################
## this file is used to test the model in a more interactive way ##
###################################################################

try:
    # Load the fine-tuned DistilBERT model
    model = DistilBertForSequenceClassification.from_pretrained('./distilBert/tuned/final_bert_model_280430_full_dataset/', local_files_only=True)
    # Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('./distilBert/tuned/final_bert_tokenizer_280430_full_dataset/', local_files_only=True)
except:
    print('Model not found')
    print('Check the path to the model and tokenizer')
    print('Correct paths are: ./distilBert/tuned/final_bert_model_280430_full_dataset/ and ./distilBert/tuned/final_bert_tokenizer_280430_full_dataset/')
    exit(1)


device = torch.device("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu")
print('device is:',device)
model.to(device)
# evaluation mode!
model.eval()
userPhrase = input('Enter a phrase: ')
while userPhrase != 'exit': 
    with torch.no_grad():
        begin = time.time()
        # Tokenize the user input, again pad and truncate to 280
        encoded_inputs = tokenizer(userPhrase, padding=True, truncation=True, return_tensors='pt', max_length=280)
        inputs = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        outputs = model(input_ids=inputs, attention_mask=attention_mask)
        # retrieve max logit. the winner!
        predicted_labels = outputs.logits.argmax(dim=1)
        part = time.time() - begin
        print(f'Predicted label: {SCHOOLS[predicted_labels]} in {part:.0f}s')
        userPhrase = input('Enter a phrase: ')