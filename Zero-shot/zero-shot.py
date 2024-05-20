import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from scoring import scorePhilosophy, SCHOOLS  # Assuming SCHOOLS is a list of school names
from utilities import getData

renamed_schools = ['Analytic Philosophy',
                   'Aristotelian Philosophy',
                   'German Idealism Philosophy',
                   'Platonic Philosophy',
                    'Continental Philosophy',
                    'Phenomenological Philosophy',
                    'Rationalist Philosophy',
                    'Empiricist Philosophy',
                    'Feminist Philosophy',
                    'Capitalist Philosophy',
                    'Communist Philosophy',
                    'Nietzschean Philosophy',
                    'Stoic Philosophy'
                    ]

# Check if CUDA is available
device = 'cuda:1' if torch.cuda.is_available() else -1  # set device to first GPU, -1 for CPU

# Load data
min_chars = 84
_, vl, _ = getData(min_chars=min_chars)
texts_vl = vl['sentence_str'].tolist()
labels_vl = vl['school'].tolist()
#texts_vl_subsets = np.array_split(texts_vl, 10)
#labels_vl_subsets = np.array_split(labels_vl, 10)

# Model name and path
#model_name = 'deberta-v3-small-tasksource-nli'
#model_name = 'bart-large-mnli'
#model_path = '../' + model_name
classifier = pipeline("zero-shot-classification", model='facebook/bart-large-mnli', device=device)

"""
# Initialize an empty list to store the predictions
predictions = []

# Process each part separately
for subset in tqdm(texts_vl_subsets, desc="Processing subsets"):
    if len(subset) > 0:
        predictions.extend(classifier(subset.tolist(), SCHOOLS))
"""

predictions = classifier(texts_vl, renamed_schools)

# Save predictions in an indented JSON file
with open('Zero-shot/predictions.json', 'w') as f:
    
    # Convert the predictions list to a JSON string with indentation
    json.dump(predictions, f, indent=4)

exit()

# Get the predicted labels
predicted_labels = [pred['labels'][0] for pred in predictions]

# Create a dictionary with the report, accuracy, and confusion matrix
results = {

    # Minimum number of characters per sentence
    'min_chars': min_chars,

    # Calculate accuracy
    'accuracy': accuracy_score(labels_vl, predicted_labels),

    # Generate classification report
    'report': classification_report(labels_vl, predicted_labels, labels=SCHOOLS, output_dict=True, zero_division=1),

    # Generate confusion matrix
    'confusion_matrix': confusion_matrix(labels_vl, predicted_labels, labels=SCHOOLS).tolist()  # Convert the numpy array to a list
}

# Write the JSON string to a file
with open('Zero-shot/results.json', 'w') as f:

    # Convert the dictionary to a JSON string
    json.dump(results, f, indent=4)

# Plot the confusion matrix
scorePhilosophy(predicted_labels, labels_vl, modelName='Zero-shot', subtitle='Zero-shot Classification', saveFolder= 'Zero-shot', saveName='zero-shot')

#{'labels': ['travel', 'dancing', 'cooking'],
# 'scores': [0.9938651323318481, 0.0032737774308770895, 0.002861034357920289],
# 'sequence': 'one day I will see the world'}