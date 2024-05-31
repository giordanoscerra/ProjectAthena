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

def zero_shot(labels_dict:dict=None, min_chars:int=None, max_chars:int=None, folder:str='results/'):

    if labels_dict is None:
        labels_dict = {label: label for label in SCHOOLS}

    # Check if all strings in SCHOOLS exist as keys in labels_dict
    if all(label in labels_dict.values() for label in SCHOOLS) and len(labels_dict) == len(SCHOOLS):
        print("All strings in SCHOOLS exist as values in labels_dict")
    else:
        print("Not all strings in SCHOOLS exist as values in labels_dict")
        return None

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else -1  # set device to first GPU, -1 for CPU

    # Load data
    _, vl, _ = getData(min_chars=min_chars, max_chars=max_chars)
    texts_vl = vl['sentence_str'].tolist()
    labels_vl = vl['school'].tolist() 

    #texts_vl_subsets = np.array_split(texts_vl, 10)
    #labels_vl_subsets = np.array_split(labels_vl, 10)

    # Model name and path
    #model_name = 'deberta-v3-small-tasksource-nli'
    #model_name = 'bart-large-mnli'
    #model_path = '../' + model_name
    classifier = pipeline("zero-shot-classification", model='facebook/bart-large-mnli', device=device)

    
    # Initialize an empty list to store the predictions
    predictions = []
    texts_vl_subsets = np.array_split(texts_vl, 10)

    # Process each part separately
    for subset in tqdm(texts_vl_subsets, desc="Processing subsets"):
        if len(subset) > 0:
            predictions.extend(classifier(subset.tolist(), SCHOOLS))
    
    # predictions = classifier(texts_vl, list(labels_dict.keys()))

    # Save predictions in an indented JSON file
    os.makedirs('Zero-shot/' + folder, exist_ok=True)
    with open('Zero-shot/' + folder + 'predictions.json', 'w') as f:

        # Convert the predictions list to a JSON string with indentation
        json.dump(predictions, f, indent=4)

    # Get the predicted labels
    predicted_labels = [labels_dict[pred['labels'][0]] for pred in predictions]

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
    with open('Zero-shot/' + folder + 'results.json', 'w') as f:

        # Convert the dictionary to a JSON string
        json.dump(results, f, indent=4)

    # Plot the confusion matrix
    scorePhilosophy(predicted_labels, labels_vl, modelName='Zero-shot', subtitle='Zero-shot Classification', saveFolder= 'Zero-shot/'+ folder, saveName='zero-shot')

    #{'labels': ['travel', 'dancing', 'cooking'],
    # 'scores': [0.9938651323318481, 0.0032737774308770895, 0.002861034357920289],
    # 'sequence': 'one day I will see the world'}

if __name__ == '__main__':

    labels_dict = {
        'Analytic Philosophy': 'analytic',
        'Aristotelian Philosophy': 'aristotle',
        'German Idealism Philosophy': 'german_idealism',
        'Platonic Philosophy': 'plato',
        'Continental Philosophy': 'continental',
        'Phenomenological Philosophy': 'phenomenology',
        'Rationalist Philosophy': 'rationalism',
        'Empiricist Philosophy': 'empiricism',
        'Feminist Philosophy': 'feminism',
        'Capitalist Philosophy': 'capitalism',
        'Communist Philosophy': 'communism',
        'Nietzschean Philosophy': 'nietzsche',
        'Stoic Philosophy': 'stoicism'
    }

    #zero_shot(min_chars=84, folder='bart-large-mnli_84+_chars_noLabels/')
    #zero_shot(labels_dict=labels_dict, min_chars=84, folder='bart-large-mnli_84+_chars_phiLabels/')
    zero_shot(max_chars=83, folder='bart-large-mnli_83-_chars_noLabels/')
    zero_shot(labels_dict=labels_dict, max_chars=83, folder='bart-large-mnli_83-_chars_phiLabels/')