#import pandas as pd
import numpy as np
import os 
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from datetime import datetime

from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ParameterGrid

from sklearn.metrics import precision_recall_fscore_support, f1_score
from scoring import scorePhilosophy
from utilities import getData

# file, folder, path... Where to record our numbers
# Pathing is a nightmare in python...
FOLDER = os.path.join('NaiveBayes','Results')
FILENAME = 'NB_Count_results.txt'
FILEPATH = os.path.join(FOLDER,FILENAME)

# data import and splitting
tr, vl, _ = getData()
X_train = tr['sentence_str']
y_train = tr['school']

X_val = vl['sentence_str']
y_val = vl['school']


# declare the "builder" of the Tf-Idf weighted term-document matrix
vectorizer = CountVectorizer()
# create the vocabulary and obtain the document-term matrix
# for the training data...
X_train = vectorizer.fit_transform(X_train)
# ... and "test" data (without changing the vocabulary)
X_val = vectorizer.transform(X_val)


with open(FILEPATH, 'w') as file:
    file.write('Naive Bayes on the Tf-Idf weighted term-document matrix\n')
    file.write('Scores for different values of the smoothing parameter\n\n')
    file.write(f'Ran on {datetime.now()}\n\n')


# declare the parameters for grid search
full_grid = {
    'alpha': list(np.append(1/np.logspace(0,4,num=10),0)),
    'fit_prior': [True, False]
}

pars_results = []
for parameters in ParameterGrid(full_grid):
    model = MultinomialNB(**parameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    with open(FILEPATH, 'a') as file:
        for key, value in parameters.items():
            rest = f'{value}\t' if isinstance(value,bool) else f'{value:<10.6f}\t'
            file.write(f'\t{key} = ' + rest)
        file.write('\n')

        macroPrec, macroRec, macroF1, _ = precision_recall_fscore_support(y_val, y_pred, zero_division=0.0, average='macro')
        file.write(f'Macroaverage precision = {macroPrec:<10.5f}\t') 
        file.write(f'Macroaverage recall = {macroRec:<10.5f}\t')
        file.write(f'Macroaverage F1 = {macroF1:<10.5f}\n')

        weightedPrec, weightedRec, weightedF1, _ = precision_recall_fscore_support(y_val, y_pred, zero_division=0.0, average='weighted')
        file.write(f'Macroaverage precision = {weightedPrec:<10.5f}\t')
        file.write(f'Macroaverage recall = {weightedRec:<10.5f}\t')
        file.write(f'Macroaverage F1 = {weightedF1:<10.5f}\n')

        micro = f1_score(y_true=y_val, y_pred=y_pred, average='micro')
        file.write(f'Microaverage precision = microaverage recall = microaverage F1 = {micro:<10.5f}\n')
        file.write('\n')

    # I choose as a reference metric the microaverage F1
    pars_results.append((f1_score(y_true=y_val, y_pred=y_pred, average='micro'),parameters))

# Let's look for the best
best_f1, best_parameters = sorted(pars_results)[-1]

# print the best parameters on file
with open(FILEPATH, 'a') as file:
    file.write('\n\n\t\tBest parameters found:\n')
    for key, value in best_parameters.items():
        rest = f'{value}\t' if isinstance(value,bool) else f'{value:<10.6f}\t'
        file.write(f'\t{key} = ' + rest)
    file.write('\n')
    file.write(f'F1 score on the validation set = {best_f1:.4f}\n')

# print the best parameters on screen
print('Found that best parameters are: ')
for key, value in best_parameters.items():
    print(f'{key} = {value}')
print(f'This combination gave a F1 score on the validation set of {best_f1}')

# we can retrain now. Shall we do it on Tr + Val? How??
model = MultinomialNB(**best_parameters)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

# show final score
scorePhilosophy(prediction=y_pred, 
                ground_truth=y_val,
                modelName='Naive Bayes (Tf)',
                subtitle=f'alpha = {best_parameters['alpha']:.4f}, fit_prior = {best_parameters['fit_prior']}',
                showConfusionMatrix=True,
                saveFolder='NaiveBayes/Images',
                saveName='NB_Count')