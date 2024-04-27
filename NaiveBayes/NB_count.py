#import pandas as pd
import numpy as np
import os 
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from datetime import datetime

from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import CountVectorizer
from scoring import scorePhilosophy
from utilities import getData

from sklearn.metrics import precision_recall_fscore_support, f1_score

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


smoothing = np.append(1/np.logspace(0,4,num=10),0) #0,4,num=10 or 0,3,num=4
pars_results = []
for smooth_par in smoothing:
    model = MultinomialNB(alpha=smooth_par, fit_prior=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    with open(FILEPATH, 'a') as file:
        file.write(f'\tSmoothing parameter = {smooth_par}\n')
        macroPrec, macroRec, macroF1, _ = precision_recall_fscore_support(y_val, y_pred, zero_division=0.0, average='macro')
        file.write(f'Macroaverage precision = {macroPrec}\t') 
        file.write(f'Macroaverage recall = {macroRec}\t')
        file.write(f'Macroaverage F1 = {macroF1}\n')

        weightedPrec, weightedRec, weightedF1, _ = precision_recall_fscore_support(y_val, y_pred, zero_division=0.0, average='weighted')
        file.write(f'Macroaverage precision = {weightedPrec}\t')
        file.write(f'Macroaverage recall = {weightedRec}\t')
        file.write(f'Macroaverage F1 = {weightedF1}\n')

        micro = f1_score(y_true=y_val, y_pred=y_pred, average='micro')
        file.write(f'Microaverage precision = microaverage recall = microaverage F1 = {micro}\n')
        file.write('\n')

    # I choose as a reference metric the microaverage F1
    pars_results.append((f1_score(y_true=y_val, y_pred=y_pred, average='micro'),smooth_par))

# Let's look for the best
best_f1, best_alpha = sorted(pars_results)[-1]
print(f'Found that best smoothing parameter is {best_alpha}')
print(f'Which gave a F1 score on the validation set of {best_f1}')

# we can retrain now. Shall we do it on Tr + Val? How??
model = MultinomialNB(alpha=best_alpha, fit_prior=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

# show final score
scorePhilosophy(prediction=y_pred, 
                ground_truth=y_val,
                modelName='Naive Bayes (Tf)',
                subtitle=f'smoothing parameter = {best_alpha:.4f}',
                showConfusionMatrix=True,
                saveFolder='NaiveBayes/Images',
                saveName='NB_Count')