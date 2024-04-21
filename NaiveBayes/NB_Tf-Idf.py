import pandas as pd
import numpy as np
import os 
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from sklearn.naive_bayes import MultinomialNB 
#from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scoring import scorePhilosophy
from utilities import getData

from sklearn.metrics import precision_recall_fscore_support, f1_score


# data splitting
tr, vl, _ = getData()
X_train = tr['sentence_str']
y_train = tr['school']

X_val = vl['sentence_str']
y_val = vl['school']


# declare the "builder" of the Tf-Idf weighted term-document matrix
vectorizer = TfidfVectorizer()
# create the vocabulary and obtain the document-term matrix
# for the training data...
X_train = vectorizer.fit_transform(X_train)
# ... and "test" data (without changing the vocabulary)
X_val = vectorizer.transform(X_val)

smoothing = 1/np.logspace(0,3,num=4) #0,4,num=10
pars_results = []
for smooth_par in smoothing:
    model = MultinomialNB(alpha=smooth_par, fit_prior=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print()
    print(f'\tSmoothing par = {smooth_par}')
    print('Macroaverage:')
    print(precision_recall_fscore_support(y_val, y_pred, zero_division=0.0, average='macro'))
    print('Microaverage:')
    print(precision_recall_fscore_support(y_val, y_pred, zero_division=0.0, average='micro'))
    print('Weighted Macroaverage??')
    print(precision_recall_fscore_support(y_val, y_pred, zero_division=0.0, average='weighted'))

    # I choose as a reference metric the microaverage F1
    pars_results.append((f1_score(y_true=y_val, y_pred=y_pred, average='micro'),smooth_par))

# Let's look for the best
best_f1, best_alpha = sorted(pars_results)[-1]
print(f'Found that best smoothing parameter is {best_alpha}')
print(f'Which gave a F1 score on the validation set of {best_f1}')

# we can retrain now. Shall we do it?



#SCHOOLS = df['school'].unique()

# scorePhilosophy(prediction=y_pred, 
#                 ground_truth=y_val,
#                 showConfusionMatrix=True,
#                 saveFolder='NaiveBayes/Images',
#                 saveName='NB_TF-Idf')