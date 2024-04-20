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


# import data
df = pd.read_csv('philosophy_data.csv')

# data splitting
tr, vl, _ = getData()
X_train = tr['sentence_str']
y_train = tr['school']

X_val = vl['sentence_str']
y_val = vl['school']

# declare the "builder" of the Tf-Idf weighted
# term-document matrix
vectorizer = TfidfVectorizer()
# create the vocabulary and obtain the document-term matrix
# for the training data...
X_train = vectorizer.fit_transform(X_train)
# ... and "test" data (without changing the vocabulary)
X_val = vectorizer.transform(X_val)

# I copied the alpha from DavideB. maybe can be fine-tuned with 
# a model selection approach...
model = MultinomialNB(alpha=0.06, fit_prior=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

#SCHOOLS = df['school'].unique()

scorePhilosophy(prediction=y_pred, 
                ground_truth=y_val,
                showConfusionMatrix=True,
                saveFolder='NaiveBayes/Images',
                saveName='NB_TF-Idf')