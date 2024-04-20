import pandas as pd
import numpy as np
import os 
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from sklearn.naive_bayes import MultinomialNB 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scoring import scorePhilosophy

# import data
df = pd.read_csv('philosophy_data.csv')

# data splitting
X_train, X_test, y_train, y_test = train_test_split(df['sentence_str'], df['school'], test_size=0.25, random_state=42)

# declare the "builder" of the Tf-Idf weighted
# term-document matrix
vectorizer = TfidfVectorizer()
# create the vocabulary and obtain the document-term matrix
# for the training data...
X_train = vectorizer.fit_transform(X_train)
# ... and test data (without changing the vocabulary)
X_test = vectorizer.transform(X_test)

# I copied the alpha from DavideB. maybe can be fine-tuned with 
# a model selection approach...
model = MultinomialNB(alpha=0.06, fit_prior=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#SCHOOLS = df['school'].unique()

scorePhilosophy(prediction=y_pred, 
                ground_truth=y_test,
                showConfusionMatrix=True,
                saveName='NB_TF-Idf')