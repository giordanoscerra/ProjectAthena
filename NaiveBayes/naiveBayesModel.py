from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from nltk.corpus import stopwords
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from scoring import scorePhilosophy
from utilities import *
from scoring import SCHOOLS  # Assuming SCHOOLS is a list of school names

def getStopWords()->list:
    return stopwords.words('english')

def addNegationsToken(phrase:str)->str:
    '''
    Adds a NEG_ token to all words following a negation word
    '''
    # Replace n't with not
    phrase = re.sub(r"n't", " not", phrase)
    # \b is to ensure that the negation word is a whole word
    # \w+ is to match the word following the negation word
    # \s+ is to match the space between the negation word and the following word
    # \1 and \2 are backreferences to the matched groups. \1 is the space and \2 is the word
    # The r before the string is to make it a raw string, so that the backslashes are not treated as escape characters
    # The lambda function is to replace the matched string with the original space and the word with NEG_ added to it
    # The flags=re.IGNORECASE is to make the regex case-insensitive
    transformed = re.sub(r'\b(?:not|never|no)\b[\w\s]+', 
                         lambda match: re.sub(r'(\s+)(\w+)', r'\1NEG_\2', match.group(0)), 
                         phrase,
                         flags=re.IGNORECASE)
    return transformed

def addNegationsToData(data:pd.DataFrame)->pd.DataFrame:
    '''
    Adds a NEG_ token to all words following a negation word
    '''
    data['sentence_str'] = data['sentence_str'].apply(addNegationsToken)
    return data

if __name__ == '__main__':
    min_chars = 84
# prepare data
    tr, vl, ts = getData(min_chars=min_chars)
    x_train = tr['sentence_str']
    y_train = tr['school']
    x_val = vl['sentence_str']
    y_val = vl['school']
    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_val = vectorizer.transform(x_val)
# train the model. alpha is the smoothing parameter. fit_prior is whether to learn class prior probabilities or not
    model = MultinomialNB(alpha=0.06, fit_prior=True)
    model.fit(x_train, y_train)
# test the model
    pred = model.predict(x_val)
    scorePhilosophy(pred, y_val, showConfusionMatrix=True)
# use the model
    while True:
        print("give me a sentence: ")
        sentence = input()
        if sentence == 'exit':
            break
        sentence = vectorizer.transform([sentence])
        print(model.predict(sentence))