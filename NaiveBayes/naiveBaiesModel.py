from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import re
from nltk.corpus import stopwords
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from scoring import *

def getStopWords()->list:
    return stopwords.words('english')

def addNegationsToken(phrase:str)->str:
    phrase = re.sub(r"n't", " not", phrase)
    transformed = re.sub(r'\b(?:not|never|no)\b[\w\s]+', 
                         lambda match: re.sub(r'(\s+)(\w+)', r'\1NEG_\2', match.group(0)), 
                         phrase,
                         flags=re.IGNORECASE)
    return transformed

def addNegationsToData(data:pd.DataFrame)->pd.DataFrame:
    data['sentence_str'] = data['sentence_str'].apply(addNegationsToken)
    return data

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    
    ############################################
    ######### Beging of the main code  #########
    ############################################

    #x_train, x_test, y_train, y_test = splitData(addNegationsToData(getData()))
    x_train, x_test, y_train, y_test = splitData(getData())
    #vectorizer = CountVectorizer(analyzer='word', stop_words=getStopWords())
    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    # fitprior (to set whether to learn class prior probabilities or not)
    # alpha (smoothing parameter)
    model = MultinomialNB(alpha=0.06, fit_prior=True)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    ############################################
    ########### End of the main code ###########
    ############################################
    

    scorePhilosophy(pred, y_test, showConfusionMatrix=False)
    exit()
    cm = confusion_matrix(y_test, pred, labels=SCHOOLS, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SCHOOLS)
    disp.plot()
    #make plot bigger
    plt.gcf().set_size_inches(12, 12)
    plt.xticks(rotation=45)
    #plt.savefig('bestNaive.png', dpi=300)
    plt.show()
    #save the plot in hd
    print(model.score(x_train, y_train))
    print(model.score(x_test, y_test))

    exit()
    while True:
        print("give me a sentence: ")
        sentence = input()
        sentence = addNegationsToken(sentence)
        sentence = vectorizer.transform([sentence])
        print(model.predict(sentence))