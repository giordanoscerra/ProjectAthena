from sklearn.naive_bayes import MultinomialNB
import pandas as pd

SCHOOLS = ['analytic','aristotle','german_idealism',
           'plato','continental','phenomenology',
           'rationalism','empiricism','feminism',
           'capitalism','communism','nietzsche',
           'stoicism']
    
def getData()->pd.DataFrame:
    return pd.read_csv('philosophy_data.csv')

def splitData(data:pd.DataFrame)->tuple:
    return train_test_split(data['sentence_str'], data['school'], test_size=0.25, random_state=42)

def vectorizeData(x_train:pd.DataFrame, x_test:pd.DataFrame)->tuple:
    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    return x_train, x_test

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    
    x_train, x_test, y_train, y_test = splitData(getData())
    x_train, x_test = vectorizeData(x_train, x_test)

    # fitprior (to set whether to learn class prior probabilities or not)
    # alpha (smoothing parameter)

    model = MultinomialNB()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    
    cm = confusion_matrix(y_test, pred, labels=SCHOOLS, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SCHOOLS)
    disp.plot()
    plt.xticks(rotation=45)
    plt.show()
    
