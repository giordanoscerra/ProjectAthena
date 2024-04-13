from typing import List, Dict, Optional

from sklearn.feature_extraction.text import CountVectorizer
import wordcloud 


def create_bow(school:str, dataframe,
               stopwords:Optional[str|List] = None) -> Dict[str,int]:
    """Returns a bag of words of the given school.

    Returns:
        Dict[str,int]: bag of words
    """
    texts = []
    # Put together all the sentences of a given school
    for stringa in dataframe.loc[dataframe['school'] == school,'sentence_str']:
        texts.append(stringa)

    # We now create the bag of word vocabulary

    vectorizer = CountVectorizer(
        analyzer='word',
        stop_words=stopwords    # leap of faith here
        )
    # learn vocabulary, return term-document matrix
    vector = vectorizer.fit_transform(texts)
    # create the dictionary
    wc_dict = dict(zip(vectorizer.get_feature_names_out(), vector.toarray().sum(axis=0)))

    return wc_dict

# function that creates the wordcloud
def make_wordcloud(text:str, 
                   stopwords:Optional[List[str]] = None, 
                   bgcolor:str = 'white'):
    cloud = wordcloud.WordCloud(
                width = 800, height = 800,
                background_color = bgcolor,
                stopwords = stopwords,
                min_font_size = 8).generate(text)
    
    return cloud


def create_bigstring(school: str, dataframe) -> str:
    """Concatenates all the sentences of a given school 
    (as stored in the dataframe) into one big string

    Args:
        school (str): the philosophical school
        dataframe (pandas.core.frame.DataFrame): the dataframe with all the philosophical data

    Returns:
        str: the concatenated sentences relative to the school
    """
    text = ''
    separator = ' '
    for stringa in dataframe.loc[dataframe['school'] == school,'sentence_str']:
        text += separator + stringa
    
    return text