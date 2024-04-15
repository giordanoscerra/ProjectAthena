from typing import List, Dict, Optional, Tuple

from sklearn.feature_extraction.text import CountVectorizer
import wordcloud 
import os


def create_bow(school:str, 
               dataframe,
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
def make_wordcloud(bow:Dict[str,int], 
                   stopwords:Optional[List[str]] = None, 
                   bgcolor:str = 'white'):
    """Creates a wordcloud for a given text

    Args:
        text (str): the text to generate the wordcloud from
        stopwords (List[str], optional): words to exclude from the wordcloud. Defaults to None.
        bgcolor (str, optional): The background color of the wordcloud. Defaults to 'white'.

    Returns:
        wordcloud.WordCloud: the generated wordcloud
    """
    cloud = wordcloud.WordCloud(
                width = 2000, height = 1100,
                background_color = bgcolor,
                stopwords = stopwords,
                min_font_size = 8).generate_from_frequencies(frequencies = bow)
    
    return cloud

def save_plot(plot, 
              filename:str,
              folder:str='Images', 
              format:str='png') -> None:
    if not filename.endswith(format):
        filename = filename.split('.')[0] + '.' + format

    fig_path = os.path.join(folder, filename)
    plot.savefig(fig_path, format=format)