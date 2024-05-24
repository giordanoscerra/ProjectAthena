import os
from typing import List, Dict, Optional, Tuple

from sklearn.feature_extraction.text import CountVectorizer
import wordcloud 

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import nltk

from itertools import chain


def create_bow_legacy(school:str, 
               dataframe,
               stopwords:Optional[str|List] = None) -> Dict[str,int]:
    """Returns a bag of words of the given school.

    Returns:
        Dict[str,int]: bag of words
    """

    texts = [stringa for stringa in dataframe.loc[dataframe['school'] == school,'sentence_str']]

    # We now create the bag of word vocabulary

    vectorizer = CountVectorizer(
        analyzer='word',
        stop_words=stopwords
        )
    # learn vocabulary, return term-document matrix
    vector = vectorizer.fit_transform(texts)
    # create the dictionary
    wc_dict = dict(zip(vectorizer.get_feature_names_out(), vector.toarray().sum(axis=0)))

    return wc_dict

def create_bow(school:str,
                dataframe,
                lowercase:bool = True,
                stopwords:Optional[str|List] = []) -> Dict[str,int]:
    """Returns a bag of words of the given school.

    Returns:
        Dict[str,int]: bag of words
    """
    # Get the tokenized sentences of the school
    tokenized_sentences = dataframe.loc[dataframe['school'] == school,'tokenized_txt']

    # Create and return the bag of words
    return nltk.FreqDist([(w.lower() if lowercase else w) for w in chain.from_iterable(tokenized_sentences) if (w not in stopwords and w.isalnum())])


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

# Function that plots the two distributions (contrastive analysis)
def plot_superimposed_distributions(bag_of_words1, 
                                    bag_of_words2,
                                    ax=None,
                                    **kwargs) -> matplotlib.axes.Axes:
    """Plot the superimposed distributions of two bags of words.

    Args:
        bag_of_words1 (dict): First bag of words.
        bag_of_words2 (dict): Second bag of words.
        **kwargs: Additional keyword arguments for customization.
            range (int): Range for x-axis ticks.
            color1 (str): Color for the first distribution plot.
            color2 (str): Color for the second distribution plot.
            label1 (str): Label for the first distribution plot.
            label2 (str): Label for the second distribution plot.
            distribution1 (str): Name of the first distribution.
            distribution2 (str): Name of the second distribution.

    Returns:
        ax (matplotlib.axes.Axes): The created ax of the figure.
    """    
    
    RANGE = kwargs.get('range', kwargs.get('RANGE',0))
    # Combine the vocabularies of both bags of words
    if RANGE:     
        shared_support = {k for (_,k) in zip(range(RANGE),bag_of_words1.keys())}&{k for (_,k) in zip(range(RANGE),bag_of_words2.keys())}
        exclusive_words1 = {k for (_,k) in zip(range(RANGE),bag_of_words1.keys())} - shared_support
        exclusive_words2 = {k for (_,k) in zip(range(RANGE),bag_of_words2.keys())} - shared_support                         
    else:
        shared_support = set(bag_of_words1.keys()) & set(bag_of_words2.keys())
        exclusive_words1 = set(bag_of_words1.keys()) - shared_support
        exclusive_words2 = set(bag_of_words2.keys()) - shared_support

    # isolate the exclusive words, and the shared part
    all_words = list(exclusive_words1) + list(shared_support) + list(exclusive_words2)

    # Get frequencies of each word in the "merged" bag of words
    freq1 = [bag_of_words1.get(word,0) for word in all_words]
    freq2 = [bag_of_words2.get(word,0) for word in all_words]

    # Plot the two distributions
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    ax.plot(range(len(all_words)), freq1, color=kwargs.get('color1','b'), alpha=0.5, label=kwargs.get('label1','Bag of Words 1'))
    ax.plot(range(len(all_words)), freq2, color=kwargs.get('color2','r'), alpha=0.5, label=kwargs.get('label2','Bag of Words 2'))
    
    ax.set_xlabel('Words (ordinal number)', fontsize=12)
    ax.set_ylabel('Normalized frequency', fontsize=12)
    step = 10**round(np.log10(len(all_words)/10))
    ax.set_xticks(np.arange(0, len(all_words), step))
    ax.set_xticklabels(np.arange(0, len(all_words), step))

    # Create the title string if both distributions are provided
    distribution1 = kwargs.get("distribution1", "")
    distribution2 = kwargs.get("distribution2", "")
    title_string = f'{distribution1.capitalize()} vs {distribution2}' if (distribution1 and distribution2) else 'Superimposed distributions'

    ax.set_title(title_string, fontsize=20)
    ax.legend(fontsize=16)

    return ax   

def save_plot(plot, 
              filename:str,
              folder:str='Images', 
              format:str='png') -> None:
    if not filename.endswith(format):
        filename = filename.split('.')[0] + '.' + format

    fig_path = os.path.join(folder, filename)
    plot.savefig(fig_path, format=format)