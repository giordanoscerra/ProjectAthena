import os
import sys
BASE_PATH = '..'

# More imports:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

import nltk

from typing import List, Dict, Optional, Tuple

df = pd.read_csv('philosophy_data.csv')


MAX_TOKENS = 1074

def check_conditions(df:pd.DataFrame, index:int) -> bool:
    check = True
    if index > 0:
        prev = index-1
        check = check and (df['title'].loc[prev] == df['title'].loc[index])
        # I skip checks on authors and schools.
        # If the book changes then the author changes, 
        # and therefore the school changes as well (this has been checked)
        
        # Let's skip the controls on info such as original publication date 
        # (who needs those anyway)
        
        # 
        check = check and (df['sentence_length'].loc[prev] + df['sentence_length'].loc[index] < MAX_TOKENS + 2)

    return check

df_new = pd.DataFrame(columns=df.columns)

i = 0
while i < 100:#df.shape[0]:
    print(f'iteration {i}')
    sentence_spacy = ''
    sentence_length = 0
    sentence_lowered = ''
    tokenized_txt = ''      # the fact that pandas turns it into a string is ridiculous
    lemmatized_str = ''

    # hopefully I don't mess up with the indexing...
    while check_conditions(df,i):
        sentence_spacy += df['sentence_spacy'].loc[i] + ' '
        sentence_length += 1 + df['sentence_length'].loc[i]
        sentence_lowered += df['sentence_lowered'].loc[i] + ' '
        tokenized_txt += df['tokenized_txt'].loc[i][1:-1] + ', '
        lemmatized_str += df['lemmatized_str'].loc[i]
        i += 1
        print(f'i increased to {i}')


    row = {
        'title': df['title'].loc[i],
        'author': df['author'].loc[i],
        'school': df['school'].loc[i],
        'sentence_spacy': sentence_spacy[:-1],
        'sentence_str': sentence_spacy[:-1],
        'original_publication_date': df['original_publication_date'].loc[i],
        'corpus_edition_date': df['corpus_edition_date'].loc[i],
        'sentence_lenght': sentence_length,
        'sentence_lowered': sentence_lowered[:-1],
        'tokenized_txt': '[' + tokenized_txt[:-2] + ']',
        'lemmatized_str': lemmatized_str
    }

    df_new = df_new._append(row, ignore_index=True)