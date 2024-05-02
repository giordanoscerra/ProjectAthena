from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple

import os

# Load your datasets
def getData(validation_size:float=0.2, 
            test_size:float=0.1,
            from_folder:str='',
            min_chars:int=None, 
            max_chars:int=None)-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(os.path.join(from_folder,'philosophy_data.csv'))
    if min_chars is not None:
        df = df[(df['sentence_length'] >= min_chars)]
    if max_chars is not None:
        df = df[(df['sentence_length'] <= max_chars)]
    development, test = train_test_split(df, test_size=test_size, stratify=df['school'], random_state=42, shuffle=True)
    #split vslidation ant train from train
    if validation_size == 0:
        return development, None, test
    train, validation = train_test_split(development, 
                                         test_size=validation_size/(1 - test_size), 
                                         stratify=development['school'], 
                                         random_state=42, 
                                         shuffle=True)
    return train, validation, test
