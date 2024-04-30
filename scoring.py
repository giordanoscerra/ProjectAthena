import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from typing import List

SCHOOLS = ['analytic','aristotle','german_idealism',
           'plato','continental','phenomenology',
           'rationalism','empiricism','feminism',
           'capitalism','communism','nietzsche',
           'stoicism']

def scorePhilosophy(prediction: List[str], 
                    ground_truth: List[str], 
                    saveName:str=None, 
                    saveFolder:str='..',
                    showConfusionMatrix:bool=False) -> None:
    cm = confusion_matrix(ground_truth, prediction, labels=SCHOOLS, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SCHOOLS)
    disp.plot()
    plt.gcf().set_size_inches(12, 12)
    plt.xticks(rotation=45)
    if saveName:
        fig_path = os.path.join(saveFolder, saveName)
        plt.savefig(fig_path, dpi=300)
    if showConfusionMatrix:
        plt.show()
    #classificaiton report
    print(classification_report(ground_truth, 
                                prediction, 
                                target_names=SCHOOLS,
                                #labels=list(range(1,14))
                                ))

def getData()->pd.DataFrame:
    return pd.read_csv('philosophy_data.csv')

def splitData(data:pd.DataFrame, test_size=0.25)->tuple:
    return train_test_split(data['sentence_str'], data['school'], test_size=test_size, random_state=42)

def filterShortPhrases(data:pd.DataFrame, numWords)->pd.DataFrame:
    return data[data['sentence_str'].apply(lambda x: len(x.split())>numWords)]
