import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from typing import List

SCHOOLS = ['analytic','aristotle','german_idealism',
           'plato','continental','phenomenology',
           'rationalism','empiricism','feminism',
           'capitalism','communism','nietzsche',
           'stoicism']

def scorePhilosophy(prediction: List[str], 
                    ground_truth: List[str], 
                    modelName:str = '',
                    subtitle:str = '',
                    saveName:str=None, 
                    saveFolder:str='..',
                    showConfusionMatrix:bool=False) -> None:
    '''
    This function prints the classification report and plots the confusion matrix.
    modelName: str, the name of the model (for the title of the confusion matrix)
    subtitle: str, a subtitle for the confusion matrix
    saveName: str, the name of the file where the confusion matrix will be saved
    saveFolder: str, the folder where the confusion matrix will be saved
    showConfusionMatrix: bool, whether to show the confusion matrix or not
    '''
    cm = confusion_matrix(ground_truth, prediction, labels=SCHOOLS, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SCHOOLS)
    disp.plot()
    plt.gcf().set_size_inches(12, 12)
    plt.xticks(rotation=45)
    plt.suptitle(f'{modelName} Confusion matrix', fontweight='bold', fontsize=20)
    if subtitle != '':
        plt.title(subtitle, fontsize=16)
    if saveName:
        fig_path = os.path.join(saveFolder, saveName)
        plt.savefig(fig_path, dpi=300)
    if showConfusionMatrix:
        plt.show()
    
    report = classification_report(ground_truth, 
                                   prediction, 
                                   target_names=SCHOOLS,
                                   )
    print(report)
    return report

def getData()->pd.DataFrame:
    return pd.read_csv('philosophy_data.csv')