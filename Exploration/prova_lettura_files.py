import os
import sys
import pandas as pd
import numpy as np
from itertools import chain
import nltk 

# read files in a directory
BASE_PATH = ''
DIRECTORY = 'Exploration/SimpleEnglishWikipedia'
subdirectory = '1of2'

for i,filename in zip(range(10), os.listdir(os.path.join(DIRECTORY,'1of2'))):
    print(filename)


for i in range(10):
    filename = 'wiki_0'+str(i) if i < 10 else 'wiki_'+str(i)
    filepath = os.path.join(DIRECTORY,subdirectory,filename)
    with open(filepath, 'r') as f:
        print(f.readline())

stopwords_list = nltk.corpus.stopwords.words('english')

# We start with our data
df = pd.read_csv(os.path.join(BASE_PATH,'philosophy_data.csv'))
# all this mess because .csv saves lists as strings
df['tokenized_txt'] = df['tokenized_txt'].apply(lambda frase: [word.strip('\'') for word in frase[1:-1].split(', ')])

SCHOOLS = df['school'].unique()

schools_bow = {}
schools_distribution = {}
for school in SCHOOLS:
    tokenized_sentences = df.loc[df['school']==school,'tokenized_txt'] 
    schools_bow[school] = nltk.FreqDist([w for w in chain.from_iterable(tokenized_sentences) if w not in stopwords_list])
    mass = schools_bow[school].N()
    schools_distribution[school] = {key:val/mass for (key,val) in schools_bow[school].items()}