import random
import pandas as pd

# Leggi il file csv
df = pd.read_csv('philosophy_data.csv')

#take 
# Ottieni la colonna delle scuole
schools = df['school'].to_list()
schools = pd.Series(schools).value_counts().items()

# Stampa le scuole
for school, count in schools:
    print(school, count)

#long phrases
# Filtra le frasi lunghe
df = df[df['sentence_str'].apply(lambda x: len(x.split())>20)]
# Stampa il numero di frasi lunghe
print(len(df))
#print couple of them
# Stampa un paio di frasi lunghe
for i in range(4):
    print('_'*50)
    random_index = random.randint(0, len(df))
    print(df['sentence_str'].iloc[random_index])
    print(df['school'].iloc[random_index])
    print('_'*50)

# Stampa quante parole sono usate in modo univoco da una scuola
words:dict = {}
for school in df['school'].unique():
    words[school] = set(' '.join(df[df['school']==school]['sentence_str'].to_list()).split())
    print(school, len(words[school]))

# Stampa le parole usate in modo univoco da una scuola
print('\n\n\n')
# initialize with all words
uniquely_used_words = set()
for school in words.keys():
    uniquely_used_words.update(words[school])
for school in words.keys():
    for school2 in words.keys():
        if school != school2:
            print(school, school2, len(words[school].intersection(words[school2])))
            for word in words[school].intersection(words[school2]):
                if word in uniquely_used_words:
                    uniquely_used_words.remove(word)
            
print('\n\n\n')
for school in words.keys():
    # count phrases with unique words
    school_phrases = df[df['school']==school]
    print(school, len(school_phrases['sentence_str'].apply(lambda x: len(set(x.split()).intersection(uniquely_used_words))>0)))
