import pandas as pd
import numpy as np
# Leggi il file csv
df = pd.read_csv('philosophy_data.csv')

# Calcola la lunghezza di ogni frase
lenghth_of_phrases = df['sentence_str'].apply(lambda x: len(x.split())).to_list()

# Stampa il dataframe
print(np.max(lenghth_of_phrases))
#print mean and other stats
print(np.mean(lenghth_of_phrases))
print(np.median(lenghth_of_phrases))
print(np.std(lenghth_of_phrases))
print(np.var(lenghth_of_phrases))
print(np.percentile(lenghth_of_phrases, 75))
print(np.percentile(lenghth_of_phrases, 25))
print(np.percentile(lenghth_of_phrases, 90))
print(np.percentile(lenghth_of_phrases, 10))
#plot histogram
import matplotlib.pyplot as plt

plt.hist(lenghth_of_phrases, bins=100)
#name the axis
#name the title
plt.xlabel('Length of phrases')
plt.ylabel('Frequency')
plt.show()