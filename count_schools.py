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