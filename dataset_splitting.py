from sklearn.model_selection import train_test_split
import pandas as pd

# Load your datasets
df = pd.read_csv('philosophy_data.csv')

# Calculate the distribution of schools, authors, and titles
school_distribution = df['school'].value_counts(normalize=True)
author_distribution = df['author'].value_counts(normalize=True)
title_distribution = df['title'].value_counts(normalize=True)

# Split data while maintaining the distribution of schools, authors, and titles
train, test = train_test_split(df, test_size=0.2, stratify=df[['school', 'author', 'title']])

# Check if distributions are maintained
train_school_distribution = train['school'].value_counts(normalize=True)
train_author_distribution = train['author'].value_counts(normalize=True)
train_title_distribution = train['title'].value_counts(normalize=True)

test_school_distribution = test['school'].value_counts(normalize=True)
test_author_distribution = test['author'].value_counts(normalize=True)
test_title_distribution = test['title'].value_counts(normalize=True)

# Combine the subsets of data
# final_train = combine_data(train)
# final_test = combine_data(test)

# You can then use final_train and final_test for your machine learning model

print(f"school Distribution (Train): {train_school_distribution}")
print(f"Author Distribution (Train): {train_author_distribution}")
print(f"title Distribution (Train): {train_title_distribution}")
print(f"school Distribution (Test): {test_school_distribution}")
print(f"Author Distribution (Test): {test_author_distribution}")
print(f"title Distribution (Test): {test_title_distribution}")