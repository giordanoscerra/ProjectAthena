from sklearn.model_selection import train_test_split
import pandas as pd

# Load your dataset
df = pd.read_csv('philosophy_data.csv')

# Calculate the distribution of currents, authors, and books
current_distribution = df['current'].value_counts(normalize=True)
author_distribution = df['author'].value_counts(normalize=True)
book_distribution = df['book'].value_counts(normalize=True)

# Split data while maintaining the distribution of currents, authors, and books
train, test = train_test_split(df, test_size=0.2, stratify=df[['current', 'author', 'book']])

# Check if distributions are maintained
train_current_distribution = train['current'].value_counts(normalize=True)
train_author_distribution = train['author'].value_counts(normalize=True)
train_book_distribution = train['book'].value_counts(normalize=True)

test_current_distribution = test['current'].value_counts(normalize=True)
test_author_distribution = test['author'].value_counts(normalize=True)
test_book_distribution = test['book'].value_counts(normalize=True)

# Combine the subsets of data
# final_train = combine_data(train)
# final_test = combine_data(test)

# You can then use final_train and final_test for your machine learning model

print(f"Current Distribution (Train): {train_current_distribution}")
print(f"Author Distribution (Train): {train_author_distribution}")
print(f"Book Distribution (Train): {train_book_distribution}")
print(f"Current Distribution (Test): {test_current_distribution}")
print(f"Author Distribution (Test): {test_author_distribution}")
print(f"Book Distribution (Test): {test_book_distribution}")