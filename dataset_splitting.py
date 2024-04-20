from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Load your datasets
def getData(validation_size:float=0.2, test_size:float=0.1)-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv('philosophy_data.csv')
    development, test = train_test_split(df, test_size=test_size, stratify=df['school'], random_state=42, shuffle=True)
    #split vslidation ant train from train
    if validation_size == 0:
        return development, None, test
    train, validation = train_test_split(development, test_size=validation_size/(1 - test_size), stratify=development['school'], random_state=42, shuffle=True)
    return train, validation, test

def assess_split(train, test):
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

    '''
    print(f"school Distribution (Train): {train_school_distribution}")
    print(f"Author Distribution (Train): {train_author_distribution}")
    print(f"title Distribution (Train): {train_title_distribution}")
    print(f"school Distribution (Test): {test_school_distribution}")
    print(f"Author Distribution (Test): {test_author_distribution}")
    print(f"title Distribution (Test): {test_title_distribution}")
    '''
    
    # Check if sums of distributions are equal to 1 in train and test
    train_school_sum = train_school_distribution.sum()
    train_author_sum = train_author_distribution.sum()
    train_title_sum = train_title_distribution.sum()
    test_school_sum = test_school_distribution.sum()
    test_author_sum = test_author_distribution.sum()
    test_title_sum = test_title_distribution.sum()

    '''
    print(f"Sum of school Distribution (Train): {train_school_sum}")
    print(f"Sum of Author Distribution (Train): {train_author_sum}")
    print(f"Sum of title Distribution (Train): {train_title_sum}")
    print(f"Sum of school Distribution (Test): {test_school_sum}")
    print(f"Sum of Author Distribution (Test): {test_author_sum}")
    print(f"Sum of title Distribution (Test): {test_title_sum}")
    '''
    
    # Check the distances between the distributions of train and test
    school_distribution_distance = train_school_distribution.subtract(test_school_distribution).abs().sum()
    author_distribution_distance = train_author_distribution.subtract(test_author_distribution).abs().sum()
    title_distribution_distance = train_title_distribution.subtract(test_title_distribution).abs().sum()

    print(f"School Distribution Distance: {school_distribution_distance}")
    print(f"Author Distribution Distance: {author_distribution_distance}")
    print(f"Title Distribution Distance: {title_distribution_distance}")

    print('###################')

if __name__ == '__main__':
    


    # Split data while maintaining the distribution of schools, authors, and titles
    train, _, test = getData(validation_size=0)

    assess_split(train, test)

    df = pd.read_csv('philosophy_data.csv')

    # Split data using StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for train_index, test_index in skf.split(df['sentence_str'], df['school']):
        train = df.iloc[train_index]
        test = df.iloc[test_index]
        assess_split(train, test)
    #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)