from typing import List, Dict, Optional

from sklearn.feature_extraction.text import CountVectorizer
import wordcloud 

def create_bow(school:str, df,
               stopwords:Optional[str|List] = None) -> Dict[str,int]:
    """Returns a bag of words of the given school.
    WIP

    Returns:
        Dict[str,int]: bag of words
    """
    texts = []
    # Put together all the sentences of a given school
    for stringa in df.loc[df['school'] == school,'sentence_str']:
        texts.append(stringa)

    # We now create the bag of word vocabulary

    vectorizer = CountVectorizer(
        analyzer='word',
        stop_words=stopwords    # leap of faith here
        )
    # learn vocabulary, return term-document matrix
    vector = vectorizer.fit_transform(texts)
    # create the dictionary
    wc_dict = dict(zip(vectorizer.get_feature_names_out(), vector.toarray().sum(axis=0)))

    return wc_dict





# function that creates the wordcloud
def make_wordcloud(text:str, 
                   stopwords:Optional[List[str]] = None, 
                   bgcolor:str = 'white'):
    cloud = wordcloud.WordCloud(
                width = 800, height = 800,
                background_color = bgcolor,
                stopwords = stopwords,
                min_font_size = 8).generate(text)
    
    return cloud




import matplotlib.pyplot as plt

def create_bigstring(school: str) -> str:
    text = ''
    separator = ' '
    for stringa in df.loc[df['school'] == school,'sentence_str']:
        text += separator + stringa
    
    return text

schools = df['school'].unique()

# Create a figure for the subplots
fig, axs = plt.subplots(7, 2, figsize=(20, 20)) 
axs = axs.ravel()       # flatten axs indices for easier iteration

# Iterate over the schools
for i, school in enumerate(schools):
    # Create a big string for the sentences of each school
    text = create_bigstring(school)

    # Create a WordCloud object
    wc = make_wordcloud(text)

    # Display the word cloud in a subplot
    axs[i].imshow(wc, interpolation="bilinear")
    axs[i].set_title(f'{school.capitalize()} Word Cloud')  # Set title to the school name
    axs[i].axis('off')

# Add a title for the whole plot
plt.suptitle('Word Clouds for philosophical schools', fontsize=20, y=1.02)

# Hide unused subplots
for i in range(len(schools), 14):
    axs[i].axis('off')

plt.tight_layout(pad=0)
plt.show()