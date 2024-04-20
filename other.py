from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
import keras
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import re

SCHOOLS = ['analytic','aristotle','german_idealism',
           'plato','continental','phenomenology',
           'rationalism','empiricism','feminism',
           'capitalism','communism','nietzsche',
           'stoicism']
    
def getData()->pd.DataFrame:
    return pd.read_csv('philosophy_data.csv')

def splitData(data:pd.DataFrame)->tuple:
    return train_test_split(data['sentence_str'], data['school'], test_size=0.25, random_state=42)

print("creating model")

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.KerasLayer(module_url)
nn = keras.Sequential([
    keras.Input(shape=(512,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(13, activation='softmax')
])
#define the input as a string passed through the encoder
def preprocess_input(input_string):
  encoded_input = embed([input_string])
  return encoded_input

def preprocess_phrases(phrases):
  encoded_phrases = []
  for i, phrase in enumerate(phrases):
    print(f'processed percentage {i/len(phrases)*100:.2f} %           ', end='\r ')
    encoded_phrase = preprocess_input(phrase)
    encoded_phrases.append(encoded_phrase)
  print('processed percentage 100%        ')
  return encoded_phrases

nn.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(1e-4),
        metrics=['accuracy'])

# Load the dataset
x_train, x_test, y_train, y_test = splitData(getData())
print("encoding training data")
x_train = preprocess_phrases(x_train)
print("encoding testing data")
x_test = preprocess_phrases(x_test)
print("encoding training labels")
#assign each school a number
y_train = y_train.apply(lambda x: SCHOOLS.index(x))
y_test = y_test.apply(lambda x: SCHOOLS.index(x))
y_train = keras.utils.to_categorical(y_train, num_classes=13)
print("encoding testing labels")
y_test = keras.utils.to_categorical(y_test, num_classes=13)

#convert everything to np array
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
x_train = x_train.reshape(x_train.shape[0], 512)
x_test = x_test.reshape(x_test.shape[0], 512)
print("fitting model")
history = nn.fit(x_train, y_train, epochs=74, validation_data=(x_test, y_test), batch_size=64)

print("plotting")
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 3])
plt.legend(loc='lower right')
plt.show()