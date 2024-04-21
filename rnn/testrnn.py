from rnn.rnn import RNN
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import TextVectorization, Embedding
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

save_dir = os.path.join(os.getcwd(), "dset")
checkpoint_prefix = "checkpoint"
use_tensorflow_dataset = False

if use_tensorflow_dataset:
  if (os.path.exists(save_dir)):
    dataset = tf.data.Dataset.load("dset.dataset")
  else:
    df = pd.read_csv('philosophy_data.csv')

    dataset_features = df.copy()
    dset_schools = dataset_features.pop('school')
    dset_sentences = dataset_features.pop('sentence_str')

    features_dict = {name: np.array(value) for name, value in dset_sentences.items()}

    features_dict = tf.placeholder(tf.string, [None])
    bs = 1024
    seed = 23
    dataset = tf.data.Dataset.from_tensor_slices(features_dict)

    step_counter = tf.Variable(0, trainable=False)
    checkpoint_args = {
      "checkpoint_interval": 50,
      "step_counter": step_counter,
      "directory": checkpoint_prefix,
      "max_to_keep": 20,
    }
    dataset.save(dataset, save_dir, checkpoint_args=checkpoint_args)

  dset_batches = dataset.shuffle(len(dset_schools)).batch(bs)
else:
  df = pd.read_csv('philosophy_data.csv')
  # shuffle the rows of the dataset
  df = df.sample(frac=1)
  
  x_train, x_test, y_train, y_test = train_test_split(df['sentence_str'], df['school'], test_size=0.25, random_state=42)
  lb_encoder = LabelEncoder()
  y_train = lb_encoder.fit_transform(y_train)
  y_train = to_categorical(y_train)
  y_test = lb_encoder.transform(y_test)
  y_test = to_categorical(y_test)

unique_words = set()
for sentence in df['sentence_str']:
    unique_words.update(sentence.split())

encoder = TextVectorization(max_tokens=len(unique_words), output_mode='int', pad_to_max_tokens=True)
encoder.adapt(list(unique_words))

path_to_glove_file = "glove.42B.300d.txt"

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

voc = encoder.get_vocabulary()

num_tokens = len(voc) + 2
embedding_dim = 300
hits = 0
misses = 0
word_index = dict(zip(voc, range(len(voc))))

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    trainable=False,
)
# embedding_layer.build((1,))
# embedding_layer.set_weights([embedding_matrix])

rnn = RNN(output_units = 13, embedding = embedding_layer, encoder = encoder, hidden_units = 25)

history = rnn.fit(x_train, y_train, epochs = 8, batch_size = 128,  validation_split = 0.25)

# test_loss, test_acc = rnn.evaluate(x_test, y_test)
# print('Test Loss:', test_loss)
# print('Test Accuracy:', test_acc)

# plt.figure(figsize=(16, 8))
# plt.subplot(1, 2, 1)
# plot_graphs(history, 'accuracy')
# plt.ylim(None, 1)
# plt.subplot(1, 2, 2)
# plot_graphs(history, 'loss')
# plt.ylim(0, None)

# evaluate micro and macro averages
y_pred = rnn.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=lb_encoder.classes_))