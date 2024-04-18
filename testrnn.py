from rnn.rnn import RNN
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import TextVectorization
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
    print("read csv")

    # one-hot encode the schools
    # df['school'] = pd.Categorical(df['school'])

    dataset_features = df.copy()
    dset_schools = dataset_features.pop('school')
    dset_sentences = dataset_features.pop('sentence_str')

    print("got schools and senteces")

    features_dict = {name: np.array(value) for name, value in dset_sentences.items()}

    features_dict = tf.placeholder(tf.string, [None])

    print("created features dict")

    bs = 1024
    seed = 23
    dataset = tf.data.Dataset.from_tensor_slices(features_dict)

    print("sliced")

    step_counter = tf.Variable(0, trainable=False)
    checkpoint_args = {
      "checkpoint_interval": 50,
      "step_counter": step_counter,
      "directory": checkpoint_prefix,
      "max_to_keep": 20,
    }
    dataset.save(dataset, save_dir, checkpoint_args=checkpoint_args)

  print("created dataset")
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

# Split the dataset
# train_size = int(0.8 * len(dataset))
# val_size = int(0.1 * len(dataset))
# train = dataset.take(train_size)
# remaining_dataset = dataset.skip(train_size)
# val = remaining_dataset.take(val_size)
# test = remaining_dataset.skip(val_size)

# Set vocabulary size
unique_words = set()
for sentence in df['sentence_str']:
    unique_words.update(sentence.split())

print("got unique words")

encoder = TextVectorization(max_tokens=len(unique_words), output_mode='int', pad_to_max_tokens=True)
print("created encoder")
encoder.adapt(list(unique_words))
print("adapted encoder")

rnn = RNN(output_units = 13, encoder = encoder, hidden_units = 20)
print("created rnn")

history = rnn.fit(x_train, y_train, epochs = 5, validation_split = 0.3)

# test_loss, test_acc = rnn.evaluate(test)
# print('Test Loss:', test_loss)
# print('Test Accuracy:', test_acc)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)

sample_text = ('We should eat all landlords')
print("predicting \"we should eat all landlords\"")
predictions = rnn.model.predict(sample_text)