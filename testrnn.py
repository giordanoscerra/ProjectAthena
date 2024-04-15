from rnn.rnn import RNN
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

df = pd.read_csv('philosophy_data.csv')

# one-hot encode the schools
df['school'] = pd.Categorical(df['school'])

dataset = tf.data.Dataset.from_tensor_slices((df['sentence_str'], df['school']))

# Split the dataset
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
train = dataset.take(train_size)
remaining_dataset = dataset.skip(train_size)
val = remaining_dataset.take(val_size)
test = remaining_dataset.skip(val_size)

# Set vocabulary size
unique_words = set()
for sentence in df['sentence_str']:
    unique_words.update(sentence.split())

VOCAB_SIZE = 1000
SEQ_LENGTH = 100

encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode = 'int',
    output_sequence_length=SEQ_LENGTH)
encoder.adapt(test.map(lambda sentence_str, label: sentence_str))

rnn = RNN(output_units = 13, encoder = encoder, hidden_units = 100)

history = rnn.fit(train, epochs=10, validation_data=val, batch_size=100)

test_loss, test_acc = rnn.evaluate(test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)

sample_text = ('We should eat all landlords')
print("predicting \"we should eat all landlords\"")
predictions = rnn.model.predict(np.array([sample_text]))