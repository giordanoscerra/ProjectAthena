import numpy as np
import tensorflow as tf
from keras import layers
import keras
import matplotlib.pyplot as plt
import tensorflow_hub as hub

from scoring import *

SCHOOLS = ['analytic','aristotle','german_idealism',
           'plato','continental','phenomenology',
           'rationalism','empiricism','feminism',
           'capitalism','communism','nietzsche',
           'stoicism']


print("creating model")
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.KerasLayer(module_url)
print(embed.get_config())
embed.trainable = True
print(embed.get_config())
regularizer = keras.regularizers.l2(0.0001)
class MyModel(keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.embedding_layer = embed
    self.dense_layer1 = layers.Dense(256, activation='relu', kernel_regularizer=regularizer)
    self.dense_layer2 = layers.Dense(128, activation='relu', kernel_regularizer=regularizer)
    self.dense_layer3 = layers.Dense(64, activation='relu', kernel_regularizer=regularizer)
    self.output_layer = layers.Dense(13, activation='softmax', kernel_regularizer=regularizer)

  def call(self, inputs):
    x = self.embedding_layer(inputs)
    x = self.dense_layer1(x)
    x = self.dense_layer2(x)
    x = self.dense_layer3(x)
    return self.output_layer(x)

nn = MyModel()

print("compiling model")
nn.compile(loss=keras.losses.CategoricalCrossentropy(),
  optimizer=keras.optimizers.Adam(0.001),
  metrics=['accuracy'],
  )

# Load the dataset
x_train, x_test, y_train, y_test = splitData(filterShortPhrases(getData(), 30))
#print("encoding training data")
#x_train = preprocess_phrases(x_train)
#print("encoding testing data")
#x_test = preprocess_phrases(x_test)
#print("encoding training labels")
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
#x_train = x_train.reshape(x_train.shape[0], 512)
#x_test = x_test.reshape(x_test.shape[0], 512)
print("fitting model")
#with weight decay
history = nn.fit(x_train, y_train, 
         validation_data=(x_test, y_test), 
         epochs=74, 
         batch_size=128,
         verbose=1,
         callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        )

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