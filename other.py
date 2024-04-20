import os
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
module_url = "./archive/"
embed = hub.KerasLayer(module_url)
print(embed.get_config())
embed.trainable = True
print(embed.get_config())

class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        regularizer = keras.regularizers.l2(0.0000)
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
x_train, x_test, y_train, y_test = splitData(filterShortPhrases(getData(), numWords=0), test_size=0.30)
#keep 20% of test
x_test = x_test[:int(len(x_test)*0.66)]
y_test = y_test[:int(len(y_test)*0.66)]
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
print("fitting model")

history = nn.fit(x_train, y_train, 
         validation_data=(x_test, y_test), 
         epochs=74, 
         batch_size=256,
         verbose=1,
         callbacks=[keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)],
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


print("parsing predictions")
predictions = nn.predict(x_test)
predictions = np.argmax(predictions, axis=1)
predictions = [SCHOOLS[pred] for pred in predictions]
y_test = np.argmax(y_test, axis=1)
y_test = [SCHOOLS[pred] for pred in y_test]
print("scoring")
accuracy_per_lenght = {}
tot_len = {}
for i in range(2):
    accuracy_per_lenght[i] = 0
    tot_len[i] = 0
for i in range(len(predictions)):
    if(len(x_test[i].split()) < 20):
        tot_len[0] += 1
        if(predictions[i] == y_test[i]):
            accuracy_per_lenght[0] += 1
    elif(len(x_test[i].split()) < 40):
        tot_len[1] += 1
        if(predictions[i] == y_test[i]):
            accuracy_per_lenght[1] += 1

#now we have the accuracy per length
for i in range(2):
    if tot_len[i] > 0:
        accuracy_per_lenght[i] /= tot_len[i]
        print(f"Accuracy for length {i}: {accuracy_per_lenght[i]}")
    else:
        accuracy_per_lenght.pop(i)

#plot the accuracy per length
plt.plot(accuracy_per_lenght.keys(), accuracy_per_lenght.values())
plt.xlabel('Length of sentence')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.show()