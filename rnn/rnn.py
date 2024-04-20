import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.base import BaseEstimator

class RNN(BaseEstimator):
    def __init__(self, output_units, encoder, hidden_units=100):
        self.hidden_units = hidden_units
        self.encoder = encoder
        self.model = keras.Sequential([
            self.encoder,
            layers.Embedding(
            input_dim=encoder._output_sequence_length,
            output_dim=self.hidden_units,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        layers.Bidirectional(layers.LSTM(self.hidden_units)),
        layers.Dense(self.hidden_units, activation='relu'),
        layers.Dense(activation='softmax', units=output_units) # 13 for the number of schools
        ])
        self.model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


    def fit(self, train_dataset, epochs=10, batch_size=100, validation_data = None):
        return self.model.fit(train_dataset, epochs=epochs, validation_data = validation_data, batch_size=batch_size)


    def predict(self, X):
        return self.model.predict(X)