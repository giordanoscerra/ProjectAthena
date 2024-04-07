import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.base import BaseEstimator

class RNN(BaseEstimator):
    def __init__(self, output_units, encoder, hidden_units=100, learning_rate=0.001, batch_size=100, epochs=100):
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.encoder = encoder
        self.model = keras.Sequential([
            self.encoder,
            layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
        ])
        self.model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


    def fit(self, train_dataset, validation_set):
        history = self.model.fit(train_dataset, epochs=10,
                    validation_data=validation_set,
                    validation_steps=30)
        return history


    def predict(self, X):
        return self.model.predict(X)