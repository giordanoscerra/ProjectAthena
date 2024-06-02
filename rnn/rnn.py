import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.base import BaseEstimator

class RNN(BaseEstimator):
    def __init__(self, output_units, encoder, embedding = None, hidden_units=100):
        self.hidden_units = hidden_units
        self.encoder = encoder
        if embedding is None:
            embedding = layers.Embedding(
                input_dim=len(self.encoder.get_vocabulary()),
                output_dim=self.hidden_units
            )
        self.model = keras.Sequential([
            self.encoder,
            embedding,
            layers.Bidirectional(layers.LSTM(40)),
            # layers.Dense(26, activation='relu'),
            layers.Dense(activation='softmax', units=output_units) # 13 for the number of schools
        ])
        self.model.compile(loss=keras.losses.CategoricalCrossentropy(),
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose='auto',
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1
    ):
        return self.model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            shuffle=shuffle,
            class_weight=class_weight,
            sample_weight=sample_weight,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_batch_size=validation_batch_size,
            validation_freq=validation_freq
            )


    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X):
        return self.model.evaluate(X)