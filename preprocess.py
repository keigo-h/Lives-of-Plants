import numpy as np
import tensorflow as tf
import keras
import keras_preprocessing
import tensorflow_datasets as tfds
from get_data import get

class Preprocess:
    def __init__(self):
        pass

    def get_and_process(self):
        # TODO: get the data and preprocess it here
        # TODO: return X0, L0 training data and labels
        # TODO: return X1, L1 testing data and labels
        train_ds, val_ds = get()
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        normalized_train_ds = train_ds.map(lambda x,y: (normalization_layer(x), y))
        normalized_val_ds = val_ds.map(lambda x,y: (normalization_layer(x), y))
        return normalized_train_ds, normalized_val_ds