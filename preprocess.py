import numpy as np
import tensorflow as tf
import keras
import keras_preprocessing
import tensorflow_datasets as tfds
from get_data import get
import matplotlib.pyplot as plt
"""
Class which handles preprocessing data
"""
class Preprocess:
    def __init__(self):
        pass

    def get_and_process(self, label_mode='categorical', img_height=256, img_width=256, batch_size=3):
        train_ds, val_ds = get(label_mode=label_mode, img_height=img_height, img_width=img_width, batch_size=batch_size)
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        augment_fn = tf.keras.Sequential([tf.keras.layers.RandomFlip(mode="horizontal"),tf.keras.layers.RandomRotation(factor=0.1)])
        normalized_train_ds = train_ds.map(lambda x,y: (normalization_layer(x), y))
        normalized_train_ds_augment=normalized_train_ds.map(lambda x,y: (augment_fn(x), y))
        normalized_val_ds = val_ds.map(lambda x,y: (normalization_layer(x), y))
        return normalized_train_ds_augment, normalized_val_ds