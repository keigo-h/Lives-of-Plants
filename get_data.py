import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
#import tensorflow_datasets as tfds

def get():

    batch_size = 3
    img_height = 180
    img_width = 180

    data_dir = "dataset"

    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels = 'inferred',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    
    

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels = 'inferred',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)
    return (train_ds, val_ds)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")