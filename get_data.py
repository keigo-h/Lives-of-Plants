import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
#import tensorflow_datasets as tfds

def get():

    batch_size = 3
    img_height = 256
    img_width = 256

    data_dir = "dataset"

    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels = 'inferred',
    label_mode='categorical',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    
    

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels = 'inferred',
    label_mode='categorical',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)
    return (train_ds, val_ds)

    # import matplotlib.pyplot as plt

    # image_batch, label_batch = next(iter(train_ds))

    # plt.figure(figsize=(10, 10))
    # for i in range(9):
    #     ax = plt.subplot(3, 3, i + 1)
    #     plt.imshow(image_batch[i].numpy().astype("uint8"))
    #     label = label_batch[i]
    #     plt.title(class_names[label])
    #     plt.axis("off")

if __name__ == "__main__":
    get()