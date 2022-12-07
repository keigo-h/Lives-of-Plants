import tensorflow as tf
import keras
from preprocess import Preprocess
import matplotlib.pyplot as plt

class Plant_Model(tf.keras.Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.l = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=16, kernel_size=3,strides=(2,2), padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3,strides=(2,2),activation='leaky_relu', padding="same"),
                tf.keras.layers.Conv2D(filters=128, kernel_size=3,strides=(2,2),activation='leaky_relu', padding="same"),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1000, activation='leaky_relu'),
                tf.keras.layers.Dense(100, activation='leaky_relu'),
                tf.keras.layers.Dense(10, activation='softmax')
        ])

    def call(self, inputs):
        return self.l(inputs)

    def run(self,data):
        train_ds, val_ds = data
        model = Plant_Model()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss =tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        
        m = model.fit(
            train_ds,
            epochs = 10,
            validation_data= val_ds
        )
        # img = "dataset/acer macrophyllum/medium (1).jpg"
        model.save('models')
        image, _ = next(iter(val_ds))
        print(model.predict_on_batch(image))
        plt.imshow(image[0])
        return m.history['val_accuracy'][-1]


class Plant_Model_v2(tf.keras.Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.l = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=16, kernel_size=3,strides=(2,2), padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3,strides=(2,2),activation='leaky_relu', padding="same"),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1000, activation='leaky_relu'),
                tf.keras.layers.Dense(100, activation='leaky_relu'),
                tf.keras.layers.Dense(10, activation='softmax')
        ])

    def call(self, inputs):
        return self.l(inputs)

    def run(self,data):
        train_ds, val_ds = data
        model = Plant_Model()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss =tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        
        m = model.fit(
            train_ds,
            epochs = 10,
            validation_data= val_ds
        )
        # img = "dataset/acer macrophyllum/medium (1).jpg"
        model.save('models')
        image, _ = next(iter(val_ds))
        print(model.predict_on_batch(image))
        plt.imshow(image)
        return m.history['val_accuracy'][-1]




