import tensorflow as tf
import keras
from preprocess import Preprocess

class Plant_Model(tf.keras.Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.l = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='leaky_relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='leaky_relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='leaky_relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='leaky_relu'),
            tf.keras.layers.Dense(10)
        ])

    def call(self, inputs):
        return self.l(inputs)

def run(data):
    train_ds, val_ds = data
    model = Plant_Model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss =tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    model.fit(
        train_ds,
        epochs = 3,
        validation_data= val_ds
    )
    




