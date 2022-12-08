import tensorflow as tf
from tensorflow.keras import regularizers
import keras
from preprocess import Preprocess
import matplotlib.pyplot as plt

class Plant_Model(tf.keras.Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.l = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=16, kernel_size=3,strides=(2,2), padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3,strides=(2,2),activation='leaky_relu', padding="same", kernel_regularizer=regularizers.l2(l=0.01)),
                tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Conv2D(filters=128, kernel_size=3,strides=(2,2),activation='leaky_relu', padding="same",
                kernel_regularizer=regularizers.l2(l=0.01)),
                tf.keras.layers.Conv2D(filters=256, kernel_size=3,strides=(2,2),activation='leaky_relu', padding="same",
                kernel_regularizer=regularizers.l2(l=0.01)),
                tf.keras.layers.Conv2D(filters=512, kernel_size=3,strides=(2,2),activation='leaky_relu', padding="same",
                kernel_regularizer=regularizers.l2(l=0.01)),
                tf.keras.layers.Conv2D(filters=1028, kernel_size=3,strides=(2,2),activation='leaky_relu', padding="same",
                kernel_regularizer=regularizers.l2(l=0.01)),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1000, activation='leaky_relu'),
                tf.keras.layers.Dense(100, activation='leaky_relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation='softmax')
        ])

    def call(self, inputs):
        return self.l(inputs)

    def run(self,data):
        train_ds, val_ds = data
        model = Plant_Model()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss =tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        
        history = model.fit(
            train_ds,
            epochs = 75,
            validation_data= val_ds
        )
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model 1 Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.savefig('accuracy.png')
        plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model 1 Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.savefig('loss.png')
        plt.show()
        return history.history['val_accuracy'][-1]


class Plant_Model_Transfer(tf.keras.Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def run(self,data):
        train_ds, val_ds = data
        model = tf.keras.Sequential()

        prev_model= tf.keras.applications.ResNet50(include_top=False,
                        input_shape=(256,256,3),
                        pooling='avg',classes=10,
                        weights='imagenet')
        for layer in prev_model.layers:
                layer.trainable=False

        model.add(prev_model)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss ='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(
            train_ds,
            epochs = 10,
            validation_data= val_ds
        )
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Transfer Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.savefig('transfer_acc.png')
        plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Transfer Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.savefig('transfer_loss.png')
        plt.show()
        return history.history['val_accuracy'][-1]




