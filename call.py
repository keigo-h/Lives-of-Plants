
import tensorflow as tf
import keras
from preprocess import Preprocess
from model import Plant_Model, run

def main():
    preprocess = Preprocess()
    train_ds, val_ds = preprocess.get_and_process()
    # train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    accuracy = run((train_ds,val_ds))
    print("accuracy is ", accuracy)

if __name__ == "__main__":
    main()