
import tensorflow as tf
import keras
from preprocess import Preprocess
from model import Plant_Model,Plant_Model_Transfer

def main():
    preprocess = Preprocess()
    model_1 = Plant_Model()
    model_2 = Plant_Model_Transfer()
    train_ds, val_ds = preprocess.get_and_process()
    accuracy = model_1.run((train_ds,val_ds))
    print("accuracy is ", accuracy)

if __name__ == "__main__":
    main()