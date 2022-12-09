
import tensorflow as tf
import keras
from preprocess import Preprocess
from model import Plant_Model,Plant_Model_Transfer, Plant_Model_Two, Test_Model
"""
file which calls the models
"""
def main():
    preprocess = Preprocess()
    model_1 = Plant_Model()
    model_2 = Plant_Model_Transfer()
    model_3 = Plant_Model_Two()
    test_model = Test_Model()
    model_3_bool = False;
    if(model_3_bool):
        train_ds, val_ds = preprocess.get_and_process(label_mode='int',img_height=224, img_width=224, batch_size=50)
    else:
        train_ds, val_ds = preprocess.get_and_process()
    accuracy = test_model.run((train_ds,val_ds))
    print("accuracy is ", accuracy)

if __name__ == "__main__":
    main()