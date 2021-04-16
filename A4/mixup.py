#import necessary libraries
import os
import keras
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from A4.mixup_generator import MixupImageDataGenerator


# setup current work path
baseDir = os.path.abspath('.')
modelPath = os.path.join(baseDir,'A4','efficientNetMixUp_best.h5')
img = os.path.join(baseDir ,'Datasets','train') 
# load pre-trained model
final_model = keras.models.load_model(modelPath)
# configure image generator
img_gen = ImageDataGenerator(rescale = 1.0/255.0, horizontal_flip = True, vertical_flip   = True,
                               fill_mode = 'nearest', rotation_range = 10, width_shift_range = 0.2,
                               height_shift_range= 0.2, shear_range= 0.2, brightness_range= (0.5,1.2),
                               zoom_range = 0.2)

def preProcessing():
    # read image and label csv file of train, valid, test set
    train = os.path.join(baseDir ,'Datasets','train_img.csv')
    test = os.path.join(baseDir ,'Datasets','test_img.csv')
    train_img=pd.read_csv(train)
    test_img=pd.read_csv(test)
    
    train_img.drop(columns=['Unnamed: 0'],inplace=True)
    test_img.drop(columns=['Unnamed: 0'],inplace=True)
    
    # map string type labels to int type
    real_labels = {'cbb':0,'cbsd':1,'cgm':2,'cmd':3,'healthy':4}
    train_img['class_num'] = train_img['label'].map(real_labels)
    test_img['class_num'] = test_img['label'].map(real_labels)
    return train_img,test_img


def train(train_img):
    # feed images into neural network
    train = MixupImageDataGenerator(train_img, generator=img_gen,directory=img,x_col = 'filename', y_col = 'label',
                                batch_size=12,target_size =(220, 220))
    # evaluate_generator will return loss and acc of the prediction
    loss,acc = final_model.evaluate_generator(train,steps=train.n//12)
    return acc
    
def test(test_img):
    test = img_gen.flow_from_dataframe(test_img, directory = img,x_col = 'filename', y_col = 'label', 
                                  target_size =(220, 220), class_mode = 'categorical', shuffle = False,
                                  batch_size = 12, color_mode = 'rgb')
    loss,acc = final_model.evaluate_generator(test,steps=test.n//12)
    return acc
