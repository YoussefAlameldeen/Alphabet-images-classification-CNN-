import cv2
from cv2 import *
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import skimage
from skimage.transform import resize
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import glob
import cv2 as cv
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tensorflow import keras
from keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

train_dir = 'E:\\ASL_Data\\train'
test_dir =  'E:\\ASL_Data\\test'

# Augementation
gen = ImageDataGenerator(rotation_range=40, width_shift_range=0.02, shear_range=0.02,
                         height_shift_range=0.02, horizontal_flip=True, fill_mode='nearest')

train_generator = gen.flow_from_directory(train_dir, 
                target_size = (200,200), color_mode = "rgb", batch_size = 16, class_mode='categorical')

test_generator = gen.flow_from_directory(test_dir,
                target_size = (200,200), color_mode = "rgb", batch_size = 16, class_mode='categorical')



def ResNet50_model(train,test):
    base_model=keras.applications.ResNet50(weights='imagenet',include_top=False, input_shape=(200, 200, 3))
    base_model.trainable = False
    
    model=Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(29, activation='softmax'))
    
    #compile
    model.compile(optimizer='sgd',
          loss='categorical_crossentropy',
          metrics=['accuracy'])
    
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)
    cp = ModelCheckpoint(filepath='cifar10.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    #fitting
    model.fit_generator(train,
                    steps_per_epoch=100,
                    epochs = 6,
                    validation_data = test,
                    validation_steps = 10,
                    callbacks=[cp, es])

ResNet50_model(train_generator,test_generator)

def VGG16_model(train,test):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(200,200,3))
    base_model.trainable = False 
    
    model=Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(29, activation='softmax'))
    
    #compile
    model.compile(optimizer='sgd',
          loss='categorical_crossentropy',
          metrics=['accuracy'])
    
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)
    cp = ModelCheckpoint('vgg16_1.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    #fitting
    model.fit_generator(train,
                    steps_per_epoch=100,
                    epochs = 6,
                    validation_data = test,
                    validation_steps = 10,
                    callbacks=[cp, es])


VGG16_model(train_generator,test_generator)

    




















