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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets, layers, models
import tensorflow as tf



train_dir = "E:\\ASL_Alphabet_Dataset\\asl_alphabet_train"
test_dir =  "E:\\ASL_Alphabet_Dataset\\asl_alphabet_test"

thisdict = {"A_test.jpg": 0,"B_test.jpg": 1,"C_test.jpg": 2,"D_test.jpg": 3,"E_test.jpg":4,"F_test.jpg":5,"G_test.jpg":6,"H_test.jpg":7,"I_test.jpg":8,"J_test.jpg":9,"K_test.jpg":10,"L_test.jpg":11,"M_test.jpg":12,"N_test.jpg":13,"nothing_test.jpg":14,"O_test.jpg":15,"P_test.jpg":16,"Q_test.jpg":17,"R_test.jpg":18,"S_test.jpg":19,"space_test.jpg":20,"T_test.jpg":21,"U_test.jpg":22,"V_test.jpg":23,"W_test.jpg":24,"X_test.jpg":25,"Y_test.jpg":26,"Z_test.jpg":27,"del_test.jpg":28,"A": 0,"B": 1,"C": 2,"D": 3,"E":4,"F":5,"G":6,"H":7,"I":8,"J":9,"K":10,"L":11,"M":12,"N":13,"nothing":14,"O":15,"P":16,"Q":17,"R":18,"S":19,"space":20,"T":21,"U":22,"V":23,"W":24,"X":25,"Y":26,"Z":27,"del":28}


def load_train(folder):
    images = []
    Y=[]
    ListOfimages=[]
    for foldername in os.listdir(folder):
        ListOfimages=os.listdir("E:\\ASL_Alphabet_Dataset\\asl_alphabet_train\\"+foldername)
        for image_filename in range(len(ListOfimages)):
                if image_filename>5500 and image_filename<6500:
                    img_file = cv2.imread(os.path.join(folder,foldername,ListOfimages[image_filename]))
                    if img_file is not None:
                        img_file=cv2.resize(img_file,(200,200))
                        img_file=cv2.resize(img_file, (0,0), fx=0.25, fy=0.25)
                        images.append(img_file)
                        Y.append(thisdict[foldername])
    return images,Y


X_, Y_ = load_train(train_dir) 

cv2.imshow('Original img',X_[5])
cv2.waitKey(0)
cv2.destroyAllWindows()

len(X_)


def load_test(folder):
    images = []
    Y=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img=cv2.resize(img,(200,200))
            img =cv2.resize(img, (0,0), fx=0.25, fy=0.25)
            images.append(img)
            Y.append(thisdict[filename])
    return images,Y

Xtest,Ytest= load_test(test_dir)

####### convert to RGB and Gray
def converting(X):
     X_RGB=[]
     X_Grey=[]
     for image in X:
        X_RGB.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        X_Grey.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
     return X_RGB,X_Grey

RGB_train,Gray_train=converting(X_)
RGB_test,Gray_test=converting(Xtest)



###### Normalize the images.
def Normalize(rgb, gray):
    RGB=[]
    Gray=[]
    for img in range(len(rgb)):
        RGB.append((rgb[img] / 255) - 0.5)
        Gray.append((gray[img] / 255) - 0.5)
    RGB=np.array(RGB)
    Gray=np.array(Gray)
    return RGB,Gray


RGB_train_images,Gray_train_images=Normalize(RGB_train,Gray_train)
RGB_test_images,Gray_test_images=Normalize(RGB_test,Gray_test)

Gray_train_images = np.expand_dims(Gray_train_images, axis=3)
Gray_test_images = np.expand_dims(Gray_test_images, axis=3)
Y_=np.array(Y_)
Ytest=np.array(Ytest)

#print(RGB_train_images.shape)
#print(Gray_test_images.shape)
 


def processing1(X_train, Y_train, X_test, Y_test, Type):
    model = models.Sequential()
    if Type=='RGB':
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(150, activation='relu'))
        model.add(layers.Dense(29, activation='softmax'))
        #compile
        model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        #fitting
        model.fit(X_train, Y_train, epochs=6, 
                    validation_data=(X_test, Y_test))
        #Precision & Recall
           # metrics can't handle a multiclass or multioutput so,
           # we use argmax to convert y_pred from 2d to 1d array
        y_pred = model.predict(X_test).argmax(axis=1)
        print("Accuracy:  %.2f%%" % (metrics.accuracy_score(Y_test, y_pred)*100))
        print ('Recall:', recall_score(Y_test,  y_pred, average='micro'))
        print ('Precision:', precision_score(Y_test, y_pred, average='micro'))
        
    elif Type=='Gray':
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(150, activation='relu'))
        model.add(layers.Dense(29, activation='softmax'))
        #compile
        model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        #fitting
        model.fit(X_train, Y_train, epochs=6, 
                    validation_data=(X_test, Y_test))
        #Precision & Recall
           # metrics can't handle a multiclass or multioutput so,
           # we use argmax to convert y_pred from 2d to 1d array
        y_pred = model.predict(X_test).argmax(axis=1)
        print("Accuracy:  %.2f%%" % (metrics.accuracy_score(Y_test, y_pred)*100))
        print ('Recall:', recall_score(Y_test,  y_pred, average='micro'))
        print ('Precision:', precision_score(Y_test, y_pred, average='micro'))


processing1(RGB_train_images, Y_, RGB_test_images, Ytest, 'RGB')
processing1(Gray_train_images, Y_, Gray_test_images, Ytest, 'Gray')

#########  ##########   #########   #########   ############   ##########

def processing2(X_train, Y_train, X_test, Y_test, Type):
    model = models.Sequential()
    if Type=='RGB':
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(50, 50, 3)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(200, activation='relu'))
        model.add(layers.Dense(29, activation='softmax'))
        #compile
        model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        #fitting
        model.fit(X_train, Y_train, epochs=6, 
                    validation_data=(X_test, Y_test))
        #Precision & Recall
           # metrics can't handle a multiclass or multioutput so,
           # we use argmax to convert y_pred from 2d to 1d array
        y_pred = model.predict(X_test).argmax(axis=1)
        print("Accuracy:  %.2f%%" % (metrics.accuracy_score(Y_test, y_pred)*100))
        print ('Recall:', recall_score(Y_test,  y_pred, average='micro'))
        print ('Precision:', precision_score(Y_test, y_pred, average='micro'))
        
    elif Type=='Gray':
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(50, 50, 1)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(200, activation='relu'))
        model.add(layers.Dense(29, activation='softmax'))
        #compile
        model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        #fitting
        model.fit(X_train, Y_train, epochs=6, 
                    validation_data=(X_test, Y_test))
        #Precision & Recall
           # metrics can't handle a multiclass or multioutput so,
           # we use argmax to convert y_pred from 2d to 1d array
        y_pred = model.predict(X_test).argmax(axis=1)
        print("Accuracy:  %.2f%%" % (metrics.accuracy_score(Y_test, y_pred)*100))
        print ('Recall:', recall_score(Y_test,  y_pred, average='micro'))
        print ('Precision:', precision_score(Y_test, y_pred, average='micro'))


processing2(RGB_train_images, Y_, RGB_test_images, Ytest, 'RGB')
processing2(Gray_train_images, Y_, Gray_test_images, Ytest, 'Gray')

