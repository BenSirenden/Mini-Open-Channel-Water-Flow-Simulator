import pdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time, datetime

import os
import pandas as pd
import random

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
#from keras.datasets import boston_housing
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping


training_dataset = "/home/bern002/disertasi/CNN/TrainingDfCPSAT.csv"
validation_dataset = "/home/bern002/disertasi/CNN/ValidationDfCPSAT.csv"
testing_dataset = "/home/bern002/disertasi/CNN/TestingDfCPSAT.csv"

testing_result = "/home/bern002/disertasi/CNN/KLASIFIKASI/CNN-Klasifikasi-Testing.csv"
training_result = "/home/bern002/disertasi/CNN/KLASIFIKASI/CNN-KLASIFIKASI-Training.csv"
weight = "/home/bern002/disertasi/CNN/KLASIFIKASI/CNN-Klasifikasi.h5"

#get datasets
DFtraining = pd.read_csv(training_dataset)
DFvalid = pd.read_csv(validation_dataset)
DFtesting = pd.read_csv(testing_dataset)

print(DFtraining.head())
print(DFvalid.head())
print(DFtesting.head())

DFtraining["score"] = DFtraining["score"].replace([4.2], "4.2")
DFtraining["score"] = DFtraining["score"].replace([3.1], "3.1")
DFtraining["score"] = DFtraining["score"].replace([1.7], "1.7")
print(DFtraining.head())

DFvalid["score"] = DFvalid["score"].replace([4.2], "4.2")
DFvalid["score"] = DFvalid["score"].replace([3.1], "3.1")
DFvalid["score"] = DFvalid["score"].replace([1.7], "1.7")
print(DFvalid.head())

DFtesting["score"] = DFtesting["score"].replace([4.2], "4.2")
DFtesting["score"] = DFtesting["score"].replace([3.1], "3.1")
DFtesting["score"] = DFtesting["score"].replace([1.7], "1.7")
print(DFtesting.head())


batch_size = 64#32
img_width = 128
img_height = 128

# create generator
datagen = ImageDataGenerator(rescale = 1/.255)

#train generator
train_generator=datagen.flow_from_dataframe(
                dataframe=DFtraining,
                directory=None,
                x_col="path",
                y_col="score",
                batch_size=batch_size,
                shuffle=False,
                color_mode = "grayscale",
                class_mode="categorical",
                target_size=(img_width,img_height))
                
#validation data using flow_flow_dataframe
valid_generator=datagen.flow_from_dataframe(
        dataframe=DFvalid,
        x_col="path",
        y_col="score",
        batch_size=batch_size,
        shuffle=False,
        color_mode = "grayscale",
        class_mode="categorical",
        target_size=(img_width,img_height))
        

test_it=datagen.flow_from_dataframe(
        dataframe=DFtesting,
        x_col="path",
        y_col="score",
        batch_size=1,
        shuffle=False,
        color_mode = "grayscale",
        class_mode="categorical",
        target_size=(img_width,img_height))
        
#early stop design
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

#check point
checkpoint = ModelCheckpoint(weight, monitor='val_loss', save_best_only=True, save_weights_only =False, verbose=1)

#create model
model = Sequential()
model.add(Conv2D(batch_size, (5, 5), input_shape=(img_width, img_height,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(batch_size, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(32))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation="sigmoid"))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
history = model.fit_generator(generator=train_generator,
                    callbacks=[checkpoint, early_stopping],
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=100)
                    
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.to_csv(training_result, index=False)

#test the model
       
STEP_SIZE_TEST=test_it.n//test_it.batch_size
test_it.reset()
pred=model.predict_generator(test_it,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (test_it.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
print("predicted class")
print(labels)

filenames=test_it.filenames

#pdb.set_trace()
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv(testing_result,index=False)

print("selesai CNN-KLAS-NEWDATASET.py")
