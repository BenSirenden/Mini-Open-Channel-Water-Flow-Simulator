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

weight = "/home/bern002/disertasi/CNN/REGRESI/CNN-Regresi.h5"
training_dataset = "/home/bern002/disertasi/CNN/TrainingDfCPSAT.csv"
validation_dataset = "/home/bern002/disertasi/CNN/ValidationDfCPSAT.csv"
testing_dataset = "/home/bern002/disertasi/CNN/TestingDfCPSAT.csv"

testing_result = "/home/bern002/disertasi/CNN/REGRESI/CNN-Regresi.csv"
training_result = "/home/bern002/disertasi/CNN/REGRESI/CNN-REGRESI-Training.csv"

#get datasets
DFtraining = pd.read_csv(training_dataset)
DFvalid = pd.read_csv(validation_dataset)
DFtesting = pd.read_csv(testing_dataset)

print(DFtraining.head())
print(DFvalid.head())
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
                class_mode="raw",
                target_size=(img_width,img_height))
                
#validation data using flow_flow_dataframe
valid_generator=datagen.flow_from_dataframe(
        dataframe=DFvalid,
        x_col="path",
        y_col="score",
        batch_size=batch_size,
        shuffle=False,
        color_mode = "grayscale",
        class_mode="raw",
        target_size=(img_width,img_height))
        

test_it=datagen.flow_from_dataframe(
        dataframe=DFtesting,
        x_col="path",
        y_col="score",
        batch_size=1,
        color_mode = "grayscale",
        class_mode="raw",
        target_size=(img_width,img_height))
       
#early stop design
early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

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

model.add(Dense(1, activation="linear"))

model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_percentage_error'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
history = model.fit_generator(generator=train_generator,
                    callbacks=[checkpoint, early_stopping],
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=100)


#save the training history
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

filenames=test_it.filenames
filenames = np.array(filenames)

pred=pred.reshape(-1,)
#pdb.set_trace()
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":pred})


results.to_csv(testing_result,index=False)

print("selesai CNN-REG-NEWDATAset.py")
