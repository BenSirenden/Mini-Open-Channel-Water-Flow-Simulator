import pdb
import cv2
import numpy as np
import time, datetime

import os
import pandas as pd
import random

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

csv_image = "/home/bern002/disertasi/CNN/dfCPSAT.csv"
training_dataset = "/home/bern002/disertasi/CNN/TrainingDfCPSAT.csv"
validation_dataset = "/home/bern002/disertasi/CNN/ValidationDfCPSAT.csv"
testing_dataset = "/home/bern002/disertasi/CNN/TestingDfCPSAT.csv"

def random_data(DFimg):
    acak = False
    while acak is False:
        x_rand = random.sample(list(DFimg['id']), len(DFimg['id']))
        x_rand = [x - 1 for x in x_rand]
        DFimgRand = DFimg.iloc[x_rand, [2,3]]
        UnikVal = set(DFimgRand['score'])
        if 1.7 in UnikVal and 3.1 in UnikVal and 4.2 in UnikVal:
            acak = True
            print ("acak is True")
        else:
            acak = False
    
    return DFimgRand
    
    
#create dataframe for clasification problem
DFimg = pd.read_csv(csv_image)
print(DFimg.head())

#shuffle the original dataframe
DFimgRand = random_data(DFimg)
print(DFimgRand.head())

nTest = round(len(DFimg)/5) #20% dataset 
nTrain = len(DFimg)-(2*nTest)

dfTrain = DFimgRand[0:nTrain]
dfValid = DFimgRand[nTrain:(len(DFimg)-nTest)]
dfTest = DFimgRand[(nTrain+nTest):len(DFimg)]

print(len(dfTrain))
print(len(dfValid))
print(len(dfTest))

print(dfTrain.head())
print(dfValid.head())
print(dfTest.head())

#create csv
dfTrain.to_csv(training_dataset, index=False)
dfValid.to_csv(validation_dataset, index=False)
dfTest.to_csv(testing_dataset, index=False)

print("end Create-CNN-dataset.py")