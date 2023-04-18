import pdb
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import time, datetime


import tensorflow as tf
import keras
from keras import layers
from keras.layers import Input,Dense, Activation
from keras.models import Model, Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

dataset_csv = "/home/bern002/disertasi/HISTO-HOG-FFT/histo-hog-fft.csv"
training_csv = "/home/bern002/disertasi/HISTO-HOG-FFT/KLASIFIKASI/LOGREG-training.csv"
testing_csv = "/home/bern002/disertasi/HISTO-HOG-FFT/KLASIFIKASI/LOGREG--testing.csv"
koef_csv = "/home/bern002/disertasi/HISTO-HOG-FFT/KLASIFIKASI/LOGREG-koefisien.csv"


def vact(pwm):
   
    if pwm == "m05000s":
        wsv = 4.2
    elif pwm == "m03000s":
        wsv = 3.1
    else :
        wsv = 1.7
    
    return wsv


df = pd.read_csv(dataset_csv) #original data from CSV
print(df.head())
#x_attr = df[['meanHisto','medianHisto','meanHOG','medianHOG','maxFFT']]
x_data = np.array(df[['meanHisto','medianHisto','meanHOG','medianHOG','maxFFT']]) #data for x
x_data = x_data/255.0

y_data = np.array(df['wsv']) #data for y
y_new = [str(numeric_string) for numeric_string in y_data]

#index 
indices = np.arange(len(x_data))


#split data 
X_train, X_test, y_train, y_test, idx_train, idx_test  = train_test_split(x_data, y_new,indices, test_size=0.3, random_state=16)

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)
# fit the model with data
logreg.fit(X_train, y_train)

reg_coef = logreg.intercept_[0] # intercept or beta 0
reg_coef = np.append(reg_coef,logreg.coef_[0]) # other coeficient
#print(reg_coef)
reg_koef = np.array(reg_coef)

x_attr = ['intercept','meanHisto','medianHisto','meanHOG','medianHOG','maxFFT']
x_attr = np.array(x_attr)

data = {'atrribute':x_attr, 'koefisien': reg_koef}

coef_logreg_df =pd.DataFrame(data)
coef_logreg_df.to_csv(koef_csv, index=False)
print(coef_logreg_df)

#testing the model
y_pred = logreg.predict(X_test)

#report
dfR = pd.DataFrame({"index tes": idx_test, "prediksi": y_pred, "actual": y_test})

#to_csv
dfR.to_csv(testing_csv, index=False)

target_names = ['1.7', '3.1', '4.2']
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))

#pdb.set_trace()
print("selesai LOGREG-HOG-HISTO-FFT-CPSAT logistic")