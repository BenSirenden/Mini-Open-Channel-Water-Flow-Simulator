import pdb
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset_csv = "/home/bern002/disertasi/HISTO-HOG-FFT/histo-hog-fft.csv"
testing_csv = "/home/bern002/disertasi/HISTO-HOG-FFT/REGRESI LINEAR/LINREG-testing.csv"
coeficient_csv = "/home/bern002/disertasi/HISTO-HOG-FFT/REGRESI LINEAR/LINREG-coeficient.csv"

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

df = pd.read_csv(dataset_csv) #original data from CSV
print(df.head())

#x_attr = df[['meanHisto','medianHisto','meanHOG','medianHOG','maxFFT']]
x_data = np.array(df[['meanHisto','medianHisto','meanHOG','medianHOG','maxFFT']]) #data for x
x_data = x_data/255.0

y_data = np.array(df['wsv']) #data for y

#index 
indices = np.arange(len(x_data))

#split data 
X_train, X_test, y_train, y_test, idx_train, idx_test  = train_test_split(x_data, y_data,indices, test_size=0.3, random_state=16)

print(len(y_train))
print(len(y_test))
#linear regression

regressor = LinearRegression().fit(X_train, y_train)

reg_coef = regressor.intercept_ # intercept or beta 0
reg_coef = np.append(reg_coef,regressor.coef_) # other coeficient
reg_koef = np.array(reg_coef)
#print(reg_koef)

x_attr = ['intercept','meanHisto','medianHisto','meanHOG','medianHOG','maxFFT']
x_attr = np.array(x_attr)
data = {'atrribute':x_attr, 'koefisien': reg_koef}

coef_df =pd.DataFrame(data)
print(coef_df)
#coef_df = pd.concat([w,v], axis = 1, join='inner')
coef_df.to_csv(coeficient_csv, index=False)

#Test the model

y_pred = regressor.predict(X_test)

dfR = pd.DataFrame({"index tes": idx_test, "prediksi": y_pred, "actual": y_test})
dfR.to_csv(testing_csv, index=False)

MAE= metrics.mean_absolute_error(y_test, y_pred)
MAPE = mean_absolute_percentage_error(y_test, y_pred)
MSE= metrics.mean_squared_error(y_test, y_pred)
RSME = np.sqrt(MSE)
R2 = metrics.r2_score(y_test, y_pred)


print('Mean Absolute Error : ', MAE )
print('Mean Absolute Percentage Error : ', MAPE )
print('Mean Squared Error : ', MSE )
print('Root Mean Squared Error : ', RSME )
print('R2 : ', R2 )


print("selesai LINREG-HHF-CPSAT linear")