# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 21:13:24 2023

@author: HP
"""
import pdb
#import random
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

f_pwm = "m01000s"
dataset = "./dataset/IMG/NEW Image/" #+f_pwm+"/"
csvoutput = "./hasil training HPC/HISTO-HOG-FFT/histo-hog-fft.csv"

def vact(file):
    nwsv = file[0:7]
    if nwsv == "m05000s":
        wsv = 4.2
    elif nwsv == "m03000s":
        wsv = 3.1
    else :
        wsv = 1.7
    
    #print(nwsv + "-" + str(wsv))
    return wsv

def HOG(img):
    im = np.float32(img) / 255.0
    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    
    # Python Calculate gradient magnitude and direction ( in degrees )
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    
    return mag

def FFT(img):
    ## Furier Transform
    f = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_shift = np.fft.fftshift(f)
    f_complex = f_shift[:,:,0] + 1j*f_shift[:,:,1]
    f_abs = np.abs(f_complex) + 1 # lie between 1 and 1e6
    f_bounded = 20 * np.log(f_abs)
    f_img = 255 * f_bounded / np.max(f_bounded)
    f_img = f_img.astype(np.uint8)
    
    #FFT IMage middle
    n_row = len(f_img)
    n_col = len(f_img[0]) 
    n_fft = n_row
    mid = int(n_fft/2)
    FFT_mid = f_img[0:n_fft,mid]
    #print(FFT_mid)
    
    return FFT_mid

nIMG = []
wsvact = []
meanGflat = []
medianGflat = []
meanHOG = []
medianHOG = []
maXFFT = []

for base, dirs, files in os.walk(dataset):
    for file in files:
        file_path = os.path.join(base, file)
        #print(file_path)
        
        wsv = vact(file)
        # Membuka berkas gambar
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #normal histo
        #gflat = np.float(gray)/255.0
        gflat = gray.flatten()
        
        mean_gflat = np.mean(gflat)
        medn_gflat = np.median(gflat)
        
        
        #HOG histo
        imgHog =HOG(gray)
        
        #HOG Flatten
        iMflat = imgHog.flatten()
        
        mean_iMflat = np.mean(iMflat)
        medn_iMflat = np.median(iMflat)
        
        #FFT
        midFFT =  FFT(gray)
        #max mid FFT
        maxMIdFFT = np.amax(midFFT)
        
        nIMG.append(file) # add file name in dataframe
        wsvact.append(wsv) #add wsv in dataframe
        meanGflat.append(mean_gflat) # add mean histo in dataframe
        medianGflat.append(medn_gflat) # add median histo in dataframe
        meanHOG.append(mean_iMflat) # add mean HOG in dataframe
        medianHOG.append(medn_iMflat) # add median HOG in dataframe
        maXFFT.append(maxMIdFFT) # add max FFT in dataframe  
        
        
        #print("meang gflat : " + str(mean_gflat) + " - median gflat : " + str(medn_gflat))
        #print("meang iMflat : " + str(mean_iMflat) + " - median iMflat : " + str(medn_iMflat))
        #print("max mid FFT : "+ str(maxMIdFFT))
        
        # plotting first histogram
        plt.hist(gflat, label='normal histogram', alpha=.7, color='red', bins =9)
 
        # plotting second histogram
        plt.hist(iMflat, label="HOG", alpha=.5,
         edgecolor='black', color='yellow' , bins =9)
        
        plt.legend()
 
        # Showing the plot using plt.show()
        plt.show()


df = pd.DataFrame(columns=['id', 'wsv','meanHisto', 'medianHisto','meanHOG','medianHOG','maxFFT'])
df['id'] = nIMG # add file name in dataframe
df['wsv'] = wsvact # add wsv in dataframe
df['meanHisto'] = meanGflat # add mean histo in dataframe
df['medianHisto'] = medianGflat # add median histo in dataframe
df['meanHOG'] = meanHOG # add mean HOG in dataframe
df['medianHOG'] = medianHOG # add median HOG in dataframe
df['maxFFT'] = maXFFT # add max FFT in dataframe       
print(df.head)
df.to_csv(csvoutput, index=False)
        
        