# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 19:46:10 2022

@author: HP
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
#import pandas as p
import multiprocessing as mp

f_dataset = "./dataset/"
n_video = "m05hn000s/"
f_video = f_dataset + n_video
f_csv =  f_dataset + "CSV/" + n_video
f_hasil = f_dataset + "IMG/"+n_video
totalFiles = 0

def xor_arrbit(bval):
    bvr = int(bval[0])^int(bval[1])^int(bval[2])^int(bval[3])^int(bval[4])^int(bval[5])^int(bval[6])^int(bval[7]) #binary array value xor result
    return bvr

def xor_frame(frame, bframe,rframe,idr):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            r =  int(rframe[i,j])
            a = frame[i,j]
            b = bframe[i,j]
            c = a ^ b
            d = '{0:08b}'.format(c)
            e =  xor_arrbit(d)
            e = e << idr
            r = r ^ e
            rframe[i,j]=int(r)
            #e = e << 6
            #e = '{0:08b}'.format(e)
            #print("a:", a, ",b:",b,"->","a ^ b =", c, "-binary :", d, "xor bit :", r)
    
    return rframe

def run_vid(v_loc,n_csv):
    #print(v_loc)
    n_csv = f_csv+n_csv+".csv"
    print(n_csv)
    cap = cv2.VideoCapture(v_loc)
    nw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    nh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    idfr = 0 #id frame
    bframe = [] #buffer frame
    rframe = np.zeros((nh,nw)) #result frame
    rflat = [] #result frame flatten
    idr = 0 # id for result 
    result = [] #empty result array
    idIMG = 0 #result image Id
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame1 = cap.read()
        if ret == True:
            # Display the resulting frame
            #cv2.imshow('Frame',frame)
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #fh,fw,fc = gray.shape
            frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            if idfr > 0:
                rframe = xor_frame(frame,bframe,rframe,idr)
                idr+=1
                if idr>7:
                    idIMG = idIMG + 1
                    n_img = n_csv[len(n_csv)-15:len(n_csv)-4]
                    n_img  = n_img+"-"+str(idIMG)+".jpg"
                    n_img = f_hasil +"/"+n_img
                    print( n_img)
                    cv2.imwrite(n_img, rframe)
                    #rflat = rframe.flatten()
                    #plt.hist(rflat,bins = [0,25,50,75,100,125,150,175,200,225,250, 255])
                    #plt.hist(rflat,bins =256)
                    #plt.show()
                    idr = 0
                    #tes = '{0:08b}'.format(int(rframe[0,0]))
                    #print(tes)
                    #hist,bins = np.histogram(rflat,bins = [0,25,50,75,100,125,150,175,200,225,250, 255]) 
                    #hist,bins = np.histogram(rflat,bins = 256)
                    #result.append(hist)
                    #print(hist[1])
                    #np.savetxt(n_csv, hist, delimiter=',')
                    #print(hist[0])
                    
                    #cv2.imshow('Frame',rframe)
                    #plt.imshow(rframe, cmap = 'gray')
                    #plt.show() 
                    #print(rframe)
                    #print(rframe[1][1])
                    rframe = np.zeros((nh,nw))
            
            bframe = frame
            idfr+=1
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
        
    #np.savetxt(n_csv, result, delimiter=',')
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    return

for base, dirs, files in os.walk(f_video):
    for file in files:
        totalFiles += 1
        #if totalFiles ==1:
        n_csv = file[0:(len(file)-4)]
        #print(file, "to csv:",n_csv,"panjang file:",len(file))
        v_loc = f_video+file
        run_vid(v_loc,n_csv)
        

#print(totalFiles)
    
