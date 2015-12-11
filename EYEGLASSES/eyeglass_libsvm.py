import numpy as np
import cv2
import cv2.cv as cv
import csv
import sys
import argparse
import mlpy as ml
import os 
import time
import glob
import pickle

'''
In this program we create a Perceptron for eyeglasses detection.

We load our positive and negative values for training.

For more details take a look at the documentation.
'''

############
''' MAIN '''
############

start=time.time()

# Creates Eye Perceptron
eyeperceptron = ml.Perceptron(alpha=0.1, thr=0.05, maxiters=100) 

# Loads necessary arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--apppath", required = True, help = "The path to the App Folder.")
args = vars(ap.parse_args())

path = args["apppath"]

# Loads all the data for training
posedg1 = np.loadtxt(open(path+'EYEGLASSES/positive glasses/GLA_TRAIN_1_p1_pos_eyes.csv' ,"rb"),delimiter=";",skiprows=0)
posedg2 = np.loadtxt(open(path+'EYEGLASSES/positive glasses/GLA_TRAIN_2_p1_pos_eyes.csv' ,"rb"),delimiter=";",skiprows=0)
posedg3 = np.loadtxt(open(path+'EYEGLASSES/positive glasses/GLA_TRAIN_3_p1_pos_eyes.csv' ,"rb"),delimiter=";",skiprows=0)
posedg4 = np.loadtxt(open(path+'EYEGLASSES/positive glasses/GLA_TRAIN_4_p1_pos_eyes.csv' ,"rb"),delimiter=";",skiprows=0)
posedg5 = np.loadtxt(open(path+'EYEGLASSES/positive glasses/GLA_TRAIN_5_p1_pos_eyes.csv' ,"rb"),delimiter=";",skiprows=0)
#posedg6 = np.loadtxt(open(path+'EYEGLASSES/positive glasses/GLA_TRAIN_7_p1_pos_eyes.csv' ,"rb"),delimiter=";",skiprows=0)
 
negedg1 = np.loadtxt(open(path+'EYEGLASSES/negative glasses/GLA_TRAIN_NEG_8_m1_pos_eyes.csv' ,"rb"),delimiter=";",skiprows=0)
negedg2 = np.loadtxt(open(path+'EYEGLASSES/negative glasses/GLA_TRAIN_NEG_9_m1_pos_eyes.csv' ,"rb"),delimiter=";",skiprows=0)
negedg3 = np.loadtxt(open(path+'EYEGLASSES/negative glasses/GLA_TRAIN_NEG_10_m1_pos_eyes.csv',"rb"),delimiter=";",skiprows=0)
negedg4 = np.loadtxt(open(path+'EYEGLASSES/negative glasses/GLA_TRAIN_NEG_11_m1_pos_eyes.csv',"rb"),delimiter=";",skiprows=0)
negedg5 = np.loadtxt(open(path+'EYEGLASSES/negative glasses/GLA_TRAIN_NEG_12_m1_pos_eyes.csv',"rb"),delimiter=";",skiprows=0)
negedg6 = np.loadtxt(open(path+'EYEGLASSES/negative glasses/GLA_TRAIN_NEG_8_m1_pos_eyes.csv' ,"rb"),delimiter=";",skiprows=0)
negedg7 = np.loadtxt(open(path+'EYEGLASSES/negative glasses/GLA_TRAIN_NEG_8_m1_pos_eyes.csv' ,"rb"),delimiter=";",skiprows=0)


# Prepares data for training
# x part is the feature vectors
# y part is the 1s or -1s columns
posedg1x,posedg1y = posedg1[:, :6],posedg1[:, 6]
posedg2x,posedg2y = posedg2[:, :6],posedg2[:, 6]
posedg3x,posedg3y = posedg3[:, :6],posedg3[:, 6]
posedg4x,posedg4y = posedg4[:, :6],posedg4[:, 6]
posedg5x,posedg5y = posedg5[:, :6],posedg5[:, 6]
#posedg6x,posedg6y = posedg6[:, :6],posedg6[:, 6]

negedg1x,negedg1y = negedg1[:, :6],negedg1[:, 6]
negedg2x,negedg2y = negedg2[:, :6],negedg2[:, 6]
negedg3x,negedg3y = negedg3[:, :6],negedg3[:, 6]
negedg4x,negedg4y = negedg4[:, :6],negedg4[:, 6]
negedg5x,negedg5y = negedg5[:, :6],negedg5[:, 6]
negedg6x,negedg6y = negedg6[:, :6],negedg6[:, 6]
negedg7x,negedg7y = negedg7[:, :6],negedg7[:, 6]

# Trains Eye Perceptron
ex = np.vstack([posedg1x,posedg2x,negedg2x,posedg3x,negedg3x,posedg4x,negedg4x,negedg7x,posedg5x,negedg5x,negedg6x])
ey = np.concatenate((posedg1y,posedg2y,negedg2y,posedg3y,negedg3y,posedg4y,negedg4y,negedg7y,posedg5y,negedg5y,negedg6y))
eyeperceptron.learn(ex,ey)

# Loads the Eye Perceptron  
pickle.dump(eyeperceptron,open('eyeperceptron_t1.xml','wb'))

print '\nEyeperceptron has been created and saved successfully!'

# Calculates the training time
end=time.time()
secs = end-start
print 'Time elapsed:%s seconds\n' %secs

# end-of-program #