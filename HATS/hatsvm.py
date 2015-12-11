import numpy as np
import cv2
import cv2.cv as cv
import csv
import argparse
import mlpy as ml
import time
import pickle

'''
In this program we train a Perceptron for hat detection.

We load the positive and negative files and we train our hat perceptron

For more details take a look at the documentation.
'''

############
''' MAIN '''
############

start=time.time()

# creates basic perceptron
hatperceptron = ml.Perceptron(alpha=0.1, thr=0.05, maxiters=100) 

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--apppath", required = True, help = "The path to the App Folder")
args = vars(ap.parse_args())

path = args["apppath"]

# Loads training data files
posedg1 = np.loadtxt(open(path + 'HATS/positivehats/PURPLEHAT1_p1_pos_hat.csv',"rb"),delimiter=";", skiprows=0)
posedg2 = np.loadtxt(open(path + 'HATS/positivehats/PURPLEHAT2_p1_pos_hat.csv',"rb"),delimiter=";", skiprows=0)
posedg3 = np.loadtxt(open(path + 'HATS/positivehats/PURPLEHAT3_p1_pos_hat.csv',"rb"),delimiter=";", skiprows=0)
posedg4 = np.loadtxt(open(path + 'HATS/positivehats/PURPLEHAT4_p1_pos_hat.csv',"rb"),delimiter=";", skiprows=0)
#posedg5 = np.loadtxt(open(path + 'HATS/positivehats/marinoshat3_p1_pos_hat.csv',"rb"),      delimiter=";", skiprows=0)

negedg1 = np.loadtxt(open(path + 'HATS/negativehats/NOHAT1_m1_neg_hat.csv',"r",),delimiter=";", skiprows=0)
negedg2 = np.loadtxt(open(path + 'HATS/negativehats/NOHAT2_m1_neg_hat.csv',"rb"),delimiter=";", skiprows=0)
negedg3 = np.loadtxt(open(path + 'HATS/negativehats/NOHAT3_m1_neg_hat.csv',"rb"),delimiter=";", skiprows=0)
negedg4 = np.loadtxt(open(path + 'HATS/negativehats/NOHAT3_m1_neg_hat.csv',"rb"),delimiter=";", skiprows=0)
#negedg5 = np.loadtxt(open(path + 'HATS/negativehats/kostasnohat3_m1_neg_hat.csv',"rb"),     delimiter=";", skiprows=0)

# prepares data. x part is all the feature vectors and y part is the 1s or -1s
posedg1x,posedg1y = posedg1[:, :12],posedg1[:, 12]
posedg2x,posedg2y = posedg2[:, :12],posedg2[:, 12]
posedg3x,posedg3y = posedg3[:, :12],posedg3[:, 12]
posedg4x,posedg4y = posedg4[:, :12],posedg4[:, 12]
#posedg5x,posedg5y = posedg5[:, :12],posedg5[:, 12]

negedg1x,negedg1y = negedg1[:, :12],negedg1[:, 12]
negedg2x,negedg2y = negedg2[:, :12],negedg2[:, 12]
negedg3x,negedg3y = negedg3[:, :12],negedg3[:, 12]
negedg4x,negedg4y = negedg4[:, :12],negedg4[:, 12]
#negedg5x,negedg5y = negedg5[:, :12],negedg5[:, 12]

# Training Perceptron
ex = np.vstack([posedg1x,posedg2x,posedg3x,posedg4x,negedg1x,negedg2x,negedg3x,negedg4x])
ey = np.concatenate((posedg1y,posedg2y,posedg3y,posedg4y,negedg1y,negedg2y,negedg3y,negedg4y))
hatperceptron.learn(ex,ey)

# Saves Perceptron to a xml file
pickle.dump(hatperceptron,open('hatperceptron_t2.xml','wb'))

print '\nHatperceptron has been created and saved successfully!'

# Calculates the training time
end=time.time()
secs = end-start
print 'Time elapsed:%s seconds\n' %secs

# end-of-program #