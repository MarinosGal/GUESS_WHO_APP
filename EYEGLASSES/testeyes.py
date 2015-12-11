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
In this program we decide if a person wears or not eyeglasses.

We load our trained classifier and with Prediction() function we decide.

For more details take a look at the documentation.
'''

############
''' MAIN '''
############

# Loads the necessary arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--test", required = True, help = "The test file's name.")
ap.add_argument("-p", "--apppath", required = True, help = "The path to the App Folder.")
args = vars(ap.parse_args())

# Creates the Eye Perceptron
eyeperceptron = ml.Perceptron(alpha=0.1, thr=0.05, maxiters=100) # basic perceptron

path = args["apppath"]

# Loads the trained Eye Percetron
eyeperceptron = pickle.load(open('eyeperceptron.xml','rb'))

# Loads the test set
test  = np.loadtxt(open(path+'EYEGLASSES/test_glasses/'+args['test'],"rb"),delimiter=";",skiprows=0)
test_ = test[:, :6]

# Prediction #

RES = eyeperceptron.pred(test_)
print '\nRESULT:'
wears=0
doesntwear=0
for r in RES:
	if r==1:
		wears+=abs(r)      # holds all the 1s
	elif r==-1:
		doesntwear+=abs(r) # holds all the -1s
	else:
		print 'NOT OK AT ALL !!!'

if (wears>doesntwear):	                                                     # If 'wears' prob is > than 'doesnwear'...
	print 'WEARS EYEGLASSES with probability of: %s %%' %wears
elif(doesntwear>wears):                                                      # If 'doesntwear' prob is > than 'wears'...
	print 'DOESN\'T WEARS EYEGLASSES with probability of: %s %%' %doesntwear
else:                                                                        # If the 2 probabilities are equal...
	print 'Algorithm can\'t decide!'

# end-of-program #