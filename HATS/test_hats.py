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
In this program we detect if a person wears or not a hat.

We load our trained perceptron and with prediction() function we decide if the person
wears or not a hat. The hat must be a certain type!

For more details take a look at the documentation.
'''

############
''' MAIN '''
############

# Loads arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--test"    , required = True, help = "The test file's name")
ap.add_argument("-p", "--apppath" , required = True, help = "The path to App Folder")
args = vars(ap.parse_args())

# Creates Perceptron
eyeperceptron = ml.Perceptron(alpha=0.1, thr=0.05, maxiters=100) # basic perceptron

path = args["apppath"]

# Loads the trained Hat Perceptron
hatperceptron = pickle.load(open('hatperceptron.xml','rb'))

# Loads the test set
test  = np.loadtxt(open(path+'HATS/test_hats/'+args['test'],"rb"),delimiter=";",skiprows=0)
test_ = test[:, :6]

##################
''' Prediction '''
##################

RES = hatperceptron.pred(test_)
print '\nRESULT:'
wears=0
doesntwear=0
for r in RES:
	if r==1:
		wears+=abs(r)       # holds all 1s from prediction array
	elif r==-1:
		doesntwear+=abs(r)  # holds all -1s from prediction array
	else:
		print 'NOT OK AT ALL !!!'

if (wears>doesntwear and (wears-doesntwear)>20):                                                # if 'wears' probability is > than 'doesn't...
	print 'WEARS HAT with probability of: %s %%' %wears
elif(doesntwear>wears and (doesntwear-wears)>20):
	print 'DOESN\'T WEAR HAT with probability of :%s %%' %doesntwear  # if 'doesn't wear' probability is > than 'wears'...
elif(doesntwear>wears and (doesntwear-wears)<20):
	print 'MAYBE DOESN\'T WEAR HAT with probability of :%s %%' %doesntwear
elif(doesntwear>wears and (doesntwear-wears)>20):
	print 'MAYBE WEARS HAT with probability of :%s %%' %doesntwear
else:
	print 'Algorithm can\'t decide! '                                 # if the 2 probabilities are equal...

# end-of-program #