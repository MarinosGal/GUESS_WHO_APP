import numpy as np
import mlpy as ml
import csv
import argparse

'''
In this program we create 3 libsvms, one for every hair color (brown,blonde and black).

We load all the positive and negative test sets and we train the libsvms.

Most of the code is commented, because of lack of training data!

For more details look at the documentation.
'''

############
''' MAIN '''
############

# Loads arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--apppath", required = True, help = "The path to App's Folder")
args = vars(ap.parse_args())

# Creates 3 libsvms one for each hair color
brownsvm  = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
blondesvm = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
blacksvm  = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)

# prepares training data for Brown SVM
BROWN_BRO_p1  = np.loadtxt(open(path+'HAIR/BROWN/positive brown/geo_38_browngeo1_p1_hair.csv',"rb"),delimiter=";",skiprows=0)
BROWN_BRO_p2  = np.loadtxt(open(path+'HAIR/BROWN/positive brown/mar_35_brown35_p1_hair.csv',"rb"),delimiter=";",skiprows=0)
BROWN_BRO_p3  = np.loadtxt(open(path+'HAIR/BROWN/positive brown/ore_37_brownore1_p1_hair.csv',"rb"),delimiter=";",skiprows=0)
#BROWN_BRO_p4  = np.loadtxt(open(path+'HAIR/BROWN/positive brown/',"rb"),delimiter=";",skiprows=0)
#BROWN_BLO_m1  = np.loadtxt(open(path+'HAIR/BROWN/negative brown/',"rb"),delimiter=";",skiprows=0)
#BROWN_BLO_m2  = np.loadtxt(open(path+'HAIR/BROWN/negative brown/',"rb"),delimiter=";",skiprows=0)
#BROWN_BLO_m3  = np.loadtxt(open(path+'HAIR/BROWN/negative brown/',"rb"),delimiter=";",skiprows=0)
#BROWN_BLO_m4  = np.loadtxt(open(path+'HAIR/BROWN/negative brown/',"rb"),delimiter=";",skiprows=0)
#BROWN_BLA_m1  = np.loadtxt(open(path+'HAIR/BROWN/negative brown/',"rb"),delimiter=";",skiprows=0)
#BROWN_BLA_m2  = np.loadtxt(open(path+'HAIR/BROWN/negative brown/',"rb"),delimiter=";",skiprows=0)
#BROWN_BLA_m3  = np.loadtxt(open(path+'HAIR/BROWN/negative brown/',"rb"),delimiter=";",skiprows=0)
#BROWN_BLA_m4  = np.loadtxt(open(path+'HAIR/BROWN/negative brown/',"rb"),delimiter=";",skiprows=0)

# prepares training data for Blonde SVM
#BLONDE_BRO_m1  = np.loadtxt(open(path+'',"rb"),delimiter=";",skiprows=0)
#BLONDE_BRO_m2  = np.loadtxt(open(path+'',"rb"),delimiter=";",skiprows=0)
#BLONDE_BRO_m3  = np.loadtxt(open(path+'',"rb"),delimiter=";",skiprows=0)
#BLONDE_BRO_m4  = np.loadtxt(open(path+'',"rb"),delimiter=";",skiprows=0)
#BLONDE_BLO_p1  = np.loadtxt(open(path+'',"rb"),delimiter=";",skiprows=0)
#BLONDE_BLO_p2  = np.loadtxt(open(path+'',"rb"),delimiter=";",skiprows=0)
#BLONDE_BLO_p3  = np.loadtxt(open(path+'',"rb"),delimiter=";",skiprows=0)
#BLONDE_BLO_p4  = np.loadtxt(open(path+'',"rb"),delimiter=";",skiprows=0)
#BLONDE_BLA_m1  = np.loadtxt(open(path+'',"rb"),delimiter=";",skiprows=0)
#BLONDE_BLA_m2  = np.loadtxt(open(path+'',"rb"),delimiter=";",skiprows=0)
#BLONDE_BLA_m3  = np.loadtxt(open(path+'',"rb"),delimiter=";",skiprows=0)
#BLONDE_BLA_m4  = np.loadtxt(open(path+'',"rb"),delimiter=";",skiprows=0)

# prepares  training data for Black SVM
BLACK_BRO_m1   = np.loadtxt(open(path+'HAIR/BLACK/negative black/geo_38_browngeo1_m1_hair',"rb"),delimiter=";",skiprows=0)
BLACK_BRO_m2   = np.loadtxt(open(path+'HAIR/BLACK/negative black/mar_35_brown35_m1_hair.csv',"rb"),delimiter=";",skiprows=0)
BLACK_BRO_m3   = np.loadtxt(open(path+'HAIR/BLACK/negative black/ore_37_brownore1_m1_hair.csv',"rb"),delimiter=";",skiprows=0)
#BLACK_BRO_m4   = np.loadtxt(open(path+'HAIR/BLACK/negative black/',"rb"),delimiter=";",skiprows=0)
#BLACK_BLO_m1   = np.loadtxt(open(path+'HAIR/BLACK/negative black/',"rb"),delimiter=";",skiprows=0)
#BLACK_BLO_m2   = np.loadtxt(open(path+'HAIR/BLACK/negative black/',"rb"),delimiter=";",skiprows=0)
#BLACK_BLO_m3   = np.loadtxt(open(path+'HAIR/BLACK/negative black/',"rb"),delimiter=";",skiprows=0)
#BLACK_BLO_m4   = np.loadtxt(open(path+'HAIR/BLACK/negative black/',"rb"),delimiter=";",skiprows=0)
#BLACK_BLA_p1   = np.loadtxt(open(path+'HAIR/BLACK/positive black/',"rb"),delimiter=";",skiprows=0)
#BLACK_BLA_p2   = np.loadtxt(open(path+'HAIR/BLACK/positive black/',"rb"),delimiter=";",skiprows=0)
#BLACK_BLA_p3   = np.loadtxt(open(path+'HAIR/BLACK/positive black/',"rb"),delimiter=";",skiprows=0)
#BLACK_BLA_p4   = np.loadtxt(open(path+'HAIR/BLACK/positive black/',"rb"),delimiter=";",skiprows=0)

# training set for brownsvm
BROWN_BRO_p1x,BROWN_BRO_p1y    = BROWN_BRO_p1[:, :180],BROWN_BRO_p1[:, 180]
BROWN_BRO_p2x,BROWN_BRO_p2y    = BROWN_BRO_p2[:, :180],BROWN_BRO_p2[:, 180]
BROWN_BRO_p3x,BROWN_BRO_p3y    = BROWN_BRO_p3[:, :180],BROWN_BRO_p3[:, 180]
#BROWN_BRO_p4x,BROWN_BRO_p4y    = BROWN_BRO_p4[:, :180],BROWN_BRO_p4[:, 180]
#BROWN_BLO_m1x,BROWN_BLO_m1y    = BROWN_BLO_m1[:, :180],BROWN_BLO_m1[:, 180]
#BROWN_BLO_m2x,BROWN_BLO_m2y    = BROWN_BLO_m2[:, :180],BROWN_BLO_m2[:, 180]
#BROWN_BLO_m3x,BROWN_BLO_m3y    = BROWN_BLO_m3[:, :180],BROWN_BLO_m3[:, 180]
#BROWN_BLO_m4x,BROWN_BLO_m4y    = BROWN_BLO_m4[:, :180],BROWN_BLO_m4[:, 180]
#BROWN_BLA_m1x,BROWN_BLA_m1y    = BROWN_BLA_m1[:, :180],BROWN_BLA_m1[:, 180]
#BROWN_BLA_m2x,BROWN_BLA_m2y    = BROWN_BLA_m2[:, :180],BROWN_BLA_m2[:, 180]
#BROWN_BLA_m3x,BROWN_BLA_m3y    = BROWN_BLA_m3[:, :180],BROWN_BLA_m3[:, 180]
#BROWN_BLA_m4x,BROWN_BLA_m4y    = BROWN_BLA_m4[:, :180],BROWN_BLA_m4[:, 180]

BROWNtx = np.vstack([BROWN_BRO_p1x,BROWN_BRO_p2x,BROWN_BRO_p3x])
BROWNty = np.concatenate((BROWN_BRO_p1y,BROWN_BRO_p2y,BROWN_BRO_p3y))
brownsvm.learn(BROWNtx,BROWNty)


#training set for blondesvm
BLONDE_BRO_m1x,BLONDE_BRO_m1y  = BLONDE_BRO_m1[:, :180],BLONDE_BRO_m1[:, 180]
BLONDE_BRO_m2x,BLONDE_BRO_m2y  = BLONDE_BRO_m2[:, :180],BLONDE_BRO_m2[:, 180]
BLONDE_BRO_m3x,BLONDE_BRO_m3y  = BLONDE_BRO_m3[:, :180],BLONDE_BRO_m3[:, 180]
BLONDE_BRO_m4x,BLONDE_BRO_m4y  = BLONDE_BRO_m4[:, :180],BLONDE_BRO_m4[:, 180]
#BLONDE_BLO_p1x,BLONDE_BLO_p1y  = BLONDE_BLO_p1[:, :180],BLONDE_BLO_p1[:, 180]
#BLONDE_BLO_p2x,BLONDE_BLO_p2y  = BLONDE_BLO_p2[:, :180],BLONDE_BLO_p2[:, 180]
#BLONDE_BLO_p3x,BLONDE_BLO_p3y  = BLONDE_BLO_p3[:, :180],BLONDE_BLO_p3[:, 180]
#BLONDE_BLO_p4x,BLONDE_BLO_p4y  = BLONDE_BLO_p4[:, :180],BLONDE_BLO_p4[:, 180]
BLONDE_BLA_m1x,BLONDE_BLA_m1y  = BLONDE_BLA_m1[:, :180],BLONDE_BLA_m1[:, 180]
BLONDE_BLA_m2x,BLONDE_BLA_m2y  = BLONDE_BLA_m2[:, :180],BLONDE_BLA_m2[:, 180]
BLONDE_BLA_m3x,BLONDE_BLA_m3y  = BLONDE_BLA_m3[:, :180],BLONDE_BLA_m3[:, 180]
BLONDE_BLA_m4x,BLONDE_BLA_m4y  = BLONDE_BLA_m4[:, :180],BLONDE_BLA_m4[:, 180]

Gtx = np.vstack([BLONDE_BLA_m1x,BLONDE_BLA_m2x,BLONDE_BLA_m3x,BLONDE_BLA_m4x,BLONDE_BRO_m1x,BLONDE_BRO_m2x,BLONDE_BRO_m3x,BLONDE_BRO_m4x,BLONDE_BLO_p1x,BLONDE_BLO_p2x,BLONDE_BLO_p3x,BLONDE_BLO_p4x])
Gty = np.concatenate((BLONDE_BLA_m1y,BLONDE_BLA_m2y,BLONDE_BLA_m3y,BLONDE_BLA_m4y,BLONDE_BRO_m1y,BLONDE_BRO_m2y,BLONDE_BRO_m3y,BLONDE_BRO_m4y,BLONDE_BLO_p1y,BLONDE_BLO_p2y,BLONDE_BLO_p3y,BLONDE_BLO_p4y))
blondesvm.learn(Gtx,Gty)

# training set for blacksvm
BLACK_BRO_m1x,BLACK_BRO_m1y    = BLACK_BRO_m1[:, :180],BLACK_BRO_m1[:, 180]
BLACK_BRO_m2x,BLACK_BRO_m2y    = BLACK_BRO_m2[:, :180],BLACK_BRO_m2[:, 180]
BLACK_BRO_m3x,BLACK_BRO_m3y    = BLACK_BRO_m3[:, :180],BLACK_BRO_m3[:, 180]
#BLACK_BRO_m4x,BLACK_BRO_m4y    = BLACK_BRO_m4[:, :180],BLACK_BRO_m4[:, 180]
#BLACK_BLO_m1x,BLACK_BLO_m1y    = BLACK_BLO_m1[:, :180],BLACK_BLO_m1[:, 180]
#BLACK_BLO_m2x,BLACK_BLO_m2y    = BLACK_BLO_m2[:, :180],BLACK_BLO_m2[:, 180]
#BLACK_BLO_m3x,BLACK_BLO_m3y    = BLACK_BLO_m3[:, :180],BLACK_BLO_m3[:, 180]
#BLACK_BLO_m4x,BLACK_BLO_m4y    = BLACK_BLO_m4[:, :180],BLACK_BLO_m4[:, 180]
#BLACK_BLA_p1x,BLACK_BLA_p1y    = BLACK_BLA_p1[:, :180],BLACK_BLA_p1[:, 180]
#BLACK_BLA_p2x,BLACK_BLA_p2y    = BLACK_BLA_p2[:, :180],BLACK_BLA_p2[:, 180]
#BLACK_BLA_p3x,BLACK_BLA_p3y    = BLACK_BLA_p3[:, :180],BLACK_BLA_p3[:, 180]
#BLACK_BLA_p4x,BLACK_BLA_p4y    = BLACK_BLA_p4[:, :180],BLACK_BLA_p4[:, 180]

Btx = np.vstack([BLACK_BRO_m1x,BLACK_BRO_m2x,BLACK_BRO_m3x])
Bty = np.concatenate((BLACK_BRO_m1y,BLACK_BRO_m2y,BLACK_BRO_m3y)) 
blacksvm.learn(Btx,Bty)

#saves the three svms
brownsvm.save_model('rgb_brownsvm.xml')
blacksvm.save_model('rgb_blacksvm.xml')
blondesvm.save_model('rgb_blondesvm.xml')

print '\nOK !\n'

# end-of-program #