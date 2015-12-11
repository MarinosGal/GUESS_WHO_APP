import numpy as np
import mlpy as ml
import csv
import argparse

'''
In this program we test our trained hair color libsvms.

The program loads the test file and with the prediction() function 
we classify our test set.

For more details take a look at the documentation.
'''

#######################
''' Functions' side '''
#######################

'''
Calculates the hard and the soft probability of libsvm

Arguments: pred_array: the prediction array from LibSvm
           prob_array: the pred_probability array from LibSvm

Returns: pred: the 1s from pred array summarized
         prob: the positive probability of the pred_probability array
'''
def show(pred_array,prob_array):
	a=0
	for i in pred_array:
		if i==1:
			a=a+i	

	pred = (a*100)/pred_array.size
	pred = pred
 
	suma=0
	psuma=0
	for i in prob_array[:,1]:
		suma=suma+i
	
	suma=suma/len(prob_array)
	prob=suma*100

	return (pred,prob)

###------###

############
''' ΜΑΙΝ '''
############

#set testing data
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--test", required = True, help = "The test file's name with its path.")
ap.add_argument("-p", "--apppath", required = True, help = "The path to the App Folder.")
args = vars(ap.parse_args())

path = args["apppath"]

test  = np.loadtxt(open(path + args["test"]+'.csv',"rb"),delimiter=";",skiprows=0)
test_ = test[:, :180]

##################
''' Prediction '''
##################

# Creates libsvms
brownsvm  = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
blondesvm = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
blacksvm  = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)

# Loads libsvms - user provides the names of the trained libsvms for each color
brownsvm  = brownsvm.load_model(  path + '')
blondesvm = blondesvm.load_model( path + '')
blacksvm  = blacksvm.load_model(  path + '')

# brown probability
Brown = brownsvm.pred(test_)
brown = brownsvm.pred_probability(test_)
(predBrown,probBrown) = show(Brown,brown)
print '\nBrown Section:'
print 'Brown_Pred_Array        :%s %%' %predBrown
print 'Brown_Probability_Array :%s %%' %probBrown

# blonde probability
Blonde = blondesvm.pred(test_)
blonde = blondesvm.pred_probability(test_)
(predBlonde,probBlonde) = show(Blonde,blonde)
print '\nBlonde Section:'
print 'Blonde_Pred_Array       :%s %%' %predBlonde
print 'Blonde_Probability_Array:%s %%' %probBlonde

# black probability
Black = blacksvm.pred(test_)
black = blacksvm.pred_probability(test_)
(predBlack,probBlack) = show(Black,black)
print '\nBlack Section:'
print 'Black_Pred_Array        :%s %%' %predBlack
print 'Black_Probability_Array :%s %%' %probBlack

print 'OK.'

# end-of-program #