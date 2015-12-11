import numpy as np
import mlpy as ml
import csv
import argparse

'''
In this program we load each test set and we use the trained libsvms for every color.

For more details take a look at the documentation.
'''

#######################
''' Functions' side '''
#######################

'''
Counts the 1s and -1s for a given prediction array

Arguments: prediction array

Returns: the hard probability of 1s and -1s
'''
def returnpred(array):
	a=0
	b=0
	for i in array:
		if i==1:
			a=a+i
		else:
			b=b+1	

	ppred = (a*100)/array.size
	mpred = (b*100)/array.size
	return (ppred,mpred)

'''
Counts the positive and negative soft probabilities for an array with 2 columns.
The 1st column refers to the probability the test set not to belong to the LibSvm 
and the 2nd column refers to the probability the test to belong to the LibSvm.

Arguments: the pred_probability array

Returns: the probability of belonging and not belonging
'''
def returnpredprob(array):
	suma=0
	for i in array[:,0]:
		suma=suma+i

	d_belong=suma/100
	d_belong=d_belong*100
	
	suma=0
	for i in array[:,1]:
		suma=suma+i

	belong = suma/100
	belong = belong*100
	return (d_belong,belong)
	
'''
Detects the correct color of a shirt, given that we don't have standard brightness in the Library
we have to make some adjustments to the results of each color libsvm. For example we have told that the 
BLUE color is: 'that' percent of blue and 'that' percent of black. Of course if you have trained your libsvms in
a standard setup

Arguments: the prediction arrays for every color LibSvm

Returns: SHIRT_ID: a list of all the ids that correspond to a value of the Perceptron
SHIRT_ID=1.10 : Person wears red shirt.
SHIRT_ID=1.15 : Person maybe wears red shirt.
SHIRT_ID=1.20 : Person wears green shirt.
SHIRT_ID=1.25 : Person maybe wears green shirt.
SHIRT_ID=1.30 : Person wears blue shirt.
SHIRT_ID=1.35 : Person maybe wears blue shirt.
SHIRT_ID=1.40 : Person wears white shirt.
SHIRT_ID=1.45 : Person maybe wears white shirt.
SHIRT_ID=1.50 : Person wears black shirt.
SHIRT_ID=1.55 : Person maybe wears black shirt.
SHIRT_ID=0    : Algorithm can't decide
'''
def detect_heuristic_shirt_color(Rpred,Gpred,Bpred,Wpred,BLpred):
	
	SHIRT_ID=[]
	rflag  = False
	gflag  = False
	bflag  = False
	wflag  = False
	blflag = False

	# the values that give the Red color
	if( (Rpred>3 and Wpred>80 and Gpred<2 and Bpred<6 and BLpred<25) or 
		(Rpred-Gpred>10 and Rpred-Bpred>10 and Rpred-BLpred>10 and Wpred>80) or 
		(0.5*Rpred<BLpred and 0.1*Rpred<Gpred and 0.1*Rpred<Bpred and Wpred>80 and Rpred>5) or 
		(Rpred>Gpred and Rpred>Bpred and Wpred>95 and BLpred>94)or
		(Rpred>90 and Bpred<80)or(Rpred>3 and Wpred>40 and BLpred<7)or
		(Rpred>=18)):
  	
  		rflag=True
  		if(Wpred==100 and BLpred<2):
  			SHIRT_ID.append(1.15)
  		else:
			SHIRT_ID.append(1.1)	

	# the values that give the Green color
	if( (Gpred>7 and Wpred>85 and BLpred>65 and Rpred<4 and Bpred<4)or
		(Bpred-Gpred>0 and Bpred-Gpred<5 and Wpred>50 and BLpred<10 and Rpred<4)or
		(BLpred>50 and Gpred!=0 and Bpred<13 and Wpred<14 and Rpred<3)or
		(Gpred>Rpred and Gpred>Bpred and Wpred>90 and BLpred>80)or
		(Gpred>30 and Bpred<5 and Rpred<1 and Wpred<40)or
		(Gpred>Rpred and Gpred>Bpred and Wpred>8 and BLpred<50)or
		(Gpred>80)):

		gflag=True
		if(Gpred-Bpred<10 and Gpred-Bpred>0)or(Bpred-Gpred>0 and Bpred-Gpred<5)or(BLpred>50 and Gpred<2 and Bpred<13 and Wpred<14 and Rpred<3):
			SHIRT_ID.append(1.25)
		else:
			SHIRT_ID.append(1.2)

	# the values that give the Blue color
	if( (Bpred>50 and Gpred>20 and Gpred<35 and BLpred<30 and Wpred>40 and Wpred<60) or 
		(Bpred>50 and 0.5*Bpred<=Gpred and 0.3*Bpred<=BLpred and Wpred>40 and Wpred<60 and Rpred<10) or 
		(Bpred>5 and 0.3*Bpred<=Gpred and 0.3*Bpred<=BLpred and Wpred>40 and Wpred<60 and Rpred<10)or
		(Bpred>4 and BLpred>60 and Wpred==100 and Rpred<=6 and Gpred<=6)or
		(Bpred>0 and Rpred==0 and Gpred==0 and Wpred>50 and BLpred>50)or
		(Bpred>60 and Gpred<10 and Wpred<30 and Rpred<=1 and BLpred<=1)or
		(Bpred>20 and Rpred==0 and Gpred==0)or
		(Bpred>Rpred and Bpred>Gpred)or
		(Bpred>50)):
	
		bflag=True
		if(Bpred-Gpred<10 and Bpred-Gpred>0)or(Rpred>90 and Bpred>90 and Wpred>90):
			SHIRT_ID.append(1.35)
		else:
			SHIRT_ID.append(1.3)

	# the values that give the White color
	if( (Wpred>80 and Rpred<=2 and Gpred<=2 and Bpred<=1 and BLpred<=1)or
		(Wpred>60 and Rpred==0 and Gpred==0 and Bpred==0 and BLpred==0)or
		(Wpred>98 and Rpred<=2 and Gpred<=2 and Bpred<=2 and BLpred==0)or
		(Wpred==100 and Rpred==0 and Gpred==0 and Bpred<=10 and BLpred==0)or
		(Wpred==100 and BLpred==0 and Rpred<10 and Bpred==0 and Gpred==0 )or
		(Wpred==100 and BLpred==100)):
	
		wflag=True
		SHIRT_ID.append(1.4)

	# the values that give the Black color
	if( (BLpred>80 and Wpred>15 and Rpred<5 and Gpred<2 and Bpred<2)or
		(BLpred>5 and Wpred<10 and Rpred<1 and Bpred<1 and Gpred<1)or 
		(BLpred>40 and Wpred>40 and Wpred<60 and 0.1*BLpred<Bpred and Gpred<1)or
		(BLpred>80 and Wpred<2 and Bpred<2 and Gpred<2 and Rpred<2)or 
		(BLpred>70 and Wpred<60 and Rpred==0 and Gpred==0 and Bpred<3)or
		(BLpred>50 and Wpred>90 and Rpred<=5 and Gpred<=5 and Bpred<=5)or
		(BLpred>Wpred and Rpred<7 and Gpred<7 and Bpred<7)or
		(BLpred>Wpred and Rpred<10)or
		(BLpred==100 and Rpred<20)):
	
		blflag=True
		if(BLpred>70 and Wpred<60 and Rpred==0 and Gpred==0 and Bpred<3)or(BLpred>70 and Wpred>98):
			SHIRT_ID.append(1.55)
		else:
			SHIRT_ID.append(1.5)

	# if heuristic algorithm can't decide then SHIRT_ID=0
	if(rflag==gflag==bflag==wflag==blflag==False):
		SHIRT_ID.append(0)	

	return SHIRT_ID

###------###

############
''' MAIN '''
############

# loads the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s1", "--test1"  , required = True, help = "The csv's rgb test's filename and the path to it.")
ap.add_argument("-s2", "--test2"  , required = True, help = "The csv's rgb hsv test's filename and the path to it.")
#ap.add_argument("-s3", "--test3"  , required = True, help = "The csv's rgbs test's filename and the path to it.")
ap.add_argument("-p", "--apppath" , required = True, help = "The path to the App Folder.")

# loads all arguments
args = vars(ap.parse_args())

path = args["apppath"]

# sets testing data
test1  = np.loadtxt(open(path+'SHIRTS/'+args["test1"],"rb"),delimiter=";",skiprows=0) # RGB
test1x = test1[:, :360]

test2  = np.loadtxt(open(path+'SHIRTS/'+args["test2"],"rb"),delimiter=";",skiprows=0) # RGB HSV
test2x = test2[:, :360]

#test3  = np.loadtxt(open(path+'SHIRTS/'+args["test3"],"rb"),delimiter=";",skiprows=0) # RGB S
#test3x = test3[:, :230]

# Creates RGB Libsvms
rgb_redsvm   = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgb_greensvm = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgb_bluesvm  = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgb_whitesvm = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgb_blacksvm = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)	
# Loads RGB Libsvms
rgb_redsvm   = rgb_redsvm.load_model(path   + 'SHIRTS/rgb_redsvm.xml')
rgb_greensvm = rgb_greensvm.load_model(path + 'SHIRTS/rgb_greensvm.xml')
rgb_bluesvm  = rgb_bluesvm.load_model(path  + 'SHIRTS/rgb_bluesvm.xml')
rgb_whitesvm = rgb_whitesvm.load_model(path + 'SHIRTS/rgb_whitesvm.xml')
rgb_blacksvm = rgb_blacksvm.load_model(path + 'SHIRTS/rgb_blacksvm.xml')
print 'fdddasdfghjklhgfdsdfghjklkjhgfdssdfghjkjhgfdssdfghjkjhgfd'

# Creates RGB S Libsvms
rgbs_redsvm   = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgbs_greensvm = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgbs_bluesvm  = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgbs_whitesvm = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgbs_blacksvm = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)	
# Loads RGB S Libsvms
rgbs_redsvm   = rgbs_redsvm.load_model(path   + 'SHIRTS/rgbs_redsvm.xml')
rgbs_greensvm = rgbs_greensvm.load_model(path + 'SHIRTS/rgbs_greensvm.xml')
rgbs_bluesvm  = rgbs_bluesvm.load_model(path  + 'SHIRTS/rgbs_bluesvm.xml')
rgbs_whitesvm = rgbs_whitesvm.load_model(path + 'SHIRTS/rgbs_whitesvm.xml')
rgbs_blacksvm = rgbs_blacksvm.load_model(path + 'SHIRTS/rgbs_blacksvm.xml')


# Creates RGB HSV Libsvms
rgbhsv_redsvm   = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgbhsv_greensvm = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgbhsv_bluesvm  = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgbhsv_whitesvm = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgbhsv_blacksvm = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)	
# Loads RGB HSV Libsvms
rgbhsv_redsvm   = rgbhsv_redsvm.load_model(path   + 'SHIRTS/rgbhsv_redsvm.xml')
rgbhsv_greensvm = rgbhsv_greensvm.load_model(path + 'SHIRTS/rgbhsv_greensvm.xml')
rgbhsv_bluesvm  = rgbhsv_bluesvm.load_model(path  + 'SHIRTS/rgbhsv_bluesvm.xml')
rgbhsv_whitesvm = rgbhsv_whitesvm.load_model(path + 'SHIRTS/rgbhsv_whitesvm.xml')
rgbhsv_blacksvm = rgbhsv_blacksvm.load_model(path + 'SHIRTS/rgbhsv_blacksvm.xml')

## Predictions ##

################
''' RGB SVMS '''
################

print '\nRGB RESULTS:\n'

# red prediction hard and soft probability
Red = rgb_redsvm.pred(test1x)
#rp = rgb_redsvm.pred_probability(test1x)
#(d_belong,Rbelong) = returnpredprob(rp)
(Rpred,mpred) = returnpred(Red)
print 'RED'
print 'Does have red with HARD prob    :%s' %Rpred
#print 'Doesn\'t have red with HARD prob:%s' %mpred
#print 'Does have red with prob         :%s' %Rbelong
#print 'Doesn\'t have red with prob     :%s' %d_belong

# green prediction hard and soft probability
Green = rgb_greensvm.pred(test1x)
#gp = rgb_greensvm.pred_probability(test1x)
#(d_belong,Gbelong) = returnpredprob(gp)
(Gpred,mpred) = returnpred(Green)
print 'GREEN'
print 'Does have green with HARD prob    :%s' %Gpred
#print 'Doesn\'t have green with HARD prob:%s' %mpred
#print 'Does have green with prob         :%s' %Gbelong
#print 'Doesn\'t have green with prob     :%s' %d_belong

# blue prediction hard and soft probability
Blue = rgb_bluesvm.pred(test1x)
#bp = rgb_bluesvm.pred_probability(test1x)
#(d_belong,Bbelong) = returnpredprob(bp)
(Bpred,mpred) = returnpred(Blue)
print 'BLUE'
print 'Does have blue with HARD prob    :%s' %Bpred
#print 'Doesn\'t have blue with HARD prob:%s' %mpred
#print 'Does have blue with prob         :%s' %Bbelong
#print 'Doesn\'t have blue with prob     :%s' %d_belong

# white prediction hard and soft probability
White = rgb_whitesvm.pred(test1x)
#wp = rgb_whitesvm.pred_probability(test1x)
#(d_belong,Wbelong) = returnpredprob(wp)
(Wpred,mpred) = returnpred(White)
print 'WHITE'
print 'Does have white with HARD prob    :%s' %Wpred
#print 'Doesn\'t have white with HARD prob:%s' %mpred
#print 'Does have white with prob         :%s' %Wbelong
#print 'Doesn\'t have white with prob     :%s' %d_belong

# black prediction hard and soft probability
Black = rgb_blacksvm.pred(test1x)
#blp = rgb_blacksvm.pred_probability(test1x)
#(d_belong,BLbelong) = returnpredprob(blp)
(BLpred,mpred) = returnpred(Black)
print 'BLACK'
print 'Does have black with HARD prob    :%s' %BLpred
#print 'Doesn\'t have black with HARD prob:%s' %mpred
#print 'Does have black with prob         :%s' %BLbelong
#print 'Doesn\'t have black with prob     :%s' %d_belong

RGBshirtIds = []
print 'RGB heuristic results:'
RGBshirtIds = detect_heuristic_shirt_color(Rpred,Gpred,Bpred,Wpred,BLpred)
print 'Person wears:'
for s in RGBshirtIds:
	if s==1.1:
		print 'RED'
	elif s==1.15:
		print 'MAYBE RED'
	elif s==1.2:
		print 'GREEN'
	elif s==1.25:
		print 'MAYBE GREEN'
	elif s==1.3:
		print 'BLUE'
	elif s==1.35:
		print 'MAYBE BLUE'
	elif s==1.4:
		print 'WHITE'
	elif s==1.45:
		print 'MAYBE WHITE'
	elif s==1.5:
		print 'BLACK'
	elif s==1.55:
		print 'MAYBE BLACK'
	else:
		print 'System can\'t decide...'
print 'shirt.'
###----------###

##################
''' RGB S SVMS '''
##################
'''
print '\nRGB S RESULTS:\n'

# red prediction hard and soft probability
Red = rgbs_redsvm.pred(test3x)
#rp = rgbs_redsvm.pred_probability(test3x)
#(d_belong,Rbelong) = returnpredprob(rp)
(Rpred,mpred) = returnpred(Red)
print 'RED'
print 'Does have red with HARD prob    :%s' %Rpred
#print 'Doesn\'t have red with HARD prob:%s' %mpred
#print 'Does have red with prob         :%s' %Rbelong
#print 'Doesn\'t have red with prob     :%s' %d_belong

# green prediction hard and soft probability
Green = rgbs_greensvm.pred(test3x)
#gp = rgbs_greensvm.pred_probability(test3x)
#(d_belong,Gbelong) = returnpredprob(gp)
(Gpred,mpred) = returnpred(Green)
print 'GREEN'
print 'Does have green with HARD prob    :%s' %Gpred
#print 'Doesn\'t have green with HARD prob:%s' %mpred
#print 'Does have green with prob         :%s' %Gbelong
#print 'Doesn\'t have green with prob     :%s' %d_belong

# blue prediction hard and soft probability
Blue = rgbs_bluesvm.pred(test3x)
#bp = rgbs_bluesvm.pred_probability(test3x)
#(d_belong,Bbelong) = returnpredprob(bp)
(Bpred,mpred) = returnpred(Blue)
print 'BLUE'
print 'Does have blue with HARD prob    :%s' %Bpred
#print 'Doesn\'t have blue with HARD prob:%s' %mpred
#print 'Does have blue with prob         :%s' %Bbelong
#print 'Doesn\'t have blue with prob     :%s' %d_belong

# white prediction hard and soft probability
White = rgbs_whitesvm.pred(test3x)
#wp = rgbs_whitesvm.pred_probability(test3x)
#(d_belong,Wbelong) = returnpredprob(wp)
(Wpred,mpred) = returnpred(White)
print 'WHITE'
print 'Does have white with HARD prob    :%s' %Wpred
#print 'Doesn\'t have white with HARD prob:%s' %mpred
#print 'Does have white with prob         :%s' %Wbelong
#print 'Doesn\'t have white with prob     :%s' %d_belong

# black prediction hard and soft probability
Black = rgbs_blacksvm.pred(test3x)
#blp = rgbs_blacksvm.pred_probability(test3x)
#(d_belong,BLbelong) = returnpredprob(blp)
(BLpred,mpred) = returnpred(Black)
print 'BLACK'
print 'Does have black with HARD prob    :%s' %BLpred
#print 'Doesn\'t have black with HARD prob:%s' %mpred
#print 'Does have black with prob         :%s' %BLbelong
#print 'Doesn\'t have black with prob     :%s' %d_belong

RGBSshirtIds = []
print 'RGB S heuristic results:'
RGBSshirtIds = detect_heuristic_shirt_color(Rpred,Gpred,Bpred,Wpred,BLpred)
print 'Person wears:'
for s in RGBSshirtIds:
	if s==1.1:
		print 'RED'
	elif s==1.15:
		print 'MAYBE RED'
	elif s==1.2:
		print 'GREEN'
	elif s==1.25:
		print 'MAYBE GREEN'
	elif s==1.3:
		print 'BLUE'
	elif s==1.35:
		print 'MAYBE BLUE'
	elif s==1.4:
		print 'WHITE'
	elif s==1.45:
		print 'MAYBE WHITE'
	elif s==1.5:
		print 'BLACK'
	elif s==1.55:
		print 'MAYBE BLACK'
	else:
		print 'System can\'t decide...'
print 'shirt.'

###----------###
'''
####################
''' RGB HSV SVMS '''
####################

print '\nRGB HSV RESULTS:\n'

# red prediction hard and soft probability
Red = rgbhsv_redsvm.pred(test2x)
#rp = rgbhsv_redsvm.pred_probability(test2x)
#(d_belong,Rbelong) = returnpredprob(rp)
(Rpred,mpred) = returnpred(Red)
print 'RED'
print 'Does have red with HARD prob    :%s' %Rpred
#print 'Doesn\'t have red with HARD prob:%s' %mpred
#print 'Does have red with prob         :%s' %Rbelong
#print 'Doesn\'t have red with prob     :%s' %d_belong

# green prediction hard and soft probability
Green = rgbhsv_greensvm.pred(test2x)
#gp = rgbhsv_greensvm.pred_probability(test2x)
#(d_belong,Gbelong) = returnpredprob(gp)
(Gpred,mpred) = returnpred(Green)
print 'GREEN'
print 'Does have green with HARD prob    :%s' %Gpred
#print 'Doesn\'t have green with HARD prob:%s' %mpred
#print 'Does have green with prob         :%s' %Gbelong
#print 'Doesn\'t have green with prob     :%s' %d_belong

# blue prediction hard and soft probability
Blue = rgbhsv_bluesvm.pred(test2x)
#bp = rgbhsv_bluesvm.pred_probability(test2x)
#(d_belong,Bbelong) = returnpredprob(bp)
(Bpred,mpred) = returnpred(Blue)
print 'BLUE'
print 'Does have blue with HARD prob    :%s' %Bpred
#print 'Doesn\'t have blue with HARD prob:%s' %mpred
#print 'Does have blue with prob         :%s' %Bbelong
#print 'Doesn\'t have blue with prob     :%s' %d_belong

# white prediction hard and soft probability
White = rgbhsv_whitesvm.pred(test2x)
#wp = rgbhsv_whitesvm.pred_probability(test2x)
#(d_belong,Wbelong) = returnpredprob(wp)
(Wpred,mpred) = returnpred(White)
print 'WHITE'
print 'Does have white with HARD prob    :%s' %Wpred
#print 'Doesn\'t have white with HARD prob:%s' %mpred
#print 'Does have white with prob         :%s' %Wbelong
#print 'Doesn\'t have white with prob     :%s' %d_belong

# black prediction hard and soft probability
Black = rgbhsv_blacksvm.pred(test2x)
#blp = rgbhsv_blacksvm.pred_probability(test2x)
#(d_belong,BLbelong) = returnpredprob(blp)
(BLpred,mpred) = returnpred(Black)
print 'BLACK'
print 'Does have black with HARD prob    :%s' %BLpred
#print 'Doesn\'t have black with HARD prob:%s' %mpred
#print 'Does have black with prob         :%s' %BLbelong
#print 'Doesn\'t have black with prob     :%s' %d_belong

RGBHSVshirtIds = []
print 'RGB HSV heuristic results:'
RGBHSVshirtIds = detect_heuristic_shirt_color(Rpred,Gpred,Bpred,Wpred,BLpred)
print 'Person wears:'
for s in RGBHSVshirtIds:
	if s==1.1:
		print 'RED'
	elif s==1.15:
		print 'MAYBE RED'
	elif s==1.2:
		print 'GREEN'
	elif s==1.25:
		print 'MAYBE GREEN'
	elif s==1.3:
		print 'BLUE'
	elif s==1.35:
		print 'MAYBE BLUE'
	elif s==1.4:
		print 'WHITE'
	elif s==1.45:
		print 'MAYBE WHITE'
	elif s==1.5:
		print 'BLACK'
	elif s==1.55:
		print 'MAYBE BLACK'
	else:
		print 'System can\'t decide...'
print 'shirt.'

###----------###

print '\nOK.'

# end-of_program #