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
This program can detect in real time the shirt's and hair' color of a person and
also can detect if he wears or not eyeglasses and hat.

There are Region of Interests (ROIs) for collecting these data. More specifically, we have 
drawn a shirt ROI, a hat ROI, a hair ROI and an eyes ROI. We devide every one of these ROIs in
6 sub ROIs with the same dimensions in order to collect more features. 

The user can trigger the detection in real time and also he can load a video and detect from it.

It creates one shirt file, one hair file, one hat file and one eyes file. After testing the data in 
each of these files, the program deletes them.

It loads the trained libsvms and perceptrons for each ROI (shirt,hair etc.) and uses the prediction()
function in order to detect the correct answer.

Important note 1: the hairs ROI doesn't collect any data, due to the fact that we didn't have enough data for 
training the libsvm. So the code is commented, but if you can find data and train your classifier you can
uncommented the code and run the program. Also, we provide functions for hat's color detection, but you have 
to create your own classifier first.

Important note 2: for shirt's color detection we use a heuristic algorithm, because of the poor lighting conditions 
in our testing area. If you have a fixed setup, then you can create your own heuristic algorithm or ignore it at all and
depend your decision to the prediction() function for every libsvm.

For more details take a look at the documentation. 
'''

#######################
''' Functions' Side '''
#######################

'''
Counts all the 1s and -1s in a given prediction array

Arguments: prediction array

Returns: the sum for 1s and the sum for -1s
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
Calculates the prediction soft probability of a libsvm.
The first column represent the negative probability and
the second column represents the positive probability.

Arguments: array, which correspondes to the produced array by 
		   the pred_probability() function

Returns: d_belong: the probability that the test set doesn't belong
	               to the given LibSvm
	     belong: the probability that the test set does belong to 
	     	     the given LibSvm
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
Calculates RGB Histograms of a RGB image

Arguments: RGBimage,bins: the number of bins that you want to 
		   calculate.(1-256)

Returns: histR,histG,histB, which are the histogram values for 
         every Red, Green and Blue

Provided by Dr Theodore Giannakopoulos
'''
def getRGBHistograms(RGBimage):
	# compute histograms:
	[histR, bin_edges] = np.histogram(RGBimage[:,:,0], bins=10)
	[histG, bin_edges] = np.histogram(RGBimage[:,:,1], bins=10)
	[histB, bin_edges] = np.histogram(RGBimage[:,:,2], bins=10)
	
	# normalize histograms:
	histR = histR.astype(float); histR = histR / np.sum(histR);
	histG = histG.astype(float); histG = histG / np.sum(histG);
	histB = histB.astype(float); histB = histB / np.sum(histB);
	return (histR, histG, histB)

'''
Calculates HSV Histograms of a HSV image

Arguments: HSVimage,bins: the number of bins that you want to 
		   calculate.(1-256)

Returns: histR,histG,histB, which are the histogram values for 
         every Red, Green and Blue

Provided by Dr Theodore Giannakopoulos
'''
def getHSVHistograms(HSVimage):
	# compute histograms:
	[histH, bin_edges] = np.histogram(HSVimage[:,:,0], bins=10)
	[histS, bin_edges] = np.histogram(HSVimage[:,:,1], bins=10)
	[histV, bin_edges] = np.histogram(HSVimage[:,:,2], bins=10)
	
	# normalize histograms:
	histH = histH.astype(float); histH = histH / np.sum(histH);
	histS = histS.astype(float); histS = histS / np.sum(histS);
	histV = histV.astype(float); histV = histV / np.sum(histV);
	return (histH, histS, histV)

'''
Creates a csv file and writes 'eyes' data in it

Arguments: name: the preferable name for the file
           DLE,DRE,LCE,RCE,LUE,RUE: the list of data that correspond to 
           each eye sub roi.            
'''
def write_eyes_file(name,DLE,DRE,LCE,RCE,LUE,RUE):
	with open(name, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=';',quotechar='', quoting=csv.QUOTE_NONE)
		a=[DLE]
		b=[DRE]
		c=[LCE]
		d=[RCE]
		e=[LUE]
		f=[RUE]
		writer.writerow(a+b+c+d+e+f)

'''
Creates a csv file and writes 'shirt' data in it

Arguments: name: the preferable name for the file
	       R1,G1,B1,H1,S1,V1,R2,G2,B2,H2,S2,V2,R3,G3,B3,H3,S3,V3,
	       R4,G4,B4,H4,S4,V4,R5,G5,B5,H5,S5,V5,R6,G6,B6,H6,S6,V6:
	       the values for every sub shirt roi. i.e. R1,G1,B1 correspond
	       to the R,G,B histogram values for the 1st shirt sub roi
'''
def write_file(name,R1,G1,B1,H1,S1,V1,R2,G2,B2,H2,S2,V2,R3,G3,B3,H3,S3,V3,R4,G4,B4,H4,S4,V4,R5,G5,B5,H5,S5,V5,R6,G6,B6,H6,S6,V6):
	with open(name, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=';',quotechar='', quoting=csv.QUOTE_NONE)		
		writer.writerow(R1.tolist()+G1.tolist()+B1.tolist()+H1.tolist()+S1.tolist()+V1.tolist()+R2.tolist()+G2.tolist()+B2.tolist()+H2.tolist()+S2.tolist()+V2.tolist()+R3.tolist()+G3.tolist()+B3.tolist()+H3.tolist()+S3.tolist()+V3.tolist()+R4.tolist()+G4.tolist()+B4.tolist()+H4.tolist()+S4.tolist()+V4.tolist()+R5.tolist()+G5.tolist()+B5.tolist()+H5.tolist()+S5.tolist()+V5.tolist()+R6.tolist()+G6.tolist()+B6.tolist()+H6.tolist()+S6.tolist()+V6.tolist())

'''
The 3 functions below are responsible for face detection.

Provided by Dr Theodore Giannakopoulos
'''
def intersect_rectangles(r1, r2):
	x11 = r1[0]; y11 = r1[1]; x12 = r1[0]+r1[2]; y12 = r1[1]+r1[3];
	x21 = r2[0]; y21 = r2[1]; x22 = r2[0]+r2[2]; y22 = r2[1]+r2[3];
		
	X1 = max(x11, x21); X2 = min(x12, x22);
	Y1 = max(y11, y21); Y2 = min(y12, y22);

	W = X2 - X1
	H = Y2 - Y1
	if (H>0) and (W>0):
		E = W * H;
	else:
		E = 0.0;
	Eratio = 2.0*E / (r1[2]*r1[3] + r2[2]*r2[3])
	return Eratio

def initialize_face():
	cascadeFrontal = cv2.cv.Load(HAAR_CASCADE_PATH_FRONTAL);
	storage = cv2.cv.CreateMemStorage()
	return (cascadeFrontal, storage)

def detect_faces(image, cascadeFrontal, storage):
	facesFrontal = []; 	

	detectedFrontal = cv2.cv.HaarDetectObjects(image, cascadeFrontal, storage, 1.3, 2, cv2.cv.CV_HAAR_DO_CANNY_PRUNING, (image.width/10,image.width/10))
	
	if detectedFrontal:
		for (x,y,w,h),n in detectedFrontal:
			facesFrontal.append((x,y,w,h))

	# remove overlaps:
	while (1):
		Found = False
		for i in range(len(facesFrontal)):
			for j in range(len(facesFrontal)):
				if i != j:
					interRatio = intersect_rectangles(facesFrontal[i], facesFrontal[j])
					if interRatio>0.3:
						Found = True;
						del facesFrontal[i]
						break;
			if Found:
				break;

		if not Found:	# not a single overlap has been detected -> exit loop
			break;
	return (facesFrontal)
	
'''
Summarizes all the values for a fiven array. In our case we use it for counting all the values
for canny edged arrays and contour arrays

Arguments: an array with the edges for a given sub roi

Returns: the sum of all the values in the array
'''
def returnEdges(edges):
	finaledges=0
	for e in edges:
		sumedges=0
		for i in e:
			sumedges+=i
		finaledges+=sumedges
	return finaledges

'''
Finds the biggest contour for a given image. In our case we give each subroi and we want to find the 
biggest of all the contours in that roi

Arguments: sub hat roi image

Returns: outline which is the values for the biggest contours. If the algorithm can't find the biggest
		 contour in the sub hat roi, then we return an all zero array
'''
def findbigcontour(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 11, 7)
	outline = np.zeros(gray.shape, dtype = "uint8")
	(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	if not cnts:
		outline = np.zeros(gray.shape, dtype = "uint8")
		return outline
	else:
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
		cv2.drawContours(outline, [cnts], -1, 255, -1)
		return outline

'''
Creates a csv file and writes 'hat' data in it

Arguments: the name of the file and the summarized canny edged values for every subhat roi 
'''
def write_hat_file(name,DLE,DRE,LCE,RCE,LUE,RUE,hatfore1,hatfore2,hatfore3,hatfore4,hatfore5,hatfore6):
	with open(name, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=';',quotechar='', quoting=csv.QUOTE_NONE)
		a=[DLE]
		b=[DRE]
		c=[LCE]
		d=[RCE]
		e=[LUE]
		f=[RUE]
		g=[hatfore1] 
		h=[hatfore2]
		i=[hatfore3]
		j=[hatfore4]
		k=[hatfore5]
		l=[hatfore6]
		writer.writerow(a+b+c+d+e+f+g+h+i+j+k+l)


'''
Founds the object that has differrent color from the background and it paints it white.
In our case we use it for hat detection. The hat must not be the color of your background, in our case
white.

Arguments: the one of the six smaller rois that we have created from the original hat roi

Returns: an array with the values that correspond the detected area
'''
def getForeground(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	th, im_th = cv2.threshold(gray, 85, 100, cv2.THRESH_BINARY_INV)
	im_floodfill = im_th.copy()
	h, w = im_th.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)	    			
	floodhatroi = cv2.floodFill(im_floodfill, mask, (0,0), 255)
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)
	im_out = im_th | im_floodfill_inv
	#cv2.imshow("Thresholded Image", im_th)
	#cv2.imshow("Floodfilled Image", im_floodfill)
	#cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
	return im_out

'''
(currently not used)
Creates a csv file and writes 'hat' color data in it

Arguments: name: the name of the file and the R,G,B,H,S,V values for every frame 

def write_hat_color_file(name,R1,G1,B1,H1,S1,V1,R2,G2,B2,H2,S2,V2,R3,G3,B3,H3,S3,V3,R4,G4,B4,H4,S4,V4,R5,G5,B5,H5,S5,V5,R6,G6,B6,H6,S6,V6):
	with open(name, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=';', quotechar='', quoting=csv.QUOTE_NONE)		
		writer.writerow(R1.tolist()+G1.tolist()+B1.tolist()+H1.tolist()+S1.tolist()+V1.tolist()+R2.tolist()+G2.tolist()+B2.tolist()+H2.tolist()+S2.tolist()+V2.tolist()+R3.tolist()+G3.tolist()+B3.tolist()+H3.tolist()+S3.tolist()+V3.tolist()+R4.tolist()+G4.tolist()+B4.tolist()+H4.tolist()+S4.tolist()+V4.tolist()+R5.tolist()+G5.tolist()+B5.tolist()+H5.tolist()+S5.tolist()+V5.tolist()+R6.tolist()+G6.tolist()+B6.tolist()+H6.tolist()+S6.tolist()+V6.tolist())
'''

'''
Detects the initial values that color libsvms provide

Arguments: the five color libsvms and the test set

Returns: the five prediction arrays for each color
'''
def detect_shirt_color(redsvm,greensvm,bluesvm,whitesvm,blacksvm,shirttest_):
	
	Red = redsvm.pred(shirttest_)     # red probability
	(Rpred,mpred) = returnpred(Red)	  # calculates the % of hard probability	

	Green = greensvm.pred(shirttest_) # green probability
	(Gpred,mpred) = returnpred(Green) # calculates the % of hard probability	

	Blue = bluesvm.pred(shirttest_)   # blue probability
	(Bpred,mpred) = returnpred(Blue)  # calculates the % of hard probability 

	White = whitesvm.pred(shirttest_) # white probability
	(Wpred,mpred) = returnpred(White) # calculates the % of hard probability	

	Black = blacksvm.pred(shirttest_) # black probability
	(BLpred,mpred) = returnpred(Black)# calculates the % of hard probability

	return (Rpred,Gpred,Bpred,Wpred,BLpred)

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

'''
Detects the values for every hat LibSvm

Arguments: the 3 hat libsvms and the test set

Returns: the 3 pred arrays for every hat libsvm
'''
def detect_hair_color(rgbhsv_brownhairsvm,rgbhsv_blondehairsvm,rgbhsv_blackhairsvm,hairtest_):
	#brown probability
	Brown = brown.pred(hairtest_)
	(predBrown,probBrown) = returnpred(Brown)
	
	#blonde probability
	Blonde = blonde.pred(hairtest_)
	(predBlonde,probBlonde) = returnpred(Blonde)
	
	#black probability
	Black = black.pred(hairtest_)
	(predBlack,probBlack) = returnpred(Black)

	return (predBrown,predBlonde,predBlack)

'''
Detects if a person wears or not eyeglasses

Arguments: eyeperceptron: the name of the Perceptron ml algorithm which we have
           previously trained
           eyetest_: the test set in order to check it with Perceptron

Returns: EYE_ID: a list of all the ids that correspond to a value of the Perceptron
EYE_ID = 3.1 : Person wears eyeglasses
EYE_ID = 3.15: Person maybe wears eyeglasses
EYE_ID = 3.2 : Person doesn't wear eyeglasses
EYE_ID = 3.25: Person maybe doesn't wear eyeglasses
EYE_ID = 0   : Algorithm can't detect if the Person wears or not eyeglasses
'''
def detect_eyeglasses(eyeperceptron,eyetest_):

	EYE_ID=[]   
	one=False
	two=False
	three=False
	four=False
	#results
	RES = eyeperceptron.pred(eyetest_) # returns the prediction array of a given test set
	wears=0
	doesntwear=0
	for r in RES:                      # the RES array contains 1 and -1 
		if r==1:
			wears+=abs(r)              # holds all the 1s
		elif r==-1:
			doesntwear+=abs(r)         # holds all the -1s
		else:
			print 'NOT OK AT ALL !!!'  # something went wrong with the eyeperseptron result
	print '%s' %wears
	print '%s' %doesntwear
	if (wears>doesntwear and (wears-doesntwear)>=20):    # if the 1s > -1s and the abstract between them are >= than 20 
		EYE_ID.append(3.1)                               # Person wears eyeglasses for sure
		one=True
	if (abs(wears-doesntwear)<20 and wears>doesntwear): # if the 1s > -1s and the abstract between them are < than 20
		EYE_ID.append(3.15)                              # Person maybe wears eyeglasses
		two=True
	if (abs(doesntwear-wears)<20 and doesntwear>wears): # if the -1s > 1s and the abstract between them are < than 20
		EYE_ID.append(3.25)								 # Person maybe doesn't wear eyeglasses
		three=True
	if(doesntwear>wears and (doesntwear-wears)>=20):     # if the -1s > 1s and the abstract between them are >= than 20
		EYE_ID.append(3.2)                               # Person doesn't wear eyeglasses for sure
		four=True		
	if(one==two==three==four==False):                    # if none of the above are True then
		EYE_ID.append(0)                                 # the algorithm can't decide
	else:
		print '\nEverything went OK.'
	
	return EYE_ID                                        

'''
Detects if a person wears or not a specific type of hat.

Arguments: hatperceptron: the name of the Perceptron ml algorithm which we have
           previously trained
           hattest_ : the test set in order to check it with Perceptron

Returns: HAT_ID: a list of all the ids that correspond to a value of the Perceptron
HAT_ID=4.10 : Person wears glasses.
HAT_ID=4.15 : Person maybe wears glasses.
HAT_ID=4.20 : Person doesn't wear glasses.
HAT_ID=4.25 : Person maybe doesn't wear glasses.
HAT_ID=0    : Algorithm can't detect if the Person wears or not eyeglasses
'''
def detect_hats(hatperceptron,hattest_):
	
	HAT_ID=[]
	one=False
	two=False
	three=False
	four=False

	RES = hatperceptron.pred(hattest_) # returns the prediction array of a given test set
	wears=0
	doesntwear=0
	for r in RES:                      # the RES array contains 1 and -1 
		if r==1:
			wears+=abs(r)              # holds all the 1s
		elif r==-1:
			doesntwear+=abs(r)         # holds all the -1s
		else:
			print 'NOT OK AT ALL !!!'  # something went wrong with the eyeperseptron result
	print '\n%s' %wears
	print '%s' %doesntwear
	if (wears>doesntwear and (wears-doesntwear)>=20):                               # if the 1s > -1s and the abstract between them are >= than 20 
		HAT_ID.append(4.1)                               # Person wears eyeglasses for sure
		one=True
	if (abs(wears-doesntwear)<20 and wears>doesntwear): # if the 1s > -1s and the abstract between them are < than 20
		HAT_ID.append(4.15)                              # Person maybe wears eyeglasses
		two=True
	if (abs(doesntwear-wears)<20 and doesntwear>wears): # if the -1s > 1s and the abstract between them are < than 20
		HAT_ID.append(4.25)                              # Person maybe doesn't wear eyeglasses
		three=True
	if(doesntwear>wears and (doesntwear-wears)>=20):                                # if the -1s > 1s and the abstract between them are >= than 20
		HAT_ID.append(4.2)                               # Person doesn't wear eyeglasses for sure
		four=True
	if(one==two==three==four==False):                    # if none of the above are True then
		HAT_ID.append(0)                                 # the algorithm can't decide
	else:
		print '\nEverything went OK.'

	return HAT_ID

'''
Returns the Color Ids from rgb and rgbhsv svms. If the two ways have similar 
Ids, then it returns them, otherwise it returns all the Ids from the two ways.

Arguments: the rgb and rgbhsv arrays with Ids

Returns: 
'''
def returncolorvalues(rgbshirtIds,rgbhsvshirtIds):
	
	temp=set(rgbhsvshirtIds)&set(rgbshirtIds)
	
	if len(temp)>0:
		return temp
	temp=[]
	for r1 in rgbshirtIds:
		temp.append(r1)
	for r2 in rgbhsvshirtIds:
		temp.append(r2)
	return temp

'''-----------'''
#################

#################
''' MAIN CODE '''
#################

font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8) # font in order to show a message on screen
#bins = np.arange(256).reshape(256,1)                          

# pass necessary arguments to the program
ap = argparse.ArgumentParser()
ap.add_argument("-pn"     , "--pn"   , required = True,  help = "Person's name.")
ap.add_argument("-q"      , "--q"       , required = True,  help = "Do you have any video to detect?")
ap.add_argument("-v"      , "--v"       , required = False, help = "Name of the video.")
ap.add_argument("-path"   , "--path"    , required = True,  help = "Path to the opencv folder (i.e. /home/yourname/opencv-2.4.9/ )")
ap.add_argument("-apppath", "--apppath" , required = True,  help = "Path to the app folder (i.e. /home/yourname/appname/ )")
ap.add_argument("-c"      , "--c"       , required = True,  help = "Which camera do you want to be opened? (i.e. 0 means the default camera)")
args = vars(ap.parse_args())

globalpath = args["path"]

HAAR_CASCADE_PATH_FRONTAL = globalpath + "/data/haarcascades/haarcascade_frontalface_default.xml" # loads the face detector cascade classifier

(cascadeFrontal, storage) = initialize_face() # initialize face detector

# Sets video format
fourcc = cv2.cv.CV_FOURCC(*'XVID')
#######################################
''' Creates libsvms and perceptrons '''
#######################################

#SHIRT RGB svms

rgb_redsvm        = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True) # red color
rgb_greensvm      = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True) # green color
rgb_bluesvm       = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True) # blue color
rgb_whitesvm      = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True) # white color
rgb_blacksvm      = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True) # black color
#SHIRT RGB HSV svms
rgbhsv_redsvm     = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True) # red color
rgbhsv_greensvm   = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True) # green color
rgbhsv_bluesvm    = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True) # blue color
rgbhsv_whitesvm   = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True) # white color
rgbhsv_blacksvm   = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True) # black color

#HAIR svms
#rgb_brownhair     = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True) 
#rgb_blackhair     = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
#rgb_blondehair    = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)

# EYE perceptron
eyeperceptron = ml.Perceptron(alpha=0.1, thr=0.05, maxiters=100) # eyes
#HAT perceptron
hatperceptron = ml.Perceptron(alpha=0.1, thr=0.05, maxiters=100) # hat

path = args["apppath"] # path to the app folder

rgb_redsvm  	  = rgb_redsvm.load_model(path  + 'TRAINED_ALGORITHMS/rgb_redsvm.xml')
rgb_greensvm 	  = rgb_greensvm.load_model(path+ 'TRAINED_ALGORITHMS/rgb_greensvm.xml')
rgb_bluesvm  	  = rgb_bluesvm.load_model(path + 'TRAINED_ALGORITHMS/rgb_bluesvm.xml')
rgb_whitesvm 	  = rgb_whitesvm.load_model(path+ 'TRAINED_ALGORITHMS/rgb_whitesvm.xml')
rgb_blacksvm 	  = rgb_blacksvm.load_model(path+ 'TRAINED_ALGORITHMS/rgb_blacksvm.xml')

rgbhsv_redsvm     = rgbhsv_redsvm.load_model(path  + 'TRAINED_ALGORITHMS/rgbhsv_redsvm.xml')
rgbhsv_greensvm   = rgbhsv_greensvm.load_model(path+ 'TRAINED_ALGORITHMS/rgbhsv_greensvm.xml')
rgbhsv_bluesvm    = rgbhsv_bluesvm.load_model(path + 'TRAINED_ALGORITHMS/rgbhsv_bluesvm.xml')
rgbhsv_whitesvm   = rgbhsv_whitesvm.load_model(path+ 'TRAINED_ALGORITHMS/rgbhsv_whitesvm.xml')
rgbhsv_blacksvm   = rgbhsv_blacksvm.load_model(path+ 'TRAINED_ALGORITHMS/rgbhsv_blacksvm.xml')

#rgb_brownhairsvm   = rgb_brownhair.load_model(path  +'TRAINED_ALGORITHMS/rgb_brownhairsvm.xml')
#rgb_blackhairsvm   = rgb_blackhair.load_model(path  +'TRAINED_ALGORITHMS/rgb_blackhairsvm.xml')
#rgb_blondehairsvm  = rgb_blondehair.load_model(path +'TRAINED_ALGORITHMS/rgb_blondehairsvm.xml')

eyeperceptron     = pickle.load(open(path+ 'TRAINED_ALGORITHMS/eyeperceptron_t1.xml','rb'))
hatperceptron     = pickle.load(open(path+ 'TRAINED_ALGORITHMS/hatperceptron_t2.xml','rb'))

###-----------------###

if args["q"]=='true':                       # if you want to detect a recorded video...
	cap = cv2.VideoCapture(args["v"])
elif args["q"]=='false':                    # if you want to detect in real time
	cap = cv2.VideoCapture(int(args["c"]))	
	out = cv2.VideoWriter(args["pn"]+'_video.avi',fourcc, 20.0, (640,480)) # Gives name to the new video
else:
	print 'Wrong input value! Insert \'true\' or \'false\' without quotes.' # if you put something else rather than 'true' or 'false'

print '\n----> If \'OK!\' press \'ESC\' key to start recording.\n'

frame_counter = 0     # counts the frames
processed     = False # if the recording stops processed=True
recording     = False # if 'ESC' is pressed then recording=True
index         = 0     # counts the number of while loops

# Cheat sheet:
# for rois: frame[up:down,left:right]
# for rectangle: rectangles(frames(left,up),(right,down))
while(True):
    
    # Boolean values to synchronize the rectangles
	face_checker  = False
	hair_checker  = False
	shirt_checker = False  
	eye_checker   = False 
	hat_checker   = False

	# Capture frame-by-frame
	ret, frames = cap.read()  

	if(args["q"]=='false'): # If users doesn't have any video, then start recording each frame
		out.write(frames)
		
	#calculates the height and the width of every raw frame
	height, width, channels = frames.shape
     
	# Get faces
	facesFrontal = detect_faces(cv2.cv.fromarray(frames), cascadeFrontal, storage)                 
	
	for f in facesFrontal:	

		#boolean values to synchronize the rectangles
		face_checker  = False
		hair_checker  = False
		shirt_checker = False  
		eye_checker   = False 
		hat_checker   = False						
		
		##################
		# FACE rectangle #
		##################
		cv2.rectangle(frames, (f[0], f[1]+7), (f[0]+f[2],f[1]+f[3]), (0,255,255), 3)
		face_roi = frames[f[1]:f[1]+f[3],f[0]:f[0]+f[2]]
		
		face_height, face_width, channels = face_roi.shape # finds the height and the width of the face roi
		if(face_height>0 and face_width>0):
			face_checker=True	 

		###------------### 
	  
		#################
		# HAT rectangle #
		#################
		cv2.rectangle(frames, (f[0]-15, f[1]+15-height/6), (f[0]+f[2]+15,f[1]-7), (0,255,0), 2)
		hat_roi = frames[f[1]+2+15-height/6:f[1]-2-7,f[0]-15+2:f[0]+f[2]+15-2]
	 		
		hat_height, hat_width, channels = hat_roi.shape	# finds the height and the width of the hair roi	
		
		hat_left  =0
		hat_up    =0
		hat_right =5
		hat_down  =5
	 	 
		if(hat_height>0 and hat_width>0 and (f[1]-height/6)>0 and (f[0]+f[2]+15)>0 and (f[1])>0 and (f[0]-15)>0): # checks if the rectangle actually appears
			hat_checker = True
	    	
			#cv2.imshow('hat_roi',hat_roi) # for debugging

			hat_left  = f[0]-15+2
			hat_right = f[0]+f[2]+15-2
			hat_up    = f[1]+2+15-height/6
			hat_down  = f[1]-2-7
	    	hat_height = hat_up-hat_down
	    	hat_width  = hat_left-hat_right
	    
	    	# draw hat roi rectangles 
	    	#cv2.rectangle(frames,(hat_left,hat_up),(hat_right+(hat_width/2),hat_down+(2*hat_height/3)),(255,0,0),1)              # up left
	    	#cv2.rectangle(frames,(hat_left,hat_up-hat_height/3),(hat_right+(hat_width/2),hat_down+(hat_height/3)),(255,0,255),1) # center left
	    	#cv2.rectangle(frames,(hat_left,hat_up-(2*hat_height/3)),(hat_right+(hat_width/2),hat_down),(0,0,255),1)              # down left
	    	#cv2.rectangle(frames,(hat_left-hat_width/2,hat_up),(hat_right,hat_down+(2*hat_height/3)),(0,100,0),1)                # up right
	    	#cv2.rectangle(frames,(hat_left-hat_width/2,hat_up-(hat_height/3)),(hat_right,hat_down+(hat_height/3)),(0,0,0),1)     # center right
	    	#cv2.rectangle(frames,(hat_left-hat_width/2,hat_up-(2*hat_height/3)),(hat_right,hat_down),(255,255,255),1)            # down right

	    	#split hat roi into 6 sub rois
	    	sub_hat_roi_1 = frames[hat_up+2:hat_down+(2*hat_height/3),hat_left+2:hat_right+(hat_width/2)]              # up left 
	    	sub_hat_roi_2 = frames[hat_up-hat_height/3+2:hat_down+(hat_height/3),hat_left+2:hat_right+(hat_width/2)]   # up center	    
	    	sub_hat_roi_3 = frames[hat_up-(2*hat_height/3)+2:hat_down,hat_left+2:hat_right+(hat_width/2)]              # up right	    
	    	sub_hat_roi_4 = frames[hat_up+2:hat_down+(2*hat_height/3),hat_left-(hat_width/2)+2:hat_right]              # down left
	    	sub_hat_roi_5 = frames[hat_up-(hat_height/3)+2:hat_down+(hat_height/3),hat_left-(hat_width/2)+2:hat_right] # down center
	    	sub_hat_roi_6 = frames[hat_up-(2*hat_height/3)+2:hat_down,hat_left-(hat_width/2)+2:hat_right]              # down right
    			
	    ###------------###

		##################
		# HAIR rectangle #       
		##################
		cv2.rectangle(frames, (f[0]+7, f[1]+10-height/10), (f[0]+f[2]-7,f[1]+7), (0,255,0), 2)
		hair_roi = frames[f[1]+10+2-height/10:f[1]-2+7,f[0]+2+7:f[0]+f[2]-2-7]
	 		
		hair_height, hair_width, channels = hair_roi.shape # Finds the height and the width of the hair roi
		h_up=0
		h_down=5
		h_left=0
		h_right=5
	 	 
		if(hair_height>0 and hair_width>0 and (f[1]+2-height/10)>0 and (f[0]+f[2]-2-20)>0 and (f[1]-2-height/10)>0 and (f[0]+2*f[2]/3-2)>0):#checks if the rectangle actually appears
			hair_checker=True
	    	
			h_left  = f[0]+2+7
			h_right = f[0]+f[2]-2-7
			h_up    = f[1]+2+10-height/10
			h_down  = f[1]-2+7
	    	h_height = h_up-h_down
	    	h_width  = h_left-h_right
	    	
	    	# Draw hair roi rectangles for debugging
	    	#cv2.rectangle(frames,(h_left,h_up),(h_right+(h_width/2),h_down+(2*h_height/3)),(255,0,0),1)            # up left
	    	#cv2.rectangle(frames,(h_left,h_up-h_height/3),(h_right+(h_width/2),h_down+(h_height/3)),(255,0,255),1) # center left
	    	#cv2.rectangle(frames,(h_left,h_up-(2*h_height/3)),(h_right+(h_width/2),h_down),(0,0,255),1)            # down left
	    	#cv2.rectangle(frames,(h_left-h_width/2,h_up),(h_right,h_down+(2*h_height/3)),(0,100,0),1)              # up right
	    	#cv2.rectangle(frames,(h_left-h_width/2,h_up-(h_height/3)),(h_right,h_down+(h_height/3)),(0,0,0),1)     # center right
	    	#cv2.rectangle(frames,(h_left-h_width/2,h_up-(2*h_height/3)),(h_right,h_down),(255,255,255),1)          # down right

	    	# Split hair roi into 6 sub rois 
	    	sub_hair_roi_1 = frames[h_up+2:h_down+(2*h_height/3),h_left+2:h_right+(h_width/2)]            # up left 
	    	sub_hair_roi_2 = frames[h_up-h_height/3+2:h_down+(h_height/3),h_left+2:h_right+(h_width/2)]   # up center	    
	    	sub_hair_roi_3 = frames[h_up-(2*h_height/3)+2:h_down,h_left+2:h_right+(h_width/2)]            # up right	    
	    	sub_hair_roi_4 = frames[h_up+2:h_down+(2*h_height/3),h_left-(h_width/2)+2:h_right]            # down left
	    	sub_hair_roi_5 = frames[h_up-(h_height/3)+2:h_down+(h_height/3),h_left-(h_width/2)+2:h_right] # down center
	    	sub_hair_roi_6 = frames[h_up-(2*h_height/3)+2:h_down,h_left-(h_width/2)+2:h_right]            # down right
	    	
		###-------------###
	 
		###################
		# SHIRT rectangle #
		###################
		cv2.rectangle(frames, (f[0]+30-height/7, f[1]+f[3]+20+height/12), (f[0]+f[2]-30+height/7,f[1]+f[3]+2+height/3), (0,0,255), 2)
		shirt_roi = frames[f[1]+f[3]+20+height/12:f[1]+f[3]+2+height/3, f[0]+30-height/7:f[0]+f[2]-30+height/7]
	 
		# Finds the height and the width of the shirt roi
		shirt_height, shirt_width, channels = shirt_roi.shape

		# Initialize shirt dimensions, in order the program not to crash
		s_left   = 0
		s_up     = 0
		s_right  = 5
		s_down   = 5	 
	 	s_height = 0
	 	s_width  = 0

		if(shirt_height>0 and shirt_width>0): # checks if the rectangle actually appears
		   	shirt_checker=True
	    		    
		   	s_left   = f[0]+30-height/7
		   	s_right  = f[0]+f[2]-30+height/7
		   	s_up     = f[1]+f[3]+20+height/12
		   	s_down   = f[1]+f[3]+2+height/3
		   	s_height = s_up-s_down
		   	s_width  = s_left-s_right
		   	
		   	# Draw shirt roi rectangles for debugging
	    	#cv2.rectangle(frames,(s_left,s_up),(s_right+(s_width/2),s_down+(2*s_height/3)),(255,0,0),1)              # up left	    	
	    	#cv2.rectangle(frames,(s_left,s_up-(s_height/3)),(s_right+(s_width/2),s_down+(s_height/3)),(255,0,255),1) # center left
	    	#cv2.rectangle(frames,(s_left,s_up-(2*s_height/3)),(s_right+(s_width/2),s_down),(0,0,255),1)              # down left
	    	#cv2.rectangle(frames,(s_left-(s_width/2),s_up),(s_right,s_down+(2*s_height/3)),(0,100,0),1)              # up right
	    	#cv2.rectangle(frames,(s_left-(s_width/2),s_up-(s_height/3)),(s_right,s_down+(s_height/3)),(0,0,0),1)     # center right
	    	#cv2.rectangle(frames,(s_left-(s_width/2),s_up-(2*s_height/3)),(s_right,s_down),(255,255,255),1)          # down right

	    	# Split shirt roi into 6 sub rois 
	    	sub_shirt_roi_1 = frames[s_up+2:s_down+(2*s_height/3), s_left+2:s_right+(s_width/2)]            # up left
	    	sub_shirt_roi_2 = frames[s_up-(s_height/3)+2:s_down+(s_height/3), s_left+2:s_right+(s_width/2)] # up center
	    	sub_shirt_roi_3 = frames[s_up-(2*s_height/3)+2:s_down, s_left+2:s_right+(s_width/2)]            # up right
	    	sub_shirt_roi_4 = frames[s_up+2:s_down+(2*s_height/3), s_left-(s_width/2)+2:s_right]            # down left
	    	sub_shirt_roi_5 = frames[s_up-(s_height/3)+2:s_down+(s_height/3), s_left-(s_width/2)+2:s_right] # down center
	    	sub_shirt_roi_6 = frames[s_up-(2*s_height/3)+2:s_down, s_left-(s_width/2)+2:s_right]            # down right
	    	
		###-------------###	 
		
		##################
		# EYES rectangle #
		##################
		cv2.rectangle(frames, (f[0]+8, f[1]+face_height/6+6), (f[0]+f[2]-8,f[1]+f[3]+6-face_height/2), (0,0,0), 1)
		eye_roi = frames[f[1]+face_height/6+6+1:f[1]+f[3]-1+6-face_height/2,f[0]+8+1:f[0]+f[2]-8-1]

		#cv2.imshow('eyes new',eye_roi)      # show eyes roi for debugging
		eye_h,eye_w,eye_chan = eye_roi.shape # find eye roi dimensions

		# Initialize eyes dimensions
		e_left     = f[0]+8
		e_up       = f[1]+2+face_height/6+6
		e_right    = f[0]+f[2]-8
		e_down     = f[1]+f[3]+6-face_height/2
		eye_height = e_up-e_down
		eye_width  = e_left-e_right

		if(eye_w>0 and eye_h>0): # if eye roi appears
			eye_checker=True

			# Draw rectangles for eyes for debugging 		
 			#cv2.rectangle(frames,(e_left,e_up-(2*eye_height/3)),(e_right+(eye_width/2),e_down),(255,0,0),1)                # down-left rectangle
 			#cv2.rectangle(frames,(e_left,e_up-(eye_height/3)),(e_right+(eye_width/2),e_down+(eye_height/3)),(0,255,255),1) # left-center rectangle
 			#cv2.rectangle(frames,(e_left,e_up),(e_right+(eye_width/2),e_down+(2*eye_height/3)),(255,0,255),1)              # up-left rectangle
 			#cv2.rectangle(frames,(e_left-eye_width/2,e_up-(eye_height/3)),(e_right,e_down+(eye_height/3)),(255,255,255),1) # right-center rectangle
 			#cv2.rectangle(frames,(e_left-eye_width/2,e_up),(e_right,e_down+(2*eye_height/3)),(255,255,0),1)                # up-right rectangle
 			#cv2.rectangle(frames,(e_left-eye_width/2,e_up-(2*eye_height/3)),(e_right,e_down),(0,0,255),1)                  # right-down rectangle

 			# All eye rois
 			eyeroi_downleft    = frames[e_up-(2*eye_height/3)+1:e_down,e_left+1:e_right+(eye_width/2)]			
			eyeroi_leftcenter  = frames[e_up-(eye_height/3)+1:e_down+(eye_height/3),e_left+1:e_right+(eye_width/2)]	
			eyeroi_upleft      = frames[e_up+1:e_down+(2*eye_height/3),e_left+1:e_right+(eye_width/2)]			
			eyeroi_rightcenter = frames[e_up-(eye_height/3)+1:e_down+(eye_height/3),e_left-(eye_width/2)+1:e_right]
			eyeroi_rightup     = frames[e_up+1:e_down+(2*eye_height/3),e_left-(eye_width/2)+1:e_right]			
			eyeroi_rightdown   = frames[e_up-(2*eye_height/3)+1:e_down,e_left-(eye_width/2)+1:e_right]		
		
		###------------###			

		if(face_checker==True and hair_checker==True and shirt_checker==True and eye_checker==True and hat_checker==True):
			
			index+=1 # calculates the number of while loops and it can be 
					 # used for telling the program to start recording 
			
			# Shows message whether ALL 5 rois appear in main frame window
	 		cv.PutText(cv.fromarray(frames),"OK :)", (width-90,40),font, (0,255,0))  	  			
	 		
	 		# If you press 'ESC' key or wait ~5 seconds, the recording will start 		
			if cv2.waitKey(1)==27: #or index>150 (add this to the if statement and you can control the time of recording i.e. index>150 means
				recording=True 	    			  # that the recording will start ~5 seconds after the initiation of the program.

	    	if recording==True:

	    		# Just for show...
	    		if ((frame_counter%2)==0):
	    			cv.PutText(cv.fromarray(frames),"Recording", (width-600,40),font, (0,0,255))

	    		#HAIR
	    		# The function below finds the RGB Histogram values for every hair roi
	    		(HR1,HG1,HB1) = getRGBHistograms(sub_hair_roi_1)
	    		(HR2,HG2,HB2) = getRGBHistograms(sub_hair_roi_2)
	    		(HR3,HG3,HB3) = getRGBHistograms(sub_hair_roi_3)
	    		(HR4,HG4,HB4) = getRGBHistograms(sub_hair_roi_4)
	    		(HR5,HG5,HB5) = getRGBHistograms(sub_hair_roi_5)
	    		(HR6,HG6,HB6) = getRGBHistograms(sub_hair_roi_6)
	    			    	    
	    		# SHIRT	    	    
	    		if (sub_shirt_roi_1.shape[0]>0 and sub_shirt_roi_1.shape[1]>0 and
	    			sub_shirt_roi_2.shape[0]>0 and sub_shirt_roi_2.shape[1]>0 and
	    			sub_shirt_roi_3.shape[0]>0 and sub_shirt_roi_3.shape[1]>0 and
	    			sub_shirt_roi_4.shape[0]>0 and sub_shirt_roi_4.shape[1]>0 and
	    			sub_shirt_roi_5.shape[0]>0 and sub_shirt_roi_5.shape[1]>0 and
	    			sub_shirt_roi_6.shape[0]>0 and sub_shirt_roi_6.shape[1]>0):

	    			# The function below finds the RGB Histogram values for every shirt roi
	    			(SR1,SG1,SB1) = getRGBHistograms(sub_shirt_roi_1)
	    			(SR2,SG2,SB2) = getRGBHistograms(sub_shirt_roi_2)
	    			(SR3,SG3,SB3) = getRGBHistograms(sub_shirt_roi_3)
	    			(SR4,SG4,SB4) = getRGBHistograms(sub_shirt_roi_4)
	    			(SR5,SG5,SB5) = getRGBHistograms(sub_shirt_roi_5)
	    			(SR6,SG6,SB6) = getRGBHistograms(sub_shirt_roi_6)	

	    			# Convert BGR to HSV for every shirt roi
    				hsv_sub_shirt_roi_1 = cv2.cvtColor(sub_shirt_roi_1, cv2.COLOR_BGR2HSV)
    				hsv_sub_shirt_roi_2 = cv2.cvtColor(sub_shirt_roi_2, cv2.COLOR_BGR2HSV)
    				hsv_sub_shirt_roi_3 = cv2.cvtColor(sub_shirt_roi_3, cv2.COLOR_BGR2HSV)
    				hsv_sub_shirt_roi_4 = cv2.cvtColor(sub_shirt_roi_4, cv2.COLOR_BGR2HSV)
    				hsv_sub_shirt_roi_5 = cv2.cvtColor(sub_shirt_roi_5, cv2.COLOR_BGR2HSV)
    				hsv_sub_shirt_roi_6 = cv2.cvtColor(sub_shirt_roi_6, cv2.COLOR_BGR2HSV)
	    			(SH1,SS1,SV1) = getHSVHistograms(hsv_sub_shirt_roi_1)
	    			(SH2,SS2,SV2) = getHSVHistograms(hsv_sub_shirt_roi_2)
	    			(SH3,SS3,SV3) = getHSVHistograms(hsv_sub_shirt_roi_3)
	    			(SH4,SS4,SV4) = getHSVHistograms(hsv_sub_shirt_roi_4)
	    			(SH5,SS5,SV5) = getHSVHistograms(hsv_sub_shirt_roi_5)
	    			(SH6,SS6,SV6) = getHSVHistograms(hsv_sub_shirt_roi_6)	        
	    	    
	    	    # EYES
	    	    # The Canny functions find the Canny Edges for every eye roi
	    		downleft_edge    = cv2.Canny(eyeroi_downleft,250,500)
	    		downright_edge   = cv2.Canny(eyeroi_rightdown,250,500)
	    		leftcenter_edge  = cv2.Canny(eyeroi_leftcenter,250,500)
	    		rightcenter_edge = cv2.Canny(eyeroi_rightcenter,250,500)
	    		leftup_edge      = cv2.Canny(eyeroi_upleft,250,500)
	    		rightup_edge     = cv2.Canny(eyeroi_rightup,250,500)

	    		# The function below summarizes the canny edge values for every eye subroi
	    		DLE = returnEdges(downleft_edge)
	    		DRE = returnEdges(downright_edge)
	    		LCE = returnEdges(leftcenter_edge)
	    		RCE = returnEdges(rightcenter_edge)
	    		LUE = returnEdges(leftup_edge)
	    		RUE = returnEdges(rightup_edge)

	    		# HATS
	    		bigcontour1 = 0
	    		bigcontour2 = 0
	    		bigcontour3 = 0
	    		bigcontour4 = 0
	    		bigcontour5 = 0
	    		bigcontour6 = 0
	    		# If the sub hat rois have formulated all correctly and at the same time, 
	    		# then starts calculating the contours for each sub roi
	    		if (sub_hat_roi_1.shape[0]>0 and sub_hat_roi_1.shape[1]>0 and
	    			sub_hat_roi_2.shape[0]>0 and sub_hat_roi_2.shape[1]>0 and
	    			sub_hat_roi_3.shape[0]>0 and sub_hat_roi_3.shape[1]>0 and
	    			sub_hat_roi_4.shape[0]>0 and sub_hat_roi_4.shape[1]>0 and
	    			sub_hat_roi_5.shape[0]>0 and sub_hat_roi_5.shape[1]>0 and
	    			sub_hat_roi_6.shape[0]>0 and sub_hat_roi_6.shape[1]>0):

	    			hat_fore_1 = getForeground(sub_hat_roi_1)
	    			hat_fore_2 = getForeground(sub_hat_roi_2)
	    			hat_fore_3 = getForeground(sub_hat_roi_3)
	    			hat_fore_4 = getForeground(sub_hat_roi_4)
	    			hat_fore_5 = getForeground(sub_hat_roi_5)
	    			hat_fore_6 = getForeground(sub_hat_roi_6)
	    				    			
	    			hatfore1 = returnEdges(hat_fore_1)
	    			hatfore2 = returnEdges(hat_fore_2)
	    			hatfore3 = returnEdges(hat_fore_3)
	    			hatfore4 = returnEdges(hat_fore_4)
	    			hatfore5 = returnEdges(hat_fore_5)
	    			hatfore6 = returnEdges(hat_fore_6)	    						

	    			hat_downleft_edge    = cv2.Canny(sub_hat_roi_1,200,100)
	    			hat_downright_edge   = cv2.Canny(sub_hat_roi_2,200,100)
	    			hat_leftcenter_edge  = cv2.Canny(sub_hat_roi_3,200,100)
	    			hat_rightcenter_edge = cv2.Canny(sub_hat_roi_4,200,100)
	    			hat_leftup_edge      = cv2.Canny(sub_hat_roi_5,200,100)
	    			hat_rightup_edge     = cv2.Canny(sub_hat_roi_6,200,100)

	    			hatedge1 = returnEdges(hat_downleft_edge)
	    			hatedge2 = returnEdges(hat_downright_edge)
	    			hatedge3 = returnEdges(hat_leftcenter_edge)
	    			hatedge4 = returnEdges(hat_rightcenter_edge)
	    			hatedge5 = returnEdges(hat_leftup_edge)
	    			hatedge6 = returnEdges(hat_rightup_edge)

	    			'''
	    			The functions below find the hat's color with RGB and HSV Histograms. 
					You have the ability to create your own hat color classifier and 
					detect the hat's color.	    		
					'''
	    			#(HATR1,HATG1,HATB1)  = getRGBHistograms(sub_hat_roi_1,False)
	    			#(HATR2,HATG2,HATB2)  = getRGBHistograms(sub_hat_roi_2,False)
	    			#(HATR3,HATG3,HATB3)  = getRGBHistograms(sub_hat_roi_3,False)
	    			#(HATR4,HATG4,HATB4)  = getRGBHistograms(sub_hat_roi_4,False)
	    			#(HATR5,HATG5,HATB5)  = getRGBHistograms(sub_hat_roi_5,False)
	    			#(HATR6,HATG6,HATB6)  = getRGBHistograms(sub_hat_roi_6,False)

	    			#hat rois HSV values
	    			#(HATH1,HATS1,HATV1)  = getHSVHistograms(sub_hat_roi_1,False)
	    			#(HATH2,HATS2,HATV2)  = getHSVHistograms(sub_hat_roi_2,False)
	    			#(HATH3,HATS3,HATV3)  = getHSVHistograms(sub_hat_roi_3,False)
	    			#(HATH4,HATS4,HATV4)  = getHSVHistograms(sub_hat_roi_4,False)
	    			#(HATH5,HATS5,HATV5)  = getHSVHistograms(sub_hat_roi_5,False)
	    			#(HATH6,HATS6,HATV6)  = getHSVHistograms(sub_hat_roi_6,False)

	    		# Starts collecting data from 100 frames
	    		if(frame_counter<100):
	      				      			
	      			# hair file
					#write_file('temp_hair.csv',HR1,HG1,HB1,HH1,HS1,HV1,HR2,HG2,HB2,HH2,HS2,HV2,HR3,HG3,HB3,HH3,HS3,HV3,HR4,HG4,HB4,HH4,HS4,HV4,HR5,HG5,HB5,HH5,HS5,HV5,HR6,HG6,HB6,HH6,HS6,HV6)
					
					# shirt RGB file
					write_file('temp_rgbshirt.csv',SR1,SG1,SB1,SR1,SG1,SB1,SR2,SG2,SB2,SR2,SG2,SB2,SR3,SG3,SB3,SR3,SG3,SB3,SR4,SG4,SB4,SR4,SG4,SB4,SR5,SG5,SB5,SR5,SG5,SB5,SR6,SG6,SB6,SR6,SG6,SB6)
					# shirt RGB HSV file
					write_file('temp_rgbhsvshirt.csv',SR1,SG1,SB1,SH1,SS1,SV1,SR2,SG2,SB2,SH2,SS2,SV2,SR3,SG3,SB3,SH3,SS3,SV3,SR4,SG4,SB4,SH4,SS4,SV4,SR5,SG5,SB5,SH5,SS5,SV5,SR6,SG6,SB6,SH6,SS6,SV6)
										
					# eyeglasses file
					write_eyes_file('temp_eyes.csv',DLE,DRE,LCE,RCE,LUE,RUE)
					# hat file
					write_hat_file('temp_hat.csv',hatedge1,hatedge2,hatedge3,hatedge4,hatedge5,hatedge6,hatfore1,hatfore2,hatfore3,hatfore4,hatfore5,hatfore6)
					#hat color file
					#write_hat_color_file('temp_hat_color.csv',HATR1,HATG1,HATB1,HATH1,HATS1,HATV1,HATR2,HATG2,HATB2,HATH2,HATS2,HATV2,HATR3,HATG3,HATB3,HATH3,HATS3,HATV3,HATR4,HATG4,HATB4,HATH4,HATS4,HATV4,HATR5,HATG5,HATB5,HATH5,HATS5,HATV5,HATR6,HATG6,HATB6,HATH6,HATS6,HATV6)

					frame_counter+=1 # counts every frame after recording has been started

	    		if(frame_counter==100):              # if 100 frames have been captured
	    			print 'OK. 100 frames captured.'
	    			processed=True
	    			break                            # exits recording
		

	#end loop for faces
	cv2.imshow('Capturing...', frames)
	if processed==True:
		break	        # exits while loop
	
#end While
cap.release()           # closes the camera
if args["q"]=='false':
	out.release()       # stops recording
cv2.destroyAllWindows() # destroy all windows 

###--------------------------------------------------------------------------------------------------###

print 'Processing. Please wait...'
start = time.time()

###----------------###

################
'''LOAD DATA '''
################

# loads test data csv for shirt detection
rgbshirttest  = np.loadtxt(open('temp_rgbshirt.csv',"rb"),delimiter=";",skiprows=0)
rgbshirttest_ = rgbshirttest[:, :360]

rgbhsvshirttest  = np.loadtxt(open('temp_rgbhsvshirt.csv',"rb"),delimiter=";",skiprows=0)
rgbhsvshirttest_ = rgbhsvshirttest[:, :360]

# loads test data csv for hair detection
#hairtest   = np.loadtxt(open('temp_hair.csv',"rb"),delimiter=";",skiprows=0)
#hairtest_  = hairtest[:, :360]

# loads test data csv for eyes detection
eyetest    = np.loadtxt(open('temp_eyes.csv',"rb"),delimiter=";",skiprows=0)
eyetest_   = eyetest[:, :6]

# loads test data csv for hat detection
hattest    = np.loadtxt(open('temp_hat.csv',"rb"),delimiter=";",skiprows=0)
hattest_   = hattest[:, :12]

#########################
###     SHIRT side    ###
#########################

# rgb shirt results
(Rpred1,Gpred1,Bpred1,Wpred1,BLpred1) = detect_shirt_color(rgb_redsvm,rgb_greensvm,rgb_bluesvm,rgb_whitesvm,rgb_blacksvm,rgbshirttest_)
rgbshirtIds = detect_heuristic_shirt_color(Rpred1,Gpred1,Bpred1,Wpred1,BLpred1)

# rgb hsv shirt results
(Rpred2,Gpred2,Bpred2,Wpred2,BLpred2) = detect_shirt_color(rgbhsv_redsvm,rgbhsv_greensvm,rgbhsv_bluesvm,rgbhsv_whitesvm,rgbhsv_blacksvm,rgbhsvshirttest_)
rgbhsvshirtIds = detect_heuristic_shirt_color(Rpred2,Gpred2,Bpred2,Wpred2,BLpred2)

#########################
###     HAIR side     ###
#########################

# hair results
#(BROWNpred,BLONDEpred,BLACKpred) = detect_hair_color(rgbhsv_brownhairsvm,rgbhsv_blondehairsvm,rgbhsv_blackhairsvm,hairtest_)
#hairIds = detect_heuristic_hair_color(BROWNpred,BLONDEpred,BLACKpred)

###############################
###     EYEGLASSES side     ###
###############################

# eyes results
eyeIds = detect_eyeglasses(eyeperceptron,eyetest_)

#########################
###     HAT side      ###
#########################

# hat results
hatIds = detect_hats(hatperceptron,hattest_)

retval = returncolorvalues(rgbshirtIds,rgbhsvshirtIds)
print 'returned ids:%s' %retval

# the decision goes here...
print '\n###########################################\n'
print 'Person wears:'
counter=0
for s1 in retval:
	if counter>0:
		print ' OR '
	if s1==1.1:
		print 'RED'
	elif s1==1.15:
		print 'MAYBE RED'
	elif s1==1.2:
		print 'GREEN'
	elif s1==1.25:
		print 'MAYBE GREEN'
	elif s1==1.3:
		print 'BLUE'
	elif s1==1.35:
		print 'MAYBE BLUE'
	elif s1==1.4:
		print 'WHITE'
	elif s1==1.45:
		print 'MAYBE WHITE'
	elif s1==1.5:
		print 'BLACK'
	elif s1==1.55:
		print 'MAYBE BLACK'
	else:
		print 'The System can\'t decide...'
		s1=0
	counter+=1
if s1!=0:
	print 'shirt.'

print '\nPerson'
for e in eyeIds:
	if e==3.1:
		print 'WEARS'
	elif e==3.15:
		print 'MAYBE WEARS'
	elif e==3.2:
		print 'DOESN\'T WEAR'
	elif e==3.25:
		print 'MAYBE DOESN\'T WEAR'
	else:
		print 'The System can\'t decide...'
		e=0
if e!=0:
	print 'eyeglasses.'

print '\nPerson'
for h in hatIds:
	if h==4.1:
		print 'WEARS'
	elif h==4.15:
		print 'MAYBE WEARS'
	elif h==4.2:
		print 'DOESN\'T WEAR'
	elif h==4.25:
		print 'MAYBE DOESN\'T WEAR'
	else:
		print 'The System can\'t decide...'
		h=0
if h!=0:
	print 'hat.'

print '\n###########################################\n'

#remove unnecessary files
try: 
	os.remove('temp_rgbshirt.csv')
	os.remove('temp_rgbhsvshirt.csv')
	#os.remove('temp_hair.csv')	
	os.remove('temp_eyes.csv')
	os.remove('temp_hat.csv')
	print 'Files has been successfully removed.\n'
except: pass

# calculates time of detection
end      = time.time()
times = end - start
print '\nTime:%s seconds.\n' %times

print 'OK.'

###-end-of-program-###