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

'''
This program collects all the data from a Person's shirt, hair, eyeglasses and hats.

There are Region of Interests (ROIs) for collecting these data. More specifically, we have 
drawn a shirt ROI, a hat ROI, a hair ROI and an eyes ROI. We devide every one of these ROIs in
6 sub ROIs with the same dimensions in order to collect more features. 

The user can trigger the recording of a video and collect data in real time and also he can
load a video and collect data from it.

We collect data from 100 frames and they are saved in the appropriate csv files with a name, which 
correspondes to the right ROI (shirt,hat, etc.)

For every ROI, we create 3 files, one for testing and two for training:
 -The testing file contains the features from 100 frames.
 -The training files contain the features from 100 frames and also they have an extra column at the end,
  which contains 1s or -1s depending on what we want to use it for. (positive training values or negative training values)

The user can collect data by pressing the 'ESC' key once, when the sign 'OK' appears to the capturing window.	

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
	#print 'Doesn\'t belong with prob:%s' %d_belong

	suma=0
	for i in array[:,1]:
		suma=suma+i

	belong = suma/100
	belong = belong*100
	return (d_belong,belong)
	#print 'Does belong with prob:%s' %belong

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
Creates a csv file and writes 'eyes' data in it with an 
extra column at the end of the file, which contains 1s

Arguments: name: the preferable name for the file
           DLE,DRE,LCE,RCE,LUE,RUE: the list of data that correspond to 
           each eye sub roi.            
'''
def write_eyes_p1_file(name,DLE,DRE,LCE,RCE,LUE,RUE):
	with open(name, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=';',quotechar='', quoting=csv.QUOTE_NONE)
		a=[DLE]
		b=[DRE]
		c=[LCE]
		d=[RCE]
		e=[LUE]
		f=[RUE]
		writer.writerow(a+b+c+d+e+f+['1'])

'''
Creates a csv file and writes 'eyes' data in it with an 
extra column at the end of the file, which contains -1s

Arguments: name: the preferable name for the file
           DLE,DRE,LCE,RCE,LUE,RUE: the list of data that correspond to 
           each eye sub roi.            
'''
def write_eyes_m1_file(name,DLE,DRE,LCE,RCE,LUE,RUE):
	with open(name, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=';',quotechar='', quoting=csv.QUOTE_NONE)
		a=[DLE]
		b=[DRE]
		c=[LCE]
		d=[RCE]
		e=[LUE]
		f=[RUE]
		writer.writerow(a+b+c+d+e+f+['-1'])

'''
Creates a csv file and writes 'shirt' or 'hair' RGB HSV or RGB RGB data in it

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
Creates a csv file and writes 'shirt' or 'hair' RGB S or RGB S data in it

Arguments: name: the preferable name for the file
	       R1,G1,B1,S1,R2,G2,B2,S2,R3,G3,B3,S3,
	       R4,G4,B4,S4,R5,G5,B5,S5,R6,G6,B6,S6:
	       the values for every sub shirt roi. i.e. R1,G1,B1 correspond
	       to the R,G,B histogram values for the 1st shirt sub roi
'''
def s_write_file(name,R1,G1,B1,S1,R2,G2,B2,S2,R3,G3,B3,S3,R4,G4,B4,S4,R5,G5,B5,S5,R6,G6,B6,S6):
	with open(name, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=';',quotechar='', quoting=csv.QUOTE_NONE)		
		writer.writerow(R1.tolist()+G1.tolist()+B1.tolist()+S1.tolist()+R2.tolist()+G2.tolist()+B2.tolist()+S2.tolist()+R3.tolist()+G3.tolist()+B3.tolist()+S3.tolist()+R4.tolist()+G4.tolist()+B4.tolist()+S4.tolist()+R5.tolist()+G5.tolist()+B5.tolist()+S5.tolist()+R6.tolist()+G6.tolist()+B6.tolist()+S6.tolist())

'''
Creates a csv file and writes 'eyes' data in it

Arguments: name: the preferable name for the file
           DLE,DRE,LCE,RCE,LUE,RUE: the list of data that correspond to 
           each eye sub roi.            
'''
def write_glassfile(name,data):
	with open(name, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=';',quotechar='', quoting=csv.QUOTE_NONE)		
		writer.writerow(data.tolist())

'''
Creates a csv file and writes 'eyes' data in it with 
extra column at the end of the file, which contains 1s 

Arguments: name: the preferable name for the file
           DLE,DRE,LCE,RCE,LUE,RUE: the list of data that correspond to 
           each eye sub roi.            
'''
def pos_write_file(name,data):
	with open(name, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=';',quotechar='', quoting=csv.QUOTE_NONE)		
		writer.writerow(data.tolist()+['1'])

'''
Creates a csv file and writes 'eyes' data in it with 
extra column at the end of the file, which contains -1s 

Arguments: name: the preferable name for the file
           DLE,DRE,LCE,RCE,LUE,RUE: the list of data that correspond to 
           each eye sub roi.            
'''
def neg_write_file(name,data):
	with open(name, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=';',quotechar='', quoting=csv.QUOTE_NONE)		
		writer.writerow(data.tolist()+['-1'])


'''
Creates a csv file and writes 'shirt' or 'hair' RGB HSV or RGB RGB data in it with 
extra column at the end of the file, which contains 1s 

Arguments: name: the preferable name for the file
	       R1,G1,B1,H1,S1,V1,R2,G2,B2,H2,S2,V2,R3,G3,B3,H3,S3,V3,
	       R4,G4,B4,H4,S4,V4,R5,G5,B5,H5,S5,V5,R6,G6,B6,H6,S6,V6:
	       the values for every sub shirt roi. i.e. R1,G1,B1 correspond
	       to the R,G,B histogram values for the 1st shirt sub roi
'''
def write_file_plus_one(name,R1,G1,B1,H1,S1,V1,R2,G2,B2,H2,S2,V2,R3,G3,B3,H3,S3,V3,R4,G4,B4,H4,S4,V4,R5,G5,B5,H5,S5,V5,R6,G6,B6,H6,S6,V6):
	with open(name, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=';',quotechar='', quoting=csv.QUOTE_NONE)		
		writer.writerow(R1.tolist()+G1.tolist()+B1.tolist()+H1.tolist()+S1.tolist()+V1.tolist()+R2.tolist()+G2.tolist()+B2.tolist()+H2.tolist()+S2.tolist()+V2.tolist()+R3.tolist()+G3.tolist()+B3.tolist()+H3.tolist()+S3.tolist()+V3.tolist()+R4.tolist()+G4.tolist()+B4.tolist()+H4.tolist()+S4.tolist()+V4.tolist()+R5.tolist()+G5.tolist()+B5.tolist()+H5.tolist()+S5.tolist()+V5.tolist()+R6.tolist()+G6.tolist()+B6.tolist()+H6.tolist()+S6.tolist()+V6.tolist()+['1'])

'''
Creates a csv file and writes 'shirt' or 'hair' RGB HSV or RGB RGB data in it with 
extra column at the end of the file, which contains -1s 

Arguments: name: the preferable name for the file
	       R1,G1,B1,H1,S1,V1,R2,G2,B2,H2,S2,V2,R3,G3,B3,H3,S3,V3,
	       R4,G4,B4,H4,S4,V4,R5,G5,B5,H5,S5,V5,R6,G6,B6,H6,S6,V6:
	       the values for every sub shirt roi. i.e. R1,G1,B1 correspond
	       to the R,G,B histogram values for the 1st shirt sub roi
'''
def write_file_minus_one(name,R1,G1,B1,H1,S1,V1,R2,G2,B2,H2,S2,V2,R3,G3,B3,H3,S3,V3,R4,G4,B4,H4,S4,V4,R5,G5,B5,H5,S5,V5,R6,G6,B6,H6,S6,V6):
	with open(name, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=';',quotechar='', quoting=csv.QUOTE_NONE)		
		writer.writerow(R1.tolist()+G1.tolist()+B1.tolist()+H1.tolist()+S1.tolist()+V1.tolist()+R2.tolist()+G2.tolist()+B2.tolist()+H2.tolist()+S2.tolist()+V2.tolist()+R3.tolist()+G3.tolist()+B3.tolist()+H3.tolist()+S3.tolist()+V3.tolist()+R4.tolist()+G4.tolist()+B4.tolist()+H4.tolist()+S4.tolist()+V4.tolist()+R5.tolist()+G5.tolist()+B5.tolist()+H5.tolist()+S5.tolist()+V5.tolist()+R6.tolist()+G6.tolist()+B6.tolist()+H6.tolist()+S6.tolist()+V6.tolist()+['-1'])

'''
Creates a csv file and writes 'shirt' or 'hair' RGB S or RGB S data in it with 
extra column at the end of the file, which contains 1s 

Arguments: name: the preferable name for the file
	       R1,G1,B1,S1,R2,G2,B2,S2,R3,G3,B3,S3,
	       R4,G4,B4,S4,R5,G5,B5,S5,R6,G6,B6,S6:
	       the values for every sub shirt roi. i.e. R1,G1,B1 correspond
	       to the R,G,B histogram values for the 1st shirt sub roi
'''
def s_write_file_plus_one(name,R1,G1,B1,S1,R2,G2,B2,S2,R3,G3,B3,S3,R4,G4,B4,S4,R5,G5,B5,S5,R6,G6,B6,S6):
	with open(name, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=';',quotechar='', quoting=csv.QUOTE_NONE)		
		writer.writerow(R1.tolist()+G1.tolist()+B1.tolist()+S1.tolist()+R2.tolist()+G2.tolist()+B2.tolist()+S2.tolist()+R3.tolist()+G3.tolist()+B3.tolist()+S3.tolist()+R4.tolist()+G4.tolist()+B4.tolist()+S4.tolist()+R5.tolist()+G5.tolist()+B5.tolist()+S5.tolist()+R6.tolist()+G6.tolist()+B6.tolist()+S6.tolist()+['1'])

'''
Creates a csv file and writes 'shirt' or 'hair' RGB S or RGB S data in it with 
extra column at the end of the file, which contains -1s 

Arguments: name: the preferable name for the file
	       R1,G1,B1,S1,R2,G2,B2,S2,R3,G3,B3,S3,
	       R4,G4,B4,S4,R5,G5,B5,S5,R6,G6,B6,S6:
	       the values for every sub shirt roi. i.e. R1,G1,B1 correspond
	       to the R,G,B histogram values for the 1st shirt sub roi
'''
def s_write_file_minus_one(name,R1,G1,B1,S1,R2,G2,B2,S2,R3,G3,B3,S3,R4,G4,B4,S4,R5,G5,B5,S5,R6,G6,B6,S6):
	with open(name, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter=';',quotechar='', quoting=csv.QUOTE_NONE)		
		writer.writerow(R1.tolist()+G1.tolist()+B1.tolist()+S1.tolist()+R2.tolist()+G2.tolist()+B2.tolist()+S2.tolist()+R3.tolist()+G3.tolist()+B3.tolist()+S3.tolist()+R4.tolist()+G4.tolist()+B4.tolist()+S4.tolist()+R5.tolist()+G5.tolist()+B5.tolist()+S5.tolist()+R6.tolist()+G6.tolist()+B6.tolist()+S6.tolist()+['-1'])

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
		return outline,cnts
	else:
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
		#cv2.drawContours(outline, [cnts], -1, 255, -1)
		return outline,cnts

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
Creates a csv file and writes 'hat' data in it with 
extra column at the end of the file, which contains 1s 

Arguments: the name of the file and the summarized canny edged values for every subhat roi 
'''
def write_hat_p1_file(name,DLE,DRE,LCE,RCE,LUE,RUE,hatfore1,hatfore2,hatfore3,hatfore4,hatfore5,hatfore6):
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
		writer.writerow(a+b+c+d+e+f+g+h+i+j+k+l+['1'])

'''
Creates a csv file and writes 'hat' data in it with 
extra column at the end of the file, which contains -1s 

Arguments: the name of the file and the summarized canny edged values for every subhat roi 
'''
def write_hat_m1_file(name,DLE,DRE,LCE,RCE,LUE,RUE,hatfore1,hatfore2,hatfore3,hatfore4,hatfore5,hatfore6):
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
		writer.writerow(a+b+c+d+e+f+g+h+i+j+k+l+['-1'])

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
###-----------###

#################
### MAIN CODE ###
#################

ap = argparse.ArgumentParser()
ap.add_argument("-pn" , "--pn"   , required = True,  help = "Person's name.")
ap.add_argument("-gla", "--gla"  , required = True,  help = "'true' if the person wears glasses and 'false' if the person doesn't wear. true and false without quotes.")
ap.add_argument("-hc" , "--hc"   , required = True,  help = "Hair' color.")
ap.add_argument("-sc" , "--sc"   , required = True,  help = "Shirt's color.")
ap.add_argument("-hat", "--hat"  , required = True,  help = "'true' if the person wears hat and 'false' if the person doesn't wear. true and false without quotes.")
ap.add_argument("-q"  , "--q"    , required = True,  help = "Do you have any video?")
ap.add_argument("-v"  , "--v"    , required = False, help = "The name of the video with its postfix.(i.e.'.avi')")
ap.add_argument("-path", "--path", required = True,  help = "The path to the opencv folder.")
ap.add_argument("-c"   , "--c"   , required = True,  help = "Which camera do you want to be opened? (i.e. 0 means the default camera)")
args = vars(ap.parse_args())

globalpath = args["path"]

# Loads face cascade classifier
HAAR_CASCADE_PATH_FRONTAL = globalpath + "/data/haarcascades/haarcascade_frontalface_default.xml"
bins = np.arange(256).reshape(256,1)

# Necessary for writing message to window
font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8)

# Initializes face detector
(cascadeFrontal, storage) = initialize_face()

# Sets video format
fourcc = cv2.cv.CV_FOURCC(*'XVID')

if args["q"]=='true':                  # If you have a video to collect data:
	cap = cv2.VideoCapture(args["v"])  # Loads the user's video
elif args["q"]=='false':               # else if you want real time collection:
	cap = cv2.VideoCapture(int(args["c"]))  # Starts recording
	out = cv2.VideoWriter(args["pn"]+'_'+args["sc"]+'_'+args["gla"]+'_'+args["hc"]+'.avi',fourcc, 20.0, (1280,720)) # Gives name to the new video
else:
	print 'Wrong input value! Insert \'true\' or \'false\' without quotes.' # Warning message for wrong values

print 'If \'OK!\' press \'ESC\' key to start recording.'

frame_counter = 0     # counts the frames
processed     = False # if the recording stops processed=True
recording     = False # if 'ESC' is pressed then recording=True
index         = 0     # counts the number of while loops
height        = 1
width         = 1

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

	# Calculates the height and the width of every raw frame
	height, width, channels = frames.shape
     
	# Get faces
	facesFrontal = detect_faces(cv2.cv.fromarray(frames), cascadeFrontal, storage)                 
	
	for f in facesFrontal:													   		
				
		##################
		# FACE rectangle #
		##################
		cv2.rectangle(frames, (f[0], f[1]+7), (f[0]+f[2],f[1]+f[3]), (0,255,255), 3)
		face_roi = frames[f[1]:f[1]+f[3],f[0]:f[0]+f[2]]
		
		face_height, face_width, channels = face_roi.shape # Finds the height and the width of the face roi
		if(face_height>0 and face_width>0):
			face_checker=True	 

		###------------### 
	  	
		#################
		# HAT rectangle #
		#################
		cv2.rectangle(frames, (f[0]-15, f[1]+15-height/6), (f[0]+f[2]+15,f[1]-7), (0,255,0), 2)
		hat_roi = frames[f[1]+2+15-height/6:f[1]-2-7,f[0]-15+2:f[0]+f[2]+15-2]
	 		 	
		hat_height, hat_width, channels = hat_roi.shape	# finds the height and the width of the face roi	
	
		hat_left  =0
		hat_up    =0
		hat_right =5
		hat_down  =5
	 	 
		if(hat_height>0 and hat_width>0 and (f[1]+15-height/6)>0 and (f[0]+f[2]+15)>0 and (f[1])>0 and (f[0]-15)>0):#checks if the rectangle actually appears
			hat_checker = True
	    	
			#cv2.imshow('hat_roi',hat_roi) # for debugging

			hat_left  = f[0]-15+2
			hat_right = f[0]+f[2]+15-2
			hat_up    = f[1]+2+15-height/6
			hat_down  = f[1]-2-7
	    	hat_height = hat_up-hat_down
	    	hat_width  = hat_left-hat_right

	    	# Draw hat roi rectangles 
	    	#cv2.rectangle(frames,(hat_left,hat_up),(hat_right+(hat_width/2),hat_down+(2*hat_height/3)),(255,0,0),1)              # up left
	    	#cv2.rectangle(frames,(hat_left,hat_up-hat_height/3),(hat_right+(hat_width/2),hat_down+(hat_height/3)),(255,0,255),1) # center left
	    	#cv2.rectangle(frames,(hat_left,hat_up-(2*hat_height/3)),(hat_right+(hat_width/2),hat_down),(0,0,255),1)              # down left
	    	#cv2.rectangle(frames,(hat_left-hat_width/2,hat_up),(hat_right,hat_down+(2*hat_height/3)),(0,100,0),1)                # up right
	    	#cv2.rectangle(frames,(hat_left-hat_width/2,hat_up-(hat_height/3)),(hat_right,hat_down+(hat_height/3)),(0,0,0),1)     # center right
	    	#cv2.rectangle(frames,(hat_left-hat_width/2,hat_up-(2*hat_height/3)),(hat_right,hat_down),(255,255,255),1)            # down right

	    	# Split hair roi into 6 sub rois 
	    	sub_hat_roi_1 = frames[hat_up+2:hat_down+(2*hat_height/3),hat_left+2:hat_right+(hat_width/2)]              #up left 
	    	sub_hat_roi_2 = frames[hat_up-hat_height/3+2:hat_down+(hat_height/3),hat_left+2:hat_right+(hat_width/2)]   #up center	    
	    	sub_hat_roi_3 = frames[hat_up-(2*hat_height/3)+2:hat_down,hat_left+2:hat_right+(hat_width/2)]              #up right	    
	    	sub_hat_roi_4 = frames[hat_up+2:hat_down+(2*hat_height/3),hat_left-(hat_width/2)+2:hat_right]              #down left
	    	sub_hat_roi_5 = frames[hat_up-(hat_height/3)+2:hat_down+(hat_height/3),hat_left-(hat_width/2)+2:hat_right] #down center
	    	sub_hat_roi_6 = frames[hat_up-(2*hat_height/3)+2:hat_down,hat_left-(hat_width/2)+2:hat_right]              #down right
	    		    	
	    ###------------###

		##################
		# HAIR rectangle #       
		##################	 						
		#cv2.rectangle(frames, (f[0]+7, f[1]+10-height/10), (f[0]+f[2]-7,f[1]+7), (0,255,0), 2)
		hair_roi = frames[f[1]+10+2-height/10:f[1]-2+7,f[0]+2+7:f[0]+f[2]-2-7]
	 		
		hair_height, hair_width, channels = hair_roi.shape # Finds the height and the width of the hair roi
		h_up    =0
		h_down  =5
		h_left  =0
		h_right =5
	 	 
		if(hair_height>0 and hair_width>0 and (f[1]+2-height/10)>0 and (f[0]+f[2]-2-20)>0 and (f[0]+2+20)>0 and (f[1]+7)>0):#checks if the rectangle actually appears
			hair_checker=True
	    	
			h_left  = f[0]+2+7
			h_right = f[0]+f[2]-2-7
			h_up    = f[1]+10+2-height/10
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

	    	# Split hair roi into 6 sub rois // frame[up:down,left:right]
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
		s_left   = 0
		s_up     = 0
		s_right  = 5
		s_down   = 5	 
	 	s_height = 0
	 	s_width  = 0
		if(shirt_height>0 and shirt_width>0): # checks if the rectangle actually appears
		   	shirt_checker=True
	    		    
		   	s_left  = f[0]+30-height/7
		   	s_right = f[0]+f[2]-30+height/7
		   	s_up    = f[1]+f[3]+20+height/12
		   	s_down  = f[1]+f[3]+2+height/3
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
	    		    	
		###------------###	 
		
		##################
		# EYES rectangle #
		##################
		cv2.rectangle(frames, (f[0]+8, f[1]+face_height/6+6), (f[0]+f[2]-8,f[1]+f[3]+6-face_height/2), (0,0,0), 1)
		eye_roi = frames[f[1]+face_height/6+6+1:f[1]+f[3]-1+6-face_height/2,f[0]+8+1:f[0]+f[2]-8-1]

		#cv2.imshow('eyes new',eye_roi)      # show eyes roi for debugging
		eye_h,eye_w,eye_chan = eye_roi.shape # find eye roi dimensions

		e_left     = f[0]+8
		e_up       = f[1]+2+face_height/6+6
		e_right    = f[0]+f[2]-8
		e_down     = f[1]+f[3]+6-face_height/2
		eye_height = e_up-e_down
		eye_width  = e_left-e_right

		if(eye_w>0 and eye_h>0):
			eye_checker=True
			
			# Draw rectangles for eyes for debugging 		
 			#cv2.rectangle(frames,(e_left,e_up-(2*eye_height/3)),(e_right+(eye_width/2),e_down),(255,0,0),1)                # down-left rectangle
 			#cv2.rectangle(frames,(e_left,e_up-(eye_height/3)),(e_right+(eye_width/2),e_down+(eye_height/3)),(0,255,255),1) # left-center rectangle
 			#cv2.rectangle(frames,(e_left,e_up),(e_right+(eye_width/2),e_down+(2*eye_height/3)),(255,0,255),1)              # up-left rectangle
 			#cv2.rectangle(frames,(e_left-eye_width/2,e_up-(eye_height/3)),(e_right,e_down+(eye_height/3)),(255,255,255),1) # right-center rectangle
 			#cv2.rectangle(frames,(e_left-eye_width/2,e_up),(e_right,e_down+(2*eye_height/3)),(255,255,0),1)                # up-right rectangle
 			#cv2.rectangle(frames,(e_left-eye_width/2,e_up-(2*eye_height/3)),(e_right,e_down),(0,0,255),1)                  # right-down rectangle

 			# All eyerois
 			eyeroi_downleft    = frames[e_up-(2*eye_height/3)+1:e_down,e_left+1:e_right+(eye_width/2)]			    # down-left rectangle
			eyeroi_leftcenter  = frames[e_up-(eye_height/3)+1:e_down+(eye_height/3),e_left+1:e_right+(eye_width/2)]	# left-center rectangle
			eyeroi_upleft      = frames[e_up+1:e_down+(2*eye_height/3),e_left+1:e_right+(eye_width/2)]			    # up-left rectangle
			eyeroi_rightcenter = frames[e_up-(eye_height/3)+1:e_down+(eye_height/3),e_left-(eye_width/2)+1:e_right] # right-center rectangle
			eyeroi_rightup     = frames[e_up+1:e_down+(2*eye_height/3),e_left-(eye_width/2)+1:e_right]			    # up-right rectangle
			eyeroi_rightdown   = frames[e_up-(2*eye_height/3)+1:e_down,e_left-(eye_width/2)+1:e_right]		        # right-down rectangle

		###------------###	  		

		if(face_checker==True and shirt_checker==True):
			
			index+=1 # calculates the number of while loops and it can be 
					 # used for telling the program to start recording 
			
			# Show message whether or not ALL 3 rois appear in frame
	 		cv.PutText(cv.fromarray(frames),"OK :)", (width-90,40),font, (0,255,0))  	  			
	 			 		
	 		# If you press 'ESC' key, the recording will start
			if cv2.waitKey(1)==27:
				recording=True 	    	

	    	if recording==True:

	    		# Just for the show...
	    		if ((frame_counter%2)==0):
	    			cv.PutText(cv.fromarray(frames),"Recording", (width-600,40),font, (0,0,255))

	    		# SHIRT
	    		if (sub_shirt_roi_1.shape[0]>0 and sub_shirt_roi_1.shape[1]>0 and # if the shirt roi has formulated completely
	    			sub_shirt_roi_2.shape[0]>0 and sub_shirt_roi_2.shape[1]>0 and # then start collecting data
	    			sub_shirt_roi_3.shape[0]>0 and sub_shirt_roi_3.shape[1]>0 and
	    			sub_shirt_roi_4.shape[0]>0 and sub_shirt_roi_4.shape[1]>0 and
	    			sub_shirt_roi_5.shape[0]>0 and sub_shirt_roi_5.shape[1]>0 and
	    			sub_shirt_roi_6.shape[0]>0 and sub_shirt_roi_6.shape[1]>0):

	    			#shirt rois RGB values
	    			(SR1,SG1,SB1) = getRGBHistograms(sub_shirt_roi_1)
	    			(SR2,SG2,SB2) = getRGBHistograms(sub_shirt_roi_2)
	    			(SR3,SG3,SB3) = getRGBHistograms(sub_shirt_roi_3)
	    			(SR4,SG4,SB4) = getRGBHistograms(sub_shirt_roi_4)
	    			(SR5,SG5,SB5) = getRGBHistograms(sub_shirt_roi_5)
	    			(SR6,SG6,SB6) = getRGBHistograms(sub_shirt_roi_6)	
	    			#shirt rois HSV values
	    		
	    			# Convert BGR to HSV
    				hsv_sub_shirt_roi_1 = cv2.cvtColor(sub_shirt_roi_1, cv2.COLOR_BGR2HSV)
    				hsv_sub_shirt_roi_2 = cv2.cvtColor(sub_shirt_roi_2, cv2.COLOR_BGR2HSV)
    				hsv_sub_shirt_roi_3 = cv2.cvtColor(sub_shirt_roi_3, cv2.COLOR_BGR2HSV)
    				hsv_sub_shirt_roi_4 = cv2.cvtColor(sub_shirt_roi_4, cv2.COLOR_BGR2HSV)
    				hsv_sub_shirt_roi_5 = cv2.cvtColor(sub_shirt_roi_5, cv2.COLOR_BGR2HSV)
    				hsv_sub_shirt_roi_6 = cv2.cvtColor(sub_shirt_roi_6, cv2.COLOR_BGR2HSV)
	    			(SH1,SS1,SV1) 		= getHSVHistograms(hsv_sub_shirt_roi_1)
	    			(SH2,SS2,SV2)	 	= getHSVHistograms(hsv_sub_shirt_roi_2)
	    			(SH3,SS3,SV3) 		= getHSVHistograms(hsv_sub_shirt_roi_3)
	    			(SH4,SS4,SV4) 		= getHSVHistograms(hsv_sub_shirt_roi_4)
	    			(SH5,SS5,SV5) 		= getHSVHistograms(hsv_sub_shirt_roi_5)
	    			(SH6,SS6,SV6) 		= getHSVHistograms(hsv_sub_shirt_roi_6)

	    		# HAIR 
	    		#hair rois RGB values
	    		#(HR1,HG1,HB1) = getRGBHistograms(sub_hair_roi_1)
	    		#(HR2,HG2,HB2) = getRGBHistograms(sub_hair_roi_2)
	    		#(HR3,HG3,HB3) = getRGBHistograms(sub_hair_roi_3)
	    		#(HR4,HG4,HB4) = getRGBHistograms(sub_hair_roi_4)
	    		#(HR5,HG5,HB5) = getRGBHistograms(sub_hair_roi_5)
	    		#(HR6,HG6,HB6) = getRGBHistograms(sub_hair_roi_6)
	    		#hair rois HSV values
	    		#(HH1,HS1,HV1) = getHSVHistograms(sub_hair_roi_1)
	    		#(HH2,HS2,HV2) = getHSVHistograms(sub_hair_roi_2)
	    		#(HH3,HS3,HV3) = getHSVHistograms(sub_hair_roi_3)
	    		#(HH4,HS4,HV4) = getHSVHistograms(sub_hair_roi_4)
	    		#(HH5,HS5,HV5) = getHSVHistograms(sub_hair_roi_5)
	    		#(HH6,HS6,HV6) = getHSVHistograms(sub_hair_roi_6)	    		

	    		# EYES	    	
	    		# The Canny functions find the Canny Edges for every eye roi	
	    		downleft_edge    = cv2.Canny(eyeroi_downleft,100,400)
	    		downright_edge   = cv2.Canny(eyeroi_rightdown,100,400)
	    		leftcenter_edge  = cv2.Canny(eyeroi_leftcenter,100,400)
	    		rightcenter_edge = cv2.Canny(eyeroi_rightcenter,100,400)
	    		leftup_edge      = cv2.Canny(eyeroi_upleft,100,400)
	    		rightup_edge     = cv2.Canny(eyeroi_rightup,100,400)

	    		#eyecannyedge    = cv2.Canny(eye_roi,100,400) # for debugging
	    		#cv2.imshow('eyes edges',eyecannyedge)

	    		# The function below summarizes the canny edge values for every eye subroi
	    		DLE = returnEdges(downleft_edge)
	    		DRE = returnEdges(downright_edge)
	    		LCE = returnEdges(leftcenter_edge)
	    		RCE = returnEdges(rightcenter_edge)
	    		LUE = returnEdges(leftup_edge)
	    		RUE = returnEdges(rightup_edge)

	    		# HATS
	    		# If the sub hat rois have formulated all correctly and at the same time, 
	    		# starts calculating the contours for each sub roi
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

	    			#hat_roi_foreground = getForeground(hat_roi) # for debugging
	    			#cv2.imshow("Foreground1", hat_roi_foreground) # for debugging

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

	    			hat_cannyedge = cv2.Canny(hat_roi,200,100)
	    			cv2.imshow('hat edges',hat_cannyedge)

	    			'''
	    			The functions below find the hat's color with RGB and HSV Histograms. 
					You have the ability to create your own hat color classifier and 
					detect the hat's color.	    		
					'''
	    			#(HATR1,HATG1,HATB1)  = getRGBHistograms(sub_hat_roi_1)
	    			#(HATR2,HATG2,HATB2)  = getRGBHistograms(sub_hat_roi_2)
	    			#(HATR3,HATG3,HATB3)  = getRGBHistograms(sub_hat_roi_3)
	    			#(HATR4,HATG4,HATB4)  = getRGBHistograms(sub_hat_roi_4)
	    			#(HATR5,HATG5,HATB5)  = getRGBHistograms(sub_hat_roi_5)
	    			#(HATR6,HATG6,HATB6)  = getRGBHistograms(sub_hat_roi_6)

	    			#hat rois HSV values
	    			#(HATH1,HATS1,HATV1)  = getHSVHistograms(sub_hat_roi_1)
	    			#(HATH2,HATS2,HATV2)  = getHSVHistograms(sub_hat_roi_2)
	    			#(HATH3,HATS3,HATV3)  = getHSVHistograms(sub_hat_roi_3)
	    			#(HATH4,HATS4,HATV4)  = getHSVHistograms(sub_hat_roi_4)
	    			#(HATH5,HATS5,HATV5)  = getHSVHistograms(sub_hat_roi_5)
	    			#(HATH6,HATS6,HATV6)  = getHSVHistograms(sub_hat_roi_6)
	    		
	    		# start collecting data from 100 frames
	    		if(frame_counter<100):
	    					
	    			#shirt RGB HSV files
					shirtname = args["pn"]+'_'+args["sc"]+'_rgbhsv_shirt.csv'					
					write_file(shirtname,SR1,SG1,SB1,SH1,SS1,SV1,SR2,SG2,SB2,SH2,SS2,SV2,SR3,SG3,SB3,SH3,SS3,SV3,SR4,SG4,SB4,SH4,SS4,SV4,SR5,SG5,SB5,SH5,SS5,SV5,SR6,SG6,SB6,SH6,SS6,SV6)
					shirtname=args["pn"]+'_'+args["sc"]+'_p1_rgbhsv_shirt.csv'
					write_file_plus_one(shirtname,SR1,SG1,SB1,SH1,SS1,SV1,SR2,SG2,SB2,SH2,SS2,SV2,SR3,SG3,SB3,SH3,SS3,SV3,SR4,SG4,SB4,SH4,SS4,SV4,SR5,SG5,SB5,SH5,SS5,SV5,SR6,SG6,SB6,SH6,SS6,SV6)
					shirtname=args["pn"]+'_'+args["sc"]+'_m1_rgbhsv_shirt.csv'
					write_file_minus_one(shirtname,SR1,SG1,SB1,SH1,SS1,SV1,SR2,SG2,SB2,SH2,SS2,SV2,SR3,SG3,SB3,SH3,SS3,SV3,SR4,SG4,SB4,SH4,SS4,SV4,SR5,SG5,SB5,SH5,SS5,SV5,SR6,SG6,SB6,SH6,SS6,SV6)
	      								
					#shirt RGB files
					shirtname = args["pn"]+'_'+args["sc"]+'_rgb_shirt.csv'					
					write_file(shirtname,SR1,SG1,SB1,SR1,SG1,SB1,SR2,SG2,SB2,SR2,SG2,SB2,SR3,SG3,SB3,SR3,SG3,SB3,SR4,SG4,SB4,SR4,SG4,SB4,SR5,SG5,SB5,SR5,SG5,SB5,SR6,SG6,SB6,SR6,SG6,SB6)
					shirtname=args["pn"]+'_'+args["sc"]+'_p1_rgb_shirt.csv'
					write_file_plus_one(shirtname,SR1,SG1,SB1,SR1,SG1,SB1,SR2,SG2,SB2,SR2,SG2,SB2,SR3,SG3,SB3,SR3,SG3,SB3,SR4,SG4,SB4,SR4,SG4,SB4,SR5,SG5,SB5,SR5,SG5,SB5,SR6,SG6,SB6,SR6,SG6,SB6)
					shirtname=args["pn"]+'_'+args["sc"]+'_m1_rgb_shirt.csv'
					write_file_minus_one(shirtname,SR1,SG1,SB1,SR1,SG1,SB1,SR2,SG2,SB2,SR2,SG2,SB2,SR3,SG3,SB3,SR3,SG3,SB3,SR4,SG4,SB4,SR4,SG4,SB4,SR5,SG5,SB5,SR5,SG5,SB5,SR6,SG6,SB6,SR6,SG6,SB6)

	      			# Hair files
					#hairname = args["pn"]+'_'+args["hc"]+'_hair.csv'
					#write_file(hairname,HR1,HG1,HB1,HR2,HG2,HB2,HR3,HG3,HB3,HR4,HG4,HB4,HR5,HG5,HB5,HR6,HG6,HB6)
					#hairname=args["pn"]+'_'+args["hc"]+'_p1_hair.csv'
					#write_file_plus_one(hairname,HG1,HB1,HR2,HG2,HB2,HR3,HG3,HB3,HR4,HG4,HB4,HR5,HG5,HB5,HR6,HG6,HB6)
					#hairname=args["pn"]+'_'+args["hc"]+'_m1_hair.csv'
					#write_file_minus_one(hairname,HG1,HB1,HR2,HG2,HB2,HR3,HG3,HB3,HR4,HG4,HB4,HR5,HG5,HB5,HR6,HG6,HB6)
					
					# Eyeglasses files
					if args["gla"]=='true':
						write_eyes_file(args["pn"]+'_pos_eyes.csv',DLE,DRE,LCE,RCE,LUE,RUE)
						write_eyes_p1_file(args["pn"]+'_p1_pos_eyes.csv',DLE,DRE,LCE,RCE,LUE,RUE)						
					elif args["gla"]=='false':
						write_eyes_file(args["pn"]+'_neg_eyes.csv',DLE,DRE,LCE,RCE,LUE,RUE)
						write_eyes_m1_file(args["pn"]+'_m1_neg_eyes.csv',DLE,DRE,LCE,RCE,LUE,RUE)
					else:
						print 'Wrong value for eyeglasses! Please put the correct ones.'
								
					# Hats' color files
					#hatname = args["pn"]+'_hat_color.csv'
					#write_hat_color_file(hatname,HATR1,HATG1,HATB1,HATH1,HATS1,HATV1,HATR2,HATG2,HATB2,HATH2,HATS2,HATV2,HATR3,HATG3,HATB3,HATH3,HATS3,HATV3,HATR4,HATG4,HATB4,HATH4,HATS4,HATV4,HATR5,HATG5,HATB5,HATH5,HATS5,HATV5,HATR6,HATG6,HATB6,HATH6,HATS6,HATV6)

					# Hats' files
					if args["hat"]=='true':
						write_hat_file(args["pn"]+'_pos_hat.csv',hatedge1,hatedge2,hatedge3,hatedge4,hatedge5,hatedge6,hatfore1,hatfore2,hatfore3,hatfore4,hatfore5,hatfore6)
						write_hat_p1_file(args["pn"]+'_p1_pos_hat.csv',hatedge1,hatedge2,hatedge3,hatedge4,hatedge5,hatedge6,hatfore1,hatfore2,hatfore3,hatfore4,hatfore5,hatfore6)						
					elif args["hat"]=='false':
						write_hat_file(args["pn"]+'_neg_hat.csv',hatedge1,hatedge2,hatedge3,hatedge4,hatedge5,hatedge6,hatfore1,hatfore2,hatfore3,hatfore4,hatfore5,hatfore6)
						write_hat_m1_file(args["pn"]+'_m1_neg_hat.csv',hatedge1,hatedge2,hatedge3,hatedge4,hatedge5,hatedge6,hatfore1,hatfore2,hatfore3,hatfore4,hatfore5,hatfore6)
					else:
						print 'Wrong value for hats! Please put the correct ones.'

					frame_counter+=1

	    		if(frame_counter==100):
	    			print 'Recording stoped.'
	    			print 'OK. 100 frames captured.'
	    			processed=True
	    			break
		
	#end loop for faces
	cv2.imshow('Capture', frames)
	if processed==True:
		break	        # exits while loop
	
#end While
cap.release()           # closes the camera
if args["q"]=='false':
	out.release()       # stops recording
cv2.destroyAllWindows() # destroy all windows

print 'OK.'

# end-of-program #