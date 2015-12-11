import numpy as np
import time
import mlpy as ml
import csv
import argparse

'''
In this program we train the libsvms for red, green, blue, white and black color.

We load the positive and negative train sets for every color and libsvm.

Also, we examine 3 different ways for color classification.
1st way: we take only the rgb values for every sub shirt roi.
2nd way: we take the rgb and the s values for every sub shirt roi.
3rd way: we take the rgb and hsv values for every sub shirt roi.

For more details take a look at the documentation.
'''

############
''' MAIN '''
############

start=time.time()

# initializes ArgumentParser
ap = argparse.ArgumentParser()

# adds an argument
ap.add_argument("-s", "--apppath", required = True, help = "The path to the App Folder.")
args = vars(ap.parse_args())

path = args["apppath"]

###################
''' RGB LibSvms '''
###################

# creates color libsvms
redsvm    = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
greensvm  = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
bluesvm   = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
whitesvm  = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
blacksvm  = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)

# loads training data for Red SVM
R_rp1  = np.loadtxt(open(path+'SHIRTS/RED/positive red/NEW_mar_27_red1_p1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
R_rp2  = np.loadtxt(open(path+'SHIRTS/RED/positive red/NEW_mar_28_red2_p1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
R_rp3  = np.loadtxt(open(path+'SHIRTS/RED/positive red/NEW_mar_30_red4_p1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
R_rp4  = np.loadtxt(open(path+'SHIRTS/RED/positive red/NEW_mar_32_red6_p1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
R_rp5  = np.loadtxt(open(path+'SHIRTS/RED/positive red/NEW_mar_33_red7_p1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
R_rp6  = np.loadtxt(open(path+'SHIRTS/RED/positive red/NEW_kostas_6_red2_p1_shirt.csv'  ,"rb") ,delimiter=";",skiprows=0)
R_rp7  = np.loadtxt(open(path+'SHIRTS/RED/positive red/NEW_kostas_6_red2_p1_shirt.csv'  ,"rb") ,delimiter=";",skiprows=0)
R_gm1  = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_mar_24_green3_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
R_gm2  = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_geo_38_greengeo1_m1_shirt.csv',"rb"),delimiter=";",skiprows=0)
R_gm3  = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_mar_26_green5_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
R_gm4  = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_mar_23_green2_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
R_bm1  = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_mar_10_blue1_m1_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
R_bm2  = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_mar_12_blue3_m1_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
R_bm3  = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_mar_16_blue7_m1_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
R_bm4  = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_mar_17_blue8_m1_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
R_bm5  = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_maria_44_blue2_m1_shirt.csv' ,"rb") ,delimiter=";",skiprows=0)
R_wm1  = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_mar_06_white1_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
R_wm2  = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_mar_07_white2_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
R_wm3  = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_mar_08_white3_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
R_wm4  = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_mar_09_white4_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
R_blm1 = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_mar_01_black1_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
R_blm2 = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_mar_02_black2_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
R_blm3 = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_mar_05_black5_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
R_blm4 = np.loadtxt(open(path+'SHIRTS/RED/negative red/NEW_oresths53_black2_m1_shirt.csv',"rb"),delimiter=";",skiprows=0)

# loads training data for Green SVM
G_rm1  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_mar_27_red1_m1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
G_rm2  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_mar_28_red2_m1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
G_rm3  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_mar_30_red4_m1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
G_rm4  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_mar_32_red6_m1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
G_rm5  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_mar_33_red7_m1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
G_rm6  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_kostas_6_red2_m1_shirt.csv'  ,"rb") ,delimiter=";",skiprows=0)
G_gp1  = np.loadtxt(open(path+'SHIRTS/GREEN/positive green/NEW_mar_23_green2_p1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
G_gp2  = np.loadtxt(open(path+'SHIRTS/GREEN/positive green/NEW_mar_24_green3_p1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
G_gp3  = np.loadtxt(open(path+'SHIRTS/GREEN/positive green/NEW_geo_38_greengeo1_p1_shirt.csv',"rb"),delimiter=";",skiprows=0)
G_gp4  = np.loadtxt(open(path+'SHIRTS/GREEN/positive green/NEW_mar_26_green5_p1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
G_gp5  = np.loadtxt(open(path+'SHIRTS/GREEN/positive green/NEW_maria_39_green4_p1_shirt.csv',"rb") ,delimiter=";",skiprows=0)
G_bm1  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_mar_10_blue1_m1_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
G_bm2  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_mar_12_blue3_m1_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
G_bm3  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_mar_16_blue7_m1_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
G_bm4  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_mar_17_blue8_m1_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
G_bm5  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_maria_44_blue2_m1_shirt.csv' ,"rb") ,delimiter=";",skiprows=0)
G_wm1  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_mar_06_white1_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
G_wm2  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_mar_07_white2_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
G_wm3  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_mar_08_white3_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
G_wm4  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_mar_09_white4_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
G_blm1 = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_mar_01_black1_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
G_blm2 = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_mar_02_black2_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
G_blm3 = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_mar_05_black5_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
G_blm4 = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/NEW_oresths53_black2_m1_shirt.csv',"rb"),delimiter=";",skiprows=0)

# loads training data for Blue SVM
B_rm1  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_mar_33_red7_m1_shirt.csv',"rb")     ,delimiter=";",skiprows=0)
B_rm2  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_mar_32_red6_m1_shirt.csv',"rb")     ,delimiter=";",skiprows=0)
B_rm3  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_mar_30_red4_m1_shirt.csv',"rb")     ,delimiter=";",skiprows=0)
B_rm4  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_mar_28_red2_m1_shirt.csv',"rb")     ,delimiter=";",skiprows=0)
B_rm5  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_mar_27_red1_m1_shirt.csv',"rb")     ,delimiter=";",skiprows=0)
B_rm6  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_kostas_6_red2_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
B_gm1  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_mar_26_green5_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
B_gm2  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_geo_38_greengeo1_m1_shirt.csv',"rb"),delimiter=";",skiprows=0)
B_gm3  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_mar_24_green3_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
B_gm4  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_mar_23_green2_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
B_bp1  = np.loadtxt(open(path+'SHIRTS/BLUE/positive blue/NEW_mar_10_blue1_p1_shirt.csv',"rb")    ,delimiter=";",skiprows=0)
B_bp2  = np.loadtxt(open(path+'SHIRTS/BLUE/positive blue/NEW_mar_12_blue3_p1_shirt.csv',"rb")    ,delimiter=";",skiprows=0)
B_bp3  = np.loadtxt(open(path+'SHIRTS/BLUE/positive blue/NEW_mar_16_blue7_p1_shirt.csv',"rb")    ,delimiter=";",skiprows=0)
B_bp4  = np.loadtxt(open(path+'SHIRTS/BLUE/positive blue/NEW_mar_17_blue8_p1_shirt.csv',"rb")    ,delimiter=";",skiprows=0)
B_bp5  = np.loadtxt(open(path+'SHIRTS/BLUE/positive blue/NEW_maria_44_blue2_p1_shirt.csv',"rb")  ,delimiter=";",skiprows=0)
B_wm1  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_mar_09_white4_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
B_wm2  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_mar_08_white3_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
B_wm3  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_mar_07_white2_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
B_wm4  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_mar_06_white1_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
B_blm1 = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_mar_01_black1_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
B_blm2 = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_mar_02_black2_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
B_blm3 = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_mar_05_black5_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
B_blm4 = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/NEW_oresths53_black2_m1_shirt.csv',"rb"),delimiter=";",skiprows=0)

# loads training data for White SVM
W_rm1  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_mar_27_red1_m1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
W_rm2  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_mar_28_red2_m1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
W_rm3  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_mar_30_red4_m1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
W_rm4  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_mar_32_red6_m1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
W_rm5  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_mar_33_red7_m1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
W_rm6  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_kostas_6_red2_m1_shirt.csv'  ,"rb") ,delimiter=";",skiprows=0)
W_gm1  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_mar_23_green2_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
W_gm2  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_mar_24_green3_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
W_gm3  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_geo_38_greengeo1_m1_shirt.csv',"rb"),delimiter=";",skiprows=0)
W_gm4  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_mar_26_green5_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
W_bm1  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_mar_10_blue1_m1_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
W_bm2  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_mar_12_blue3_m1_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
W_bm3  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_mar_16_blue7_m1_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
W_bm4  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_mar_17_blue8_m1_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
W_bm5  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_maria_44_blue2_m1_shirt.csv' ,"rb") ,delimiter=";",skiprows=0)
W_wp1  = np.loadtxt(open(path+'SHIRTS/WHITE/positive white/NEW_mar_06_white1_p1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
W_wp2  = np.loadtxt(open(path+'SHIRTS/WHITE/positive white/NEW_mar_07_white2_p1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
W_wp3  = np.loadtxt(open(path+'SHIRTS/WHITE/positive white/NEW_mar_08_white3_p1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
W_wp4  = np.loadtxt(open(path+'SHIRTS/WHITE/positive white/NEW_mar_09_white4_p1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
W_blm1 = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_mar_01_black1_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
W_blm2 = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_mar_02_black2_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
W_blm3 = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_mar_05_black5_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
W_blm4 = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/NEW_oresths53_black2_m1_shirt.csv',"rb"),delimiter=";",skiprows=0)

# loads training data for Black SVM
BL_rm1  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_mar_27_red1_m1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
BL_rm2  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_mar_28_red2_m1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
BL_rm3  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_mar_30_red4_m1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
BL_rm4  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_mar_32_red6_m1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
BL_rm5  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_mar_33_red7_m1_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
BL_rm6  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_kostas_6_red2_m1_shirt.csv'  ,"rb") ,delimiter=";",skiprows=0)
BL_gm1  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_mar_23_green2_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
BL_gm2  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_mar_24_green3_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
BL_gm3  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_geo_38_greengeo1_m1_shirt.csv',"rb"),delimiter=";",skiprows=0)
BL_gm4  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_mar_26_green5_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
BL_bm1  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_mar_10_blue1_m1_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
BL_bm2  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_mar_12_blue3_m1_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
BL_bm3  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_mar_16_blue7_m1_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
BL_bm4  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_mar_17_blue8_m1_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
BL_bm5  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_maria_44_blue2_m1_shirt.csv' ,"rb") ,delimiter=";",skiprows=0)
BL_wm1  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_mar_06_white1_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
BL_wm2  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_mar_07_white2_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
BL_wm3  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_mar_08_white3_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
BL_wm4  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/NEW_mar_09_white4_m1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
BL_blp1 = np.loadtxt(open(path+'SHIRTS/BLACK/positive black/NEW_mar_01_black1_p1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
BL_blp2 = np.loadtxt(open(path+'SHIRTS/BLACK/positive black/NEW_mar_02_black2_p1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
BL_blp3 = np.loadtxt(open(path+'SHIRTS/BLACK/positive black/NEW_mar_05_black5_p1_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
BL_blp4 = np.loadtxt(open(path+'SHIRTS/BLACK/positive black/NEW_oresths53_black2_p1_shirt.csv',"rb"),delimiter=";",skiprows=0)
BL_blp5 = np.loadtxt(open(path+'SHIRTS/BLACK/positive black/NEW_kos_36_blackkos1_p1_shirt.csv',"rb"),delimiter=";",skiprows=0)

# training set for redsvm
# the x part is all the feature vectors and the y part is 1 or -1
R_rp1x,R_rp1y   = R_rp1[:, :360],R_rp1[:, 360]
R_rp2x,R_rp2y   = R_rp2[:, :360],R_rp2[:, 360]
R_rp3x,R_rp3y   = R_rp3[:, :360],R_rp3[:, 360]
R_rp4x,R_rp4y   = R_rp4[:, :360],R_rp4[:, 360]
R_rp5x,R_rp5y   = R_rp5[:, :360],R_rp5[:, 360]
R_rp6x,R_rp6y   = R_rp6[:, :360],R_rp6[:, 360]
R_rp7x,R_rp7y   = R_rp7[:, :360],R_rp7[:, 360]
R_gm1x,R_gm1y   = R_gm1[:, :360],R_gm1[:, 360]
R_gm2x,R_gm2y   = R_gm2[:, :360],R_gm2[:, 360]
R_gm3x,R_gm3y   = R_gm3[:, :360],R_gm3[:, 360]
R_gm4x,R_gm4y   = R_gm4[:, :360],R_gm4[:, 360]
R_bm1x,R_bm1y   = R_bm1[:, :360],R_bm1[:, 360]
R_bm2x,R_bm2y   = R_bm2[:, :360],R_bm2[:, 360]
R_bm3x,R_bm3y   = R_bm3[:, :360],R_bm3[:, 360]
R_bm4x,R_bm4y   = R_bm4[:, :360],R_bm4[:, 360]
R_bm5x,R_bm5y   = R_bm5[:, :360],R_bm5[:, 360]
R_wm1x,R_wm1y   = R_wm1[:, :360],R_wm1[:, 360]
R_wm2x,R_wm2y   = R_wm2[:, :360],R_wm2[:, 360]
R_wm3x,R_wm3y   = R_wm3[:, :360],R_wm3[:, 360]
R_wm4x,R_wm4y   = R_wm4[:, :360],R_wm4[:, 360]
R_blm1x,R_blm1y = R_blm1[:, :360],R_blm1[:, 360]
R_blm2x,R_blm2y = R_blm2[:, :360],R_blm2[:, 360]
R_blm3x,R_blm3y = R_blm3[:, :360],R_blm3[:, 360]
R_blm4x,R_blm4y = R_blm4[:, :360],R_blm4[:, 360]

# prepares the training data in order to give them as arguments for learning
Rtx = np.vstack([R_gm1x,R_rp1x,R_bm1x,R_rp2x,R_wm1x,R_rp6x,R_rp7x,R_blm1x,R_rp3x,R_rp4x,R_rp5x,R_blm4x,R_bm5x])
Rty = np.concatenate((R_gm1y,R_rp1y,R_bm1y,R_rp2y,R_wm1y,R_rp6y,R_rp7y,R_blm1y,R_rp3y,R_rp4y,R_rp5y,R_blm4y,R_bm5y))
redsvm.learn(Rtx,Rty)

# training set for greensvm
# the x part is all the feature vectors and the y part is 1 or -1
G_rm1x,G_rm1y   = G_rm1[:, :360],G_rm1[:, 360]
G_rm2x,G_rm2y   = G_rm2[:, :360],G_rm2[:, 360]
G_rm3x,G_rm3y   = G_rm3[:, :360],G_rm3[:, 360]
G_rm4x,G_rm4y   = G_rm4[:, :360],G_rm4[:, 360]
G_rm5x,G_rm5y   = G_rm5[:, :360],G_rm5[:, 360]
G_gp1x,G_gp1y   = G_gp1[:, :360],G_gp1[:, 360]
G_gp2x,G_gp2y   = G_gp2[:, :360],G_gp2[:, 360]
G_gp3x,G_gp3y   = G_gp3[:, :360],G_gp3[:, 360]
G_gp4x,G_gp4y   = G_gp4[:, :360],G_gp4[:, 360]
G_gp5x,G_gp5y   = G_gp5[:, :360],G_gp5[:, 360]
G_bm1x,G_bm1y   = G_bm1[:, :360],G_bm1[:, 360]
G_bm2x,G_bm2y   = G_bm2[:, :360],G_bm2[:, 360]
G_bm3x,G_bm3y   = G_bm3[:, :360],G_bm3[:, 360]
G_bm4x,G_bm4y   = G_bm4[:, :360],G_bm4[:, 360]
G_bm5x,G_bm5y   = G_bm5[:, :360],G_bm5[:, 360]
G_wm1x,G_wm1y   = G_wm1[:, :360],G_wm1[:, 360]
G_wm2x,G_wm2y   = G_wm2[:, :360],G_wm2[:, 360]
G_wm3x,G_wm3y   = G_wm3[:, :360],G_wm3[:, 360]
G_wm4x,G_wm4y   = G_wm4[:, :360],G_wm4[:, 360]
G_blm1x,G_blm1y = G_blm1[:, :360],G_blm1[:, 360]
G_blm2x,G_blm2y = G_blm2[:, :360],G_blm2[:, 360]
G_blm3x,G_blm3y = G_blm3[:, :360],G_blm3[:, 360]
G_blm4x,G_blm4y = G_blm4[:, :360],G_blm4[:, 360]

# prepares the training data in order to give them as arguments for learning
Gtx = np.vstack([G_rm1x,G_gp1x,G_bm1x,G_wm1x,G_gp2x,G_blm1x,G_gp3x,G_rm5x,G_gp5x,G_blm4x,G_bm5x])
Gty = np.concatenate((G_rm1y,G_gp1y,G_bm1y,G_wm1y,G_gp2y,G_blm1y,G_gp3y,G_rm5y,G_gp5y,G_blm4y,G_bm5y))
greensvm.learn(Gtx,Gty)

# training set for bluesvm
# the x part is all the feature vectors and the y part is 1 or -1
B_rm1x,B_rm1y   = B_rm1[:, :360],B_rm1[:, 360]
B_rm2x,B_rm2y   = B_rm2[:, :360],B_rm2[:, 360]
B_rm3x,B_rm3y   = B_rm3[:, :360],B_rm3[:, 360]
B_rm4x,B_rm4y   = B_rm4[:, :360],B_rm4[:, 360]
B_rm5x,B_rm5y   = B_rm5[:, :360],B_rm5[:, 360]
B_gm1x,B_gm1y   = B_gm1[:, :360],B_gm1[:, 360]
B_gm2x,B_gm2y   = B_gm2[:, :360],B_gm2[:, 360]
B_gm3x,B_gm3y   = B_gm3[:, :360],B_gm3[:, 360]
B_gm4x,B_gm4y   = B_gm4[:, :360],B_gm4[:, 360]
B_bp1x,B_bp1y   = B_bp1[:, :360],B_bp1[:, 360]
B_bp2x,B_bp2y   = B_bp2[:, :360],B_bp2[:, 360]
B_bp3x,B_bp3y   = B_bp3[:, :360],B_bp3[:, 360]
B_bp4x,B_bp4y   = B_bp4[:, :360],B_bp4[:, 360]
#B_bp5x,B_bp5y   = B_bp5[:, :360],B_bp5[:, 360]
B_wm1x,B_wm1y   = B_wm1[:, :360],B_wm1[:, 360]
B_wm2x,B_wm2y   = B_wm2[:, :360],B_wm2[:, 360]
B_wm3x,B_wm3y   = B_wm3[:, :360],B_wm3[:, 360]
B_wm4x,B_wm4y   = B_wm4[:, :360],B_wm4[:, 360]
B_blm1x,B_blm1y = B_blm1[:, :360],B_blm1[:, 360]
B_blm2x,B_blm2y = B_blm2[:, :360],B_blm2[:, 360]
B_blm3x,B_blm3y = B_blm3[:, :360],B_blm3[:, 360]
B_blm4x,B_blm4y = B_blm4[:, :360],B_blm4[:, 360]

# prepares the training data in order to give them as arguments for learning
Btx = np.vstack([B_rm1x,B_bp1x,B_wm1x,B_bp2x,B_gm1x,B_bp3x,B_blm1x,B_blm4x])
Bty = np.concatenate((B_rm1y,B_bp1y,B_wm1y,B_bp2y,B_gm1y,B_bp3y,B_blm1y,B_blm4y)) 
bluesvm.learn(Btx,Bty)

# training set for whitesvm
# the x part is all the feature vectors and the y part is 1 or -1
W_rm1x,W_rm1y   = W_rm1[:, :360],W_rm1[:, 360]
W_rm2x,W_rm2y   = W_rm2[:, :360],W_rm2[:, 360]
W_rm3x,W_rm3y   = W_rm3[:, :360],W_rm3[:, 360]
W_rm4x,W_rm4y   = W_rm4[:, :360],W_rm4[:, 360]
W_rm5x,W_rm5y   = W_rm5[:, :360],W_rm5[:, 360]
W_gm1x,W_gm1y   = W_gm1[:, :360],W_gm1[:, 360]
W_gm2x,W_gm2y   = W_gm2[:, :360],W_gm2[:, 360]
W_gm3x,W_gm3y   = W_gm3[:, :360],W_gm3[:, 360]
W_gm4x,W_gm4y   = W_gm4[:, :360],W_gm4[:, 360]
W_bm1x,W_bm1y   = W_bm1[:, :360],W_bm1[:, 360]
W_bm2x,W_bm2y   = W_bm2[:, :360],W_bm2[:, 360]
W_bm3x,W_bm3y   = W_bm3[:, :360],W_bm3[:, 360]
W_bm4x,W_bm4y   = W_bm4[:, :360],W_bm4[:, 360]
W_bm5x,W_bm5y   = W_bm5[:, :360],W_bm5[:, 360]
W_wp1x,W_wp1y   = W_wp1[:, :360],W_wp1[:, 360]
W_wp2x,W_wp2y   = W_wp2[:, :360],W_wp2[:, 360]
W_wp3x,W_wp3y   = W_wp3[:, :360],W_wp3[:, 360]
W_wp4x,W_wp4y   = W_wp4[:, :360],W_wp4[:, 360]
W_blm1x,W_blm1y = W_blm1[:, :360],W_blm1[:, 360]
W_blm2x,W_blm2y = W_blm2[:, :360],W_blm2[:, 360]
W_blm3x,W_blm3y = W_blm3[:, :360],W_blm3[:, 360]
W_blm4x,W_blm4y = W_blm4[:, :360],W_blm4[:, 360]   

# prepares the training data in order to give them as arguments for learning
Wtx = np.vstack([W_rm1x,W_wp1x,W_blm1x,W_wp2x,W_bm1x,W_wp3x,W_gm1x,W_blm4x,W_bm5x])
Wty = np.concatenate((W_rm1y,W_wp1y,W_blm1y,W_wp2y,W_bm1y,W_wp3y,W_gm1y,W_blm4y,W_bm5y)) 
whitesvm.learn(Wtx,Wty)

# training set for blacksvm
# the x part is all the feature vectors and the y part is 1 or -1
BL_rm1x,BL_rm1y   = BL_rm1[:, :360],BL_rm1[:, 360]
BL_rm2x,BL_rm2y   = BL_rm2[:, :360],BL_rm2[:, 360]
BL_rm3x,BL_rm3y   = BL_rm3[:, :360],BL_rm3[:, 360]
BL_rm4x,BL_rm4y   = BL_rm4[:, :360],BL_rm4[:, 360]
BL_rm5x,BL_rm5y   = BL_rm5[:, :360],BL_rm5[:, 360]
BL_gm1x,BL_gm1y   = BL_gm1[:, :360],BL_gm1[:, 360]
BL_gm2x,BL_gm2y   = BL_gm2[:, :360],BL_gm2[:, 360]
BL_gm3x,BL_gm3y   = BL_gm3[:, :360],BL_gm3[:, 360]
BL_gm4x,BL_gm4y   = BL_gm4[:, :360],BL_gm4[:, 360]
BL_bm1x,BL_bm1y   = BL_bm1[:, :360],BL_bm1[:, 360]
BL_bm2x,BL_bm2y   = BL_bm2[:, :360],BL_bm2[:, 360]
BL_bm3x,BL_bm3y   = BL_bm3[:, :360],BL_bm3[:, 360]
BL_bm4x,BL_bm4y   = BL_bm4[:, :360],BL_bm4[:, 360]
BL_bm5x,BL_bm5y   = BL_bm5[:, :360],BL_bm5[:, 360]
BL_wm1x,BL_wm1y   = BL_wm1[:, :360],BL_wm1[:, 360]
BL_wm2x,BL_wm2y   = BL_wm2[:, :360],BL_wm2[:, 360]
BL_wm3x,BL_wm3y   = BL_wm3[:, :360],BL_wm3[:, 360]
BL_wm4x,BL_wm4y   = BL_wm4[:, :360],BL_wm4[:, 360]
BL_blp1x,BL_blp1y = BL_blp1[:, :360],BL_blp1[:, 360]
BL_blp2x,BL_blp2y = BL_blp2[:, :360],BL_blp2[:, 360]
BL_blp3x,BL_blp3y = BL_blp3[:, :360],BL_blp3[:, 360]
BL_blp4x,BL_blp4y = BL_blp4[:, :360],BL_blp4[:, 360]
BL_blp5x,BL_blp5y = BL_blp5[:, :360],BL_blp5[:, 360]

# prepares the training data in order to give them as arguments for learning
BLtx = np.vstack([BL_rm1x,BL_blp1x,BL_bm1x,BL_blp2x,BL_gm1x,BL_blp3x,BL_wm1x,BL_blp4x,BL_rm5x,BL_blp5x,BL_bm5x])
BLty = np.concatenate((BL_rm1y,BL_blp1y,BL_bm1y,BL_blp2y,BL_gm1y,BL_blp3y,BL_wm1y,BL_blp4y,BL_rm5y,BL_blp5y,BL_bm5y)) 
blacksvm.learn(BLtx,BLty)

# save trainied libsvms as xml files
redsvm.save_model('rgb_redsvm.xml')
greensvm.save_model('rgb_greensvm.xml')
bluesvm.save_model('rgb_bluesvm.xml')
whitesvm.save_model('rgb_whitesvm.xml')
blacksvm.save_model('rgb_blacksvm.xml')

print '\nColor RGB svms has been created successfully!'

###--------------------###

##########################
''' Train RBG S LibSvms'''
##########################

# creates color libsvms
rgbs_redsvm    = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgbs_greensvm  = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgbs_bluesvm   = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgbs_whitesvm  = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgbs_blacksvm  = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)

# loads training data for Red SVM
rgbs_R_rp1  = np.loadtxt(open(path+'SHIRTS/RED/positive red/mar_27_red1_p1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbs_R_rp2  = np.loadtxt(open(path+'SHIRTS/RED/positive red/mar_28_red2_p1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbs_R_rp3  = np.loadtxt(open(path+'SHIRTS/RED/positive red/mar_30_red4_p1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbs_R_rp4  = np.loadtxt(open(path+'SHIRTS/RED/positive red/mar_32_red6_p1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbs_R_rp5  = np.loadtxt(open(path+'SHIRTS/RED/positive red/mar_33_red7_p1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
#rgbs_R_rp6  = np.loadtxt(open(path+'SHIRTS/RED/positive red/kostas_6_red2_p1_rgbs_shirt.csv'  ,"rb") ,delimiter=";",skiprows=0)
#rgbs_R_rp7  = np.loadtxt(open(path+'SHIRTS/RED/positive red/kostas_6_red2_p1_rgbs_shirt.csv'  ,"rb") ,delimiter=";",skiprows=0)
rgbs_R_gm1  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_24_green3_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_R_gm2  = np.loadtxt(open(path+'SHIRTS/RED/negative red/geo_38_greengeo1_m1_rgbs_shirt.csv',"rb"),delimiter=";",skiprows=0)
rgbs_R_gm3  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_26_green5_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_R_gm4  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_23_green2_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_R_bm1  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_10_blue1_m1_rgbs_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbs_R_bm2  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_12_blue3_m1_rgbs_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbs_R_bm3  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_16_blue7_m1_rgbs_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbs_R_bm4  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_17_blue8_m1_rgbs_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbs_R_bm5  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mariatest9_blue2_m1_rgbs_shirt.csv' ,"rb") ,delimiter=";",skiprows=0)
rgbs_R_wm1  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_06_white1_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_R_wm2  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_07_white2_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_R_wm3  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_08_white3_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_R_wm4  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_09_white4_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_R_blm1 = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_01_black1_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_R_blm2 = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_02_black2_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_R_blm3 = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_05_black5_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_R_blm4 = np.loadtxt(open(path+'SHIRTS/RED/negative red/oresthstest15_black2_m1_rgbs_shirt.csv',"rb"),delimiter=";",skiprows=0)

# loads training data for Green SVM
rgbs_G_rm1  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_27_red1_m1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbs_G_rm2  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_28_red2_m1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbs_G_rm3  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_30_red4_m1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbs_G_rm4  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_32_red6_m1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbs_G_rm5  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_33_red7_m1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
#rgbs_G_rm6  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/kostas_6_red2_m1_rgbs_shirt.csv'  ,"rb") ,delimiter=";",skiprows=0)
rgbs_G_gp1  = np.loadtxt(open(path+'SHIRTS/GREEN/positive green/mar_23_green2_p1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_G_gp2  = np.loadtxt(open(path+'SHIRTS/GREEN/positive green/mar_24_green3_p1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_G_gp3  = np.loadtxt(open(path+'SHIRTS/GREEN/positive green/geo_38_greengeo1_p1_rgbs_shirt.csv',"rb"),delimiter=";",skiprows=0)
rgbs_G_gp4  = np.loadtxt(open(path+'SHIRTS/GREEN/positive green/mar_26_green5_p1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_G_gp5  = np.loadtxt(open(path+'SHIRTS/GREEN/positive green/mariatest4_green4_p1_rgbs_shirt.csv',"rb") ,delimiter=";",skiprows=0)
rgbs_G_bm1  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_10_blue1_m1_rgbs_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbs_G_bm2  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_12_blue3_m1_rgbs_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbs_G_bm3  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_16_blue7_m1_rgbs_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbs_G_bm4  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_17_blue8_m1_rgbs_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbs_G_bm5  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mariatest9_blue2_m1_rgbs_shirt.csv' ,"rb") ,delimiter=";",skiprows=0)
rgbs_G_wm1  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_06_white1_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_G_wm2  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_07_white2_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_G_wm3  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_08_white3_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_G_wm4  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_09_white4_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_G_blm1 = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_01_black1_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_G_blm2 = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_02_black2_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_G_blm3 = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_05_black5_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_G_blm4 = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/oresthstest15_black2_m1_rgbs_shirt.csv',"rb"),delimiter=";",skiprows=0)

# loads training data for Blue SVM
rgbs_B_rm1  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_33_red7_m1_rgbs_shirt.csv',"rb")     ,delimiter=";",skiprows=0)
rgbs_B_rm2  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_32_red6_m1_rgbs_shirt.csv',"rb")     ,delimiter=";",skiprows=0)
rgbs_B_rm3  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_30_red4_m1_rgbs_shirt.csv',"rb")     ,delimiter=";",skiprows=0)
rgbs_B_rm4  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_28_red2_m1_rgbs_shirt.csv',"rb")     ,delimiter=";",skiprows=0)
rgbs_B_rm5  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_27_red1_m1_rgbs_shirt.csv',"rb")     ,delimiter=";",skiprows=0)
#rgbs_B_rm6  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/kostas_6_red2_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_B_gm1  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_26_green5_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_B_gm2  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/geo_38_greengeo1_m1_rgbs_shirt.csv',"rb"),delimiter=";",skiprows=0)
rgbs_B_gm3  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_24_green3_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_B_gm4  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_23_green2_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_B_bp1  = np.loadtxt(open(path+'SHIRTS/BLUE/positive blue/mar_10_blue1_p1_rgbs_shirt.csv',"rb")    ,delimiter=";",skiprows=0)
rgbs_B_bp2  = np.loadtxt(open(path+'SHIRTS/BLUE/positive blue/mar_12_blue3_p1_rgbs_shirt.csv',"rb")    ,delimiter=";",skiprows=0)
rgbs_B_bp3  = np.loadtxt(open(path+'SHIRTS/BLUE/positive blue/mar_16_blue7_p1_rgbs_shirt.csv',"rb")    ,delimiter=";",skiprows=0)
rgbs_B_bp4  = np.loadtxt(open(path+'SHIRTS/BLUE/positive blue/mar_17_blue8_p1_rgbs_shirt.csv',"rb")    ,delimiter=";",skiprows=0)
rgbs_B_bp5  = np.loadtxt(open(path+'SHIRTS/BLUE/positive blue/mariatest9_blue2_p1_rgbs_shirt.csv',"rb")  ,delimiter=";",skiprows=0)
rgbs_B_wm1  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_09_white4_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_B_wm2  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_08_white3_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_B_wm3  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_07_white2_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_B_wm4  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_06_white1_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_B_blm1 = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_01_black1_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_B_blm2 = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_02_black2_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_B_blm3 = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_05_black5_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_B_blm4 = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/oresthstest15_black2_m1_rgbs_shirt.csv',"rb"),delimiter=";",skiprows=0)

# loads training data for White SVM
rgbs_W_rm1  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_27_red1_m1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbs_W_rm2  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_28_red2_m1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbs_W_rm3  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_30_red4_m1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbs_W_rm4  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_32_red6_m1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbs_W_rm5  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_33_red7_m1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
#rgbs_W_rm6  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/kostas_6_red2_m1_rgbs_shirt.csv'  ,"rb") ,delimiter=";",skiprows=0)
rgbs_W_gm1  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_23_green2_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_W_gm2  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_24_green3_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_W_gm3  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/geo_38_greengeo1_m1_rgbs_shirt.csv',"rb"),delimiter=";",skiprows=0)
rgbs_W_gm4  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_26_green5_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_W_bm1  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_10_blue1_m1_rgbs_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbs_W_bm2  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_12_blue3_m1_rgbs_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbs_W_bm3  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_16_blue7_m1_rgbs_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbs_W_bm4  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_17_blue8_m1_rgbs_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbs_W_bm5  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mariatest9_blue2_m1_rgbs_shirt.csv' ,"rb") ,delimiter=";",skiprows=0)
rgbs_W_wp1  = np.loadtxt(open(path+'SHIRTS/WHITE/positive white/mar_06_white1_p1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_W_wp2  = np.loadtxt(open(path+'SHIRTS/WHITE/positive white/mar_07_white2_p1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_W_wp3  = np.loadtxt(open(path+'SHIRTS/WHITE/positive white/mar_08_white3_p1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_W_wp4  = np.loadtxt(open(path+'SHIRTS/WHITE/positive white/mar_09_white4_p1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_W_blm1 = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_01_black1_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_W_blm2 = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_02_black2_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_W_blm3 = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_05_black5_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_W_blm4 = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/oresthstest15_black2_m1_rgbs_shirt.csv',"rb"),delimiter=";",skiprows=0)

# loads training data for Black SVM
rgbs_BL_rm1  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_27_red1_m1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_rm2  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_28_red2_m1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_rm3  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_30_red4_m1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_rm4  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_32_red6_m1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_rm5  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_33_red7_m1_rgbs_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
#rgbs_BL_rm6  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/kostas_6_red2_m1_rgbs_shirt.csv'  ,"rb") ,delimiter=";",skiprows=0)
rgbs_BL_gm1  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_23_green2_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_gm2  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_24_green3_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_gm3  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/geo_38_greengeo1_m1_rgbs_shirt.csv',"rb"),delimiter=";",skiprows=0)
rgbs_BL_gm4  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_26_green5_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_bm1  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_10_blue1_m1_rgbs_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_bm2  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_12_blue3_m1_rgbs_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_bm3  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_16_blue7_m1_rgbs_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_bm4  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_17_blue8_m1_rgbs_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_bm5  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mariatest9_blue2_m1_rgbs_shirt.csv' ,"rb") ,delimiter=";",skiprows=0)
rgbs_BL_wm1  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_06_white1_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_wm2  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_07_white2_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_wm3  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_08_white3_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_wm4  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_09_white4_m1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_blp1 = np.loadtxt(open(path+'SHIRTS/BLACK/positive black/mar_01_black1_p1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_blp2 = np.loadtxt(open(path+'SHIRTS/BLACK/positive black/mar_02_black2_p1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_blp3 = np.loadtxt(open(path+'SHIRTS/BLACK/positive black/mar_05_black5_p1_rgbs_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbs_BL_blp4 = np.loadtxt(open(path+'SHIRTS/BLACK/positive black/oresthstest15_black2_p1_rgbs_shirt.csv',"rb"),delimiter=";",skiprows=0)
rgbs_BL_blp5 = np.loadtxt(open(path+'SHIRTS/BLACK/positive black/kos_36_blackkos1_p1_rgbs_shirt.csv',"rb"),delimiter=";",skiprows=0)

# training set for redsvm
# the x part is all the feature vectors and the y part is 1 or -1
R_rp1x,R_rp1y   = rgbs_R_rp1[:, :230],rgbs_R_rp1[:, 230]
R_rp2x,R_rp2y   = rgbs_R_rp2[:, :230],rgbs_R_rp2[:, 230]
R_rp3x,R_rp3y   = rgbs_R_rp3[:, :230],rgbs_R_rp3[:, 230]
R_rp4x,R_rp4y   = rgbs_R_rp4[:, :230],rgbs_R_rp4[:, 230]
R_rp5x,R_rp5y   = rgbs_R_rp5[:, :230],rgbs_R_rp5[:, 230]
#R_rp6x,R_rp6y   = rgbs_R_rp6[:, :230],rgbs_R_rp6[:, 230]
#R_rp7x,R_rp7y   = rgbs_R_rp7[:, :230],rgbs_R_rp7[:, 230]
R_gm1x,R_gm1y   = rgbs_R_gm1[:, :230],rgbs_R_gm1[:, 230]
R_gm2x,R_gm2y   = rgbs_R_gm2[:, :230],rgbs_R_gm2[:, 230]
R_gm3x,R_gm3y   = rgbs_R_gm3[:, :230],rgbs_R_gm3[:, 230]
R_gm4x,R_gm4y   = rgbs_R_gm4[:, :230],rgbs_R_gm4[:, 230]
R_bm1x,R_bm1y   = rgbs_R_bm1[:, :230],rgbs_R_bm1[:, 230]
R_bm2x,R_bm2y   = rgbs_R_bm2[:, :230],rgbs_R_bm2[:, 230]
R_bm3x,R_bm3y   = rgbs_R_bm3[:, :230],rgbs_R_bm3[:, 230]
R_bm4x,R_bm4y   = rgbs_R_bm4[:, :230],rgbs_R_bm4[:, 230]
R_bm5x,R_bm5y   = rgbs_R_bm5[:, :230],rgbs_R_bm5[:, 230]
R_wm1x,R_wm1y   = rgbs_R_wm1[:, :230],rgbs_R_wm1[:, 230]
R_wm2x,R_wm2y   = rgbs_R_wm2[:, :230],rgbs_R_wm2[:, 230]
R_wm3x,R_wm3y   = rgbs_R_wm3[:, :230],rgbs_R_wm3[:, 230]
R_wm4x,R_wm4y   = rgbs_R_wm4[:, :230],rgbs_R_wm4[:, 230]
R_blm1x,R_blm1y = rgbs_R_blm1[:, :230],rgbs_R_blm1[:, 230]
R_blm2x,R_blm2y = rgbs_R_blm2[:, :230],rgbs_R_blm2[:, 230]
R_blm3x,R_blm3y = rgbs_R_blm3[:, :230],rgbs_R_blm3[:, 230]
R_blm4x,R_blm4y = rgbs_R_blm4[:, :230],rgbs_R_blm4[:, 230]

# prepares the training data in order to give them as arguments for learning
rgbs_Rtx = np.vstack([R_gm1x,R_rp1x,R_bm1x,R_rp2x,R_wm1x,R_rp3x,R_blm1x,R_rp4x,R_blm4x,R_rp5x,R_bm5x])
rgbs_Rty = np.concatenate((R_gm1y,R_rp1y,R_bm1y,R_rp2y,R_wm1y,R_rp3y,R_blm1y,R_rp4y,R_blm4y,R_rp5y,R_bm5y))
rgbs_redsvm.learn(rgbs_Rtx,rgbs_Rty)

# training set for greensvm
# the x part is all the feature vectors and the y part is 1 or -1
G_rm1x,G_rm1y   = rgbs_G_rm1[:, :230],rgbs_G_rm1[:, 230]
G_rm2x,G_rm2y   = rgbs_G_rm2[:, :230],rgbs_G_rm2[:, 230]
G_rm3x,G_rm3y   = rgbs_G_rm3[:, :230],rgbs_G_rm3[:, 230]
G_rm4x,G_rm4y   = rgbs_G_rm4[:, :230],rgbs_G_rm4[:, 230]
G_rm5x,G_rm5y   = rgbs_G_rm5[:, :230],rgbs_G_rm5[:, 230]
G_gp1x,G_gp1y   = rgbs_G_gp1[:, :230],rgbs_G_gp1[:, 230]
G_gp2x,G_gp2y   = rgbs_G_gp2[:, :230],rgbs_G_gp2[:, 230]
G_gp3x,G_gp3y   = rgbs_G_gp3[:, :230],rgbs_G_gp3[:, 230]
G_gp4x,G_gp4y   = rgbs_G_gp4[:, :230],rgbs_G_gp4[:, 230]
G_gp5x,G_gp5y   = rgbs_G_gp5[:, :230],rgbs_G_gp5[:, 230]
G_bm1x,G_bm1y   = rgbs_G_bm1[:, :230],rgbs_G_bm1[:, 230]
G_bm2x,G_bm2y   = rgbs_G_bm2[:, :230],rgbs_G_bm2[:, 230]
G_bm3x,G_bm3y   = rgbs_G_bm3[:, :230],rgbs_G_bm3[:, 230]
G_bm4x,G_bm4y   = rgbs_G_bm4[:, :230],rgbs_G_bm4[:, 230]
G_bm5x,G_bm5y   = rgbs_G_bm5[:, :230],rgbs_G_bm5[:, 230]
G_wm1x,G_wm1y   = rgbs_G_wm1[:, :230],rgbs_G_wm1[:, 230]
G_wm2x,G_wm2y   = rgbs_G_wm2[:, :230],rgbs_G_wm2[:, 230]
G_wm3x,G_wm3y   = rgbs_G_wm3[:, :230],rgbs_G_wm3[:, 230]
G_wm4x,G_wm4y   = rgbs_G_wm4[:, :230],rgbs_G_wm4[:, 230]
G_blm1x,G_blm1y = rgbs_G_blm1[:, :230],rgbs_G_blm1[:, 230]
G_blm2x,G_blm2y = rgbs_G_blm2[:, :230],rgbs_G_blm2[:, 230]
G_blm3x,G_blm3y = rgbs_G_blm3[:, :230],rgbs_G_blm3[:, 230]
G_blm4x,G_blm4y = rgbs_G_blm4[:, :230],rgbs_G_blm4[:, 230]

# prepares the training data in order to give them as arguments for learning
rgbs_Gtx = np.vstack([G_rm1x,G_gp1x,G_bm1x,G_wm1x,G_gp2x,G_blm1x,G_gp3x,G_rm5x,G_gp5x,G_blm4x,G_bm5x])
rgbs_Gty = np.concatenate((G_rm1y,G_gp1y,G_bm1y,G_wm1y,G_gp2y,G_blm1y,G_gp3y,G_rm5y,G_gp5y,G_blm4y,G_bm5y))
rgbs_greensvm.learn(rgbs_Gtx,rgbs_Gty)

# training set for bluesvm
# the x part is all the feature vectors and the y part is 1 or -1
B_rm1x,B_rm1y   = rgbs_B_rm1[:, :230],rgbs_B_rm1[:, 230]
B_rm2x,B_rm2y   = rgbs_B_rm2[:, :230],rgbs_B_rm2[:, 230]
B_rm3x,B_rm3y   = rgbs_B_rm3[:, :230],rgbs_B_rm3[:, 230]
B_rm4x,B_rm4y   = rgbs_B_rm4[:, :230],rgbs_B_rm4[:, 230]
B_rm5x,B_rm5y   = rgbs_B_rm5[:, :230],rgbs_B_rm5[:, 230]
B_gm1x,B_gm1y   = rgbs_B_gm1[:, :230],rgbs_B_gm1[:, 230]
B_gm2x,B_gm2y   = rgbs_B_gm2[:, :230],rgbs_B_gm2[:, 230]
B_gm3x,B_gm3y   = rgbs_B_gm3[:, :230],rgbs_B_gm3[:, 230]
B_gm4x,B_gm4y   = rgbs_B_gm4[:, :230],rgbs_B_gm4[:, 230]
B_bp1x,B_bp1y   = rgbs_B_bp1[:, :230],rgbs_B_bp1[:, 230]
B_bp2x,B_bp2y   = rgbs_B_bp2[:, :230],rgbs_B_bp2[:, 230]
B_bp3x,B_bp3y   = rgbs_B_bp3[:, :230],rgbs_B_bp3[:, 230]
B_bp4x,B_bp4y   = rgbs_B_bp4[:, :230],rgbs_B_bp4[:, 230]
#B_bp5x,B_bp5y   = rgbs_B_bp5[:, :230],rgbs_B_bp5[:, 230]
B_wm1x,B_wm1y   = rgbs_B_wm1[:, :230],rgbs_B_wm1[:, 230]
B_wm2x,B_wm2y   = rgbs_B_wm2[:, :230],rgbs_B_wm2[:, 230]
B_wm3x,B_wm3y   = rgbs_B_wm3[:, :230],rgbs_B_wm3[:, 230]
B_wm4x,B_wm4y   = rgbs_B_wm4[:, :230],rgbs_B_wm4[:, 230]
B_blm1x,B_blm1y = rgbs_B_blm1[:, :230],rgbs_B_blm1[:, 230]
B_blm2x,B_blm2y = rgbs_B_blm2[:, :230],rgbs_B_blm2[:, 230]
B_blm3x,B_blm3y = rgbs_B_blm3[:, :230],rgbs_B_blm3[:, 230]
B_blm4x,B_blm4y = rgbs_B_blm4[:, :230],rgbs_B_blm4[:, 230]

# prepares the training data in order to give them as arguments for learning
rgbs_Btx = np.vstack([B_rm1x,B_bp1x,B_wm1x,B_bp2x,B_gm1x,B_bp3x,B_blm1x,B_blm4x])
rgbs_Bty = np.concatenate((B_rm1y,B_bp1y,B_wm1y,B_bp2y,B_gm1y,B_bp3y,B_blm1y,B_blm4y)) 
rgbs_bluesvm.learn(rgbs_Btx,rgbs_Bty)

# training set for whitesvm
# the x part is all the feature vectors and the y part is 1 or -1
W_rm1x,W_rm1y   = rgbs_W_rm1[:, :230],rgbs_W_rm1[:, 230]
W_rm2x,W_rm2y   = rgbs_W_rm2[:, :230],rgbs_W_rm2[:, 230]
W_rm3x,W_rm3y   = rgbs_W_rm3[:, :230],rgbs_W_rm3[:, 230]
W_rm4x,W_rm4y   = rgbs_W_rm4[:, :230],rgbs_W_rm4[:, 230]
W_rm5x,W_rm5y   = rgbs_W_rm5[:, :230],rgbs_W_rm5[:, 230]
W_gm1x,W_gm1y   = rgbs_W_gm1[:, :230],rgbs_W_gm1[:, 230]
W_gm2x,W_gm2y   = rgbs_W_gm2[:, :230],rgbs_W_gm2[:, 230]
W_gm3x,W_gm3y   = rgbs_W_gm3[:, :230],rgbs_W_gm3[:, 230]
W_gm4x,W_gm4y   = rgbs_W_gm4[:, :230],rgbs_W_gm4[:, 230]
W_bm1x,W_bm1y   = rgbs_W_bm1[:, :230],rgbs_W_bm1[:, 230]
W_bm2x,W_bm2y   = rgbs_W_bm2[:, :230],rgbs_W_bm2[:, 230]
W_bm3x,W_bm3y   = rgbs_W_bm3[:, :230],rgbs_W_bm3[:, 230]
W_bm4x,W_bm4y   = rgbs_W_bm4[:, :230],rgbs_W_bm4[:, 230]
W_bm5x,W_bm5y   = rgbs_W_bm5[:, :230],rgbs_W_bm5[:, 230]
W_wp1x,W_wp1y   = rgbs_W_wp1[:, :230],rgbs_W_wp1[:, 230]
W_wp2x,W_wp2y   = rgbs_W_wp2[:, :230],rgbs_W_wp2[:, 230]
W_wp3x,W_wp3y   = rgbs_W_wp3[:, :230],rgbs_W_wp3[:, 230]
W_wp4x,W_wp4y   = rgbs_W_wp4[:, :230],rgbs_W_wp4[:, 230]
W_blm1x,W_blm1y = rgbs_W_blm1[:, :230],rgbs_W_blm1[:, 230]
W_blm2x,W_blm2y = rgbs_W_blm2[:, :230],rgbs_W_blm2[:, 230]
W_blm3x,W_blm3y = rgbs_W_blm3[:, :230],rgbs_W_blm3[:, 230]
W_blm4x,W_blm4y = rgbs_W_blm4[:, :230],rgbs_W_blm4[:, 230]   

# prepares the training data in order to give them as arguments for learning
rgbs_Wtx = np.vstack([W_rm1x,W_wp1x,W_blm1x,W_wp2x,W_bm1x,W_wp3x,W_gm1x,W_blm4x,W_bm5x])
rgbs_Wty = np.concatenate((W_rm1y,W_wp1y,W_blm1y,W_wp2y,W_bm1y,W_wp3y,W_gm1y,W_blm4y,W_bm5y)) 
rgbs_whitesvm.learn(rgbs_Wtx,rgbs_Wty)

# training set for blacksvm
# the x part is all the feature vectors and the y part is 1 or -1
BL_rm1x,BL_rm1y   = rgbs_BL_rm1[:, :230],rgbs_BL_rm1[:, 230]
BL_rm2x,BL_rm2y   = rgbs_BL_rm2[:, :230],rgbs_BL_rm2[:, 230]
BL_rm3x,BL_rm3y   = rgbs_BL_rm3[:, :230],rgbs_BL_rm3[:, 230]
BL_rm4x,BL_rm4y   = rgbs_BL_rm4[:, :230],rgbs_BL_rm4[:, 230]
BL_rm5x,BL_rm5y   = rgbs_BL_rm5[:, :230],rgbs_BL_rm5[:, 230]
BL_gm1x,BL_gm1y   = rgbs_BL_gm1[:, :230],rgbs_BL_gm1[:, 230]
BL_gm2x,BL_gm2y   = rgbs_BL_gm2[:, :230],rgbs_BL_gm2[:, 230]
BL_gm3x,BL_gm3y   = rgbs_BL_gm3[:, :230],rgbs_BL_gm3[:, 230]
BL_gm4x,BL_gm4y   = rgbs_BL_gm4[:, :230],rgbs_BL_gm4[:, 230]
BL_bm1x,BL_bm1y   = rgbs_BL_bm1[:, :230],rgbs_BL_bm1[:, 230]
BL_bm2x,BL_bm2y   = rgbs_BL_bm2[:, :230],rgbs_BL_bm2[:, 230]
BL_bm3x,BL_bm3y   = rgbs_BL_bm3[:, :230],rgbs_BL_bm3[:, 230]
BL_bm4x,BL_bm4y   = rgbs_BL_bm4[:, :230],rgbs_BL_bm4[:, 230]
BL_bm5x,BL_bm5y   = rgbs_BL_bm5[:, :230],rgbs_BL_bm5[:, 230]
BL_wm1x,BL_wm1y   = rgbs_BL_wm1[:, :230],rgbs_BL_wm1[:, 230]
BL_wm2x,BL_wm2y   = rgbs_BL_wm2[:, :230],rgbs_BL_wm2[:, 230]
BL_wm3x,BL_wm3y   = rgbs_BL_wm3[:, :230],rgbs_BL_wm3[:, 230]
BL_wm4x,BL_wm4y   = rgbs_BL_wm4[:, :230],rgbs_BL_wm4[:, 230]
BL_blp1x,BL_blp1y = rgbs_BL_blp1[:, :230],rgbs_BL_blp1[:, 230]
BL_blp2x,BL_blp2y = rgbs_BL_blp2[:, :230],rgbs_BL_blp2[:, 230]
BL_blp3x,BL_blp3y = rgbs_BL_blp3[:, :230],rgbs_BL_blp3[:, 230]
BL_blp4x,BL_blp4y = rgbs_BL_blp4[:, :230],rgbs_BL_blp4[:, 230]
BL_blp5x,BL_blp5y = rgbs_BL_blp5[:, :230],rgbs_BL_blp5[:, 230]

# prepares the training data in order to give them as arguments for learning
rgbs_BLtx = np.vstack([BL_rm1x,BL_blp1x,BL_bm1x,BL_blp2x,BL_gm1x,BL_blp3x,BL_wm1x,BL_blp4x,BL_rm5x,BL_blp5x,BL_bm5x])
rgbs_BLty = np.concatenate((BL_rm1y,BL_blp1y,BL_bm1y,BL_blp2y,BL_gm1y,BL_blp3y,BL_wm1y,BL_blp4y,BL_rm5y,BL_blp5y,BL_bm5y)) 
rgbs_blacksvm.learn(rgbs_BLtx,rgbs_BLty)

# save trainied libsvms as xml files
rgbs_redsvm.save_model('rgbs_redsvm.xml')
rgbs_greensvm.save_model('rgbs_greensvm.xml')
rgbs_bluesvm.save_model('rgbs_bluesvm.xml')
rgbs_whitesvm.save_model('rgbs_whitesvm.xml')
rgbs_blacksvm.save_model('rgbs_blacksvm.xml')

print '\nColor RGB S svms has been created successfully!'

###--------------------###

############################
''' Train RGB HSV Libsvms'''
############################

# creates color libsvms
rgbhsv_redsvm    = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgbhsv_greensvm  = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgbhsv_bluesvm   = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgbhsv_whitesvm  = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)
rgbhsv_blacksvm  = ml.LibSvm(svm_type='c_svc', kernel_type='linear', C=1, probability=True)

# loads training data for Red SVM
rgbhsv_R_rp1  = np.loadtxt(open(path+'SHIRTS/RED/positive red/mar_27_red1_p1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_rp2  = np.loadtxt(open(path+'SHIRTS/RED/positive red/mar_28_red2_p1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_rp3  = np.loadtxt(open(path+'SHIRTS/RED/positive red/mar_30_red4_p1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_rp4  = np.loadtxt(open(path+'SHIRTS/RED/positive red/mar_32_red6_p1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_rp5  = np.loadtxt(open(path+'SHIRTS/RED/positive red/mar_33_red7_p1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
#rgbhsv_R_rp6  = np.loadtxt(open(path+'SHIRTS/RED/positive red/kostas_6_red2_p1_rgbhsv_shirt.csv'  ,"rb") ,delimiter=";",skiprows=0)
#rgbhsv_R_rp7  = np.loadtxt(open(path+'SHIRTS/RED/positive red/kostas_6_red2_p1_rgbhsv_shirt.csv'  ,"rb") ,delimiter=";",skiprows=0)
rgbhsv_R_gm1  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_24_green3_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_gm2  = np.loadtxt(open(path+'SHIRTS/RED/negative red/geo_38_greengeo1_m1_rgbhsv_shirt.csv',"rb"),delimiter=";",skiprows=0)
rgbhsv_R_gm3  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_26_green5_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_gm4  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_23_green2_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_bm1  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_10_blue1_m1_rgbhsv_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_bm2  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_12_blue3_m1_rgbhsv_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_bm3  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_16_blue7_m1_rgbhsv_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_bm4  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_17_blue8_m1_rgbhsv_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_bm5  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mariatest9_blue2_m1_rgbhsv_shirt.csv' ,"rb") ,delimiter=";",skiprows=0)
rgbhsv_R_wm1  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_06_white1_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_wm2  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_07_white2_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_wm3  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_08_white3_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_wm4  = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_09_white4_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_blm1 = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_01_black1_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_blm2 = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_02_black2_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_blm3 = np.loadtxt(open(path+'SHIRTS/RED/negative red/mar_05_black5_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_R_blm4 = np.loadtxt(open(path+'SHIRTS/RED/negative red/oresthstest15_black2_m1_rgbhsv_shirt.csv',"rb"),delimiter=";",skiprows=0)

# loads training data for Green SVM
rgbhsv_G_rm1  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_27_red1_m1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_rm2  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_28_red2_m1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_rm3  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_30_red4_m1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_rm4  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_32_red6_m1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_rm5  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_33_red7_m1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
#rgbhsv_G_rm6  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/kostas_6_red2_m1_rgbhsv_shirt.csv'  ,"rb") ,delimiter=";",skiprows=0)
rgbhsv_G_gp1  = np.loadtxt(open(path+'SHIRTS/GREEN/positive green/mar_23_green2_p1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_gp2  = np.loadtxt(open(path+'SHIRTS/GREEN/positive green/mar_24_green3_p1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_gp3  = np.loadtxt(open(path+'SHIRTS/GREEN/positive green/geo_38_greengeo1_p1_rgbhsv_shirt.csv',"rb"),delimiter=";",skiprows=0)
rgbhsv_G_gp4  = np.loadtxt(open(path+'SHIRTS/GREEN/positive green/mar_26_green5_p1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_gp5  = np.loadtxt(open(path+'SHIRTS/GREEN/positive green/mariatest4_green4_p1_rgbhsv_shirt.csv',"rb") ,delimiter=";",skiprows=0)
rgbhsv_G_bm1  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_10_blue1_m1_rgbhsv_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_bm2  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_12_blue3_m1_rgbhsv_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_bm3  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_16_blue7_m1_rgbhsv_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_bm4  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_17_blue8_m1_rgbhsv_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_bm5  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mariatest9_blue2_m1_rgbhsv_shirt.csv' ,"rb") ,delimiter=";",skiprows=0)
rgbhsv_G_wm1  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_06_white1_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_wm2  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_07_white2_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_wm3  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_08_white3_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_wm4  = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_09_white4_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_blm1 = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_01_black1_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_blm2 = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_02_black2_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_blm3 = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/mar_05_black5_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_G_blm4 = np.loadtxt(open(path+'SHIRTS/GREEN/negative green/oresthstest15_black2_m1_rgbhsv_shirt.csv',"rb"),delimiter=";",skiprows=0)

# loads training data for Blue SVM
rgbhsv_B_rm1  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_33_red7_m1_rgbhsv_shirt.csv',"rb")     ,delimiter=";",skiprows=0)
rgbhsv_B_rm2  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_32_red6_m1_rgbhsv_shirt.csv',"rb")     ,delimiter=";",skiprows=0)
rgbhsv_B_rm3  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_30_red4_m1_rgbhsv_shirt.csv',"rb")     ,delimiter=";",skiprows=0)
rgbhsv_B_rm4  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_28_red2_m1_rgbhsv_shirt.csv',"rb")     ,delimiter=";",skiprows=0)
rgbhsv_B_rm5  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_27_red1_m1_rgbhsv_shirt.csv',"rb")     ,delimiter=";",skiprows=0)
#rgbhsv_B_rm6  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/kostas_6_red2_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_B_gm1  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_26_green5_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_B_gm2  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/geo_38_greengeo1_m1_rgbhsv_shirt.csv',"rb"),delimiter=";",skiprows=0)
rgbhsv_B_gm3  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_24_green3_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_B_gm4  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_23_green2_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_B_bp1  = np.loadtxt(open(path+'SHIRTS/BLUE/positive blue/mar_10_blue1_p1_rgbhsv_shirt.csv',"rb")    ,delimiter=";",skiprows=0)
rgbhsv_B_bp2  = np.loadtxt(open(path+'SHIRTS/BLUE/positive blue/mar_12_blue3_p1_rgbhsv_shirt.csv',"rb")    ,delimiter=";",skiprows=0)
rgbhsv_B_bp3  = np.loadtxt(open(path+'SHIRTS/BLUE/positive blue/mar_16_blue7_p1_rgbhsv_shirt.csv',"rb")    ,delimiter=";",skiprows=0)
rgbhsv_B_bp4  = np.loadtxt(open(path+'SHIRTS/BLUE/positive blue/mar_17_blue8_p1_rgbhsv_shirt.csv',"rb")    ,delimiter=";",skiprows=0)
rgbhsv_B_bp5  = np.loadtxt(open(path+'SHIRTS/BLUE/positive blue/mariatest9_blue2_p1_rgbhsv_shirt.csv',"rb")  ,delimiter=";",skiprows=0)
rgbhsv_B_wm1  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_09_white4_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_B_wm2  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_08_white3_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_B_wm3  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_07_white2_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_B_wm4  = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_06_white1_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_B_blm1 = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_01_black1_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_B_blm2 = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_02_black2_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_B_blm3 = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/mar_05_black5_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_B_blm4 = np.loadtxt(open(path+'SHIRTS/BLUE/negative blue/oresthstest15_black2_m1_rgbhsv_shirt.csv',"rb"),delimiter=";",skiprows=0)

# loads training data for White SVM
rgbhsv_W_rm1  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_27_red1_m1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_rm2  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_28_red2_m1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_rm3  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_30_red4_m1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_rm4  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_32_red6_m1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_rm5  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_33_red7_m1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
#rgbhsv_W_rm6  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/kostas_6_red2_m1_rgbhsv_shirt.csv'  ,"rb") ,delimiter=";",skiprows=0)
rgbhsv_W_gm1  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_23_green2_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_gm2  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_24_green3_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_gm3  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/geo_38_greengeo1_m1_rgbhsv_shirt.csv',"rb"),delimiter=";",skiprows=0)
rgbhsv_W_gm4  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_26_green5_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_bm1  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_10_blue1_m1_rgbhsv_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_bm2  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_12_blue3_m1_rgbhsv_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_bm3  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_16_blue7_m1_rgbhsv_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_bm4  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_17_blue8_m1_rgbhsv_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_bm5  = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mariatest9_blue2_m1_rgbhsv_shirt.csv' ,"rb") ,delimiter=";",skiprows=0)
rgbhsv_W_wp1  = np.loadtxt(open(path+'SHIRTS/WHITE/positive white/mar_06_white1_p1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_wp2  = np.loadtxt(open(path+'SHIRTS/WHITE/positive white/mar_07_white2_p1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_wp3  = np.loadtxt(open(path+'SHIRTS/WHITE/positive white/mar_08_white3_p1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_wp4  = np.loadtxt(open(path+'SHIRTS/WHITE/positive white/mar_09_white4_p1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_blm1 = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_01_black1_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_blm2 = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_02_black2_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_blm3 = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/mar_05_black5_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_W_blm4 = np.loadtxt(open(path+'SHIRTS/WHITE/negative white/oresthstest15_black2_m1_rgbhsv_shirt.csv',"rb"),delimiter=";",skiprows=0)

# loads training data for Black SVM
rgbhsv_BL_rm1  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_27_red1_m1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_rm2  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_28_red2_m1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_rm3  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_30_red4_m1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_rm4  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_32_red6_m1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_rm5  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_33_red7_m1_rgbhsv_shirt.csv'  ,"rb")   ,delimiter=";",skiprows=0)
#rgbhsv_BL_rm6  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/kostas_6_red2_m1_rgbhsv_shirt.csv'  ,"rb") ,delimiter=";",skiprows=0)
rgbhsv_BL_gm1  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_23_green2_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_gm2  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_24_green3_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_gm3  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/geo_38_greengeo1_m1_rgbhsv_shirt.csv',"rb"),delimiter=";",skiprows=0)
rgbhsv_BL_gm4  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_26_green5_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_bm1  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_10_blue1_m1_rgbhsv_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_bm2  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_12_blue3_m1_rgbhsv_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_bm3  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_16_blue7_m1_rgbhsv_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_bm4  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_17_blue8_m1_rgbhsv_shirt.csv' ,"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_bm5  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mariatest9_blue2_m1_rgbhsv_shirt.csv' ,"rb") ,delimiter=";",skiprows=0)
rgbhsv_BL_wm1  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_06_white1_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_wm2  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_07_white2_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_wm3  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_08_white3_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_wm4  = np.loadtxt(open(path+'SHIRTS/BLACK/negative black/mar_09_white4_m1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_blp1 = np.loadtxt(open(path+'SHIRTS/BLACK/positive black/mar_01_black1_p1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_blp2 = np.loadtxt(open(path+'SHIRTS/BLACK/positive black/mar_02_black2_p1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_blp3 = np.loadtxt(open(path+'SHIRTS/BLACK/positive black/mar_05_black5_p1_rgbhsv_shirt.csv',"rb")   ,delimiter=";",skiprows=0)
rgbhsv_BL_blp4 = np.loadtxt(open(path+'SHIRTS/BLACK/positive black/oresthstest15_black2_p1_rgbhsv_shirt.csv',"rb"),delimiter=";",skiprows=0)
rgbhsv_BL_blp5 = np.loadtxt(open(path+'SHIRTS/BLACK/positive black/kos_36_blackkos1_p1_rgbhsv_shirt.csv',"rb"),delimiter=";",skiprows=0)

# training set for redsvm
# the x part is all the feature vectors and the y part is 1 or -1
R_rp1x,R_rp1y   = rgbhsv_R_rp1[:, :360],rgbhsv_R_rp1[:, 360]
R_rp2x,R_rp2y   = rgbhsv_R_rp2[:, :360],rgbhsv_R_rp2[:, 360]
R_rp3x,R_rp3y   = rgbhsv_R_rp3[:, :360],rgbhsv_R_rp3[:, 360]
R_rp4x,R_rp4y   = rgbhsv_R_rp4[:, :360],rgbhsv_R_rp4[:, 360]
R_rp5x,R_rp5y   = rgbhsv_R_rp5[:, :360],rgbhsv_R_rp5[:, 360]
#R_rp6x,R_rp6y   = rgbhsv_R_rp6[:, :360],rgbhsv_R_rp6[:, 360]
#R_rp7x,R_rp7y   = rgbhsv_R_rp7[:, :360],rgbhsv_R_rp7[:, 360]
R_gm1x,R_gm1y   = rgbhsv_R_gm1[:, :360],rgbhsv_R_gm1[:, 360]
R_gm2x,R_gm2y   = rgbhsv_R_gm2[:, :360],rgbhsv_R_gm2[:, 360]
R_gm3x,R_gm3y   = rgbhsv_R_gm3[:, :360],rgbhsv_R_gm3[:, 360]
R_gm4x,R_gm4y   = rgbhsv_R_gm4[:, :360],rgbhsv_R_gm4[:, 360]
R_bm1x,R_bm1y   = rgbhsv_R_bm1[:, :360],rgbhsv_R_bm1[:, 360]
R_bm2x,R_bm2y   = rgbhsv_R_bm2[:, :360],rgbhsv_R_bm2[:, 360]
R_bm3x,R_bm3y   = rgbhsv_R_bm3[:, :360],rgbhsv_R_bm3[:, 360]
R_bm4x,R_bm4y   = rgbhsv_R_bm4[:, :360],rgbhsv_R_bm4[:, 360]
R_bm5x,R_bm5y   = rgbhsv_R_bm5[:, :360],rgbhsv_R_bm5[:, 360]
R_wm1x,R_wm1y   = rgbhsv_R_wm1[:, :360],rgbhsv_R_wm1[:, 360]
R_wm2x,R_wm2y   = rgbhsv_R_wm2[:, :360],rgbhsv_R_wm2[:, 360]
R_wm3x,R_wm3y   = rgbhsv_R_wm3[:, :360],rgbhsv_R_wm3[:, 360]
R_wm4x,R_wm4y   = rgbhsv_R_wm4[:, :360],rgbhsv_R_wm4[:, 360]
R_blm1x,R_blm1y = rgbhsv_R_blm1[:, :360],rgbhsv_R_blm1[:, 360]
R_blm2x,R_blm2y = rgbhsv_R_blm2[:, :360],rgbhsv_R_blm2[:, 360]
R_blm3x,R_blm3y = rgbhsv_R_blm3[:, :360],rgbhsv_R_blm3[:, 360]
R_blm4x,R_blm4y = rgbhsv_R_blm4[:, :360],rgbhsv_R_blm4[:, 360]

# prepares the training data in order to give them as arguments for learning
rgbhsv_Rtx = np.vstack([R_gm1x,R_rp1x,R_bm1x,R_rp2x,R_wm1x,R_rp3x,R_rp4x,R_blm1x,R_rp5x,R_blm4x,R_bm5x])
rgbhsv_Rty = np.concatenate((R_gm1y,R_rp1y,R_bm1y,R_rp2y,R_wm1y,R_rp3y,R_rp4y,R_blm1y,R_rp5y,R_blm4y,R_bm5y))
rgbhsv_redsvm.learn(rgbhsv_Rtx,rgbhsv_Rty)

# training set for greensvm
# the x part is all the feature vectors and the y part is 1 or -1
G_rm1x,G_rm1y   = rgbhsv_G_rm1[:, :360],rgbhsv_G_rm1[:, 360]
G_rm2x,G_rm2y   = rgbhsv_G_rm2[:, :360],rgbhsv_G_rm2[:, 360]
G_rm3x,G_rm3y   = rgbhsv_G_rm3[:, :360],rgbhsv_G_rm3[:, 360]
G_rm4x,G_rm4y   = rgbhsv_G_rm4[:, :360],rgbhsv_G_rm4[:, 360]
G_rm5x,G_rm5y   = rgbhsv_G_rm5[:, :360],rgbhsv_G_rm5[:, 360]
G_gp1x,G_gp1y   = rgbhsv_G_gp1[:, :360],rgbhsv_G_gp1[:, 360]
G_gp2x,G_gp2y   = rgbhsv_G_gp2[:, :360],rgbhsv_G_gp2[:, 360]
G_gp3x,G_gp3y   = rgbhsv_G_gp3[:, :360],rgbhsv_G_gp3[:, 360]
G_gp4x,G_gp4y   = rgbhsv_G_gp4[:, :360],rgbhsv_G_gp4[:, 360]
G_gp5x,G_gp5y   = rgbhsv_G_gp5[:, :360],rgbhsv_G_gp5[:, 360]
G_bm1x,G_bm1y   = rgbhsv_G_bm1[:, :360],rgbhsv_G_bm1[:, 360]
G_bm2x,G_bm2y   = rgbhsv_G_bm2[:, :360],rgbhsv_G_bm2[:, 360]
G_bm3x,G_bm3y   = rgbhsv_G_bm3[:, :360],rgbhsv_G_bm3[:, 360]
G_bm4x,G_bm4y   = rgbhsv_G_bm4[:, :360],rgbhsv_G_bm4[:, 360]
G_bm5x,G_bm5y   = rgbhsv_G_bm5[:, :360],rgbhsv_G_bm5[:, 360]
G_wm1x,G_wm1y   = rgbhsv_G_wm1[:, :360],rgbhsv_G_wm1[:, 360]
G_wm2x,G_wm2y   = rgbhsv_G_wm2[:, :360],rgbhsv_G_wm2[:, 360]
G_wm3x,G_wm3y   = rgbhsv_G_wm3[:, :360],rgbhsv_G_wm3[:, 360]
G_wm4x,G_wm4y   = rgbhsv_G_wm4[:, :360],rgbhsv_G_wm4[:, 360]
G_blm1x,G_blm1y = rgbhsv_G_blm1[:, :360],rgbhsv_G_blm1[:, 360]
G_blm2x,G_blm2y = rgbhsv_G_blm2[:, :360],rgbhsv_G_blm2[:, 360]
G_blm3x,G_blm3y = rgbhsv_G_blm3[:, :360],rgbhsv_G_blm3[:, 360]
G_blm4x,G_blm4y = rgbhsv_G_blm4[:, :360],rgbhsv_G_blm4[:, 360]

# prepares the training data in order to give them as arguments for learning
rgbhsv_Gtx = np.vstack([G_rm1x,G_gp1x,G_bm1x,G_wm1x,G_gp2x,G_blm1x,G_gp3x,G_rm5x,G_gp5x,G_blm4x,G_bm5x])
rgbhsv_Gty = np.concatenate((G_rm1y,G_gp1y,G_bm1y,G_wm1y,G_gp2y,G_blm1y,G_gp3y,G_rm5y,G_gp5y,G_blm4y,G_bm5y))
rgbhsv_greensvm.learn(rgbhsv_Gtx,rgbhsv_Gty)

# training set for bluesvm
# the x part is all the feature vectors and the y part is 1 or -1
B_rm1x,B_rm1y   = rgbhsv_B_rm1[:, :360],rgbhsv_B_rm1[:, 360]
B_rm2x,B_rm2y   = rgbhsv_B_rm2[:, :360],rgbhsv_B_rm2[:, 360]
B_rm3x,B_rm3y   = rgbhsv_B_rm3[:, :360],rgbhsv_B_rm3[:, 360]
B_rm4x,B_rm4y   = rgbhsv_B_rm4[:, :360],rgbhsv_B_rm4[:, 360]
B_rm5x,B_rm5y   = rgbhsv_B_rm5[:, :360],rgbhsv_B_rm5[:, 360]
B_gm1x,B_gm1y   = rgbhsv_B_gm1[:, :360],rgbhsv_B_gm1[:, 360]
B_gm2x,B_gm2y   = rgbhsv_B_gm2[:, :360],rgbhsv_B_gm2[:, 360]
B_gm3x,B_gm3y   = rgbhsv_B_gm3[:, :360],rgbhsv_B_gm3[:, 360]
B_gm4x,B_gm4y   = rgbhsv_B_gm4[:, :360],rgbhsv_B_gm4[:, 360]
B_bp1x,B_bp1y   = rgbhsv_B_bp1[:, :360],rgbhsv_B_bp1[:, 360]
B_bp2x,B_bp2y   = rgbhsv_B_bp2[:, :360],rgbhsv_B_bp2[:, 360]
B_bp3x,B_bp3y   = rgbhsv_B_bp3[:, :360],rgbhsv_B_bp3[:, 360]
B_bp4x,B_bp4y   = rgbhsv_B_bp4[:, :360],rgbhsv_B_bp4[:, 360]
#B_bp5x,B_bp5y   = rgbhsv_B_bp5[:, :360],rgbhsv_B_bp5[:, 360]
B_wm1x,B_wm1y   = rgbhsv_B_wm1[:, :360],rgbhsv_B_wm1[:, 360]
B_wm2x,B_wm2y   = rgbhsv_B_wm2[:, :360],rgbhsv_B_wm2[:, 360]
B_wm3x,B_wm3y   = rgbhsv_B_wm3[:, :360],rgbhsv_B_wm3[:, 360]
B_wm4x,B_wm4y   = rgbhsv_B_wm4[:, :360],rgbhsv_B_wm4[:, 360]
B_blm1x,B_blm1y = rgbhsv_B_blm1[:, :360],rgbhsv_B_blm1[:, 360]
B_blm2x,B_blm2y = rgbhsv_B_blm2[:, :360],rgbhsv_B_blm2[:, 360]
B_blm3x,B_blm3y = rgbhsv_B_blm3[:, :360],rgbhsv_B_blm3[:, 360]
B_blm4x,B_blm4y = rgbhsv_B_blm4[:, :360],rgbhsv_B_blm4[:, 360]

# prepares the training data in order to give them as arguments for learning
rgbhsv_Btx = np.vstack([B_rm1x,B_bp1x,B_wm1x,B_bp2x,B_gm1x,B_bp3x,B_blm1x,B_blm4x])
rgbhsv_Bty = np.concatenate((B_rm1y,B_bp1y,B_wm1y,B_bp2y,B_gm1y,B_bp3y,B_blm1y,B_blm4y)) 
rgbhsv_bluesvm.learn(rgbhsv_Btx,rgbhsv_Bty)

# training set for whitesvm
# the x part is all the feature vectors and the y part is 1 or -1
W_rm1x,W_rm1y   = rgbhsv_W_rm1[:, :360],rgbhsv_W_rm1[:, 360]
W_rm2x,W_rm2y   = rgbhsv_W_rm2[:, :360],rgbhsv_W_rm2[:, 360]
W_rm3x,W_rm3y   = rgbhsv_W_rm3[:, :360],rgbhsv_W_rm3[:, 360]
W_rm4x,W_rm4y   = rgbhsv_W_rm4[:, :360],rgbhsv_W_rm4[:, 360]
W_rm5x,W_rm5y   = rgbhsv_W_rm5[:, :360],rgbhsv_W_rm5[:, 360]
W_gm1x,W_gm1y   = rgbhsv_W_gm1[:, :360],rgbhsv_W_gm1[:, 360]
W_gm2x,W_gm2y   = rgbhsv_W_gm2[:, :360],rgbhsv_W_gm2[:, 360]
W_gm3x,W_gm3y   = rgbhsv_W_gm3[:, :360],rgbhsv_W_gm3[:, 360]
W_gm4x,W_gm4y   = rgbhsv_W_gm4[:, :360],rgbhsv_W_gm4[:, 360]
W_bm1x,W_bm1y   = rgbhsv_W_bm1[:, :360],rgbhsv_W_bm1[:, 360]
W_bm2x,W_bm2y   = rgbhsv_W_bm2[:, :360],rgbhsv_W_bm2[:, 360]
W_bm3x,W_bm3y   = rgbhsv_W_bm3[:, :360],rgbhsv_W_bm3[:, 360]
W_bm4x,W_bm4y   = rgbhsv_W_bm4[:, :360],rgbhsv_W_bm4[:, 360]
W_bm5x,W_bm5y   = rgbhsv_W_bm5[:, :360],rgbhsv_W_bm5[:, 360]
W_wp1x,W_wp1y   = rgbhsv_W_wp1[:, :360],rgbhsv_W_wp1[:, 360]
W_wp2x,W_wp2y   = rgbhsv_W_wp2[:, :360],rgbhsv_W_wp2[:, 360]
W_wp3x,W_wp3y   = rgbhsv_W_wp3[:, :360],rgbhsv_W_wp3[:, 360]
W_wp4x,W_wp4y   = rgbhsv_W_wp4[:, :360],rgbhsv_W_wp4[:, 360]
W_blm1x,W_blm1y = rgbhsv_W_blm1[:, :360],rgbhsv_W_blm1[:, 360]
W_blm2x,W_blm2y = rgbhsv_W_blm2[:, :360],rgbhsv_W_blm2[:, 360]
W_blm3x,W_blm3y = rgbhsv_W_blm3[:, :360],rgbhsv_W_blm3[:, 360]
W_blm4x,W_blm4y = rgbhsv_W_blm4[:, :360],rgbhsv_W_blm4[:, 360]   

# prepares the training data in order to give them as arguments for learning
rgbhsv_Wtx = np.vstack([W_rm1x,W_wp1x,W_blm1x,W_wp2x,W_bm1x,W_wp3x,W_gm1x,W_blm4x,W_bm5x])
rgbhsv_Wty = np.concatenate((W_rm1y,W_wp1y,W_blm1y,W_wp2y,W_bm1y,W_wp3y,W_gm1y,W_blm4y,W_bm5y)) 
rgbhsv_whitesvm.learn(rgbhsv_Wtx,rgbhsv_Wty)

# training set for blacksvm
# the x part is all the feature vectors and the y part is 1 or -1
BL_rm1x,BL_rm1y   = rgbhsv_BL_rm1[:, :360],rgbhsv_BL_rm1[:, 360]
BL_rm2x,BL_rm2y   = rgbhsv_BL_rm2[:, :360],rgbhsv_BL_rm2[:, 360]
BL_rm3x,BL_rm3y   = rgbhsv_BL_rm3[:, :360],rgbhsv_BL_rm3[:, 360]
BL_rm4x,BL_rm4y   = rgbhsv_BL_rm4[:, :360],rgbhsv_BL_rm4[:, 360]
BL_rm5x,BL_rm5y   = rgbhsv_BL_rm5[:, :360],rgbhsv_BL_rm5[:, 360]
BL_gm1x,BL_gm1y   = rgbhsv_BL_gm1[:, :360],rgbhsv_BL_gm1[:, 360]
BL_gm2x,BL_gm2y   = rgbhsv_BL_gm2[:, :360],rgbhsv_BL_gm2[:, 360]
BL_gm3x,BL_gm3y   = rgbhsv_BL_gm3[:, :360],rgbhsv_BL_gm3[:, 360]
BL_gm4x,BL_gm4y   = rgbhsv_BL_gm4[:, :360],rgbhsv_BL_gm4[:, 360]
BL_bm1x,BL_bm1y   = rgbhsv_BL_bm1[:, :360],rgbhsv_BL_bm1[:, 360]
BL_bm2x,BL_bm2y   = rgbhsv_BL_bm2[:, :360],rgbhsv_BL_bm2[:, 360]
BL_bm3x,BL_bm3y   = rgbhsv_BL_bm3[:, :360],rgbhsv_BL_bm3[:, 360]
BL_bm4x,BL_bm4y   = rgbhsv_BL_bm4[:, :360],rgbhsv_BL_bm4[:, 360]
BL_bm5x,BL_bm5y   = rgbhsv_BL_bm5[:, :360],rgbhsv_BL_bm5[:, 360]
BL_wm1x,BL_wm1y   = rgbhsv_BL_wm1[:, :360],rgbhsv_BL_wm1[:, 360]
BL_wm2x,BL_wm2y   = rgbhsv_BL_wm2[:, :360],rgbhsv_BL_wm2[:, 360]
BL_wm3x,BL_wm3y   = rgbhsv_BL_wm3[:, :360],rgbhsv_BL_wm3[:, 360]
BL_wm4x,BL_wm4y   = rgbhsv_BL_wm4[:, :360],rgbhsv_BL_wm4[:, 360]
BL_blp1x,BL_blp1y = rgbhsv_BL_blp1[:, :360],rgbhsv_BL_blp1[:, 360]
BL_blp2x,BL_blp2y = rgbhsv_BL_blp2[:, :360],rgbhsv_BL_blp2[:, 360]
BL_blp3x,BL_blp3y = rgbhsv_BL_blp3[:, :360],rgbhsv_BL_blp3[:, 360]
BL_blp4x,BL_blp4y = rgbhsv_BL_blp4[:, :360],rgbhsv_BL_blp4[:, 360]
BL_blp5x,BL_blp5y = rgbhsv_BL_blp5[:, :360],rgbhsv_BL_blp5[:, 360]

# prepares the training data in order to give them as arguments for learning
rgbhsv_BLtx = np.vstack([BL_rm1x,BL_blp1x,BL_bm1x,BL_blp2x,BL_gm1x,BL_blp3x,BL_wm1x,BL_blp4x,BL_rm5x,BL_blp5x,BL_bm5x])
rgbhsv_BLty = np.concatenate((BL_rm1y,BL_blp1y,BL_bm1y,BL_blp2y,BL_gm1y,BL_blp3y,BL_wm1y,BL_blp4y,BL_rm5y,BL_blp5y,BL_bm5y)) 
rgbhsv_blacksvm.learn(rgbhsv_BLtx,rgbhsv_BLty)

# save trainied libsvms as xml files
rgbhsv_redsvm.save_model('rgbhsv_redsvm.xml')
rgbhsv_greensvm.save_model('rgbhsv_greensvm.xml')
rgbhsv_bluesvm.save_model('rgbhsv_bluesvm.xml')
rgbhsv_whitesvm.save_model('rgbhsv_whitesvm.xml')
rgbhsv_blacksvm.save_model('rgbhsv_blacksvm.xml')

print '\nColor RGB HSV svms has been created successfully!'

###----------------------###

# calculates the training time
end=time.time()
suma=end-start
print '\nTime elapsed:%s seconds.\n' %suma

# end-of-program #