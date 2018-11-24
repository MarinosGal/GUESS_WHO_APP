# Description
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

This application is part of my intern research at NCSR Demokritos. (Sept 2015 - December 2015)

# Machine Learning Algorithms 
1. Support Vector Machines (SVM)
2. Perceptron

# Libraries
1. OpenCV https://opencv.org/
2. mlpy http://mlpy.sourceforge.net/docs/3.5/install.html
