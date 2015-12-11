import numpy as np
import cv2
import argparse

'''
Captures video and stops when you press the 'ESC' key.
'''

############
''' MAIN '''
############

# Loads the necessary arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--v", required = True, help = "The video's name.")
args = vars(ap.parse_args())

# Defines the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# Opens a window for capturing
cap = cv2.VideoCapture(0)

# Creates a video file
out = cv2.VideoWriter(args["v"]+'.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret==True:
        out.write(frame)
        cv2.imshow('Capturing...',frame)
        if cv2.waitKey(1) == 27:
            break
    

print "The video has been captured!"    

# Releases everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

# end-of-program #