# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:46:54 2021

@author: Zoe Mercury
"""


# Task 2:
# =======

# Draw a box (side length 20cm) around the head in 'face.jpg'
# Intrinsic camera parameters are cx=250, cy=375, fx=fy=716.
# Head position is (in mm relative to camera) (212, -168, 712) with rotation (Rodrigues) (.35, .59, -.30).
# Refer to 'https://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html'
# for information. The result should look like 'face_example.jpg', but you should choose thinner lines.

#------IMPORTS------#

import cv2
import numpy as np
import sys


#-----------------DETECTING FACE--------------------#
####################################################################

def detect_face(imagepath):
    imagePath = sys.argv[1]
    faceCascade = cv2.CascadeClassifier(r"C:\Users\Zoe Mercury\Desktop\Careerhack\ixp\haarcascade_frontalface_default.xml")

    # Read the image
    image = cv2.imread(imagepath)
    print('Image read')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect face in the image
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=2,minSize=(100, 100))
    print(faces)

    # Draw a rectangle around the faces & save locally     
    for (x, y, w, h) in faces:
        image= cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        #roi_color = image[y:y + h, x:x + w] 
        print("Object found. Saving locally.") 
        #cv2.imwrite(str(i)+'.jpg', roi_color) 
        cv2.imwrite('Angela.jpg',image)   
        
        
#-----------------DRAWING BOX--------------------#
####################################################################

def draw_box(imagepath):
    image = cv2.imread(imagepath)
    image= cv2.rectangle(image, (290,96), (500,303), (255, 0, 255), 2)
    image= cv2.line(image, (290, 303), (368,275), (255, 0, 255), 2)
    image= cv2.line(image, (500, 303), (547,276), (255, 0, 255), 2)
    cv2.imwrite('Angi.jpg', image)
    return ('Box created!')


#-----------------IDENTIFYING CLICK COORDINATES--------------------#
####################################################################

    
def click_event(event, x, y, flags, params): 
  
    if event == cv2.EVENT_LBUTTONDOWN:  # checking for left mouse clicks
        print(x, ' | ', y) # displaying the coordinates on the Shell  
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' + str(y), (x,y), font, 1, (255, 0, 0), 2) # displaying the coordinates on the image window
        cv2.imshow('image', img) 

  
# driver function
if __name__=="__main__": 
    img = cv2.imread('Tilted0.jpg', 1)  
    cv2.imshow('image', img) 
    
    cv2.setMouseCallback('image', click_event) #calling the click_event() function 
  
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
    
    
    
#-----------------ROTATE RECTANGLE--------------------#
#######################################################

def rotate(imagepath):
    img = cv2.imread(imagepath)
    
    # first rectangle
    cnt1 = np.array([
            [[301,164]],
            [[374, 353]],
            [[585, 306]],
            [[519, 72]]
        ])
    print("shape of cnt: {}".format(cnt1.shape))
    rect1 = cv2.minAreaRect(cnt1)
    print("rect: {}".format(rect1))
    
    box1 = cv2.boxPoints(rect1)
    box1 = np.int0(box1)

    #second rectangle
    cnt2 = np.array([
            [[376,140]],
            [[429, 297]],
            [[607, 251]],
            [[555, 63]]
        ])
    print("shape of cnt: {}".format(cnt2.shape))
    rect2 = cv2.minAreaRect(cnt2)
    print("rect: {}".format(rect2))
    
    box2 = cv2.boxPoints(rect2)
    box2 = np.int0(box2)

    print("bounding box: {}".format(box))
    img=cv2.drawContours(img, [box1], 0, (255, 0, 255), 2)
    img=cv2.drawContours(img, [box2], 0, (255, 0, 255), 2)
    img= cv2.line(img, (359, 368), (418,299), (255, 0, 255), 2)
    img= cv2.line(img, (583, 303), (605,251), (255, 0, 255), 2)
    cv2.imwrite('Tilted.jpg',img)
    
    return ('Image saved!')