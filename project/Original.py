import os
import cv2 as cv
import numpy as np
          

files= os.listdir('resized')

photos=[]
resized_photos=[]



for file in files[:20]:
    if file.endswith('jpg'):
        photos.append(cv.imread('resized\\'+file))




#color detection

lower_blue = np.array([100,40,0])
upper_blue = np.array([135,255,255])
    
for photo in photos:
    
    hsv = cv.cvtColor(photo,cv.COLOR_BGR2HSV)

    mask2 = cv.inRange(hsv,lower_blue,upper_blue)
    mask2=cv.bitwise_not(mask2)

    _,ctrs, hier = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    rects = [cv.boundingRect(ctr) for ctr in ctrs]


    out = cv.cvtColor(mask2,cv.COLOR_GRAY2RGB)
    for rect in rects:
        cv.rectangle(out, (rect[0],rect[1]) , (rect[0]+rect[2],rect[1]+rect[3]), (0, 255, 0), 3) 
    """detector=cv.SimpleBlobDetector()

    keypoints=detector.detect(mask2)

    mask_and_blob=cv.drawKeypoints(gray,keypoints,np.array([]),(100,255,255),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    """


    

    cv.imshow('Image',out)
    cv.waitKey(0)













