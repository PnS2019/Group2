#!/usr/bin/python3
import numpy as np
import cv2

# open the camera
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    rows, cols, depth = frame.shape
    M_flip = np.float32([[-1, 0, cols], [0, 1, 0]])
    frame = cv2.warpAffine(frame, M_flip, (cols, rows))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # for opencv 3.x and above
    # you will need to run the following code to install
    # pip install opencv-contrib-python -U
    sift = cv2.xfeatures2d.SIFT_create()

    kp = sift.detect(gray, None)

    # for opencv 3.x and above
    cv2.drawKeypoints(gray, kp, frame)

    cv2.imshow('frame', frame)

    # the loop breaks at pressing `q`
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
