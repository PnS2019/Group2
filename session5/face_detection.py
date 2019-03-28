#!/usr/bin/python3
import numpy as np
import cv2
from pnslib import utils

# open the camera
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    rows, cols, depth = frame.shape
    M_flip = np.float32([[-1, 0, cols], [0, 1, 0]])
    frame = cv2.warpAffine(frame, M_flip, (cols, rows))

    # load face cascade and eye cascade
    face_cascade = cv2.CascadeClassifier(
        utils.get_haarcascade_path('haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(
        utils.get_haarcascade_path('haarcascade_eye.xml'))

    # search face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    # the loop breaks at pressing `q`
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
