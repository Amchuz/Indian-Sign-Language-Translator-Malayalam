import numpy as np
from keras.models import model_save
import operator
import cv2
import sys, os

loaded_model=model_save
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
  
    frame = cv2.flip(frame, 1)
 
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])

    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)

    roi = frame[y1:y2, x1:x2]

    roi = cv2.resize(roi, (64, 64)) 
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: 
        break
        
 
cap.release()
cv2.destroyAllWindows()
