# -*- coding: utf-8 -*-

import cv2


cascade_src = 'Bus_front.xml'
video_src = 'bus1.mp4'
#video_src = 'two_wheeler1.mp4'



cap = cv2.VideoCapture(video_src)
#cap = cv2.VideoCapture(0)

bus_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    ret, img = cap.read()
    
    if (type(img) == type(None)):
        print('video file not present')
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    bus = bus_cascade.detectMultiScale(gray, 1.16, 1)

    for (x,y,w,h) in bus:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),10)
        print('Bus detected')
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
