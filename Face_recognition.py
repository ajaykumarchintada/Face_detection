import numpy as np 
import cv2 as cv 
face_cascade=cv.CascadeClassifier("haarcascade_frontalface_alt2.xml")
eyes_cascade=cv.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
#loading the harcascade using CascadeClassifier methon in cv
cap=cv.VideoCapture(0)#argument as 0 because we are using the web camera here if you have multiple cameras u can use 1,2,3,..
while True:
    ret,frame=cap.read()#ret value will be true 
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY) #converting every frame into gray cause open cv works with well with bnw
    face=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5) #detecting the face scaleFactor will be for accuarcy 
    #and here wer are using the scalefactor and minneighbors as per the documentation
    
    
    for(x,y,w,h) in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #finding the co_ordinates of detected face and drawing rectangle aroudn it
        roi_gray=gray[y:y+h,x:x+w] #region of index for the detected face 
        roi_image=frame[y:y+h,x:x+w]
        
        
        eyes=eyes_cascade.detectMultiScale(roi_gray) #detect the eyes in face so passed roi_gray 
        
        for(ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_image,(ex,ey),(ex+ew,ey+eh),(255,0,255),2) #draw rectangles around the eyes

            
    cv.imshow("Face_Detection",frame)
    k=cv.waitKey(25)&0xFF

    if k==27:
        break
cap.release()
cv.destroyAllWindows()    
