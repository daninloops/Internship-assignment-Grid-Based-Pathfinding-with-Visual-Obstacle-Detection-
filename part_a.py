import numpy as np
import cv2

webcam=cv2.VideoCapture(0)#connect to camera
if not webcam.isOpened():
    exit()#guard if camera disconnects
for i in range (40):
   webcam.read()


ret,background=webcam.read() 
background=cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)
background=cv2.GaussianBlur(background,(5,5),0)
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

while True:
    ret,frame=webcam.read() 
    if not ret:
        break
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    diff=cv2.absdiff(background,gray)
    _,mask=cv2.threshold(diff,80,255,cv2.THRESH_BINARY)
    opened=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,iterations=1)
    clean_mask=cv2.dilate(opened,kernel,iterations=2)
    output=frame.copy()
    red_layer=np.zeros_like(frame)
    red_layer[:]=(0,0,255)
    output[clean_mask==255]=cv2.addWeighted(frame,0.4,red_layer,0.6,0)[clean_mask==255]
    cv2.putText(output,"Obstacles:",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.putText(clean_mask,"Binary Mask",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,255,2)
    cv2.imshow("Frame",output)
    cv2.imshow("mask",clean_mask)

    if cv2.waitKey(1) &0xFF==ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()