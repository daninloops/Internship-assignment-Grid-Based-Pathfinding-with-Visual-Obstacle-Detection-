import numpy as np
import cv2
import heapq
webcam=cv2.VideoCapture(1)#connect to camera
if not webcam.isOpened():
    exit()#guard if camera disconnects
for i in range (50):
   webcam.read()

CELL_SIZE=20 #Chose 20 as its accurate enough to detect obstacles well, and small enough that A* runs fast 
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
    clean_mask=cv2.dilate(opened,kernel,iterations=2)#grows obstacle blobs outward-> adds safety margin

    output=frame.copy()
    red_layer=np.zeros_like(frame)
    red_layer[:]=(0,0,255)
    output[clean_mask==255]=cv2.addWeighted(frame,0.4,red_layer,0.6,0)[clean_mask==255]
    cv2.putText(output,"Obstacles:",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.putText(clean_mask,"Binary Mask",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,255,2)

    grid_rows=frame.shape[0]//CELL_SIZE
    grid_cols=frame.shape[1]//CELL_SIZE
    grid=np.zeros((grid_rows,grid_cols), dtype=np.uint8)


    for r in range(grid_rows):
        for c in range(grid_cols):
            y_start=r*(CELL_SIZE)
            y_end=(r*(CELL_SIZE))+CELL_SIZE
            x_start=c*(CELL_SIZE)
            x_end=(c*(CELL_SIZE))+CELL_SIZE
            cell=clean_mask[y_start:y_end,x_start:x_end]
            white_pixels=np.sum(cell==255)
            total_pixels=CELL_SIZE*CELL_SIZE
            if white_pixels/total_pixels>0.30:
                grid[r,c]=1
            
    for r in range(grid_rows):
        for c in range (grid_cols):
            x1=c*CELL_SIZE
            y1=r*CELL_SIZE
            x2=x1+CELL_SIZE
            y2=y1+CELL_SIZE
            if grid[r,c]==1:
                colour=(0,0,255)
            else:
                colour=(0,255,0)
            cv2.rectangle(output,(x1,y1),(x2,y2),colour,1)


    cv2.imshow("Frame",output)
    cv2.imshow("mask",clean_mask)

    if cv2.waitKey(1) &0xFF==ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()