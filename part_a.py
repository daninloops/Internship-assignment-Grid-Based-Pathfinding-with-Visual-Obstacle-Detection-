import numpy as np
import cv2
import heapq
import time 
aruco=cv2.aruco
aruco_dict=aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters=aruco.DetectorParameters()
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

start = None
goal = None

dwell_start=None
arrived= False

def mouse_callback(event, x, y, flags, param):
    global start, goal

    if event == cv2.EVENT_LBUTTONDOWN:
        r = y // CELL_SIZE
        c = x // CELL_SIZE

        if start is None:
            start = (r, c)
            print("Start:", start)

        elif goal is None:
            goal = (r, c)
            print("Goal:", goal)

        else:
            # reset on third click
            start = (r, c)
            goal = None
            print("Reset Start:", start)



def astar(grid,start,goal):
    h=abs(start[0]-goal[0])+abs(start[1]-goal[1])
    open_list=[]
    heapq.heappush(open_list,(0+h,start[0],start[1]))
    closed_set=set()
    g_score={start:0}
    parent={}
    while open_list:
        f,row,col=heapq.heappop(open_list)
        current=(row,col)

        if current==goal:
            path=[]
            while current!=start:
                path.append(current)
                current=parent[current]
            path.append(start)
            path.reverse()
            return path
        
        if current in closed_set:
            continue 
        closed_set.add(current)

        directions=[(-1,0),(1,0),(0,-1),(0,1)]
        for dr,dc in directions:
            nr=row+dr
            nc=col+dc
            neighbour=(nr,nc)

            if nr<0 or nr>=grid.shape[0] or nc<0 or nc>=grid.shape[1]:
                continue 
            if grid[nr][nc]==1:
                continue
            if neighbour in closed_set:
                continue 
            new_g=g_score[current]+1

            if new_g<g_score.get(neighbour,float('inf')):
                g_score[neighbour]=new_g
                h=abs(nr-goal[0])+abs(nc-goal[1])
                f=new_g+h
                heapq.heappush(open_list,(f,nr,nc))
                parent[neighbour]=current
    return None 
last_replan=time.time()
path=None
 
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame",mouse_callback)

while True:
    ret,frame=webcam.read() 
    if not ret:
        break
  
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    corners,ids,_=aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
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
    if ids is not None:
      for i in range(len(ids)):
        marker_id = ids[i][0]

        # Get center of marker
        corner = corners[i][0]
        cx = int(np.mean(corner[:, 0]))
        cy = int(np.mean(corner[:, 1]))

        # Convert to grid
        mr = cy//CELL_SIZE
        mc = cx//CELL_SIZE
        if mr<0 or mr>=grid_rows or mc<0 or mc>=grid_cols:
            continue
        if marker_id == 0:
            start = (mr, mc)

        elif marker_id == 1:
            goal = (mr, mc)

        # Draw marker center
        cv2.circle(output, (cx, cy), 5, (255,255,0), -1)       
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

    
    now=time.time()
    path_blocked=False

    if path is not None:
        for (r,c) in path:
            if grid[r][c]==1:
                path_blocked=True
                break
    
    
    
    if start is not None and goal is not None:
        if grid[start[0]][start[1]]==1 or grid[goal[0]][goal[1]]==1:
            path=None
        
         
        elif now-last_replan>0.5 or path_blocked:
          path=astar(grid,start,goal)
          last_replan=now

         
    if path is not None:
            for(pr,pc) in path:
               x=pc*CELL_SIZE+CELL_SIZE//2
               y=pr*CELL_SIZE+CELL_SIZE//2
               cv2.circle(output,(x,y),4,(255,0,0),-1)
    if start is not None:
        sx=start[1]*CELL_SIZE+CELL_SIZE//2
        sy=start[0]*CELL_SIZE+CELL_SIZE//2
        cv2.circle(output,(sx,sy),8,(0,255,0),-1)

    if goal is not None:
        gx=goal[1]*CELL_SIZE+CELL_SIZE//2
        gy=goal[0] *CELL_SIZE+CELL_SIZE//2
        cv2.circle(output,(gx,gy),8,(0,255,255),-1)

    if start is not None and goal is not None:
        dist = abs(start[0]-goal[0]) + abs(start[1]-goal[1])
        if dist <= 2:
            if dwell_start is None:
                dwell_start = time.time()
            elif time.time() - dwell_start >= 1.0:
                arrived = True
        else:
            dwell_start = None
            arrived = False
    if arrived:
         pulse = int(time.time() * 2) % 2 == 0
         if pulse and goal is not None:
             gx1 = goal[1] * CELL_SIZE
             gy1 = goal[0] * CELL_SIZE
             gx2 = gx1 + CELL_SIZE
             gy2 = gy1 + CELL_SIZE
             cv2.rectangle(output, (gx1-10, gy1-10), (gx2+10, gy2+10), (0,255,0), 3)
         cv2.putText(output, "Arrived - Target Reached", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)



    cv2.imshow("Frame",output)

    cv2.imshow("mask",clean_mask)

    if cv2.waitKey(1) &0xFF==ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()