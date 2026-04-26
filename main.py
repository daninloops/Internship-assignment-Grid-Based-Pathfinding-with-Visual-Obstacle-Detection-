import numpy as np#numpy for array operation on frames and grid 
import cv2#for camera capture, image processing and display
import heapq#for priority queue used in A*
import time #for tracking replan intervals and dwell timer


aruco=cv2.aruco
aruco_dict=aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters=aruco.DetectorParameters()
#ArUco setup-DICT_4X4_50 means markers are 4x4 black/white squares
#marker 0= agent position, marker 1= goal position

webcam=cv2.VideoCapture(1)#connect to phone camera via Droid camera (index 1)
if not webcam.isOpened():
    exit()#guard if camera disconnects
for i in range (50):#discard first 50 frames so camera sensor can warm up 
   webcam.read()#first 50 frames are often too dark or blurry and would corrupt the background model

CELL_SIZE=20 #Chose 20 as its accurate enough to detect obstacles well, and small enough that A* runs fast 
ret,background=webcam.read() #capture one clean frame of the empty scene as background
background=cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)
background=cv2.GaussianBlur(background,(5,5),0)#blur to reduct noise before storing a s reference 
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
start = None #None until the user clicks- prevents A* running before a point is selected 
goal = None#None until a user selects a goal point - provides A* with a point to go till from start

dwell_start=None #None as the agent hasn't entered arrival zone yet hence timer has not started 
arrived= False# is a flag- starts as false because the agent hasn't arrived anywhere yet

def mouse_callback(event, x, y, flags, param):#defining a mouse click funtion which will specify start and goal 
    global start, goal#allows this funtion to modify the global start anf goal variables 

    if event==cv2.EVENT_LBUTTONDOWN:
        r=y//CELL_SIZE#convert pixel y coordinates into grid row 
        c=x//CELL_SIZE#convert pixel x coordinate to grid column 

        if start is None:
            start = (r, c)#if start has not been defined then on clicking left define a start by recording row and column
            print("Start:", start)

        elif goal is None:
            goal = (r, c)
            print("Goal:", goal)#if goal has not been defined then on clicking left define a goal by recording a row and column 

        else:
            # third click resets- new start point, clears goal so user can pick again
            start=(r,c)
            goal=None
            print("Start:", start)



def astar(grid,start,goal):#define the A* algortihm through funtion
    h=abs(start[0]-goal[0])+abs(start[1]-goal[1])#from the start and goal obtained, calculate the distance using manhattan distance 
    open_list=[]#heap of cell to explore next, ordered by f score (lowest first)
    heapq.heappush(open_list,(0+h,start[0],start[1]))#heapq.heappush() adds an item to the open_list defined with values being the f (g+h where g is initially zero) and h is the heuristic and start's x coordinate and start's y coordinate
    closed_set=set()#cells already fully explored- skip if encountered again to prevent loops  
    g_score={start:0}#defined a g_score dictionary as it will be used again and again for calculating f values 
    parent={}#defined a parent dictionary as it will be used for trakcing the algorithm from goal to start 
    while open_list:#keep exploring until no cells are left and goal is found
        f,row,col=heapq.heappop(open_list)#pop cell with most lowest f score - most promising cell
        current=(row,col)#treat as current position 

        if current==goal:
            #goal reached- trace back through parent dict to reconstruct path
            #parent[cell]= the cell we came from to reach that cell
            path=[]
            while current!=start:
                path.append(current)
                current=parent[current]#move backwards pne step
            path.append(start)#add start as loop stops before it 
            path.reverse()#flip from goal-> start to start-> goal
            return path
        
        if current in closed_set:
            #skip if already explored - same cell can be pushed 
            #to heap multiple times if a cheaper path is found later 
            continue 
        closed_set.add(current)#marked as fully explored before checking neighbours 

        directions=[(-1,0),(1,0),(0,-1),(0,1)]#up,down,left,right directions which can be travelled in 
        for dr,dc in directions:
            nr=row+dr
            nc=col+dc
            neighbour=(nr,nc)

            if nr<0 or nr>=grid.shape[0] or nc<0 or nc>=grid.shape[1]:
                continue #skip if outside grid boundaries 
            if grid[nr][nc]==1:
                continue#skip is cell is blocked by obstacle 
            if neighbour in closed_set:
                continue #skip if already explored 

            #cost to reach nighbour via current cell 
            #each step costs 1 for cardinal movement 
            new_g=g_score[current]+1

            if new_g<g_score.get(neighbour,float('inf')):
                #found a cheaper path to this neighbour- update everything 
                #float('inf) is default for unvisited cells- any real cost beats it 
                g_score[neighbour]=new_g
                h=abs(nr-goal[0])+abs(nc-goal[1])#manhattan distance to goal 
                f=new_g+h#total score=actual +estimate
                heapq.heappush(open_list,(f,nr,nc))#add to heap with new score 
                parent[neighbour]=current#record how we got here 
    return None #open list exhausted- no path exists between start and goal 

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