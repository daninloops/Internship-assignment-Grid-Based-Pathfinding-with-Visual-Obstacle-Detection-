import cv2

aruco=cv2.aruco
aruco.dict=aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

#Generate marker ID 0 (agent)
marker0=aruco.generateImageMarker(aruco.dict,0,200)
cv2.imwrite("Marker_0.png",marker0)

#Generate marker ID 1 (goal)
marker1=aruco.generateImageMarker(aruco.dict,1,200)
cv2.imwrite("marker_1.png",marker1)

print("Markers saved ")

