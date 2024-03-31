# Import nessary packages
import cv2 as cv
import pandas as pd
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import time

# Load nessary data
model = YOLO('yolov8s.pt')
video = cv.VideoCapture('parking1.mp4')
tags  = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
log   = open('log.txt', 'a')

# Parker
park = [
    (23, 180, 48, 178, 39, 214, 8, 214),
    (48, 178, 76, 177, 71, 213, 39,214),
    (250,165, 276,163, 304,191, 273,194),
    (276,163, 301,159, 332,186, 304,191),
    (301,159, 326,157, 357,182, 332,186),
    (326,157, 345,155, 381,179, 357,182)
]
parked = []
def say_coordinates(e, x, y, f, p):
    if e == cv.EVENT_MOUSEMOVE: print([x, y])
def parker(video):
    ret, frame = video.read()
    if not ret: return False
    frame = cv.resize(frame, (1020//2, 500//2))
    cv.namedWindow('Parker')
    cv.setMouseCallback('Parker', say_coordinates)
    for p in park: cv.polylines(frame, [np.array([(p[i], p[i+1]) for i in range(0,len(p),2)], np.int32)], True, (0,255,0), 2)
    cv.imshow('Parker', frame)
    cv.waitKey(0)
for n in range(len(park)):
    park[n] = np.array([(park[n][i], park[n][i+1]) for i in range(0,len(park[n]),2)], np.int32)
    parked.append(False)

# Scanner function
def scan(video):
    global parked
    ret, frame = video.read()
    if not ret: return False

    frame = cv.resize(frame, (1020//2, 500//2))
    out = model.predict(frame)
    a = out[0].boxes.data
    px = pd.DataFrame(a).astype('float')

    # Draw parking areas
    taken = 0
    cars = [0, 0]
    for p in park: cv.polylines(frame, [p], True, (0,255,0), 1)
    parked_new = [False for p in park]

    for i, r in px.iterrows():
        x1 = int(r[0])
        y1 = int(r[1])
        x2 = int(r[2])
        y2 = int(r[3])
        d  = int(r[5])
        c  = tags[d]

        if 'car' in c:
            cars[1] += 1
            cx = int(x1+x2)//2
            cy = int(y1*0.4+y2*0.6)
            
            cr = (0,0,255)
            for n, p in enumerate(park):
                if cv.pointPolygonTest(p, ((cx,cy)), False) >= 0:
                    taken += 1
                    cr = (255,0,0)
                    cars[0] += 1
                    parked_new[n] = True
                    break
            cv.rectangle(frame, (x1, y1), (x2, y2), cr, 1)
            cv.circle(frame, (cx,cy), 2, cr, -1)
    
    for i,p in enumerate(parked):
        if p != parked_new[i]: log.write(str(i).rjust(2,'0')+' '+str(round((datetime.utcnow()-datetime(1970,1,1)).total_seconds()*1000)).rjust(13,'0')+'\n')
    parked = parked_new

    cv.rectangle(frame, (0, 0), (80, 20), (0,255,0), -1)
    cv.rectangle(frame, (1020//2-80, 0), (1020//2, 20), (0,0,255), -1)
    cv.putText(frame, str(taken).rjust(2,'0')+' / '+str(len(park)-taken).rjust(2,'0'), (5,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
    cv.putText(frame, str(cars[0]).rjust(2,'0')+' / '+str(cars[1]).rjust(2,'0'), (1020//2-75,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
    cv.putText(frame, datetime.now().strftime('Date: %b %d, %Y %H:%M:%S'), (1020//8, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
    cv.imshow('Picture', frame)
    return True

while scan(video):
    if cv.waitKey(1000)&0xFF == 27: break


video.release()
cv.destroyAllWindows()
log.close()