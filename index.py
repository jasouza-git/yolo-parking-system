# Import nessary packages
import cv2 as cv
import pandas as pd
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# CCTV
camera_ip = '192.168.1.108:554'
username = 'admin'
password = 'ECEpower'
rtsp_url = f"rtsp://{username}:{password}@{camera_ip}/cam/realmonitor?channel=1&subtype=1"
video = cv.VideoCapture(rtsp_url)

# Load nessary data
model = YOLO('yolov8s.pt')
wait  = 1000*30 # 30 Seconds for each frame
tags  = open('coco.txt', 'r').read().split('\n')
park = [[int(v) for v in l] for l in open('park.txt', 'r').read().split('\n')]

# Parker
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
        log = open('log.txt', 'a')
        if p != parked_new[i]: log.write(str(i).rjust(2,'0')+' '+str(round((datetime.utcnow()-datetime(1970,1,1)).total_seconds()*1000)).rjust(13,'0')+'\n')
        log.close()
    parked = parked_new

    cv.rectangle(frame, (0, 0), (80, 20), (0,255,0), -1)
    cv.rectangle(frame, (1020//2-80, 0), (1020//2, 20), (0,0,255), -1)
    cv.putText(frame, str(taken).rjust(2,'0')+' / '+str(len(park)-taken).rjust(2,'0'), (5,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
    cv.putText(frame, str(cars[0]).rjust(2,'0')+' / '+str(cars[1]).rjust(2,'0'), (1020//2-75,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
    cv.putText(frame, datetime.now().strftime('Date: %b %d, %Y %H:%M:%S'), (1020//8, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
    cv.imshow('Picture', frame)
    return True

while scan(video):
    if cv.waitKey(wait)&0xFF == 27: break


video.release()
cv.destroyAllWindows()