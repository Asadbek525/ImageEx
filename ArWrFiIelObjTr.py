from __future__ import print_function
import sys,os
import cv2
import numpy as np
from random import randint
from collections import deque
import pandas as pd
from datetime import datetime
import argparse

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
mybuffer = 12
points = deque(maxlen=mybuffer)
Xcor = []
Ycor = []

def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


def MulTracking(videoPath, delay, pionts, filname):
    date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    # Create a video capture object to read videos
    cap = cv2.VideoCapture(videoPath)

    # Read first frame
    success, frame = cap.read()
    # quit if unable to read the video file
    if not success:
        print('Failed to read video')
        sys.exit(1)

    ## Select boxes
    bboxes = []
    colors = []

    '''
    # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
    # So we will call this function in a loop till we are done selecting all objects
    while True:
        # draw bounding boxes over objects
        # selectROI's default behaviour is to draw box starting from the center
        # when fromCenter is set to false, you can draw box starting from top left corner
        bbox = cv2.selectROI('MultiTracker', frame)
        #print("bbox <<<---->>>>", bbox)

        bboxes.append(bbox)
        print("bbox <<<---->>>>", bboxes)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        print(bboxes, "____")
        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        k = cv2.waitKey(0) & 0xFF
        if (k == 113):  # q is pressed
            break
    '''
    if pionts == 1:
        bbox = (202, 94, 41, 45)
    elif pionts == 2:
        bbox = (401, 148, 46, 48)
    elif pionts == 3:
        bbox = (641, 72, 41, 43)
    elif pionts == 4:
        bbox = (223, 347, 42, 45)
    elif pionts == 5:
        bbox = (402, 300, 41, 45)
    elif pionts == 6:
        bbox = (610, 344, 38, 40)


    #bbox = [(199, 97, 45, 36), (405, 150, 43, 43)]
    bboxes.append(bbox)

    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

    print('Selected bounding boxes {}'.format(bboxes))

    # Specify the tracker type
    trackerType = "CSRT"

    # Create MultiTracker object
    multiTracker = cv2.MultiTracker_create()

    # Initialize MultiTracker
    for bbox in bboxes:
        # print("---------->>", bbox)
        multiTracker.add(createTrackerByName(trackerType), frame, bbox)

    mask = np.zeros_like(frame)

    filename = f""+filname+date+".csv"
    with open(filename, 'w', encoding='utf-8') as wr_file:
        wr_file.write("Xcor[1], Ycor[1], Xcor[4], Ycor[4], Xcor[3], Ycor[3], Xcor[8], Ycor[8], Xcor[6], Ycor[6], Xcor[10], Ycor[10], cnetroids\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Process video and track objects
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        canny_output = cv2.Canny(thresh, 100, 100 * 2)
        contours, _ = cv2.findContours(canny_output.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
          M = cv2.moments(c)
          # calculate x,y coordinate of center
          cX = int(M["m10"] / M["m00"])
          cY = int(M["m01"] / M["m00"])
          center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

          cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
          Xcor.append(cX)
          Ycor.append(cY)

        cv2.line(frame, (Xcor[1], Ycor[1]), (Xcor[4], Ycor[4]), (0, 255, 0), 2)
        cv2.line(frame, (Xcor[4], Ycor[4]), (Xcor[3], Ycor[3]), (0, 255, 0), 2)
        cv2.line(frame, (Xcor[3], Ycor[3]), (Xcor[8], Ycor[8]), (0, 255, 0), 2)
        cv2.line(frame, (Xcor[8], Ycor[8]), (Xcor[6], Ycor[6]), (0, 255, 0), 2)
        cv2.line(frame, (Xcor[6], Ycor[6]), (Xcor[10], Ycor[10]), (0, 255, 0), 2)
        cv2.line(frame, (Xcor[10], Ycor[10]), (Xcor[1], Ycor[1]), (0, 255, 0), 2)

        # get updated location of objects in subsequent frames
        success, boxes = multiTracker.update(frame)

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            x, y, w, h = int(newbox[0]), int(newbox[1]), int(newbox[2]), int(newbox[3])
            cv2.rectangle(frame, (x, y), ((x + w), (y + h)), colors[i], 2, 1)
            print("frame-->", frame)
            print("(x, y)-->", colors[i])

            cnetroids = (int(x + w / 2), int(y + w / 2))
            points.appendleft(cnetroids)

        for i in range(1, len(points)):
          if points[i - 1] is None or points[i] is None:
            continue
          # Calculate the thickness of the small line drawn
          thickness = int(np.sqrt(mybuffer / float(i + 1)) * 2.5)

          # Draw a small line
          cv2.line(frame, points[i - 1], points[i], (0, 0, 255), thickness)

        # show frame
        cv2.imshow('MultiTracker', frame)


        #datas = pd.DataFrame([[Xcor[1], Ycor[1], cnetroids]], columns=['xcor', 'ycor', 'cnetroids'])
        datas = pd.DataFrame([[Xcor[1], Ycor[1], Xcor[4], Ycor[4], Xcor[3], Ycor[3], Xcor[8], Ycor[8], Xcor[6], Ycor[6], Xcor[10], Ycor[10], points]])
        #print(datas)
        with open(filename, 'a', newline='\n') as fil:
            datas.to_csv(fil, header = False, index=False,)

        # quit on ESC button
        if cv2.waitKey(delay) & 0xFF == 27:  # Esc pressed
            break


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='image traking app v1')

    ap.add_argument('points', type=int, default='1', help='input the point (x,y) number: 1,2,3,4,5,6')
    ap.add_argument('filename', type=str, default=os.getcwd(), help='Input the csv file name for exmaple :input.csv')
    ap.add_argument('videoPath', type=str, default=os.getcwd(), help='Input the avi file path for exmaple : C:/Users/User/Desktop/ImageEx/project.avi')
    args = ap.parse_args()

    if os.stat(args.videoPath):
        print("files Exists")
        videoPath = args.videoPath
    else:
        print("get file auto")
        videoPath = 'project.avi'

    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    MulTracking(videoPath, delay, args.points, args.filename)