from __future__ import print_function
import sys
import cv2
import numpy as np
from random import randint
from collections import deque

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


def MulTracking(videoPath, delay):
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

    # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
    # So we will call this function in a loop till we are done selecting all objects
    while True:
        # draw bounding boxes over objects
        # selectROI's default behaviour is to draw box starting from the center
        # when fromCenter is set to false, you can draw box starting from top left corner
        bbox = cv2.selectROI('MultiTracker', frame)
        bboxes.append(bbox)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        print(bboxes, "____")
        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        k = cv2.waitKey(0) & 0xFF
        if (k == 113):  # q is pressed
            break

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
          # calculate moments of binary image
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
        print(boxes)
        # draw tracked objects
        for i, newbox in enumerate(boxes):
            x, y, w, h = int(newbox[0]), int(newbox[1]), int(newbox[2]), int(newbox[3])
            cv2.rectangle(frame, (x, y), ((x + w), (y + h)), colors[i], 2, 1)
            points.appendleft(((x + w), (y + h)))
            #cv2.line(frame,  (x, y),  ((x + w), (y + h)), (0, 255, 0), 2)

        for i in range(1, len(points)):
          if points[i - 1] is None or points[i] is None:
            continue
          # Calculate the thickness of the small line drawn
          thickness = int(np.sqrt(mybuffer / float(i + 1)) * 2.5)

          # Draw a small line
          print(points[i - 1], "<<______________>>", points[i])
          cv2.line(frame, points[i - 1], points[i], (0, 0, 255), thickness)
        # show frame
        cv2.imshow('MultiTracker', frame)
        #cv2.imshow('canny_output', canny_output)

        # quit on ESC button
        if cv2.waitKey(delay) & 0xFF == 27:  # Esc pressed
            break


if __name__ == '__main__':
    videoPath = 'C:/Users/qarsh/OneDrive/Desktop/ImageEx/project.avi'

    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    MulTracking(videoPath, delay)