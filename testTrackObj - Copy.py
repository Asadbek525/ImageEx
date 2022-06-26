import cv2
import numpy as np
from collections import deque
import argparse
import math
import time
global Xcor, Ycor

Xcor = []
Ycor = []
#centerMom = []
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
threshVal = 100  # initial threshold
#Set the red threshold, HSV space
redLower = np.array([170, 100, 100])
redUpper = np.array([179, 255, 255])
#Initialize the list of tracking points
mybuffer = 12
points = deque(maxlen=mybuffer)
centerY = deque(maxlen=12)
centerX = deque(maxlen=12)

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def draw(img, inbbox):

    x, y, w, h = int(inbbox[0]), int(inbbox[1]), int(inbbox[2]), int(inbbox[3])
    #edged = cv2.Canny(mask, 30, 200)


    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
    #blurred = cv2.GaussianBlur(img, (11, 11), 0)

    crop_img = img[y:y + h, x:x + w]

    crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    crop_gray = cv2.GaussianBlur(crop_gray, (7, 7), 0)
    crop_thresh = cv2.threshold(crop_gray, 45, 255, cv2.THRESH_BINARY)[1]
    crop_thresh = cv2.erode(crop_thresh, None, iterations=2)
    crop_thresh = cv2.dilate(crop_thresh, None, iterations=2)
    crop_canny_output = cv2.Canny(crop_thresh, threshVal, threshVal * 2)
    crop_contours, _ = cv2.findContours(crop_canny_output.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # If there is an outline


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    canny_output = cv2.Canny(thresh, threshVal, threshVal * 2)
    contours, _ = cv2.findContours(canny_output.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # calculate moments of binary image
        M = cv2.moments(c)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
        Xcor.append(cX)
        Ycor.append(cY)
        centerX.appendleft(cX)
        centerY.appendleft(cX)
        points.appendleft(center)
        # print(cX, cY)
    #center = (int(Xcor[1] / Ycor[1]), int(Xcor[4] / Ycor[4]))
    #print(centerX[1], centerY[1])
    #print("____________________")

    for i in range(1, len(points)):
        #print(i)
        if points[i - 11] is None or points[i] is None:
            continue
        #Calculate the thickness of the small line drawn
        thickness = int(np.sqrt(mybuffer / float(i + 1)) * 2.5)

        #Draw a small line
        print(points[i - 11], "<<______________>>", points[i])
        #print(centerX[1], centerY[1])
        #cv2.line(img,(centerX[1], centerY[1]), (centerX[4], centerY[4]), (0, 0, 255), thickness)
        cv2.line(img, points[i - 11], points[i], (0, 0, 255), thickness)

    cv2.line(img, (Xcor[1], Ycor[1]), (Xcor[4], Ycor[4]), (0, 255, 0), 2)

    cv2.line(img, (Xcor[4], Ycor[4]), (Xcor[3], Ycor[3]), (0, 255, 0), 2)
    cv2.line(img, (Xcor[3], Ycor[3]), (Xcor[8], Ycor[8]), (0, 255, 0), 2)
    cv2.line(img, (Xcor[8], Ycor[8]), (Xcor[6], Ycor[6]), (0, 255, 0), 2)
    cv2.line(img, (Xcor[6], Ycor[6]), (Xcor[10], Ycor[10]), (0, 255, 0), 2)
    cv2.line(img, (Xcor[10], Ycor[10]), (Xcor[1], Ycor[1]), (0, 255, 0), 2)
    cv2.imshow('Selected points', crop_img)

    cv2.putText(img, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def drwaline(camera, delay):

    traking = cv2.TrackerMOSSE_create()
    success, img = camera.read()
    inbbox = cv2.selectROI("Traking", img, False)
    traking.init(img, inbbox)

    # Wait for two seconds
    time.sleep(1)
    while camera.isOpened():
        # Read frame
        timer = cv2.getTickCount()
        success, img = capImage.read()
        success, inbbox = traking.update(img)
        if success:
            draw(img, inbbox)
        else:
            cv2.putText(img, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(img, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        cv2.imshow('tracking', img)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    camera.release()
    # Destroy all windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    avi = 'C:/Users/qarsh/OneDrive/Desktop/ImageEx/project.avi'
    capImage = cv2.VideoCapture(avi)
    cap = cv2.VideoCapture(avi)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    drwaline(capImage, delay)



