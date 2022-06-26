import cv2
import numpy as np
from collections import deque
import argparse


avi =  'C:/Users/qarsh/OneDrive/Desktop/ImageEx/project.avi'
capImage = cv2.VideoCapture(avi)


cap = cv2.VideoCapture(avi)
fps = cap.get(cv2.CAP_PROP_FPS) # Ramkalar sonini oladi
delay = int(1000/fps)

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

global pts
pts = deque(maxlen=64)

def draw(img, inbbox):

    x, y, w, h = int(inbbox[0]), int(inbbox[1]), int(inbbox[2]), int(inbbox[3])
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(hsv, None, iterations = 2)
    mask = cv2.dilate(mask, None, iterations = 2)
    edged = cv2.Canny(mask, 30, 200)

    # cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x1, y1), radius) = cv2.minEnclosingCircle(c)

        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 72:
            cv2.circle(img, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(img, center, 2, (0, 0, 255), -1)

    pts.appendleft(center)
    print(pts)
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.rectangle(img, (x,y), ((x+w), (y+h)), (255,0,255), 3,1)


    cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
    cv2.circle(img, ((x+w), (y+h)), 4, (0, 255, 0), -1)
    #cv2.line(img, (x, y),((x+w), (y+h)), (0, 0, 255), 10)


    cv2.putText(img, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)




traking = cv2.TrackerMOSSE_create()
success, img = capImage.read()
inbbox  = cv2.selectROI("Traking", img, False)
traking.init(img, inbbox)


def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame


while True:
    timer = cv2.getTickCount()
    success, img = capImage.read()
    success, inbbox = traking.update(img)

    if success:
        draw(img, inbbox)
    else:
        cv2.putText(img, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



    fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
    cv2.putText(img, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #img = draw_rect(img)
    cv2.imshow("tracking", img)
    if cv2.waitKey(delay) & 0xff == ord('q'):
        break
