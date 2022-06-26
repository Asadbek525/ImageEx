import cv2
#import imutils
import numpy as np;
import matplotlib.pyplot as plt
from PIL import Image
import random as rng
from multiprocessing import Process

global Xcor, Ycor
Xcor = []
Ycor = []

Xcor2 = []
Ycor2 = []


def drwaLine(image, image2):

    threshVal = 100  # initial threshold
    '''
    dir = 'C:/Users/User/Desktop/ImageEx/IMG_1192.jpeg'
    img = Image.open(dir)  # Open the picture
    plt.figure("youyou")
    plt.imshow(img)
    plt.show()
    '''
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    canny_output = cv2.Canny(thresh, threshVal, threshVal * 2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(canny_output.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    #im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = imutils.grab_contours(cnts)
    #c = max(cnts, key=cv2.contourArea)

    for c in contours:

    # calculate moments of binary image
        M = cv2.moments(c)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        crcPos = cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
        Xcor.append(cX)
        Ycor.append(cY)
        #print(cX, cY)

    print(Xcor, Ycor)

    cv2.line(image, (Xcor[1], Ycor[1]), (Xcor[4], Ycor[4]), (0, 255, 0), 2)
    cv2.line(image, (Xcor[4], Ycor[4]), (Xcor[3], Ycor[3]), (0, 255, 0), 2)
    cv2.line(image, (Xcor[3], Ycor[3]), (Xcor[8], Ycor[8]), (0, 255, 0), 2)
    cv2.line(image, (Xcor[8], Ycor[8]), (Xcor[6], Ycor[6]), (0, 255, 0), 2)
    cv2.line(image, (Xcor[6], Ycor[6]), (Xcor[10], Ycor[10]), (0, 255, 0), 2)
    cv2.line(image, (Xcor[10], Ycor[10]), (Xcor[1], Ycor[1]), (0, 255, 0), 2)

    ########################################################################################
    #
    #                   Image comparing
    #
    #

    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (7, 7), 0)
    thresh2 = cv2.threshold(gray2, 45, 255, cv2.THRESH_BINARY)[1]
    thresh2 = cv2.erode(thresh2, None, iterations=2)
    thresh2 = cv2.dilate(thresh2, None, iterations=2)

    canny_output2 = cv2.Canny(thresh2, threshVal, threshVal * 2)

    cnts2 = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(canny_output2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # c = max(cnts, key=cv2.contourArea)

    for cntr in contours2:
        # calculate moments of binary image
        M2 = cv2.moments(cntr)
        # calculate x,y coordinate of center
        cX2 = int(M2["m10"] / M2["m00"])
        cY2 = int(M2["m01"] / M2["m00"])
        crcPos = cv2.circle(image2, (cX2, cY2), 5, (255, 255, 255), -1)
        Xcor2.append(cX2)
        Ycor2.append(cY2)

    print(Xcor2, Ycor2)

    cv2.line(image2, (Xcor2[1], Ycor2[1]), (Xcor2[4], Ycor2[4]), (255, 255, 0), 2)
    cv2.line(image2, (Xcor2[4], Ycor2[4]), (Xcor2[3], Ycor2[3]), (255, 255, 0), 2)
    cv2.line(image2, (Xcor2[3], Ycor2[3]), (Xcor2[8], Ycor2[8]), (255, 255, 0), 2)
    cv2.line(image2, (Xcor2[8], Ycor2[8]), (Xcor2[6], Ycor2[6]), (255, 255, 0), 2)
    cv2.line(image2, (Xcor2[6], Ycor2[6]), (Xcor2[10], Ycor2[10]), (255, 255, 0), 2)
    cv2.line(image2, (Xcor2[10], Ycor2[10]), (Xcor2[1], Ycor2[1]), (255, 255, 0), 2)

    cv2.line(image2, (Xcor[1], Ycor[1]), (Xcor[4], Ycor[4]), (0, 255, 0), 2)
    cv2.line(image2, (Xcor[4], Ycor[4]), (Xcor[3], Ycor[3]), (0, 255, 0), 2)
    cv2.line(image2, (Xcor[3], Ycor[3]), (Xcor[8], Ycor[8]), (0, 255, 0), 2)
    cv2.line(image2, (Xcor[8], Ycor[8]), (Xcor[6], Ycor[6]), (0, 255, 0), 2)
    cv2.line(image2, (Xcor[6], Ycor[6]), (Xcor[10], Ycor[10]), (0, 255, 0), 2)
    cv2.line(image2, (Xcor[10], Ycor[10]), (Xcor[1], Ycor[1]), (0, 255, 0), 2)


    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])


    for i in range(len(contours)):
         color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
         cv2.drawContours(drawing, contours_poly, i, color)
         #cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
         #cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

    cv2.imshow("Orginal", image)
    cv2.imshow("image2", image2)
    #cv2.imshow("drawing", drawing)
    #drwaLine2()
    #cv2.imwrite('C:/Users/User/Desktop/ImageEx/drawing2.jpeg', canny_output)
    #fig, ax = plt.subplots()
    #ax.imshow(img)
    cv2.waitKey(0)



def drwaVideo(frame):
    vidFile = cv2.VideoCapture(frame)
    while True:
        ret, image = vidFile.read()

        fps = vidFile.get(cv2.CAP_PROP_FPS)  # 프레임 수 구하기
        delay = int(1000 / fps)

        threshVal = 100  # initial threshold
        #output = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        canny_output = cv2.Canny(thresh, threshVal, threshVal * 2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(canny_output.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        #im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = imutils.grab_contours(cnts)
        #c = max(cnts, key=cv2.contourArea)

        for c in contours:

        # calculate moments of binary image
            M = cv2.moments(c)
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            crcPos = cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
            Xcor2.append(cX)
            Ycor2.append(cY)
            #print(cX, cY)

        print(Xcor, Ycor)

        cv2.line(image, (Xcor2[1], Ycor2[1]), (Xcor2[4], Ycor2[4]), (0, 255, 0), 2)
        cv2.line(image, (Xcor2[4], Ycor2[4]), (Xcor2[3], Ycor2[3]), (0, 255, 0), 2)
        cv2.line(image, (Xcor2[3], Ycor2[3]), (Xcor2[8], Ycor2[8]), (0, 255, 0), 2)
        cv2.line(image, (Xcor2[8], Ycor2[8]), (Xcor2[6], Ycor2[6]), (0, 255, 0), 2)
        cv2.line(image, (Xcor2[6], Ycor2[6]), (Xcor2[10], Ycor2[10]), (0, 255, 0), 2)
        cv2.line(image, (Xcor2[10], Ycor2[10]), (Xcor2[1], Ycor2[1]), (0, 255, 0), 2)


        #cv2.circle(image, (Xcor[1], Ycor[1]), 20, (255, 0, 0), 3)
        #cv2.circle(image, (Xcor[2], Ycor[2]), 20, (255, 0, 0), 3)
        #cv2.circle(image, (Xcor[10], Ycor[10]), 20, (255, 0, 0), 3)

        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

        contours_poly = [None] * len(contours)
        boundRect = [None] * len(contours)
        centers = [None] * len(contours)
        radius = [None] * len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
            centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])


        for i in range(len(contours)):
             color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
             cv2.drawContours(drawing, contours_poly, i, color)
             #cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
             #cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

        cv2.imshow("Orginal2", image)
        #cv2.imshow("Canny", canny_output)
        #cv2.imshow("drawing", drawing)

        #cv2.imwrite('C:/Users/User/Desktop/ImageEx/drawing2.jpeg', canny_output)
        #fig, ax = plt.subplots()
        #ax.imshow(img)
        cv2.waitKey(delay)


def circles(image):

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)


    accum_size = 1
    # Minimum distance between the centers of the detected circles.
    minDist = 50
    # First method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).
    param1 = 50
    # Second method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
    param2 = 5
    #
    minRadius = 1
    #
    maxRadius = 10
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, accum_size, minDist,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    circles = circles.reshape(1, circles.shape[1], circles.shape[2])
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for ind, i in enumerate(circles[0, :]):
            center = (i[0], i[1])
            radius = 10
            cv2.circle(image, center, radius, (255, 0, 255), 2)

    #thresh = cv2.resize(thresh, (1280, 720))  # <---- This is just for easier display
    #image = cv2.resize(image, (1280, 720))  # <---- This is just for easier display

    cv2.imshow("circles_black_dot", image)
    cv2.imshow("threshold_black_dots", thresh)
    cv2.waitKey(0)

def circlesVideoes(vid):
    vidFile = cv2.VideoCapture(vid)
    ret, frame = vidFile.read()
    cv2.imshow("Image", frame)


    # fig, ax = plt.subplots()
    # ax.imshow(img)
    cv2.waitKey(0)


if __name__ == '__main__':


   drwaVideo("project.avi")
   '''
   dir = "IMG_1178.jpeg"
   dir2 = "IMG_1193.jpeg"
   #drwaLine(dir)
   frame = cv2.imread(dir)
   frame2 = cv2.imread(dir2)

   pro = Process(target=drwaLine, args=(frame,frame2,))
   #pro2 = Process(target=drwaLine2, args=(frame2, Xcor, Ycor, ))

   pro.start()
   #pro2.start()

   pro.join()
   #pro2.join()

   #drwaLine2()
   #drwaLine(frame)

   #th = Thread(target=drwaLine2)
   #th.start()

'''

