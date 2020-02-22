# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
        help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
        help="max buffer size")
args = vars(ap.parse_args())
# define the lower and upper boundaries of the "green" ball in the HSV color space
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
vs = cv2.VideoCapture(0)
# keep looping
while True:
        # grab the current frame
        _, frame = vs.read()
        # handle the frame from VideoCapture
        frame = frame[1] if args.get("video", False) else frame
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if frame is None:
                break
        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # only proceed if the radius meets a minimum size
                if radius > 20:
                        # draw the circle and centroid on the frame,
                        # then update the list of tracked points
                        cv2.circle(frame, (int(x), int(y)), int(radius),
                                (0, 255, 255), 2)
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)
                        print("Green ball detected")
                        res = cv2.bitwise_and(frame,frame, mask=mask)
                        ret,thrshed = cv2.threshold(cv2.cvtColor(res,cv2.COLOR_BGR2GRAY),3,255,cv2.THRESH_BINARY)
                        contours,hier = cv2.findContours(thrshed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                        cnt=max(contours,key=cv2.contourArea)
                        area = cv2.contourArea(cnt)
                        # Calculate whole area
                        _, img=vs.read()
                        h,w = img.shape[:2]
                        whole_area_mask = np.ones((h, w), np.uint8)
                        ret, thresh = cv2.threshold(whole_area_mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                        cnts = max(contours, key=cv2.contourArea)
                        whole_area = cv2.contourArea(cnts)
                        perc=(area*100/whole_area)
                        print("Percentage of ball: ",perc)
                        if(center[0]<=300 and center[1]<=200):
                                print("Top left corner")
                        elif(center[0]>=300 and center[1]<=200):
                                print("Top right corner")
                        elif(center[0]<=300 and center[1]>=200):
                                print("Bottom left corner")
                        elif(center[0]>=300 and center[1]>=200):
                                print("Bottom right corner")
                        cv2.waitKey(-1)
                        break
        # show the frame to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
                break
vs.release()
# close all windows
cv2.destroyAllWindows()

