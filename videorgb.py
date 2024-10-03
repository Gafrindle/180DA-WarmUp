# Code basis taken from https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
# Modified using https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
# Modified using code snippets from https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
# Grayscale conversion taken from https://stackoverflow.com/questions/38438646/findcontours-support-only-8uc1-and-32sc1-images, by Dainius Saltenis
# For loop taken from https://stackoverflow.com/questions/41542861/unable-to-display-bounding-rectangles-around-contours-in-opencv-python
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # define range of blue color in BGR
    lower_blue = np.array([150,0,0])
    upper_blue = np.array([255,100,50])

    mask = cv2.inRange(frame, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    # Bounding box
    img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours,hierarchy = cv2.findContours(img, 1, 2)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()