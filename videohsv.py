# Code basis taken from https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
# Modified using code snippets from https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
# Modified using code snippets from https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
# Grayscale conversion taken from https://stackoverflow.com/questions/38438646/findcontours-support-only-8uc1-and-32sc1-images, by Dainius Saltenis
# For loop taken from https://stackoverflow.com/questions/41542861/unable-to-display-bounding-rectangles-around-contours-in-opencv-python
# Dominant color code taken from https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
# Image cropping code taken from https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    # Dominant color code - see header for source
    def find_histogram(clt):
        """
        create a histogram with k clusters
        :param: clt
        :return:hist
        """
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        hist = hist.astype("float")
        hist /= hist.sum()

        return hist
    def plot_colors2(hist, centroids):
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0

        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                        color.astype("uint8").tolist(), -1)
            startX = endX

        # return the bar chart
        return bar

    # Crop frame
    # Through trial and error, image size is 650x500
    crop_frame = frame[200:300, 200:450]
    
    # Through trial and error
    # Displays rectangle for determining majority color - blue border
    cv2.rectangle(frame,(200,200),(450,300),(255,0,0),2)

    dom = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)

    dom = dom.reshape((dom.shape[0] * dom.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(dom)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    plt.axis("off")
    plt.imshow(bar)
    plt.show()

    # Bounding box
    img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours,hierarchy = cv2.findContours(img, 1, 2)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        # Filter out small rectangles
        # Credit to Ashish Basetty for the idea
        if(w > 40):
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# Lab question answers:
# 1: I did not notice a big difference between tracking for HSV and RGB. The range needs to be significant to account for differences in lighting depending on orientation.
# 2: The flashlight helps with tracking, as the object more closely matches the color threshold.
# 3: Changing the phone brightness helps to an extent, but exceeds the threshold if turned too bright.
# 4: I did not notice a large difference in the robustness to brightness of my phone compared to my non-phone object.