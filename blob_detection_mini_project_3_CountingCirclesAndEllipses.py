import cv2
import numpy as np

image = cv2.imread('images/blobs.jpg', 0)
cv2.imshow('Original Image', image)
cv2.waitKey()

detector = cv2.SimpleBlobDetector()

keypoints = detector.detect(image)

blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Total number of Blobs : " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

cv2.imshow("Blobs using default parameters", blobs)
cv2.waitKey()

######### SET OUR FILTERING PARAMETERS
# INITIALIZE PARAMETER SETTING USING CV2.SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# Set Area filtering parameters
params.filterByArea = True
params.minArea = 100

# Set Circularity filtering parameters
params.filterByCircularity = True
params.minCircularity = 0.9
# 1 being perfect circle and 0 being opposite

# Set Convexity filtering parameters
# - Area of blob / Area of Convex Hull
params.filterByConvexity = False
params.minConvexity = 0.2

# Set inertia filtering parameters
# - MEASURE OF ELLIPTICALNESS : LOW BEING MORE ELLIPTICAL, HIGH BEING MORE CIRCULAR
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector(params)

# Detect blobs
keypoints = detector.detect(params)

# Draw blobs on our image as red circles
blank = np.zeros( (1, 1) )
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Number of Circular Blobs : " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

cv2.imshow("Filtering Circular Blobs Only", blobs)
cv2.waitKey()
cv2.destroyAllWindows()
