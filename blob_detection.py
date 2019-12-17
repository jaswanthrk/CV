# BLOBS : GROUPS OF CONNECTED PIXELS THAT ALL SHARE A COMMON PROPERTY

# simpleBlobDetector : CREATE DETECTOR => INPUT IMAGE INTO DETECTOR => OBTAIN KEY POINTS => DRAW KEY POINTS

import cv2
import numpy as np

image = cv2.imread('images/Sunflowers.jpg', cv2.IMREAD_GRAYSCALE)

detector = cv2.SimpleBlobDetector()

keypoints = detector.detect(image)

# DRAW DETECTED BLOBS AS RED CIRCLES.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of
# THE CIRCLE CORRESPONDS TO THE SIZE OF BLOB.
blank = np.zeros( (1, 1) )
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# Show keypoints
cv2.imshow("Blobs", blobs)
cv2.waitKey()
cv2.destroyAllWindows()


# cv2.drawKeypoints(input image, keypoints, blank_output_array, color, flags)
# flags : 
#         cv2.DRAW_MATCHES_FLAGS_DEFAULT
#         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
#         cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
#         cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
#         cv2.DRAW_MATCHES_FLAGS_DEFAULT
#         cv2.DRAW_MATCHES_FLAGS_DEFAULT
