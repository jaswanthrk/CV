# cv2.matchShapes(contour template, contour, method, method parameter)
# OUTPUT : MATCH VALUE - lower the better

#  CONTOUR TEMPLATE : REFERENCE CONTOUR 
#  CONTOUR : THE INDIVIDUAL CONTOUR WE ARE CHECKING AGAINST
#  METHOD  : TYPE OF CONTOUR MATCHING (1, 2, 3)
#  http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
#  METHOD PARAMETER : LEAVE ALONE AS 0.0 (NOT FULLY UTILIZED IN PYTHON OPENCV)

import cv2
import numpy as np

# Load the template or reference image
template = cv2.imread('images/4star,jpg', 0)
cv2.imshow('Template', template)
cv2.waitKey()

# Load the target image with the shapes we're trying to match
target = cv2.imread('images/shapes_to_match.jpg')
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

# Threshold both images first before using cv2.findContours
ret, thresh1 = cv2.threshold(template, 127, 255, 0)
ret, thresh2 = cv2.threshold(target_gray, 127, 255, 0)

# Find contours in template
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# SORT CONTOURS SO THAT THE WE CAN REMOVE THE LARGEST CONTOUR WHICH IS THE IMAGE OUTLINE
sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
template_contour = contours[1]

# Find contours in target
contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # iterate through each contour in the target image and use cv2.matchShapes to compare contour shapes
    match = cv2.matchShapes(template_contour, c, 1, 0.0)
    print(match)
    if match < 0.15 :
        closest_contour = c
    else :
        closest_contour = []
        
cv2.drawContours(target, [closest_contour], -1, (0, 255, 0), 3)
cv2.imshow('Output', target)
cv2.waitKey()
cv2.destroyAllWindows()
