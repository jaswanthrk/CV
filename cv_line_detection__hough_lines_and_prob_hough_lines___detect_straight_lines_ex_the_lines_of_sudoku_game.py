# rho = x*cos(theta) + y*sin(theta)
# rho   : perpendicular distance from origin
# theta : angle formed by the normal of this line to the origin (in radians)

# cv2.HoughLines(binarized image, rho accuracy, theta accuracy, threshold)
# Threshold here is the minimum vote for it to be considered a line

# Probabilistic Hough Lines
# Idea is that is takes only a random subset of points sufficient enough for line detection
# Also returns the start and end points of the line unlike the previous function

# cv2.HoughLinesP(binarized image, rho accuracy, theta accuracy, threshold, minimum line length, max line gap)

import cv2
import numpy as np

image = cv2.imread('images/sudoku.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges= cv2.Canny(gray, 100, 170, apertureSize = 3)

# RUNNING HOUGHLINES USING A RHO ACCURACY OF 1 PIXEL
# THETA ACCURACY OF NP.PI / 180 WHICH IS 1 DEGREE
# OUR LINE THRESHOLD IS SET TO 240 (NUMBER OF POINTS ON LINE)
lines = cv2.HoughLines(edges, 1, np.pi/180, 240)

# WE ITERATE THROUGH EACH LINE AND CONVERT IT TO THE FORMAT
# REQUIRED BY cv.lines (i.e. requiring end points)
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))    
    x2 = int(x0 + 1000 * (-b))    
    y2 = int(y0 + 1000 * (a))    
    
    
cv2.imshow('Hough Lines', image)
cv2.waitKey()
cv2.destroyAllWindows()

############ Probabilistic Hough Lines ###########


import cv2
import numpy as np

image = cv2.imread('images/sudoku.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges= cv2.Canny(gray, 100, 170, apertureSize = 3)

# AGAIN WE USE THE SAME RHO AND THETA ACCURACIES
# HOWEVER, WE SPECIFIC A MINIMUM VOTE (PTS ALONG LINE) OF 100
# AND MIN LINE LENGTH OF 5 PIXELS AND MAX GAP BETWEEN LINES OF 10 PIXELS
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, 5, 10)
print(lines.shape)

for x1, y1, x2, y2 in lines[0]:
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

cv2.imshow('Probabilistic Hough Lines, image)
cv2.waitKey()
cv2.destroyAllWindows()



