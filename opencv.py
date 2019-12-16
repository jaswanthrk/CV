import cv2
import numpy as np

image = cv2.imread('../images/input.jpg')

cv2.imshow('TITLE', image)

cv2.waitKey(time in ms) # if 0 - wait till a key is pressed

cv2.destroyAllWindows() # Closes all open windows

cv2.imwrite('output.jpg',image) # filename and the image to be saved.
# Returns true if done.



################# COLOR IMAGE TO GREYSCALE #################

import cv2

image = cv2.imread('../images/input.jpg')
cv2.imshow('Original',image)
cv2.waitKey()

# EITHER
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# OR
gray_image = cv2.imread('../images/input.jpg', 0)

cv2.imshow('Grayscale',gray_image)
cv2.waitKey()
cv2.destroyAllWindows()

################# COLOR SPACES #################

## BGR
image = cv2.imread('../images/input.jpg')

B, G, R = image[0,0]

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
x = image[0,0]

## HSV
image = cv2.imread('../images/input.jpg')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#H: 0 - 180; S: 0 - 255; V: 0 - 255
cv2.imshow('HSV Image', hsv_image)
cv2.imshow('Hue Channel', hsv_image[:, :, 0])
cv2.imshow('Saturation Channel', hsv_image[:, :, 1])
cv2.imshow('Value Channel', hsv_image[:, :, 2])

cv2.waitKey()
cv2.destroyAllWindows()

## SPLITTING AND MERGING R G B
B, G, R = cv2.split(image)

cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)
cv2.waitKey(0)
cv2.destroyAllWindows()

merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)

blue_amplified_merge = cv2.merge([B + 100, G, R])
cv2.imshow("Merged with Blue Amplified", blue_amplified_merge)

cv2.waitKey(0)
cv2.destroyAllWindows()


# cv2.imshow("Red", R) -- Red values in grayscale will be shown. 
# BUT What if I want Red values in RED COLOUR ONLY.

B, G, R = cv2.split(image)

zeros = np.zeros(image.shape[:2], dtype = "uint8")

cv2.imshow("Red",   cv2.merge([zeros, zeros, R])
cv2.imshow("Green", cv2.merge([zeros, G, zero])
cv2.imshow("Blue",  cv2.merge([B, zeros, zeros])

cv2.waitKey()
cv2.destroyAllWindows()


###########################################################
######### Histograms ############

from matplotlib.pyplot as plt

image = cv2.imread('')
histogram = cv2.calcHist([image], [0], None, [0, 256])

plt.hist(image.ravel(), 256, [0, 256]);
plt.show()

color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histogram2, color = col)
    plt.xlim([0, 256])

plt.show()

# images : it is the source image of type uint8 or float32. It should be given in square brackets i.e., "[img]".
# Channels : It is also given in square brackets. It is in the index of channel for which we calculate histogram. For example, if input is grayscale image, its 
             value is [0]. For color image, you can pass [0], [1], [2] to calculate histogram of blue, green or red channel respectively.
# Mask   : Mask Image. To find histogram of full image, it is given as "None". But, if you want to find histogram of particular region of image, you have to create a mask
             image for that and give it as mask. (I will show an example later.)
# histSize : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].
# ranges : this is our RANGE. Normally, it is [0, 256].






#####################################################

# CREATING BLACK IMAGES : COLOR AND BLACK&WHITE
image    = np.zeros( (512, 512, 3), np.uint8)
image_bw = np.zeros( (512, 512, 3), np.uint8)

cv2.imshow("Black Rectangle (Color)", image)
cv2.imshow("Black Rectangle (B & W)", image_bw)

cv2.waitKey()
cv2.destroyAllWindows()

############ DRAW A DIAGONAL BLUE LINE OF THICKNESS OF 5 PIXELS

image = np.zeros((512, 512, 3), np.uint8)

# cv2.line(image, starting coordinates, ending coordinates, colour, thickness)
cv2.line(image, (0,0), (511,511), (255,127,0), 5)

cv2.imshow("Blue Line", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

############ DRAW A RECTANGLE

image = np.zeros((512, 512, 3), np.uint8)

# cv2.rectangle(image, starting vertex, opposite vertex, colour, thickness)
cv2.rectangle(image,  (100,100), (300,250), (127,50,127), 5)

cv2.imshow("Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

############ DRAW A CIRCLE

image = np.zeros((512, 512, 3), np.uint8)

# cv2.circle(image, center, radius, colour, fill)
cv2.circle(image, (350,350), 100, (15,75,50), -1)

cv2.imshow("Circle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

############ DRAW A POLYGON

image = np.zeros((512, 512, 3), np.uint8)

pts = np.array([[10, 50], [400, 50], [90, 200], [50, 500]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(image, [pts], True, (0, 0, 255), 3)

cv2.imshow("Polygon", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


############################################################ 

############ ADD TEXT TO DISPLAY

cv2.putText(image, 'Text to Display', bottom left starting point, Font, Font Size, Color, Thickness)
# FONT_HERSHEY_ SIMPLEX/PLAIN/COMPLEX/DUPLEX/TRIPLEX/COMPLEX_SMALL/SCRIPT_SIMPLEX/SCRIPT_COMPLEX

cv2.putText(image, 'Hello World!', (75, 290), cv2.FONT_HERSHEY_COMPLEX, 2, (100, 170, 0), 3)
cv2.imshow("Hello World!", image)

cv2.waitKey(0)
cv2.destroyAllWindows()


############################################################ ############################################################ 
############################################################ ############################################################ 
############################################################ ############################################################ 
############################################################ ############################################################ 


################# TRANSFORMATIONS #################

################# TRANSLATE
import cv2
import numpy as np

image = cv2.imread('images/input.jpg')

height, width = image.shape[:2]
quarter_height, quarter_width = height/4, width/4

# T = [[ 1 0 T_x]
#      [ 0 1 T_y]]

T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])

img_translation = cv2.warpAffine(image, T, (width, height))

cv2.imshow('Translation', img_translation)
cv2.waitKey()
cv2.destroyAllWindows()



################# ROTATIONS

# cv2.getRotationMatrix2D(rotation_center_x, rotation_center_y, angle_of_rotation, scale)

import cv2
import numpy as np

image = cv2.imread('images/input.jpg')

height, width = image.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
rotated_image   = cv2.warpAffine(image, rotation_matrix, (width, height) )

cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey()
cv2.destroyAllWindows()

rotated_image2 = cv2.transpose(image)
cv2.imshow('Rotated Image 2', rotated_image2)
cv2.waitKey()
cv2.destroyAllWindows()


################# RESIZING, SCALING AND INTERPOLATION

# cv2.INTER_AREA    - GOOD FOR SHRINKAGE OR DOWN SAMPLING
# cv2.INTER_NEAREST - FASTEST
# cv2.INTER_LINEAR  - GOOD FOR ZOOMING OR UP SAMPLING
# cv2.INTER_CUBIC   - BETTER
# cv2.INTER_LANCZOS4 - BEST


import cv2
import numpy as np

image = cv2.imread('images/input.jpg')

# 3/4 OF IT'S ORIGINAL SIZE
image_scaled = cv2.resize(image, None, fx = 0.75, fy = 0.75)
cv2.imshow('Scaling - Linear Interpolation', image_scaled)
cv2.waitKey()

# DOUBLE THE SIZE OF OUR IMAGE
image_scaled = cv2.resize(image, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Scaling - Cubic Interpolation', image_scaled)
cv2.waitKey()

# SKEW THE RESIZING BY SETTING EXACT DIMENSIONS
image_scaled = cv2.resize(image, (900, 400), interpolation = cv2.INTER_AREA)
cv2.imshow('Scaling - Skewed size', image_scaled)
cv2.waitKey()

cv2.destroyAllWindows()



################# IMAGE PYRAMIDS.   --- MAKE IMAGE LARGER AND SMALLER

import cv2
import numpy as np

image = cv2.imread('images/input.jpg')

smaller = cv2.pyrDown(image)
larger  = cv2.pyrUp(smaller)

cv2.imshow('Original', image)
cv2.imshow('Smaller', smaller)
cv2.imshow('Larger', larger)
cv2.waitKey()

cv2.destroyAllWindows()


################# CROPPING

import cv2
import numpy as np

image = cv2.imread('images/input.jpg')
height, width = image.shape[:2]

start_row, start_col = int(height * 0.25), int(width * 0.25)
end_row,     end_col = int(height * 0.75), int(width * 0.75)

cropped = image[start_row : end_row, start_col : end_col]

cv2.imshow('Original Image', image)
cv2.waitKey()
cv2.imshow('Cropped Image', cropped)
cv2.waitKey()
cv2.destroyAllWindows()




################# ARITHMETIC OPERATIONS

import cv2
import numpy as np

image = cv2.imread('images/input.jpg')

M = np.ones(image.shape, dtype = "uint8") * 75

added = cv2.add(image, M)
cv2.imshow('Added', added)

subtracted = cv2.subtract(image, M)
cv2.imshow('Subtracted', subtracted)

cv2.waitKey()
cv2.destroyAllWindows()



################# BITWISE OPERATIONS AND MASKING

import cv2
import numpy as np

image = cv2.imread('images/input.jpg')

square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, (50, 50), (250, 250), 255, -2)
cv2.imshow('Square', square)
cv2.waitKey()

ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1)
cv2.imshow('Ellipse', ellipse)
cv2.waitKey()

cv2.destroyAllWindows()


################# BITWISE OPERATIONS

And = cv2.bitwise_and(square, ellipse)
cv2.imshow("AND", And)
cv2.waitKey()

Or = cv2.bitwise_or(square, ellipse)
cv2.imshow("OR", Or)
cv2.waitKey()

Xor = cv2.bitwise_xor(square, ellipse)
cv2.imshow("XOR", Xor)
cv2.waitKey()

bitwiseNot_sq = cv2.bitwise_not(square)
cv2.imshow("NOT - square", bitwiseNot_sq)
cv2.waitKey()

cv2.destroyAllWindows()


############################################################ ############################################################ 


################# CONVOLUTIONS AND BLURRING #################


# cv2.filter2D(image, -1, kernel)

import cv2
import numpy as np

image = cv2.imread('images/input.jpg')
cv2.imshow('Image', image)
cv2.waitKey()

kernel_3x3 = np.ones((3,3), np.float32) / 9

blurred = cv2.filter2D(image, -1, kernel_3x3)
cv2.imshow('3x3 Kernel Blurring', blurred)
cv2.waitKey()

kernel_7x7 = np.ones((7,7), np.float32) / 49

blurred2 = cv2.filter2D(image, -1, kernel_7x7)
cv2.imshow('7x7 Kernel Blurring', blurred2)
cv2.waitKey()

cv2.destroyAllWindows()


################# COMMONLY USED CONVLTIONS AND BLURRING


import cv2
import numpy as np

image = cv2.imread('images/input.jpg')

blur = cv2.blur(image, (3,3)) # BOX SIZE NEEDS TO BE ODD AND POSITIVE
cv2.imshow('Averaging', blur)
cv2.waitKey()

Gaussian = cv2.GaussianBlur(image, (7,7), 0)
cv2.imshow('Gaussian Blurring', Gaussian)
cv2.waitKey()

Gaussian = cv2.medianBlur(image, 5)
cv2.imshow('Median Blurring', median)
cv2.waitKey()

bilateral = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow('Bilateral Blurring', bilateral)
cv2.waitKey()

cv2.destroyAllWindows()



# cv2.fastN1MeansDenoising()              - single grayscale images
# cv2.fastN1MeansDenoisingColored()       - color image
# cv2.fastN1MeansDenoisingMulti()         - image sequence captured in short period of time (grayscale)
# cv2.fastN1MeansDenoisingColoredMulti()  - image sequence captured in short period of time (color)


import cv2
import numpy as np

image = cv2.imread('images/input.jpg')

# after None - the filter strength 'h' (5-10 good range)
# Next is hForColorComponents, set as same value as h again
dst = cv2.fastN1MeansDenoisingColored(image, None, 6, 6, 7, 21)

cv2.imshow('Fast Means Denoising', dst)
cv2.waitKey()

cv2.destroyAllWindows()



############# SHARPENING

import cv2
import numpy as np

image = cv2.imread('images/input.jpg')
cv2.imshow('Original', image)

kernel_sharpening = np.array([[-1, -1, -1]
                              [-1,  9, -1]
                              [-1, -1, -1]])

sharpened = cv2.filter2D(image, -1, kernel_sharpening)

cv2.imshow('Image Sharpening', sharpened)

cv2.waitKey()
cv2.destroyAllWindows()



############# THRESHOLDING, BINARIZATION & ADAPTIVE THRESHOLDING

cv2.threshold(image, Threshold Value, Max Value, Threshold Type)

# cv2.THRESH_BINARY       - MOST COMMON
# cv2.THRESH_BINARY_INV   - MOST COMMON
# cv2.THRESH_TRUNC
# cv2.THRESH_TOZERO
# cv2.THRESH_TOZERO_INV

# NOTE : IMAGE NEED TO BE CONVERTED TO GRAYSCALE BEFORE THRESHOLDING

import cv2
import numpy as np

image = cv2.imread('images/input.jpg')
cv2.imshow('Original', image)

# all < 127 -> 0 and all > 255 -> 255
ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('1 Threshold Binary', thresh1)

# all < 127 -> 255 and all > 255 -> 0
ret, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('2 Threshold Binary Inverse', thresh2)

# all > 127 -> 127 and 255 not used
ret, thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
cv2.imshow('3 Threshold Truncated', thresh3)

# all < 127 -> 0 and 255 not used
ret, thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
cv2.imshow('4 Threshold Truncated', thresh4)

# all > 127 -> 0 and 255 not used
ret, thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow('5 Threshold Truncated Inverse', thresh5)

cv2.waitKey()
cv2.destroyAllWindows()

###### ADAPTIVE WAY OF THRESHOLDING: 


import cv2
import numpy as np

image = cv2.imread('images/input.jpg')
cv2.imshow('Original', image)
cv2.waitKey()


ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary', thresh1)
cv2.waitKey()

image = cv2.GaussianBlur(image, (3, 3), 0)  # ITS GOOD TO BLUR COZ IT REMOVES NOISE
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
# cv2.adaptiveThreshold(image, Max Value, Adaptive Type, Threshold Type, Block Size (Odd NUmber), Constant that is subtracted from Mean)
# Threshold Types : ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_OTSU
cv2.imshow('Adaptive Mean Thresholding', thresh)
cv2.waitKey()

_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('OTSU's Thresholding', thresh)
cv2.waitKey()

blur = cv2.GaussianBlur(image, (5, 5), 0)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('Gaussian OTSU's Thresholding', thresh)
cv2.waitKey()


############# DILATION AND EROSION

# Dilation - Adds pixels to the boundaries of objects in an image
# Erosion  - Removes pixels at the boundaries of objects in an image
# Opening  - Erosion followed by dilation
# Closing  - Dilation followed by erosion

import cv2
import numpy as np

image = cv2.imread('images/input.jpg')
cv2.imshow('Original', image)
cv2.waitKey()

kernel = np.ones( (5,5), np.uint8)

erosion = cv2.erode(image, kernel, iterations = 1)
cv2.imshow('Erosion', erosion)
cv.waitKey()

dilation = cv2.dilate(image, kernel, iterations = 1)
cv2.imshow('Dilation', dilation)
cv.waitKey()

opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
cv2.imshow('Opening', opening)
cv.waitKey()

closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closing', closing)
cv.waitKey()






############# EDGE DETECTION & IMAGE GRADIENTS

# 1. SOBEL - TO EMPHASIZE VERTICAL AND HORIZONTAL EDGES
# 2. LAPLACIAN - GETS ALL ORIENTATIONS
# 3. CANNY - OPTIMAL DUE TO LOW ERROR RATE, WELL DEFINED EDGES AND ACCURATE DETECTION
      : GAUSSIAN BLURRING
      : FINDS INTENSITY GRADIENT OF THE IMAGE
      : APPLIED NON-MAX SUPPRESSION ( REMOVES PIXELS THAT ARE NOT EDGES )
      : HYSTERESIS => APPLIES THRESHOLDS ( i.e., IF PIXEL IS WITHIN IN THE UPPER AND LOWER THRESHOLDS, IT IS CONSIDERED AN EDGE)

import cv2
import numpy as np

image = cv2.imread('images/input.jpg')
height, width = image.shape

# EXTRACT SOBEL EDGES
sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 5)

cv2.imshow('Original', image)
cv2.waitKey()
cv2.imshow('Sobel X', sobel_x)
cv2.waitKey()
cv2.imshow('Sobel Y', sobel_y)
cv2.waitKey()

sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)
cv2.imshow('Sobel_OR', sobel_OR)
cv2.waitKey()

laplacian = cv2.Laplacian(image, cv2.CV_64F)
cv2.imshow('Laplacian', laplacian)
cv2.waitKey()

## Then, Then we need to provide two values: threshold1 and threshold2. 
# Any gradient value larger than threshold2 is considered to be an edge. 
# Any value below threshold1 is considered not to be an edge. 
# Values in between threshold1 and threshold2 are either classified as edges or non-edges based on how their intensity are connected. 
# In this case, any gradient values below 60 are considered non-edges whereas any values above 120 are considered edges.

# Canny edge detection uses gradient values as thresholds. 

# The first threshold gradient
canny = cv2.Canny(image, 20, 170)
cv2.imshow('Canny', canny)

cv2.waitKey()
cv2.destroyAllWindows()











################ PERSPECTIVE TRANSFORM

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/scan.jpg')

cv2.imshow('Original',image)
cv2.waitKey()

# 4 corners coordinates of the original image
points_A = np.float32([[320, 15], [700, 215], [85, 610], [530, 780]])

# Coordinates of the 4 corners of the desired output
# We use a ratio of an A4 Paper 1 : 1.41
points_B = np.float32([[0, 0], [420, 0], [0, 594], [420, 594]])

# Use the two sets of four points to compute the Perspective Transformation Matrix, M
M = cv2.getPerspectiveTransform(points_A, points_B)

warped = cv2.warpPerspective(image, M, (420, 594) )

cv2.imshow('warPerspective', warped)
cv2.waitKey()
cv2.destroyAllWindows()



############### AFFINE TRANSFORM NEEDS ONLY 3 COORDINATES TO OBTAIN THE CORRECT TRANSFORM

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/scan.jpg')
rows, cols, ch = image.shape

cv2.imshow('Original',image)
cv2.waitKey()

points_A = np.float32([[320, 15], [700, 215], [85, 610]])
points_B = np.float32([[0, 0], [420, 0], [0, 594]])

M = cv2.getPerspectiveTransform(points_A, points_B)
warped = cv2.warpAffine(image, M, (cols, rows) )

cv2.imshow('warpPerspective', warped)
cv2.waitKey()




########################################################################
########################################################################
########################################################################

############### MINI PROJECT #1 - LIVE SKETCH USING WEBCAM

import cv2
import numpy as np

def sketch(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    canny_edges = cv2.Canny(img_gray_blur, 10, 70)
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
    return mask
           
           
           
# Initialize  webcam, cap is the object provided by VideoCapture
# It contains a boolean indicating if it was succesful (ret)
# It also contains the images collected from the webcam (frame)
cap = cv2.VideoCapture()
           
while True :
    ret, frame = cap.read()
    cv2.imshow('Our Live Sketcher', sketch(frame))
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break;
           
# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
########################################################################
########################################################################
########################################################################

############### SECTION IV : IMAGE SEGMENTATION
           
# CONTOURS ARE VERY IMPORTANT IN : (A) OBJECT DETECTION (B) SHAPE ANALYSIS
           
           
           
import cv2
import numpy as np
           
image = cv2.imread('')
cv2.imshow('Input Image', image)
cv2.waitKey()           
           
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)           
edged = cv2.Canny(gray, 30, 200)           
cv2.imshow('Canny Edges', edged)
cv2.waitKey()           
           
# Finding contours; Use a copy of your image e.g. edged.copy(), since findContours alters the image
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
           
# cv2.findContours(image, Retrieval Mode, Approximated Method)
# Approximation Methods : cv2.CHAIN_APPROX_NONE - stores all the boundary points.
           # cv2.CHAIN_APPROX_SIMPLE - only start and end points of bounding contours
# HIERARCHY TYPES : THE FIRST TWO ARE THE MOST USEFUL
           # cv2.RETR_LIST - RETRIEVES ALL CONTOURS
           # cv2.RETR_EXTERNAL - RETRIEVES EXTERNAL OR OUTER CONTOURS ONLY
           # cv2.RETR_COMP - RETRIEVES ALL IN A 2-LEVEL HIERARCHY
           # cv2.RETR_TREE - RETRIES ALL IN FULL HIERARCHY
           # STORED IN FORMAT : [NEXT, PREVIOUS, FIRST CHILD, PARENT]
# print(contours)
cv2.imshow('Canny Edges after Contouring', edged)
cv2.waitKey()
           
print("Number of Contours found = " + str(len(contours)))
           
# Draw all contours. Use '-1' as the 3rd parameter to draw all.
           
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
           
cv2.imshow("Contours", image)           
cv2.waitKey()
cv2.destroyAllWindows() 
           
           
##################################################

##### SORTING CONTOURS :
           # SORTING BY AREA : can assist in Object Recognition (using contour area)
               # Eliminate small contours that may be noise. Extract the largest contour.
           # SORTING BY SPATIAL POSITION : (using the contour centroid)
               # Sort characters left to right. Process images in specific order.
           
import cv2
import numpy as np







