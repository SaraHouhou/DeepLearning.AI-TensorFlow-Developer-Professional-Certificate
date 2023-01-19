# Load Images
# diplayiong an image
# resizing ana image
# rotating an image
# pip install opencv-python or python -m pip install opencv-python

import cv2
import random
# eTAPE 1 lOAD THE IMAGE 
img =cv2.imread('Assets/1.jpg', -1)

# POSSIBLE ARGUMENT"
#  -1 ou cv2.IMREAD_COLOR : load a color image, Any trensparency of image WILL BE NEGLEGATED
#0, cv2.IMREAD_GRAYSCALE: Load image in grayscal
#1, cv2.IMREAD_UNCHANGED/ lOAD IMAGE AS SUCH INCLUDING ALPHA


# resize images
#img= cv2.resize(img, (0,0), fx=0.5, fy=0.5)

# rotate images

# img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# #save this in a new file 

# cv2.imwrite('new_imag.jpg', img )

# # etape 2 : display the image 

# cv2.imshow('Image', img) # label window and the name of the image

# # close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Image Manipulation

# print(type(img))
# print(img.shape) # image width hight and channels 3 (green bleu, red) ( how many pixel in any value rgb), dans le cas img gray nous ne trouvons pas channels
# # IMAGE REPRESENTATION

# #print the first row of an image 

# print(img[0])

# look out to a pixel in particular 

#print(img[257][500])
# changing pixel color

#random.randint (low, hight) generates randow between the min and the max
# for i in range(100): 
#     for j in range(img.shape[1]):
#         img[i][j] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]


# copy a part of an image and past it some where else in tha image
tag=img[300:500, 400:700]
img[0:200, 600:900]=tag

cv2.imshow('Image', img) # label window and the name of the image

# # close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
