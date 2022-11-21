#import libraries
from PIL import Image
import numpy as np
import cv2
import math

#read image
path = r'/Users/sirap/Documents/Year_4/year-4-image/shiba.jpeg'
img = cv2.imread(path, 0)

#blur the image using gaussian blurc
gaussian = cv2.GaussianBlur(img, (7, 7),0)

#store images as arrays
img=np.asarray(img)
gaussian=np.asarray(gaussian)

#subtract blurred image from original, then add to original
mask =  img - gaussian 
img_unsharp = img + ( mask * 0.7 )

#output unsharp image
cv2.imshow('Window', img_unsharp)
cv2.imwrite('unsharp-shiba.png', img_unsharp)
cv2.waitKey(0)

#convert array to image
img_unsharp=Image.fromarray(img_unsharp)
img_unsharp