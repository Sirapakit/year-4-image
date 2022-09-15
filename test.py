import cv2
# python3 -m pip install opencv-python

path = r'image1.jpg'
image2 = cv2.imread(path)
image = cv2.resize(image2, [400, 400], interpolation=cv2.INTER_AREA)
window_name = 'image'

# split image useing matirces
blue_channel = image[:, :, 0]
green_channel = image[:, :, 1]
red_channel = image[:, :, 2]
print(blue_channel[0, 0])
cv2.imshow('blue_channel', blue_channel)
cv2.imshow('green_channel', green_channel)
cv2.imshow('red_channel', red_channel)

# split image using cv2.split command
blue, green, red = cv2.split(image)
# merged image back from above then show it
merged = cv2.merge([blue, green, red])
cv2.imshow('merged', merged)

cv2.waitKey(0)
cv2.destroyAllWindows()
