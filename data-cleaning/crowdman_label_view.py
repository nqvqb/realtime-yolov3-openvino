import json
import pandas
import cv2
import os, random

'''
This script will randomly view one of the image from crowdman dataset with yolo-style label
'''

source_img = '/media/n0v0b/m1/dataset/crowdman-raw/val/'
source_label = '/media/n0v0b/m1/dataset/crowdman-labels/val/'

image_name = random.choice(os.listdir(source_img)) #change dir name to whatever
image_name_full = source_img + image_name
identifier = image_name[:-4]

print(image_name)
print(image_name_full)
print(identifier)

label_name = identifier + '.txt'
label_name_full = source_label + identifier + '.txt'
print(label_name_full)

file = open(label_name_full, "r")

img = cv2.imread(image_name_full)
# row(Y), col (X)
Y = img.shape[0]
X = img.shape[1]
print(Y,X)



for row in file:
    item = row.split()
    # <x> <y> <width> <height> - float values relative to width and height of image, it can be equal from (0.0 to 1.0]
    # for example: <x> = <absolute_x> / <image_width> or <height> = <absolute_height> / <image_height>
    print(float(item[2]))

    x1 = int(float(item[1]) * X) - int(float(item[3]) * X/2)
    y1 = int(float(item[2]) * Y) - int(float(item[4]) * X/2)
    x2 = int(float(item[1]) * X) + int(float(item[3]) * X/2)
    y2 = int(float(item[2]) * Y) + int(float(item[4]) * Y/2)

    print(x1)
    print(y1)
    print(x2)
    print(y2)

    img[y1:y2, x1:x2] = 255

cv2.imshow('image',img)
cv2.waitKey(0)
