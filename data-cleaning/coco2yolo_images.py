import pandas
import cv2
from glob import iglob
import os
from shutil import copy


'''
This script will convert multi-class coco-yolo dataset to single class yolo dataset
'''

#source_txt = 'data-cleaning/dataset/crowdhuman-coco-yolo/labels-half-v2/val/'
#source_img = 'data-cleaning/dataset/coco-raw/images/val/'

# make sure destination is empty
#destination = 'data-cleaning/dataset/crowdhuman-coco-yolo/images/val/'


source_txt = 'data-cleaning/dataset/crowdhuman-coco-yolo/labels-half-v2/train/'
source_img = 'data-cleaning/dataset/coco-raw/images/train/'

# make sure destination is empty
destination = 'data-cleaning/dataset/crowdhuman-coco-yolo/images/train/'

count = 0

for filename in sorted(iglob(source_txt + '*.txt')):
    count += 1
    print(count)
    filename_base = os.path.splitext(os.path.basename(filename))[0]
    #print(filename_base)
    path_img = source_img + filename_base + '.jpg'
    #print(path_img)
    try:
        copy(path_img, destination)
    except FileNotFoundError:
        pass

