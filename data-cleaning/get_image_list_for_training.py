import pandas
import cv2
from glob import iglob
import os

'''
This script will list full path for training set indexing (ls command not working when list too long, need sequential approach)
'''


source = 'data-cleaning/dataset/crowdhuman-coco-yolo/images/train/'
# make sure destination is empty
destination = 'data-cleaning/dataset/crowdhuman-coco-yolo/train.txt'

source = 'data-cleaning/dataset/crowdhuman-coco-yolo/images/val/'
# make sure destination is empty
destination = 'data-cleaning/dataset/crowdhuman-coco-yolo/val.txt'

count = 0
for filename in sorted(iglob(source + '*.jpg')):
    count += 1

    #if count%10 == 0:
    print(count)
    with open(destination, 'a') as file:
        file.write(filename + '\n')

    #break
