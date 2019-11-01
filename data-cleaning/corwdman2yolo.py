import json
import pandas
import cv2

'''
This script will convert Crowdman dataset to Yolo format
produce original label
'''

# source_txt = '/home/dcm-admin/datasets/crowdman-raw/annotation_val.odgt'
source_txt = '/home/dcm-admin/datasets/crowdman-raw/annotation_train.odgt'
# source_img = '/home/dcm-admin/datasets/crowdman-raw/val/'
source_img = '/home/dcm-admin/datasets/crowdman-raw/train/'

# make sure destination is empty
# destination = '/home/dcm-admin/datasets/crowdman-labels/val/'
destination = '/home/dcm-admin/datasets/crowdman-labels/train/'

file = open(source_txt, "r")
counter = 1
for row in file:
    print('processing row no.',counter)
    counter += 1
    row = json.loads(row)
    #print(row)
    img = cv2.imread(source_img + row['ID'] + '.jpg')
    H = img.shape[0]
    W = img.shape[1]
    #print(H, W)
    for i in range (0, len(row['gtboxes'])):
        if row['gtboxes'][i]['tag'] == 'person':

            w = row['gtboxes'][i]['fbox'][2]/W
            h = row['gtboxes'][i]['fbox'][3]/H
            x = row['gtboxes'][i]['fbox'][0]/W
            y = row['gtboxes'][i]['fbox'][1]/H
            #print(w)

            with open(destination + row['ID'] + '.txt', 'a') as file:
                content = '0 ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
                file.write(content)


# generate list of iamge names for train and val under dataset/my_dataset/ before images and labels

