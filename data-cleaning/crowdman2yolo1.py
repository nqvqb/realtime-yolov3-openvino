import json
import pandas
import cv2

'''
This script will convert Crowdman dataset to Yolo format
'''

#source_txt = '/media/n0v0b/m1/dataset/crowdman-raw/annotation_val.odgt'
source_txt = '/media/n0v0b/m1/dataset/crowdman-raw/annotation_train.odgt'
#source_img = '/media/n0v0b/m1/dataset/crowdman-raw/val/'
source_img = '/media/n0v0b/m1/dataset/crowdman-raw/train/'

# make sure destination is empty
#destination = '/media/n0v0b/m1/dataset/crowdman-labels/val/'
destination = '/media/n0v0b/m1/dataset/crowdman-labels/train/'

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

            w = row['gtboxes'][i]['hbox'][2]/W * 1.5
            h = row['gtboxes'][i]['hbox'][3]/H * 1.5
            x = row['gtboxes'][i]['hbox'][0]/W + w/2
            y = row['gtboxes'][i]['hbox'][1]/H + h/2
            #print(w)

            with open(destination + row['ID'] + '.txt', 'a') as file:
                content = '0 ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
                file.write(content)


# generate list of iamge names for train and val under dataset/my_dataset/ before images and labels

