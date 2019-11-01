# adapted from https://github.com/mitmul/caltech-pedestrian-dataset-converter/blob/master/scripts/convert_seqs.py

import os
import glob
import cv2 as cv


def save_img(dname, fn, i, frame):
	cv.imwrite('{}/{}_{}_{}.jpg'.format(out_dir, os.path.basename(dname),os.path.basename(fn).split('.')[0], i), frame)

out_dir = 'images'
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

def convert(dir):
	for dname in sorted(glob.glob(dir)):
		print(dname)
		for fn in sorted(glob.glob('{}/*.seq'.format(dname))):
			cap = cv.VideoCapture(fn)
			i = 0
			while True:
				ret, frame = cap.read()
				if not ret:
					break
				save_img(dname, fn, i, frame)
				i += 1
			print(fn)

convert('892f17c8-81f5-42a9-ab17-fcbd7ea9476b/caltech/train/*')
