# DATA ClEANING
This is a guide of how to handle raw, labelled data downloaded from online sources, and convert it to desired format (YoloV3)
## Setup for YoloV3
##### Environment
```shell script
virtualenv --system-site-packages -p python3 ~/realtime-yolov3-openvino/venv/data-cleaning
source ~/realtime-yolov3-openvino/venv/data-cleaning/bin/activate
pip install pandas
pip install opencv-python
``` 
##### print all file name full path for images folder
	ls -d $PWD/* > ../train.txt
	ls -d $PWD/* > ../val.txt
##### 
