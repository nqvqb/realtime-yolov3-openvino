# REALTIME YOLOV3 TRAINING AND INFERENCE

##### Prerequisis
Install the gpu machine and cpu inference tool by following real-time-inference-machine-installation-guide/README.md

##### Environment
1. OpenVINO l_openvino_toolkit_p_2019.1.133
2. Ubuntu 18.04 x86_64 destop version (gcc)
3. Tensorflow cpu 1.12
4. working directory $HOME
5. opencv4.0.1

## Train on GPU Machine and Inference on CPU
This guide is designed for single-class YOLOv3 architecture training and inference on CPU with OpenVINO. In the future with balanced dataset, multiple classes can be trained to improve inference result

##### Reference
1. https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html
2. https://software.intel.com/en-us/forums/computer-vision/topic/800049
3. https://github.com/PINTO0309/OpenVINO-YoloV3/tree/master/cpp
4. https://qiita.com/PINTO/items/7dd7135085a7249bf17a#support-for-local-training-and-openvino-of-one-class-tiny-yolov3-with-a-proprietary-data-set

##### Install OpenVINO
```sh
# change to your development directory
cd $HOME
# change to your development directory
cd $HOME/realtime-yolov3-openvino
# download from openvino source
# unzip the zip file
tar -zxf l_openvino_toolkit_p_2019.1.133.tgz
# remove the zip file
rm l_openvino_toolkit_p_2019.1.133.tgz
# install openvino
cd $HOME/realtime-yolov3-openvino/l_openvino_toolkit_p_2019.1.133
sudo -E ./install_openvino_dependencies.sh
# install via GUI, if headless server, change to others
sudo ./install_GUI.sh
# after installation, install dependencies
cd /opt/intel/openvino/install_dependencies/
sudo -E ./install_openvino_dependencies.sh
cd
# add a line at the end of bashrc file so the openvino environment will be set each time you launch a terminal
vim $HOME/.bashrc
	source /opt/intel/openvino/bin/setupvars.sh
# source the bashrc file
source $HOME/.bashrc
# expected output: [setupvars.sh] OpenVINO environment initialized
# (optional) terminate current terminal and relaunch terminal
# install model optimiser prerequisites
cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites/
sudo ./install_prerequisites.sh
```

##### Install OPENCV
```sh
sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
# git clone opencv
cd ~/opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
# https://blog.csdn.net/orDream/article/details/84311697

make -j8
sudo make install
```


##### Install and build YOLOv3 using darknet architecture
yolov3 training requires CPU, CUDNN, CUDA, and opencv in this example, but it also can be trained without GPU. Below shows file system structure for training. 
The darknet folder is a direct copy from darknet author. 
Please refer to following folder structure to set up dataset for training; if any doubts, please reach darknet official guide for help.
```text
realtime-yolov3-openvino/
	darknet/
		data*/
			obj.data
			obj.names
			test.txt
			train.txt
			
```
prepare the training:
```sh
cd $HOME/realtime-yolov3-openvino/darknet
# change the YOLOv3 makefile
vim Makefile
	GPU=1
	CUDNN=1
	OPENCV=0
	OPENMP=0
	DEBUG=0
# start building yolov3
make -j8
# build finished
# download network and weights
cd $HOME/realtime-yolov3-openvino/darknet
wget https://pjreddie.com/media/files/darknet53.conv.74
# optional, if you want to fine-tune the original model
wget https://pjreddie.com/media/files/yolov3.weights
# for yolov3-tiny weights
# wget https://pjreddie.com/media/files/yolov3-tiny.weights
# check obj.names file it should only contain single class person
# in the future if need to train the network for multiple classes, google how to change the config to multiple classes
# here only provide single class training
# for multiple classes, size of each class must be simlar or use other augmentation technique
# check if obj.data in following format
	classes = 1
	train  = data/train.txt
	valid  = data/test.txt
	names = data/data.names
	backup = backup/
# change cfg file
# copy original file to your single class config file
cp $HOME/realtime-yolov3-openvino/darknet/cfg/yolov3.cfg $HOME/realtime-yolov3-openvino/darknet/cfg/yolov3_1.cfg
vim $HOME/realtime-yolov3-openvino/darknet/cfg/yolov3_1.cfg
# for yolov3-tiny
# cp $HOME/realtime-yolov3-openvino/darknet/cfg/yolov3-tiny.cfg $HOME/realtime-yolov3-openvino/darknet/cfg/yolov3_tiny_1.cfg
# vim $HOME/realtime-yolov3-openvino/darknet/cfg/yolov3_tiny_1.cfg
# change classes=80 to classes=1
# filters=(classes+5)*3 
# change filters=255 to filters=18
```

##### Run a quick test with self-labelled dataset
After darknet is compiled, we need to quickly verify that a model trained with small dataset would converge before commiting large amount of time in training with large dataset.
The easiest way to do it is by labelling your own dataset using labelling app. This section provide a guide to use VoTT - a labelling tool to verify network training.
```sh
# install VoTT image labelling tool to produce your own dataset
cd $HOME
sudo apt install -y snap
wget https://github.com/Microsoft/VoTT/releases/download/v1.7.1/vott-linux.snap
sudo snap install --dangerous vott-linux.snap
cd $HOME
mkdir $HOME/realtime-yolov3-openvino/vott
# launch the vott app and start labelling
vott
# start labelling data
# export(this is a bug: some times you need to ctrl+E to export) the labelled data from vott to darknet/data_vott
# sample data are located in data_vott can be used to quickly train a network for prove-of-concept
```

##### Start training
start training the network with your small dataset; based on experiment it only needs <1000 epochs for the netowrk to converge.
```sh
cd $HOME/realtime-yolov3-openvino/darknet
./darknet detector train data/obj.data cfg/yolov3_1.cfg
# or train with pre-trained weights 
./darknet detector train data/obj.data cfg/yolov3_1.cfg darknet53.conv.74 
# or
./darknet detector train data_vott/obj.data cfg/yolov3_1.cfg
# weights will be store in ../backup
```
	
##### From darknet weights to tensorflow weights
after traing, convert the darknet structure to tensorflow architecture for later network optimization
```sh
# copy weights
cp $HOME/realtime-yolov3-openvino/darknet/backup/yolov3_1.weights $HOME/realtime-yolov3-openvino/OpenVINO-YoloV3/weights/
cd $HOME/realtime-yolov3-openvino/OpenVINO-YoloV3/
# convert weights
# create virtual environment for tensorflow 1.12
# (once)
virtualenv --system-site-packages -p python3 ~/realtime-yolov3-openvino/venv/tf-model-convert
source ~/realtime-yolov3-openvino/venv/tf-model-convert/bin/activate
# (once)
pip install tensorflow==1.12

python3 convert_weights_pb.py

# modify convert_weights_pb for yolov3-tiny and run
# python3 convert_weights_pb.py --class_names yolov3_tiny_1.names --data_format NHWC --weights_file ../OpenVINO-YoloV3/weights/yolov3_tiny_1.weights --output_graph ../OpenVINO-YoloV3/pbmodels/yolov3_tiny_1.pb --tiny
```

##### Convert model from tensorflow to openvino
after converted to tensorflow architecture, pass the model to openvino optimizer
```sh

sudo python3 /opt/intel/openvino_2019.1.133/deployment_tools/model_optimizer/mo_tf.py \
	--input_model $HOME/realtime-yolov3-openvino/OpenVINO-YoloV3/pbmodels/yolov3_1.pb \
	--output_dir $HOME/realtime-yolov3-openvino/OpenVINO-YoloV3/lrmodels/YoloV3/FP32 \
	--data_type FP32 \
	--batch 1 \
	--tensorflow_use_custom_operations_config $HOME/realtime-yolov3-openvino/OpenVINO-YoloV3/yolo_v3_changed_1.json

# (optional)
sudo python3 /opt/intel/openvino_2019.1.133/deployment_tools/model_optimizer/mo_tf.py \
	--input_model $HOME/realtime-yolov3-openvino/OpenVINO-YoloV3/pbmodels/yolov3_1.pb \
	--output_dir $HOME/realtime-yolov3-openvino/OpenVINO-YoloV3/lrmodels/YoloV3/FP32 \
	--tensorflow_use_custom_operations_config $HOME/realtime-yolov3-openvino/OpenVINO-YoloV3/yolo_v3_changed_1.json \
	--input_shape [1,416,416,3]
```

##### Recompile the main cpp using openvino each time you change the cpp file
cpp main file need to be recompiled by openvino for each specific machine
```sh
cd $HOME/realtime-yolov3-openvino/OpenVINO-YoloV3/cpp
sudo cp main.cpp /opt/intel/openvino/deployment_tools/inference_engine/samples/object_detection_demo_yolov3_async
sudo cp object_detection_demo_yolov3_async.hpp /opt/intel/openvino/deployment_tools/inference_engine/samples/object_detection_demo_yolov3_async
cd /opt/intel/openvino/deployment_tools/inference_engine/samples
sudo ./build_samples.sh
# expected output: Build completed, you can find binaries for all samples in the $HOME/inference_engine_samples_build/intel64/Release subfolder.
cd $HOME/realtime-yolov3-openvino/OpenVINO-YoloV3/cpp
cp $HOME/inference_engine_samples_build/intel64/Release/object_detection_demo_yolov3_async .
```
##### Inference with webcam
```sh
cd $HOME/realtime-yolov3-openvino/OpenVINO-YoloV3/cpp
# inference from webcam
./object_detection_demo_yolov3_async -i cam0 -m ../lrmodels/YoloV3/FP32/yolov3_1.xml -d CPU
# inference from video
./object_detection_demo_yolov3_async -m ../lrmodels/YoloV3/FP32/yolov3_1.xml -d CPU -i path/to/video.mp4
./object_detection_demo_yolov3_async -m ../lrmodels/YoloV3/FP32/yolov3_1.xml -d CPU -i ~/video/1.mp4
# or perform on movie file
# ./object_detection_demo_yolov3_async -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/frozen_yolo_v3.xml -l ../lib/libcpu_extension.so -d CPU
# or 
```

##### Prepare large-scale training dataset
replace vott data with large scal dataset and repeat the training and inference processes
```sh
# use customised dataset, copy train and test dataset files
# the train.txt and test.txt contain all the absolute (preferred)/relativ filenames of training and testing image
cp your_yolo_training_image_names.txt $HOME/realtime-yolov3-openvino/darknet/data/train.txt
cp your_yolo_testing_image_names.txt $HOME/realtime-yolov3-openvino/darknet/data/test.txt
```



##### sample pedestrian head detection
![](sample_head_detection.gif)

	

































