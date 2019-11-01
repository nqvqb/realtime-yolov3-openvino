# REAL TIME INFERENCE MACHINE INSTALLATION GUIDE
The purpose of this installation guide is to provide a comperhensive guide to install different type of linux machine with the capability of runnning various kind of CNN packages
## OpenVINO compatable GPU Training and Inference Machine Installation Guide
In this section a Ubuntu 16.04 machine with OpenVINO and TensorFlow support will be installed
##### Notes
This installation guide only provide installation on native environment without multiple version support, all the versions of softwares are fozen
##### Reference
https://medium.com/@zhanwenchen/install-cuda-9-2-and-cudnn-7-1-for-tensorflow-pytorch-gpu-on-ubuntu-16-04-1822ab4b2421
##### Environment
1. Ubuntu 16.04 LTS x86_64
2. TensorFlow GPU 1.12
3. PyTorch GPU latest version
4. CUDA 9.0
5. CUDNN 7.2.1
6. Nvidia Linux Graphic Driver 396.54
7. OpenCV 4.1.0-openvino (optional, don't know whether is needed or not)
8. OpenCV 3.x (for YoLoV3)

##### (TODO) prepare Ubuntu 16.04 installation USB driver
##### Nvidia Driver Installation
```sh
# update and upgrade the system
sudo apt update
sudo apt upgrade
sudo apt dist-upgrade
# generate ssh key
ssh-keygen
# attatch your public key to code repos
cat .ssh/id_ras.pub
# create dev directory
cd
mkdir DCM
cd DCM
mkdir src
cd src
git clone git@bitbucket.org:ruoxiangDCM/dcm-real-time-cnn.git
# install required packages
sudo apt install vim
sudo apt install git	
# purge nvidia driver if exist
sudo apt-get remove --purge nvidia-*
# download nvidia driver NVIDIA-Linux-x86_64-396.54.run or copy it from local harddrive
# copy the run file to a easy-to-find localtion because ui will be disabled later
cp path/to/NVIDIA-Linux-x86_64-396.54.run $HOME/
# create the file to disable noveau from starting after boot
sudo vim /etc/modprobe.d/blacklist-nouveau.conf
	blacklist nouveau
	options nouveau modeset=0
# ctrl+alt+F1 to enter comamnd line terminal without x server
ctrl+alt+F1
# now you are in headless terminal
# login from the command line terminal use machine user name and password
# stop x server (optional)
sudo systemctl stop lightdm.service

# regenerate the kernal initramfs
sudo update-initramfs -u
# stop services	
sudo systemctl disable lightdm.service
sudo systemctl /etc/init.d/gdm stop
# reboot the system
sudo reboot
# after reboot the display server will be disabled
# you can not see any graphic login screen
# so you need to press ctrl+alt+F1 to enter command line login terminal
ctrl+alt+F1
# login from the command line terminal
# stop x server (optional)
sudo systemctl stop lightdm.service
# install nvidia driver via nvidia run file
cd
chmod +x NVIDIA-Linux-x86_64-396.54.run
sudo ./NVIDIA-Linux-x86_64-396.54.run
# say yes most of the time according to prompts
# enable lightdm service
sudo systemctl start lightdm.service
# reboot after installation
sudo reboot
# (optional) after reboot if can not login graphic terminal
sudo systemctl start lightdm.service
# post installation check
sudo lshw -c video | grep 'configuration'
nvidia-msi
# nvidia-msi doesn't work but the installation is still valid
ls /etc/systemd/system/display-manager.service
# ls /etc/systemd/system/display-manager.service doesn't work but the installation is still valid
# not the driver has been installed successfully
# TODO solve the problem that lightdm.service doesn't start at boot
```

##### CUDA9.0 And CUDNN7.2 Installation
```sh
# download and install cuda 9.0 run file from the link or copy from local harddrive
https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu
sudo reboot
# expose the environment to .bashrc file
vim $HOME/.bashrc
	export PATH=/usr/local/cuda-9.0/bin:$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
# download and install cudnn 7.2 in order from the link
# cuDNN v7.2.1 Runtime Library for Ubuntu16.04 (Deb): https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.2.1/prod/9.0_20180806/Ubuntu16_04-x64/libcudnn7_7.2.1.38-1_cuda9.0_amd64
# cuDNN v7.2.1 Developer Library for Ubuntu16.04 (Deb): https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.2.1/prod/9.0_20180806/Ubuntu16_04-x64/libcudnn7-dev_7.2.1.38-1_cuda9.0_amd64
# cuDNN v7.2.1 Code Samples and User Guide for Ubuntu16.04 (Deb): https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.2.1/prod/9.0_20180806/Ubuntu16_04-x64/libcudnn7-doc_7.2.1.38-1_cuda9.0_amd64
sudo dpkg -i libcudnn7_7.2.1.38–1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.2.1.38–1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-doc_7.2.1.38–1+cuda9.0_amd64.deb
# verify the installation
cp -r /usr/src/cudnn_samples_v7/ $HOME
cd $HOME/cudnn_samples_v7/mnistCUDNN
make clean && make
./mnistCUDNN
```

##### Install python packages
```sh
sudo apt install python-pip
sudo apt install python3-pip
# installl and test tensorflow gpu 1.12.0
sudo pip3 install -U virtualenv
virtualenv --system-site-packages -p python3 ~/DCM/venv/tensorflow
source ~/DCM/venv/tensorflow/bin/activate
# because inside python3 virtual environment, pip and python will be come pip3 and python3
pip install tensorflow-gpu==1.12.0
python
# in python terminal
# import tensorflow gpu module
from tensorflow.python.client import device_lib
# check devices
device_lib.list_local_devices()
# exit python terminal
exit()
# finished testing and deactivate the virtual environment
deactivate

# install and test pytorch
virtualenv --system-site-packages -p python3 ~/DCM/venv/pytorch
source ~/DCM/venv/pytorch/bin/activate
pip3 install torch torchvision
python
import torch
torch.cuda.get_device_name(0)
exit()

# install and test tensorflow-cpu in virtual environment
virtualenv --system-site-packages -p python3 ~/DCM/venv/tf-yolov3-openvino
source ~/DCM/venv/tf-yolov3-openvino/bin/activate
pip install tensorflow==1.12.0
# if install in native environment
# pip install --user tensorflow==1.12.0
python
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
exit()
deactivate
# for openvino tensorflow cpu version may need to be installed in native environment
# this gpu machine will not be used as an production inference machine, only for training and validation
```

##### (optional) Install OpenCV 4.1.0-openvino packages
```sh
# clone opencv and opencv contrib repo from official git repo
git clone opencv...
git clone opencv_contrib...
# checkout the 4.1.0 branch
cd $HOME/opencv
git checkout -b 4.1.0-openvino
cd #HOME/opencv_contrib
git checkout -b 4.1.0
# copy the repos to /opt
sudo cp -r opencv /opt
sudo cp -r opencv_contrib /opt
# install dependencies
sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python3.5-dev python3-numpy libtbb2 libtbb-dev
sudo apt-get install libjpeg-dev libpng-dev libtiff5-dev libjasper-dev libdc1394-22-dev libeigen3-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev sphinx-common libtbb-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libavutil-dev libavfilter-dev libavresample-dev
# compile
cd /opt/opencv
mkdir release
cd /opt/opencv/release
sudo cmake -D BUILD_TIFF=ON \
	-D WITH_CUDA=OFF \
	-D ENABLE_AVX=OFF \
	-D WITH_OPENGL=OFF \
	-D WITH_OPENCL=OFF \
	-D WITH_IPP=OFF \
	-D WITH_TBB=ON \
	-D BUILD_TBB=ON \
	-D WITH_EIGEN=OFF \
	-D WITH_V4L=OFF \
	-D WITH_VTK=OFF \
	-D BUILD_TESTS=OFF \
	-D BUILD_PERF_TESTS=OFF \
	-D OPENCV_GENERATE_PKGCONFIG=ON \
	-D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
	/opt/opencv/
# make with 4 jobs in parallel
sudo make -j4
# install opencv
sudo make install
sudo ldconfig
# verify if opencv4 is installed successfully
pkg-config --cflags opencv4
pkg-config --modversion opencv4
python3
import cv2 as cv
print(cv.__version__)
```
##### Install OpenCV 3.x packages
```sh
# https://docs.opencv.org/3.4.1/d2/de6/tutorial_py_setup_in_ubuntu.html
# the following command will probably install opencv4 instead of 3 if get updated in the future
sudo apt install libopencv-dev
sudo apt install python-opencv
# verify if opencv3 is installed successfully or not
pkg-config --cflags opencv
pkg-config --modversion opencv

```


	
	
