#!/usr/bin/bash

sudo apt-get -y update
sudo apt-get -y upgrade

sudo apt-get -y install python3-pip
sudo apt-get -y install python-is-python3

sudo apt-get -y install nvidia-cuda-toolkit

python --version
pip --version
nvcc --version

pip install --upgrade pip
pip cache purge

pip install numpy
pip install scipy

pip install pandas

pip install matplotlib
pip install seaborn

pip install opencv-python
pip install scikit-learn

pip install tqdm
pip install git-repo
pip install keyboard

pip install torch torchvision torchaudio
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

pip install wandb
pip install pytransform3d

pip install pyrealsense2

pip install open3d-python

pip install PySide6