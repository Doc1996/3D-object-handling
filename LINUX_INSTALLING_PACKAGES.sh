#!/usr/bin/bash

sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y install nvidia-cuda-toolkit

python3 --version
pip --version
nvcc --version

python3 -m pip install --upgrade pip
python3 -m pip cache purge

python3 -m pip install numpy
python3 -m pip install scipy

python3 -m pip install pandas

python3 -m pip install matplotlib
python3 -m pip install seaborn

python3 -m pip install opencv-python
python3 -m pip install scikit-learn

python3 -m pip install open3d
python3 -m pip install pytransform3d

python3 -m pip install install torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
python3 -m pip install torch torchvision torchaudio

python3 -m pip install wandb
python3 -m pip install git-repo

python3 -m pip install PySide6
python3 -m pip install pyrealsense2

python3 -m pip install keyboard
python3 -m pip install tqdm