@ECHO OFF

python --version
pip --version
nvcc --version

python -m pip install --upgrade pip
python -m pip cache purge

python -m pip install numpy
python -m pip install scipy

python -m pip install pandas

python -m pip install matplotlib
python -m pip install seaborn

python -m pip install opencv-python
python -m pip install scikit-learn

python -m pip install open3d
python -m pip install pytransform3d

python -m pip install torch torchvision torchaudio
python -m pip install torch==1.10.2+cu102 torchvision==0.11.3+cu102 torchaudio===0.10.2+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html

python -m pip install wandb
python -m pip install git-repo

python -m pip install PySide6
python -m pip install pyrealsense2

python -m pip install keyboard
python -m pip install tqdm