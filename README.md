# 3D Object Handling

<br>

<p align="justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This project integrates a system for intelligent robotic handling of food products, based on collecting their images and point clouds, detection in images by a neural network, determining spatial position and planning and realizing grasping based on geometry. The images and point clouds used are collected by an Intel RealSense D435 3D vision system mounted on a Universal Robots UR5 collaborative robot, with an eye-in-hand calibration of the 3D vision system against his gripper. The YOLO v5 neural network, trained on a specially designed set of annotated images is used to detect the product in images, while the Open3D library for point cloud processing is used to determine the spatial position of the product. The Principal component analysis (PCA) algorithm is used as the basis for planning of product grasping. The listed elements of the system are connected by a graphical user interface, while the communication between the robot and computer is realized by the TCP/IP protocol.</p>

<br>


## Project workflow

<br>

<b>Step 1.</b>&nbsp;&nbsp;Creating the dataset
<br>
<p align="center"><img src="https://raw.githubusercontent.com/Doc1996/3D-object-handling/master/MISCELLANEOUS/images%20for%20GitHub/creating%20dataset.jpg" width="420px"></p>
<br>

<b>Step 2.</b>&nbsp;&nbsp;Labeling the dataset
<br>
<p align="center"><img src="https://raw.githubusercontent.com/Doc1996/3D-object-handling/master/MISCELLANEOUS/images%20for%20GitHub/labeling%20dataset.png" width="420px"></p>
<br>

<b>Step 3.</b>&nbsp;&nbsp;Training the YOLO v5 neural network on labeled dataset
<br>
<p align="center"><img src="https://raw.githubusercontent.com/Doc1996/3D-object-handling/master/MISCELLANEOUS/images%20for%20GitHub/training%20neural%20network.png" width="420px"></p>
<p align="center"><img src="https://raw.githubusercontent.com/Doc1996/3D-object-handling/master/MISCELLANEOUS/images%20for%20GitHub/detection%20with%20neural%20network.jpg" width="420px"></p>
<br>

<b>Step 4.</b>&nbsp;&nbsp;Retrieving the pointcloud of scene from 3D camera and robot
<br>
<p align="center"><img src="https://raw.githubusercontent.com/Doc1996/3D-object-handling/master/MISCELLANEOUS/images%20for%20GitHub/retrieved%20pointcloud.png" width="420px"></p>
<br>

<b>Step 5.</b>&nbsp;&nbsp;Filtering the pointcloud of scene
<br>
<p align="center"><img src="https://raw.githubusercontent.com/Doc1996/3D-object-handling/master/MISCELLANEOUS/images%20for%20GitHub/filtered%20pointcloud.png" width="420px"></p>
<br>

<b>Step 6.</b>&nbsp;&nbsp;Isolating the pointcloud of object
<br>
<p align="center"><img src="https://raw.githubusercontent.com/Doc1996/3D-object-handling/master/MISCELLANEOUS/images%20for%20GitHub/isolated%20pointcloud.png" width="240px"></p>
<br>

<b>Step 7.</b>&nbsp;&nbsp;Defining the positions for object handling
<br>
<p align="center"><img src="https://raw.githubusercontent.com/Doc1996/3D-object-handling/master/MISCELLANEOUS/images%20for%20GitHub/positions%20for%20handling.png" width="360px"></p>
<br>

<b>Step 8.</b>&nbsp;&nbsp;Visualizing the object handling
<br>
<p align="center"><img src="https://raw.githubusercontent.com/Doc1996/3D-object-handling/master/MISCELLANEOUS/images%20for%20GitHub/visualized%20handling.png" width="420px"></p>
<br>

<b>Step 9.</b>&nbsp;&nbsp;Handling the object with robot
<br>
<p align="center"><img src="https://raw.githubusercontent.com/Doc1996/3D-object-handling/master/MISCELLANEOUS/images%20for%20GitHub/handling%20with%20robot.jpg" width="420px"></p>
<br>


## Run the project on Windows

<br>

<b>Step 1.</b>&nbsp;&nbsp;Clone the repository:
<pre>
cd %HOMEPATH%

git clone https://github.com/Doc1996/3D-object-handling
</pre>
<br>

<b>Step 2.</b>&nbsp;&nbsp;Create the virtual environment and install dependencies:
<pre>
cd %HOMEPATH%\3D-object-handling

python -m pip install --upgrade pip
python -m pip install --user virtualenv

python -m venv python-virtual-environment
.\python-virtual-environment\Scripts\activate

python -m pip install ipykernel
python -m ipykernel install --user --name=3D-object-handling

.\WINDOWS_INSTALLING_PACKAGES.bat
</pre>
<br>

<b>Step 3.</b>&nbsp;&nbsp;Modify the changeable variables in <i>RS_and_3D_OD_constants.py</i>
<br>
<br>

<b>Step 4.</b>&nbsp;&nbsp;Run the program:
<pre>
cd %HOMEPATH%\3D-object-handling

.\python-virtual-environment\Scripts\activate

.\WINDOWS_3D_OBJECT_HANDLING_APPLICATION.bat
</pre>
<br>

<b>Optional step</b>&nbsp;&nbsp;Run the prototyping program with Jupyter Notebook:
<pre>
cd %HOMEPATH%\3D-object-handling

.\python-virtual-environment\Scripts\activate

jupyter notebook RS_and_3D_OD_prototypes.ipynb
:: set the virtual environment kernel: "Kernel" -> "Change kernel" -> "3D-object-handling"
:: run cells one after another
</pre>
<br>


## Run the project on Linux

<br>

<b>Step 1.</b>&nbsp;&nbsp;Clone the repository:
<pre>
cd $HOME

git clone https://github.com/Doc1996/3D-object-handling
</pre>
<br>

<b>Step 2.</b>&nbsp;&nbsp;Create the virtual environment and install dependencies:
<pre>
cd $HOME/3D-object-handling

python3 -m pip install --upgrade pip
python3 -m pip install --user virtualenv

python3 -m venv python-virtual-environment
source python-virtual-environment/bin/activate

python3 -m pip install ipykernel
python3 -m ipykernel install --user --name=3D-object-handling

source LINUX_INSTALLING_PACKAGES.sh
</pre>
<br>

<b>Step 3.</b>&nbsp;&nbsp;Modify the changeable variables in <i>RS_and_3D_OD_constants.py</i>
<br>
<br>

<b>Step 4.</b>&nbsp;&nbsp;Run the program:
<pre>
cd $HOME/3D-object-handling

source python-virtual-environment/bin/activate

source LINUX_3D_OBJECT_HANDLING_APPLICATION.sh
</pre>
<br>

<b>Optional step</b>&nbsp;&nbsp;Run the prototyping program with Jupyter Notebook:
<pre>
cd $HOME/3D-object-handling

source python-virtual-environment/bin/activate

jupyter notebook RS_and_3D_OD_prototypes.ipynb
# set the virtual environment kernel: "Kernel" -> "Change kernel" -> "3D-object-handling"
# run cells one after another
</pre>