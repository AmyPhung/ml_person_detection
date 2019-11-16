# ml-person-detection

### Setup
1. `cd your_catkin_ws/src`
2. `git clone https://github.com/waymo-research/waymo-open-dataset.git`
3. `cd waymo-od && git branch -a`
4. `git checkout remotes/origin/r1.0`
5. `pip install waymo-open-dataset`
6. `pip install tensorflow`
7. `cd your_catkin_ws/src`
8. `git clone https://github.com/AmyPhung/ml_person_detection`
9. `catkin_make`


### Dependencies:
Package for converting from numpy to PointCloud2
https://github.com/eric-wieser/ros_numpy
`pip install --user .`

7. `cd your_catkin_ws/`
8. `catkin_make`

### Usage
`roslaunch ml_person_detection bringup.launch`
Note: make sure FILENAME in waymo2ros.py is set properly

### Troubleshooting
##### Tensorflow not installing properly
+ pip install --user tensorflow==1.15.0
"launchpadlib 1.10.6 requires testresources, which is not installed. tensorflow"
sudo apt install python-testresources
pip uninstall numpy
pip install numpy
<!-- For python3: `pip3 install --user tensorflow==1.14.0`
For python2 (use with ROS): `python -m pip install --user --ignore-installed tensorflow` -->
