# ml-person-detection

### Setup
1. `cd your_catkin_ws/src`
2. `git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od`
3. `cd waymo-od && git branch -a`
4. `git checkout remotes/origin/r1.0`
5. `pip install waymo-open-dataset`
6. `pip install tensorflow`


### Dependencies:
https://github.com/eric-wieser/ros_numpy
`pip install --user .`

### Usage
`roslaunch ml_person_detection bringup.launch`
Note: make sure FILENAME in waymo2ros.py is set properly

### Troubleshooting
##### Tensorflow not installing properly
+ pip install --user tensorflow==1.15.0
<!-- For python3: `pip3 install --user tensorflow==1.14.0`
For python2 (use with ROS): `python -m pip install --user --ignore-installed tensorflow` -->
