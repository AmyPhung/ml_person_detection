#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
import ros_numpy
import matplotlib.pyplot as plt
import numpy as np

class PclFeatureDetection:
    def __init__(self):
        rospy.init_node("pcl_feature_detection")
        self.pcl_sub = rospy.Subscriber("/pcl", PointCloud2, self.pclCB)
        self.update_rate = rospy.Rate(10)

        self._pcl_msg = None

    def pclCB(self, msg):
        self._pcl_msg = msg

    def computeClusters(self):
        ground_thresh = 1 # In meters

        # Convert to numpy array - each point is a tuple
        data = ros_numpy.numpify(self._pcl_msg)
        # Convert from list of tuples to list of lists
        data_np = np.array([list(pt) for pt in data])
        # Remove points below a certain threshold in z
        data_th = data_np[data_np[:,2] > ground_thresh]

        # Visualize points
        plt.plot(data_th[:,0], data_th[:,1], '*')  # data_th[:,:2]
        plt.axis("equal")
        plt.draw()
        plt.pause(0.00000000001)

    def run(self):
        plt.ion()
        plt.show()

        while not rospy.is_shutdown():
            if self._pcl_msg != None:
                self.computeClusters()
            self.update_rate.sleep()


if __name__ == "__main__":
    feat = PclFeatureDetection()
    feat.run()
