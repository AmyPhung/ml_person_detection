#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
import ros_numpy
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hcluster


class PclFeatureDetection:
    def __init__(self):
        rospy.init_node("pcl_feature_detection")
        self.pcl_sub = rospy.Subscriber("/pcl", PointCloud2, self.pclCB)
        self.update_rate = rospy.Rate(10)

        self._pcl_msg = None

    def pclCB(self, msg):
        self._pcl_msg = msg


    def computeClusters(self):

        # Convert to numpy array - each point is a tuple
        data = ros_numpy.numpify(self._pcl_msg)
        # Convert from list of tuples to list of lists
        data_np = np.array([list(pt) for pt in data])
        # Remove points below a certain threshold in z
        # TODO: Use histogram to find ground_thresh (https://hcis-journal.springeropen.com/articles/10.1186/s13673-017-0120-7)
        data_th = data_np[data_np[:,2] > ground_thresh]
        # Downsample points
        print("here")
        ds_factor = 30 # factor to downsample points by TODO: remove hardcode
        data_ds = data_th[::ds_factor]
        # Cluster points in 2D - uses hierarchical clustering (https://stackoverflow.com/questions/10136470/unsupervised-clustering-with-unknown-number-of-clusters)
        thresh = 1
        clusters = hcluster.fclusterdata(data_ds[:,:2], thresh, criterion="distance")
        num_clusters = len(np.unique(clusters))

        plt.clf()
        plt.scatter(*np.transpose(data_ds[:,:2]), c=clusters)
        plt.axis("equal")
        plt.draw()
        plt.pause(0.00000000001)

    def run(self):
        plt.ion()
        plt.show()

        while not rospy.is_shutdown():
            if self._pcl_msg != None: # Only compute for unique messages
                self.computeClusters()
                self._pcl_msg = None
            self.update_rate.sleep()


if __name__ == "__main__":
    feat = PclFeatureDetection()
    feat.run()
