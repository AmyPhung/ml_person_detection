#!/usr/bin/env python2
import rospy
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage
import ros_numpy
import numpy as np
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import tf2_ros
import tf2_py as tf2

from modules.helperFunctions import *
# from models.helperFunctions import *
# asdffasdfasd

class SimDatasetCreator(object):
    def __init__(self):
        rospy.init_node('sim_dataset_creator')
        self.update_rate = rospy.Rate(10)
        self.pcl_sub = rospy.Subscriber("lidarx_points",
            PointCloud2, self.pclCB)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.tf_pcl_msg = None # Pointcloud message in base_link frame
        self.pcl_np_filtered = None # Pointcloud in numpy form with groundplane removed
        self.clusters = None # List containing clusters in Pointcloud
        self.pcl_np = None
        self.tf_pcl_np = None

        # For Visualization
        self.verification_pub = rospy.Publisher("tf_points_verificaton",
            PointCloud2, queue_size=1)
        self.visualize = rospy.get_param('/visualize', 0)


    def pclCB(self, msg):
        """Transform pointcloud into base_link frame and save to object"""

        try:
            trans = self.tf_buffer.lookup_transform("base_link", msg.header.frame_id,
                                               msg.header.stamp,
                                               rospy.Duration(4.0))
        except tf2.LookupException as ex:
            rospy.logwarn(ex)
            return
        except tf2.ExtrapolationException as ex:
            rospy.logwarn(ex)
            return
        self.tf_pcl_msg = do_transform_cloud(msg, trans)

    def parsePCL(self):

        # Convert to numpy array
        pc = ros_numpy.numpify(self.tf_pcl_msg)
        self.pcl_np=np.zeros((pc.shape[0],4))
        self.pcl_np[:,0]=pc['x']
        self.pcl_np[:,1]=pc['y']
        self.pcl_np[:,2]=pc['z']
        self.pcl_np[:,3]=pc['intensity']
        print(self.pcl_np.shape)

        # Remove groundplane from data
        self.pcl_np_filtered = remove_groundplane(self.pcl_np)

        # Cluster data
        self.clusters = compute_clusters(self.pcl_np_filtered, thresh=0.001)


    def visualizeOutput(self):
        """Publishes data to visualize different stages of the dataset based
        on the ROS param /visualize

        Visualization Parameters:
            0 = unfiltered, relative to base_link
            1 = ground plane removed
        """

        self.visualize = int(
            rospy.get_param("/visualize", self.visualize))

        if self.visualize == 0: # Publish unfiltered data in base_link frame
            self.verification_pub.publish(self.tf_pcl_msg)

        if self.visualize == 1: # Publish ground-filtered data in base_link frame
            self.verification_pub.publish(self.np2msg(self.pcl_np_filtered))


    def np2msg(self, points):
        """Convert numpy array of points into ros PointCloud2 msg.

        Args:
            points: numpy (x * 3) array of xyz points or numpy (n * 4) array of
                xyz points and intensities

        Returns:
            ROS PointCloud2 msg with points and frame_id
        """

        data = np.zeros(points.shape[0], dtype=[
          ('x', np.float32),
          ('y', np.float32),
          ('z', np.float32),
        ])
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]

        return ros_numpy.msgify(
            PointCloud2, data, stamp=None, frame_id='base_link')


    def run(self):
        while not rospy.is_shutdown():
            self.visualize = int(
                rospy.get_param("/visualize", self.visualize))

            # Wait for valid pointcloud message
            if self.tf_pcl_msg == None:
                self.update_rate.sleep()
                continue

            self.parsePCL()
            self.visualizeOutput()
            self.update_rate.sleep()

if __name__ == "__main__":
    sdc = SimDatasetCreator()
    sdc.run()
