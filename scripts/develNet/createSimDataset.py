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
        self.pcl_np = None
        self.tf_pcl_np = None

        # For Visualization
        self.verification_pub = rospy.Publisher("tf_points_verificaton",
            PointCloud2, queue_size=1)
        # Visualization Parameters:
        # 0 = unfiltered, relative to base_link
        # 1 = ground plane removed
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

        # # Convert to numpy array
        # pc = ros_numpy.numpify(self.tf_pcl_msg)
        # self.pcl_np=np.zeros((pc.shape[0],3))
        # self.pcl_np[:,0]=pc['x']
        # self.pcl_np[:,1]=pc['y']
        # self.pcl_np[:,2]=pc['z']
        # print(self.pcl_np.shape)
        #
        # # Remove groundplane from data
        # pcl_out = remove_groundplane(np.array([list(pt) for pt in pcl]))

        self.visualize = int(
            rospy.get_param("/visualize", self.visualize))

        if self.visualize == 0: # Publish unfiltered data in base_link frame
            self.verification_pub.publish(self.tf_pcl_msg)

    def run(self):
        while not rospy.is_shutdown():
            self.visualize = int(
                rospy.get_param("/visualize", self.visualize))

            # Wait for valid pointcloud message
            if self.tf_pcl_msg == None:
                self.update_rate.sleep()
                continue

            self.parsePCL()
            self.update_rate.sleep()

if __name__ == "__main__":
    sdc = SimDatasetCreator()
    sdc.run()
