#!/usr/bin/env python2
import rospy
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage
import ros_numpy
import numpy as np
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import tf2_ros
import tf2_py as tf2

class SimDatasetCreator(object):
    def __init__(self):
        rospy.init_node('sim_dataset_creator')
        self.update_rate = rospy.Rate(10)
        self.pcl_sub = rospy.Subscriber("lidarx_points",
            PointCloud2, self.pclCB)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pcl_np = None
        self.tf_pcl_np = None

        # FOR VERIFICATION PURPOSES ONLY
        self.verification_pub = rospy.Publisher("tf_points_verificaton",
            PointCloud2, queue_size=1)

    def pclCB(self, msg):
        """Transform pointcloud into base_link frame and convert to numpy
        array"""

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
        cloud_out = do_transform_cloud(msg, trans)

        # Convert to numpy array
        pc = ros_numpy.numpify(cloud_out)
        self.pcl_np=np.zeros((pc.shape[0],3))
        self.pcl_np[:,0]=pc['x']
        self.pcl_np[:,1]=pc['y']
        self.pcl_np[:,2]=pc['z']
        print(self.pcl_np.shape)

        # FOR VERIFICATION PURPOSES ONLY
        print(cloud_out.header.frame_id)
        self.verification_pub.publish(cloud_out)

    def run(self):
        while not rospy.is_shutdown():
            self.update_rate.sleep()

if __name__ == "__main__":
    sdc = SimDatasetCreator()
    sdc.run()
