#!/usr/bin/env python

import tensorflow as tf
import rospy as rp
from modules.helperFunctions import *
from modules.waymo2ros import Waymo2Numpy, Waymo2Ros
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray

class DatasetIntrospectorVis(object):
    """Class for introspecting a dataset."""

    def __init__(self, dir_data, logger=None, verbosity=None):

        # Initialize main attributes
        self.dir_data = dir_data
        self.waymo_converter = Waymo2Numpy()

        # Initialize ROS attributes
        rp.init_node('dataset_introspector_vis')
        self.ros_converter = Waymo2Ros()
        self.marker_pub = rp.Publisher('/bboxes', MarkerArray, queue_size=1)
        self.pcl_pub = rp.Publisher('/pcl', PointCloud2, queue_size=1)
        self.visualize = rp.get_param('/visualize', 1)

    def visualize_cluster(self, tfrecord_id, frame_ndx, bbox_id):
        """Shows single cluster in Rviz given fully defined bbox id.

        Args:
            tfrecord_id: id of tfrecord containing cluster.
            frame_ndx: index of frame containing cluster.
            cluster_id: id of bbox for which to visualize cluster.

        """
        print('Search criteria:\ntfrecord: %s\nframe index: %i\nbbox id: %s'\
            % (tfrecord_id, frame_ndx, bbox_id))
        # Get specified tfrecord
        filename = '%s/segment-%s_with_camera_labels.tfrecord'\
            % (self.dir_data, tfrecord_id)
        tfrecord = tf.data.TFRecordDataset(filename, compression_type='')

        # Get specified frame from tfrecord
        scan = next(f for i,f in enumerate(tfrecord) if i==frame_ndx)
        frame = self.waymo_converter.create_frame(scan)

        # Get specified bbox from frame
        pcl, bboxes = self.waymo_converter.unpack_frame(frame)
        if self.visualize == 1:
            self.pcl_pub.publish(
                self.ros_converter.convert2pcl(pcl))
            self.marker_pub.publish(
                self.ros_converter.convert2markerarray(bboxes))

        pcl = remove_groundplane(np.array([list(pt) for pt in pcl]))
        if self.visualize == 2:
            self.pcl_pub.publish(
                self.ros_converter.convert2pcl(pcl))
            self.marker_pub.publish(
                self.ros_converter.convert2markerarray(bboxes))
        print(len(pcl))
        bbox = next(b for i, b in enumerate(bboxes) if b.id==bbox_id)
        
        # Calculate cluster for specified bbox
        cluster = get_pts_in_bbox(pcl, bbox)
        print("bbox class=%i, id=%s, pt_count=%i"
            % (bbox.type, bbox.id, len(cluster)))

        # Publish cluster and bbox to rviz
        if self.visualize == 3 and cluster is not None:
            self.pcl_pub.publish(self.ros_converter.convert2pcl(cluster))
            self.marker_pub.publish(self.ros_converter.convert2markerarray(
                [bbox]))

    def play_cluster(self, tfrecord, frame, id):
        """Shows every instance of cluster given fully defined cluster id."""
        pass

if __name__ == '__main__':

    dataset = 'validation_0000'
    dir_data = '/home/cnovak/Data/waymo-od/%s' % dataset

    introspector = DatasetIntrospectorVis(dir_data=dir_data)
    introspector.visualize_cluster(
        tfrecord_id='4816728784073043251_5273_410_5293_410',
        frame_ndx=9,
        bbox_id='71FpMY77d23_mPWjWUOt7Q')
