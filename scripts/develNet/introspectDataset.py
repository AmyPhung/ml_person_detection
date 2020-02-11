#!/usr/bin/env python


import pdb
import tensorflow as tf
import rospy as rp
from modules.helperFunctions import *
from modules.waymo2ros import Waymo2Numpy, Waymo2Ros
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray

class FrameError(Exception):
    """Error to raise when a given frame causes an error.

    Note:
        Frame doesn't exist in tfrecord
        Frame does not contain cluster specified

    """
    pass


class DatasetIntrospectorVis(object):
    """Class for introspecting a dataset.

    Note: int rosparam /visualize changes display option as follows:
        1. Display all bounding boxes & full pcl
        2. Display all bounding boxes & groundplane removed
        3. Display single bounding box and cluster points
        4. Display single bounding box and full pcl"""

    def __init__(self, dir_data):
        """Initialize main attributes and set up Ros node."""
        self.dir_data = dir_data
        self.waymo_converter = Waymo2Numpy()

        # Initialize ROS attributes
        rp.init_node('dataset_introspector_vis')
        self.rate_dur = rp.get_param('/framerate', 1)
        self.rate = rp.Rate(self.rate_dur)
        self.visualize = rp.get_param('/visualize', 3)
        self.ros_converter = Waymo2Ros()
        self.marker_pub = rp.Publisher('/bboxes', MarkerArray, queue_size=1)
        self.pcl_pub = rp.Publisher('/pcl', PointCloud2, queue_size=1)
        self.visualize = rp.get_param('/visualize', 4)

    def update_rosparams(self):
        """Update rosparam attributes of class from parameter server."""
        self.visualize = rp.get_param('/visualize', self.visualize)
        self.rate_dur = rp.get_param('/framerate', self.rate_dur)
        self.rate = rp.Rate(self.rate_dur)

    def show_cluster(self, tfrecord_id, frame_ndx, bbox_id):
        """Show single cluster in rviz given fully defined bbox id.

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
        frame_num = sum(1 for _ in tf.python_io.tf_record_iterator(filename))
        if frame_ndx == frame_num:
            raise FrameError  # Index is out of bounds of frames in tfrecord
        print('Found %s frames in tfrecord' % frame_num)

        # Get specified frame from tfrecord
        scan = next(f for i, f in enumerate(tfrecord) if i==frame_ndx)
        frame = self.waymo_converter.create_frame(scan)
        frame.context.name = '%s-%i' % (frame.context.name, frame_ndx)

        # Unpack frame into pointcloud and bounding boxes
        pcl, bboxes = self.waymo_converter.unpack_frame(frame)
        if self.visualize == 1:
            self.display_in_rviz(pcl, bboxes)
        
        # Remove groundplane from pointcloud
        pcl = remove_groundplane(pcl)
        if self.visualize == 2:
            self.display_in_rviz(pcl, bboxes)

        # Get bbox and calculate relevant cluster from pcl
        try:
            bbox = next(b for i, b in enumerate(bboxes) if b.id==bbox_id)
        except StopIteration:
            raise FrameError  # No bbox with id in current frame
        cluster = get_pts_in_bbox(pcl, bbox)

        # Publish cluster and bbox to rviz
        if self.visualize == 3 and cluster is not None:
            self.display_in_rviz(cluster, [bbox])

        # Publish pcl and bbox to rviz
        elif self.visualize == 4 and cluster is not None:
            self.display_in_rviz(pcl, [bbox])

    def play_cluster(self, tfrecord_id, frame_ndx, bbox_id):
        """Publishes cluster data in all frames consecutively for rviz display.

        Todo:
            Does not stop when the frame is done.

        Args:
            tfrecord_id: (str) id of tfrecord containing cluster.
            frame_ndxs: (int) index of 1st frame containing cluster.
            cluster_id: (str) id of bbox for which to visualize cluster.

        """

        while not rp.is_shutdown():
            self.update_rosparams()
            try:
                self.show_cluster(tfrecord_id, frame_ndx, bbox_id)
            except FrameError:
                return
            frame_ndx += 1
            self.rate.sleep()

    def display_in_rviz(self, pcl, bboxes):
        """Convert messages and publish to rviz."""
        self.pcl_pub.publish(
            self.ros_converter.convert2pcl(pcl))
        self.marker_pub.publish(
            self.ros_converter.convert2markerarray(bboxes))
        
    def run(self):
        while not rp.is_shutdown():
            self.update_rosparams()
            introspector.show_cluster(
                tfrecord_id='4816728784073043251_5273_410_5293_410',
                frame_ndx=0,
                bbox_id='OzEvFP55h3PLVLTOvzrV1g')
            self.rate.sleep()

if __name__ == '__main__':

    dataset = 'validation_0000'
    dir_data = '/home/cnovak/Data/waymo-od/%s' % dataset
    introspector = DatasetIntrospectorVis(dir_data=dir_data)
    introspector
    introspector.play_cluster(
        tfrecord_id='4816728784073043251_5273_410_5293_410',
        frame_ndx=0,
        bbox_id='OzEvFP55h3PLVLTOvzrV1g')
