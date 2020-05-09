#!/usr/bin/env python
"""Module for conversion from Waymo dataset constructs to numpy and Ros.

This module is intended to hold all awkward conversion code from the
Waymo dataset, which uses tensorflow .frame data storage. The module
contains classes for conversion, both to Ros (PointCloud2 & MarkerArray)
and numpy (ndarray & list)
"""

import ros_numpy

import tensorflow.compat.v1 as tf
import numpy as np
import rospy as rp

from waymo_open_dataset import dataset_pb2 as open_dataset

from waymo2numpy import Waymo2Numpy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import quaternion_from_euler

tf.enable_eager_execution()


class Waymo2Ros(Waymo2Numpy):
    """Converter class for translating Waymo frame data to Ros msgs."""

    def __init__(self):
        """Initialize parent class."""
        Waymo2Numpy.__init__(self)

    def convert2pcl(self, points, frame_id='base_link'):
        """Convert list of points into ros PointCloud2 msg.

        Args:
            points: numpy (x * 3) array of xyz points or numpy (n * 4) array of
                xyz points and intensities
            frame_id: str of ros tf frame to attach to pcl

        Returns:
            Ros PointCloud2 msg with points and frame_id

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
            PointCloud2, data, stamp=None, frame_id=frame_id)

    def convert2markerarray(self, labels, frame_id='base_link'):
        """Convert list of waymo dataset labels to bounding boxes in rviz.

        Args:
            labels: iterable of waymo_open_dataset.label_pb2.Labels
            frame_id: str of ros tf frame to attach to pcl

        Returns:
            ROS MarkerArray msg with each label as a Marker of type CUBE

        Todo:
            Tag with type, rather than id

        """
        msg = MarkerArray()

        for i, label in enumerate(labels):
            m = Marker()
            m.header.frame_id = frame_id
            # m.header.stamp = rp.get_time()
            m.ns = "waymo2ros"

            name = str(label.id)
            m.id = self.get_label_id(name)
            m.type = m.CUBE

            m.action = m.MODIFY if name in self.label_ids.keys() else m.ADD
            m.pose.position.x = label.box.center_x
            m.pose.position.y = label.box.center_y
            m.pose.position.z = label.box.center_z

            q = quaternion_from_euler(0, 0, np.pi/2 + label.box.heading)
            m.pose.orientation.x = q[0]
            m.pose.orientation.y = q[1]
            m.pose.orientation.z = q[2]
            m.pose.orientation.w = q[3]
            m.scale.x = label.box.width
            m.scale.y = label.box.length
            m.scale.z = label.box.height
            m.color.a = 0.5  # Opacity
            m.color.r, m.color.g, m.color.b = self.get_label_color(name)

            msg.markers.append(m)

        return msg


class Waymo2RosViz(Waymo2Ros):
    """Converter class for visualizing Waymo frame data as Ros msgs."""

    def __init__(self, frame_id="base_link"):
        """Initialize node, publishers, and parent classes."""
        self.frame_id = frame_id
        rp.init_node("waymo2ros")
        self.pcl_pub = rp.Publisher(
            "/pcl", PointCloud2, queue_size=1)
        self.box_pub = rp.Publisher(
            "/visualization_marker_array", MarkerArray, queue_size=1)
        Waymo2Ros.__init__(self)

    def update(self, frame):
        """Publish pointcloud and bounding boxes from given frame."""
        points_raw = self.frame2points(frame)
        pcl_msg = self.convert2pcl(points_raw)
        self.pcl_pub.publish(pcl_msg)

        bbox_msg = self.convert2markerarray(frame.laser_labels)
        self.box_pub.publish(bbox_msg)


class DatasetCreatorVis(DatasetCreator):
    """Class for visualizing DatasetCreator tasks with rviz."""

    def __init__(
            self, dir_load, dir_save, logger=None, dir_log=None,
            save_data=True, verbosity=None, visualize=0, density_thresh=0):
        """Initialize Ros components, DatasetCreator, visualize setting."""
        self.density_thresh = density_thresh
        self.visualize = visualize
        self.ros_converter = Waymo2Ros()
        rp.init_node('dataset_creator_vis', disable_signals=True)
        self.marker_pub = rp.Publisher('/bboxes', MarkerArray, queue_size=1)
        self.pcl_pub = rp.Publisher('/pcl', PointCloud2, queue_size=1)
        DatasetCreator.__init__(
            self, dir_load=dir_load, dir_save=dir_save, logger=logger,
            dir_log=dir_log, save_data=save_data, verbosity=verbosity,
            density_thresh=density_thresh)
        self.logger.debug('Exit:__init__')

    def pubData(self, pcl=None, bboxes=None):
        """Publish pointcloud and bounding boxes for rviz visualization."""
        self.pcl_pub.publish(self.ros_converter.convert2pcl(pcl))
        self.marker_pub.publish(self.ros_converter.convert2markerarray(bboxes))

    def parseFrame(self, frame, frame_id):
        """Extract and save data from a single given frame, viz if specified.

        Note:
            self.visualize attribute settings:
                1: shows original data.
                2: shows ground filtered data.
                3: shows clustered data.
                4: shows density filtered data.

        Args:
            frame: waymo open dataset Frame with loaded data
            frame_id: index of waymo Frame in tfrecord

        """
        self.logger.debug('Entr:parseFrame')
        self.visualize = int(
            rp.get_param("/visualize", self.visualize))
        self.density_thresh = int(
            rp.get_param("/density_thresh", self.density_thresh))

        pcl, bboxes = self.waymo_converter.unpack_frame(frame)  # 1

        if self.visualize == 1:
            self.pubData(pcl, bboxes)

        pcl = self.filterPcl(pcl)  # 2

        if self.visualize == 2:
            self.pubData(pcl, bboxes)

        clusters, bboxes = self.clusterByBBox(pcl, bboxes)  # 3

        if self.visualize == 3:
            if len(clusters) > 0:
                self.pubData(np.concatenate(clusters.values()), bboxes)
            else:
                self.logger.warning("No pcl with count > 10 pts")

        metadata = [self.computeClusterMetadata(
                    clusters[bbox.id], bbox, frame_id)
                    for bbox in bboxes]  # 4

        metadata, clusters = self.filterMetadata(
            metadata, clusters, self.density_thresh)  # 5

        # Show density filtered data if any clusters found
        if self.visualize == 4:
            if len(clusters) > 0:
                # Decided to plot full pointcloud since it makes it easier to
                # tell that nothing important is being removed. To only plot
                # points that are a part of the new sub-selected clusters,
                # uncomment this code
                # self.pcl_pub.publish(self.ros_converter.convert2pcl(
                #     np.concatenate(sub_clusters.values())))

                self.pubData(
                    pcl, [b for b in bboxes if str(b.id) in clusters.keys()])
            else:
                self.logger.warning("No pcl with density > 100 pts/m^3")

        if self.save_data:
            self.saveClusterMetadata(metadata, frame.context.name)  # 6
        self.logger.debug('Exit:parseFrame')
        return
