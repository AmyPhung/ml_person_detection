#!/usr/bin/env python
"""Module for conversion from Waymo dataset constructs to numpy and Ros.

This module is intended to hold all awkward conversion code from the
Waymo dataset, which uses tensorflow .frame data storage. The module
contains classes for conversion, both to Ros (PointCloud2 & MarkerArray)
and numpy (ndarray & list)
"""

import itertools
import math
import os
import random
import ros_numpy

import tensorflow as tf
import numpy as np
import rospy as rp

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import \
    range_image_utils, transform_utils, frame_utils

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import quaternion_from_euler

tf.enable_eager_execution()


class Waymo2Numpy(object):
    """Converter class for translating Waymo frame data to numpy arrays."""

    def __init__(self):
        """Initialize state vars."""
        self.label_ids = {}  # stores label - int correspondence

    def get_label_id(self, label):
        """Return numeric id of label from label_ids.

        Checks for label in hash map. If exists, returns int.
        Otherwise, adds label and returns newly populated consecutive id.
        """
        if label not in self.label_ids:
            self.label_ids[label] = len(self.label_ids)
        return self.label_ids[label]

    def get_label_color(self, label):
        """Return seeded random color for label."""
        random.seed(self.get_label_id(label))
        return [random.uniform(0, 1) for i in range(3)]

    def frame2labels(self, frame):
        """Extract laser labels from waymo data frame."""
        return frame.laser_labels

    def frame2points(self, frame):
        """Extract points from waymo frame.

        Args:
            frame: waymo data frame

        Returns:
            numpy array (x * 3) of xyz points

        """
        frame.lasers.sort(key=lambda laser: laser.name)

        (range_images, camera_projections, range_image_top_pose) = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        points, __ = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)

        return np.concatenate(points, axis=0)


class Waymo2Ros(Waymo2Numpy):
    """Converter class for translating Waymo frame data to Ros msgs."""

    def __init__(self):
        """Initialize parent class."""
        super(Waymo2Numpy, self).__init__()

    def convert2pcl(self, points, frame_id='base_link'):
        """Convert list of points into ros PointCloud2 msg.

        Args:
            points: numpy (x * 3) array of xyz points
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
            Ros MarkerArray msg with each label as a Marker of type CUBE

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
        super(Waymo2RosViz, self).__init__()

    def update(self, frame):
        """Publish pointcloud and bounding boxes from given frame."""
        temp = self.frame2labels(frame)
        points_raw = self.frame2points(frame)
        pcl_msg = self.convert2pcl(points_raw)
        self.pcl_pub.publish(pcl_msg)

        bbox_msg = self.convert2markerarray(frame.laser_labels)
        self.box_pub.publish(bbox_msg)


if __name__ == "__main__":

    converter = Waymo2RosViz()

    # TODO: Make this more general
    DIRECTORY = '/home/cnovak/Data/waymo-od/'
    FILE = 'segment-15578655130939579324_620_000_640_000' \
        + '_with_camera_labels.tfrecord'
    dataset = tf.data.TFRecordDataset(DIRECTORY+'/'+FILE, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        converter.update(frame)
