#!/usr/bin/env python

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


class Waymo2ROS:

    def __init__(self, frame_id="base_link"):
        """Initialize node, publishers, state vars."""
        self.frame_id = frame_id
        self.label_ids = {}  # stores label - int correspondence

        rp.init_node("waymo2ros")
        self.pcl_pub = rp.Publisher(
            "/pcl", PointCloud2, queue_size=1)
        self.box_pub = rp.Publisher(
            "/visualization_marker_array", MarkerArray, queue_size=1)

    def update(self, frame):
        frame.ParseFromString(bytearray(data.numpy()))
        frame.lasers.sort(key=lambda laser: laser.name)

        (range_images, camera_projections, range_image_top_pose) = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)

        points_all = np.concatenate(points, axis=0)
        print(points_all.shape)

        pcl_msg = self.convert2pcl(points_all)
        self.pcl_pub.publish(pcl_msg)

        markerarray_msg = self.convert2markerarray(frame.laser_labels)
        self.box_pub.publish(markerarray_msg)

    def convert2pcl(self, points):
        """Convert list of points into ros PointCloud2 msg."""
        data = np.zeros(points.shape[0], dtype=[
          ('x', np.float32),
          ('y', np.float32),
          ('z', np.float32),
        ])
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]

        msg = ros_numpy.msgify(
            PointCloud2, data, stamp=None, frame_id=self.frame_id)
        return msg

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

    def convert2markerarray(self, labels):
        """Convert list of waymo dataset labels to bounding boxes in rviz."""
        msg = MarkerArray()

        for i, label in enumerate(labels):
            m = Marker()
            m.header.frame_id = self.frame_id
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


if __name__ == "__main__":
    conversion = Waymo2ROS()

    # TODO: Make this more general
    FILENAME = '/home/amy/test_ws/src/waymo-od/tutorial/frames'
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        conversion.update(frame)
