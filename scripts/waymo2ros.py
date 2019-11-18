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


class Waymo2ROSViz:

    def __init__(self, frame_id="base_link"):
        """Initialize node, publishers."""
        rp.init_node("waymo2ros")
        self.pcl_pub = rp.Publisher(
            "/pcl", PointCloud2, queue_size=1)
        self.box_pub = rp.Publisher(
            "/visualization_marker_array", MarkerArray, queue_size=1)

        self.converter = Waymo2ROS()

    def update(self, frame):
        """Publish pointcloud and bounding boxes from given frame."""
        points_raw = self.converter.frame2points(frame)
        pcl_msg = self.converter.convert2pcl(points_raw)
        self.pcl_pub.publish(pcl_msg)

        bbox_msg = self.converter.convert2markerarray(frame.laser_labels)
        self.box_pub.publish(bbox_msg)


class Waymo2ROS:

    def __init__(self):
        """Initialize state vars."""
        self.label_ids = {}  # stores label - int correspondence

    def convert2pcl(self, points, frame_id='base_link'):
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
            PointCloud2, data, stamp=None, frame_id=frame_id)
        return msg

    def convert2markerarray(self, labels, frame_id='base_link'):
        """Convert list of waymo dataset labels to bounding boxes in rviz."""
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

    def frame2points(self, frame):
        """Extract points from waymo frame."""
        #frame.ParseFromString(bytearray(data.numpy()))
        frame.lasers.sort(key=lambda laser: laser.name)

        (range_images, camera_projections, range_image_top_pose) = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)

        return np.concatenate(points, axis=0)

    def get_label_color(self, label):
        """Return seeded random color for label."""
        random.seed(self.get_label_id(label))
        return [random.uniform(0, 1) for i in range(3)]

    def get_label_id(self, label):
        """Return numeric id of label from label_ids.

        Checks for label in hash map. If exists, returns int.
        Otherwise, adds label and returns newly populated consecutive id.
        """
        if label not in self.label_ids:
            self.label_ids[label] = len(self.label_ids)
        return self.label_ids[label]


if __name__ == "__main__":
    conversion = Waymo2ROSViz()

    # TODO: Make this more general
    FILENAME = '/home/cnovak/Workspaces/catkin_ws/src/waymo-od/tutorial/frames'
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        conversion.update(frame)

"""CPP Source Code
    visualization_msgs::Marker marker;
    marker.header.frame_id = "base_link";
    marker.header.stamp = ros::Time();
    marker.ns = "my_namespace";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 1;
    marker.pose.position.y = 1;
    marker.pose.position.z = 1;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.a = 1.0; // Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    //only if using a MESH_RESOURCE marker type:
    marker.mesh_resource = "package://pr2_description/meshes/base_v0/base.dae";
    vis_pub.publish( marker );
    """

"""Other Source Code
    (range_images, camera_projections,
     range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
        frame)


    print(frame.context)

    plt.figure(figsize=(64, 20))
    def plot_range_image_helper(data, name, layout, vmin = 0, vmax=1, cmap='gray'):
      ""\"Plots range image.

      Args:
        data: range image data
        name: the image title
        layout: plt layout
        vmin: minimum value of the passed data
        vmax: maximum value of the passed data
        cmap: color map
      ""\"
      plt.subplot(*layout)
      plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
      plt.title(name)
      plt.grid(False)
      plt.axis('off')
      # plt.show()

    def get_range_image(laser_name, return_index):
      ""\"Returns range image given a laser name and its return index.""\"
      return range_images[laser_name][return_index]

    def show_range_image(range_image, layout_index_start = 1):
      ""\"Shows range image.

      Args:
        range_image: the range image data from a given lidar of type MatrixFloat.
        layout_index_start: layout offset
      ""\"
      range_image_tensor = tf.convert_to_tensor(range_image.data)
      range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
      lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
      range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
                                    tf.ones_like(range_image_tensor) * 1e10)
      range_image_range = range_image_tensor[...,0]
      range_image_intensity = range_image_tensor[...,1]
      range_image_elongation = range_image_tensor[...,2]
      plot_range_image_helper(range_image_range.numpy(), 'range',
                       [8, 1, layout_index_start], vmax=75, cmap='gray')
      plot_range_image_helper(range_image_intensity.numpy(), 'intensity',
                       [8, 1, layout_index_start + 1], vmax=1.5, cmap='gray')
      plot_range_image_helper(range_image_elongation.numpy(), 'elongation',
                       [8, 1, layout_index_start + 2], vmax=1.5, cmap='gray')


    frame.lasers.sort(key=lambda laser: laser.name)
    show_range_image(get_range_image(open_dataset.LaserName.TOP, 0), 1)
    show_range_image(get_range_image(open_dataset.LaserName.TOP, 1), 4)

    #
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)

    points_all = np.concatenate(points, axis=0)
    print(points_all)
    """
