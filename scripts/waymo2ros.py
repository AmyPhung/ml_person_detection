#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import ros_numpy

import os
import tensorflow as tf
import math
import numpy as np
import itertools

import matplotlib.pyplot as plt
import matplotlib.patches as patches

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

class Waymo2ROS:
    def __init__(self, frame_id="base_link"):
        self.frame_id = frame_id

        rospy.init_node("waymo2ros")
        self.pcl_pub = rospy.Publisher("/pcl", PointCloud2, queue_size=1)
        self.box_pub = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=1)

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
        # points
        # pcl_msg = PointCloud2()
        # pcl_msg.points = points_all

    def convert2pcl(self, points):
        data = np.zeros(points.shape[0], dtype=[
          ('x', np.float32),
          ('y', np.float32),
          ('z', np.float32),
        ])
        data['x'] = points[:,0]
        data['y'] = points[:,1]
        data['z'] = points[:,2]

        msg = ros_numpy.msgify(PointCloud2, data, stamp=None, frame_id=self.frame_id)
        return msg

    def convert2markerarray(self, labels):
        msg = MarkerArray()

        for i in range(0, len(labels)):
            m = Marker()
            m.header.frame_id = self.frame_id
            # m.header.stamp = rospy.get_time()
            m.ns = "waymo2ros"
            m.id = i #str(label.id) # TODO: Add unique IDs for each item (add id permanance). Can't use label.id since it's a string
            m.type = m.CUBE
            m.action = m.ADD # Can also be add, delete, delteall TODO: make this MODIFY after IDs are added
            m.pose.position.x = labels[i].box.center_x
            m.pose.position.y = labels[i].box.center_y
            m.pose.position.z = labels[i].box.center_z
            m.pose.orientation.x = 0.0 # TODO: fill these out (use labels[i].box.pose for heading in radians)
            m.pose.orientation.y = 0.0 # I hate quaternions....
            m.pose.orientation.z = 0.0 # I hate tfs.....
            m.pose.orientation.w = 1.0 # math.cos(labels[i].box.pose)
            m.scale.x = labels[i].box.width
            m.scale.y = labels[i].box.length
            m.scale.z = labels[i].box.height
            m.color.a = 0.5 # Opacity

            # TODO: Use unique color for each ID and item
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0

            msg.markers.append(m)
            print(m)

        return msg
        # print(labels[0].box)

def heading2quat(heading):
    cos
    return w, x, y, z

  #  1 visualization_msgs::Marker markerr;
  #  2 marker.header.frame_id = "base_link";
  #  3 marker.header.stamp = ros::Time();
  #  4 marker.ns = "my_namespace";
  #  5 marker.id = 0;
  #  6 marker.type = visualization_msgs::Marker::SPHERE;
  #  7 marker.action = visualization_msgs::Marker::ADD;
  #  8 marker.pose.position.x = 1;
  #  9 marker.pose.position.y = 1;
  # 10 marker.pose.position.z = 1;
  # 11 marker.pose.orientation.x = 0.0;
  # 12 marker.pose.orientation.y = 0.0;
  # 13 marker.pose.orientation.z = 0.0;
  # 14 marker.pose.orientation.w = 1.0;
  # 15 marker.scale.x = 1;
  # 16 marker.scale.y = 0.1;
  # 17 marker.scale.z = 0.1;
  # 18 marker.color.a = 1.0; // Don't forget to set the alpha!
  # 19 marker.color.r = 0.0;
  # 20 marker.color.g = 1.0;
  # 21 marker.color.b = 0.0;
  # 22 //only if using a MESH_RESOURCE marker type:
  # 23 marker.mesh_resource = "package://pr2_description/meshes/base_v0/base.dae";
  # 24 vis_pub.publish( marker );


conversion = Waymo2ROS()

# TODO: Make this more general
FILENAME = '/home/amy/test_ws/src/waymo-od/tutorial/frames'
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    conversion.update(frame)

    # break




# (range_images, camera_projections,
#  range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
#     frame)


# print(frame.context)
#
# plt.figure(figsize=(64, 20))
# def plot_range_image_helper(data, name, layout, vmin = 0, vmax=1, cmap='gray'):
#   """Plots range image.
#
#   Args:
#     data: range image data
#     name: the image title
#     layout: plt layout
#     vmin: minimum value of the passed data
#     vmax: maximum value of the passed data
#     cmap: color map
#   """
#   plt.subplot(*layout)
#   plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
#   plt.title(name)
#   plt.grid(False)
#   plt.axis('off')
#   # plt.show()
#
# def get_range_image(laser_name, return_index):
#   """Returns range image given a laser name and its return index."""
#   return range_images[laser_name][return_index]
#
# def show_range_image(range_image, layout_index_start = 1):
#   """Shows range image.
#
#   Args:
#     range_image: the range image data from a given lidar of type MatrixFloat.
#     layout_index_start: layout offset
#   """
#   range_image_tensor = tf.convert_to_tensor(range_image.data)
#   range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
#   lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
#   range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
#                                 tf.ones_like(range_image_tensor) * 1e10)
#   range_image_range = range_image_tensor[...,0]
#   range_image_intensity = range_image_tensor[...,1]
#   range_image_elongation = range_image_tensor[...,2]
#   plot_range_image_helper(range_image_range.numpy(), 'range',
#                    [8, 1, layout_index_start], vmax=75, cmap='gray')
#   plot_range_image_helper(range_image_intensity.numpy(), 'intensity',
#                    [8, 1, layout_index_start + 1], vmax=1.5, cmap='gray')
#   plot_range_image_helper(range_image_elongation.numpy(), 'elongation',
#                    [8, 1, layout_index_start + 2], vmax=1.5, cmap='gray')
#
#
# frame.lasers.sort(key=lambda laser: laser.name)
# show_range_image(get_range_image(open_dataset.LaserName.TOP, 0), 1)
# show_range_image(get_range_image(open_dataset.LaserName.TOP, 1), 4)
#
# #
# points, cp_points = frame_utils.convert_range_image_to_point_cloud(
#     frame,
#     range_images,
#     camera_projections,
#     range_image_top_pose)
#
# points_all = np.concatenate(points, axis=0)
# print(points_all)
