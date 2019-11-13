#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
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
    def __init__(self):
        rospy.init_node("waymo2ros")
        self.pcl_pub = rospy.Publisher("/pcl", PointCloud2, queue_size=1)

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

        pcl_msg = convert2pcl(points_all)
        self.pcl_pub.publish(pcl_msg)
        # points
        # pcl_msg = PointCloud2()
        # pcl_msg.points = points_all

def convert2pcl(points):
    data = np.zeros(points.shape[0], dtype=[
      ('x', np.float32),
      ('y', np.float32),
      ('z', np.float32),
    ])
    data['x'] = points[:,0]
    data['y'] = points[:,1]
    data['z'] = points[:,2]

    msg = ros_numpy.msgify(PointCloud2, data, stamp=None, frame_id="base_link")
    return msg


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
