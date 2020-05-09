"""Module for conversion from Waymo dataset constructs to numpy and Ros.

This module is intended to hold all awkward conversion code from the
Waymo dataset, which uses tensorflow .frame data storage. The module
contains classes for conversion, both to Ros (PointCloud2 & MarkerArray)
and numpy (ndarray & list)
"""

import random  # Used for random label colors

import tensorflow as tf
import numpy as np

"""Used to create custom convert_range_image_to_pcl function
    Exposes lidar intensity data in the form of pointclouds
    Used instead of waymo_open_dataset corresponding function
"""
from waymo_open_dataset import dataset_pb2 as open_dataset

"""Found error on range_image_utils import:
    https://github.com/waymo-research/waymo-open-dataset/issues/155
    Fixed locally (2020-05-08) until further action on issue
"""
from waymo_open_dataset.utils import \
    range_image_utils, transform_utils, frame_utils


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

    def create_frame(self, scan):
        """Creates frame from scan in tfrecord.

        Example:
            tfrecord = tf.data.TFRecordDataset(<.tfrecord>, compression_type='')
            for scan in tfrecord:
                frame = create_frame(scan)
        """
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(scan.numpy()))
        return frame

    def unpack_frame(self, frame):
        """Get pcl, intensities, and labels from frame."""
        return self.frame2pcl(frame), self.frame2labels(frame)

    def frame2labels(self, frame):
        """Extract laser labels from waymo data frame."""
        return frame.laser_labels

    def frame2points(self, frame):
        """Extract points from waymo frame.

        Args:
            frame: waymo data frame

        Returns:
            points: numpy (n * 3) array of xyz points
        """
        frame.lasers.sort(key=lambda laser: laser.name)

        (range_images, camera_projections, range_image_top_pose) = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        pcl = convert_range_image_to_pcl(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)
        return pcl[:,:3] # Only return first three columns for xyz data

    def frame2pcl(self, frame):
        """Extract points and intensities from waymo frame.

        Args:
            frame: waymo data frame

        Returns:
            pcl: numpy (n * 4) array of xyz points and intensities
        """
        frame.lasers.sort(key=lambda laser: laser.name)

        (range_images, camera_projections, range_image_top_pose) = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        pcl = convert_range_image_to_pcl(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)
        return pcl


def convert_range_image_to_pcl(frame,
                               range_images,
                               camera_projections,
                               range_image_top_pose,
                               ri_index=0):

    """Convert range images to point cloud with point intensity info
    Based on frame_utils.convert_range_image_to_point_cloud

    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return,
                                 range_image_second_return]}.
        camera_projections: A dict of {laser_name,
                                       [camera_projection_from_first_return,
                                       camera_projection_from_second_return]}.
       range_image_top_pose: range image pixel pose for top lidar.
       ri_index: 0 for the first return, 1 for the second return.

    Returns:
       pcl: numpy (n * 4) array of xyz points and intensities
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    intensities = []

    frame_pose = tf.convert_to_tensor(
        value=np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data),
        range_image_top_pose.shape.dims)
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None

        if c.name == open_dataset.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)
        range_image_intensity = range_image_tensor[...,1]

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

        points_tensor = tf.gather_nd(range_image_cartesian,
                                 tf.compat.v1.where(range_image_mask))
        intensity_tensor = tf.gather_nd(range_image_intensity,
                                 tf.compat.v1.where(range_image_mask))

        points.append(points_tensor.numpy())
        intensities.append(intensity_tensor.numpy())

    points = np.concatenate(points, axis=0)
    intensities = np.concatenate(intensities, axis=0)

    # Adding intensities to numpy array
    pcl = np.hstack((points, np.atleast_2d(intensities).T))
    return pcl
