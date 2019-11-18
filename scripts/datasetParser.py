#!/usr/bin/env python
import rospy
import ros_numpy
import time

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from collections import namedtuple
from sensor_msgs.msg import PointCloud2
from tf.transformations import euler_from_quaternion
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo2ros import Waymo2ROS

tf.logging.set_verbosity(tf.logging.ERROR)

XYPair = namedtuple('XYPair', 'x y')

class DatasetParser:
    def __init__(self):
        pass

    def compute_clusters(self, pcl, bboxes):

        obj_pcls = {}  # Hash map of bbox label : pcl

        # Convert to numpy array - each point is a tuple
        data = ros_numpy.numpify(pcl)
        # Convert from list of tuples to list of XYPairs
        data_pts = [XYPair(pt[0], pt[1]) for pt in data]

        print("Found %i markers" % len(bboxes.markers))
        for m in bboxes.markers:
            # Sub-select points into new PointCloud2 if within marker rect
            print("Parsing marker %i" % m.id)
            t = time.time()
            obj_pcls[m.id] = [pt for pt in data_pts if self.is_in_bbox(m, pt)]
            print("Took %.2f sec" % (time.time() - t))

        return obj_pcls

    def is_in_bbox(self, bbox, point):
        """Return True if point within bbox in xy-plane.

        Args:
            bbox: Marker of type CUBE representing bounding box.
            point: obj representing point to check, has x & y attr

        Returns:
            bool True if point in box else False
        """
        
        # Simplify variables for some marker attributes
        quat = bbox.pose.orientation
        angle = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]
        cntr = bbox.pose.position

        # Calculate offsets in xy-coordinates for each side
        l = XYPair(
            0.5 * bbox.scale.y * np.cos(np.radians(angle)),
            0.5 * bbox.scale.y * np.sin(np.radians(angle)))
        w = XYPair(
            0.5 * bbox.scale.x * np.cos(np.radians(90 + angle)),
            0.5 * bbox.scale.x * np.sin(np.radians(90 + angle)))

        # Calculate corner points
        p1 = XYPair(cntr.x + l.x + w.x, cntr.y + l.y + w.y)
        p2 = XYPair(cntr.x - l.x - w.x, cntr.y - l.y - w.y)
        p3 = XYPair(cntr.x + l.x - w.x, cntr.y + l.y - w.y)
        p4 = XYPair(cntr.x - l.x + w.x, cntr.y - l.y + w.y)

        # Create functions for lines representing bbox sides
        def w1(x):
            return ((p4.y - p2.y) / (p4.x - p2.x)) * (x - p4.x) + p4.y

        def w2(x):
            return ((p3.y - p1.y) / (p3.x - p1.x)) * (x - p3.x) + p3.y

        def l1(x):
            return ((p2.y - p3.y) / (p2.x - p3.x)) * (x - p2.x) + p2.y

        def l2(x):
            return ((p1.y - p4.y) / (p1.x - p4.x)) * (x - p1.x) + p1.y

        # Check that point is between both sets of parallel lines
        return True if self.is_between_lines(w1, w2, point) \
            and self.is_between_lines(l1, l2, point) else False

    def is_between_lines(self, l1, l2, p):
        """Return true if point is between parallel lines.

        Args:
            l1: function that returns y for any x
            l2: function that returns y for any x
            p: some obj with x and y attr

        Returns:
            bool if point between parallel lines.

        """
        if not abs(round(l1(-2) - l1(2), 3)) \
                == abs(round(l2(-2) - l2(2), 3)):
            raise ValueError('lines are not parallel!')

        y0, y1, y2 = p.y, l1(p.x), l2(p.x)
        # If line 1 is above line 2
        if y1 > y2:
            return True if y2 < y0 < y1 else False
        else:
            return True if y1 < y0 < y2 else False

    def run(self):
        plt.ion()
        plt.show()


if __name__ == "__main__":

    conversion = Waymo2ROS()
    parser = DatasetParser()

    FILENAME = '/home/cnovak/Workspaces/catkin_ws/src/waymo-od/tutorial/frames'
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        pcl = conversion.convert2pcl(conversion.frame2points(frame))
        bboxes = conversion.convert2markerarray(frame.laser_labels)
        parser.computeClusters(pcl, bboxes)
