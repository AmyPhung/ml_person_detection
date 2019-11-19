#!usr/bin/env python
import logging
import rospy
import time

import numpy as np
import tensorflow as tf

from collections import namedtuple
from modules.helperFunctions import *
from modules.waymo2ros import Waymo2Numpy


XYPair = namedtuple('XYPair', 'x y')
XYZPair = namedtuple('XYZPair', 'x y z')

class DatasetCreator(object):
    """TODO ADD DOCSTRING"""

    def __init__(self):
        """Provide directory location to find frames."""
        self.waymo_converter = Waymo2Numpy()
        pass

    def filterPcl(self, pcl):
        """Downsample and remove groundplane from pcl."""
        print('Filtering pointcloud')
        return remove_groundplane(np.array([list(pt) for pt in pcl]))

    def clusterByBBox(self, pcl, bboxes):
        """Extract points from pcl within bboxes as clusters.

        Args:
            pcl: (n * 3) numpy array of xyz points
            bboxes:

        Returns: list of (n * 3) numpy arrays of xyz points

        """
        print('Computing clusters')
        obj_pcls = {}  # Hash map of bbox label : pcl

        # Convert from list of tuples to list of XYPairs
        pcl = [XYPair(pt[0], pt[1]) for pt in pcl]
        print("Found %i bounding boxes" % len(bboxes))

        for bbox in bboxes:
            # Sub-select points into new PointCloud2 if within marker rect
            print("Parsing bounding box %s" % bbox.id)
            t = time.time()
            obj_pcls[bbox.id] = [pt for pt in pcl if self.is_in_bbox(pt, bbox)]
            print("Took %.2f sec" % (time.time() - t))

        return obj_pcls

    def computeClusterMetadata(self):
        print('Computing metadata from cluster')
        pass

    def saveClusterMetadata(self):
        print('Saving cluster metadata')
        pass

    def parseFrame(self, frame):
        """Extract and save data from a single given frame.

        Args:
            frame: waymo open dataset Frame with loaded data

        """
        print('Saving dataset points from frame')
        pcl, bboxes = self.waymo_converter.unpack_frame(frame)
        pcl = self.filterPcl(pcl)
        clusters = self.clusterByBBox(pcl, bboxes)
        metadata = [self.computeClusterMetadata(c) for c in clusters]
        self.saveClusterMetadata(metadata)
        return

    def run(self):
        """Generate data for all scans in all .tfrecord files in dir.

        Todo:
            put glob + directory stuff here

        """
        DIRECTORY = '/home/cnovak/Data/waymo-od/'
        FILE = 'segment-15578655130939579324_620_000_640_000' \
            + '_with_camera_labels.tfrecord'
        tfrecord = tf.data.TFRecordDataset(DIRECTORY+'/'+FILE, compression_type='')
        for scan in tfrecord:
            frame = self.waymo_converter.create_frame(scan)
            self.parseFrame(frame)
        return

    def is_in_bbox(self, point, label):
        """Return True if point within bbox in xy-plane.

        Args:
            point: obj representing point to check, has x & y attr
            bbox: laser scan thing from waymo?

        Returns:
            bool True if point in box else False
        """
        
        # Simplify variables for some marker attributes
        angle = label.box.heading
        cntr = XYZPair(
            label.box.center_x, label.box.center_y, label.box.center_z)

        # Calculate offsets in xy-coordinates for each side
        l = XYPair(
            0.5 * label.box.length * np.cos(np.radians(angle)),
            0.5 * label.box.length * np.sin(np.radians(angle)))
        w = XYPair(
            0.5 * label.box.width * np.cos(np.radians(90 + angle)),
            0.5 * label.box.width * np.sin(np.radians(90 + angle)))

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


class DatasetCreatorVis(DatasetCreator):

    def __init__(self):

        # Do ros stuff here
        super(DatasetCreator, self).__init__()
        pass

    def parseFrame(self):
        """Overwrite CreateDataset run function, insert viz."""

        # load frame (from file)
        # publish to ros
        # publsih markers and such to ros
        # filter frame
        # cluster
        # compute
        # save
        pass

if __name__ == "__main__":
    visualize = False
    creator = DatasetCreatorVis() if visualize else DatasetCreator()
    creator.run()  # TODO Setup directory choosing
