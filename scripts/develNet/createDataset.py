#!usr/bin/env python
import sys # Needed for relative imports
sys.path.append('../') # Needed for relative imports

import json
import logging
import os.path
import rospy

import numpy as np
import tensorflow as tf

from collections import namedtuple
from modules.helperFunctions import *
from modules.waymo2ros import Waymo2Numpy


XYPair = namedtuple('XYPair', 'x y')
XYZPair = namedtuple('XYZPair', 'x y z')


def is_between_lines(l1, l2, p):
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


def is_in_bbox(point, label, logger=None):
    """Return True if point within bbox in xy-plane.

    Args:
        point: obj representing point to check, has x & y attr
        bbox: laser scan thing from waymo?
        logger: python logging object

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
    try:
        return True if is_between_lines(w1, w2, point) \
            and is_between_lines(l1, l2, point) else False
    except ValueError as e:
        if logger is not None:
            logger.debug('Warn:non-parallel_lines=True')
            # TODO show points making lines as debug
            raise


class DatasetCreator(object):
    """Class for creating labeled cluster metadata from raw waymo data.

    Use this class to iterate through .tfrecord files, pull out frames,
    extract clusters from the frames using provided bounding boxes,
    calculate features of clusters, and save resulting features and
    metadata to create a cleaned dataset.

    """

    def __init__(self, logger=None, verbosity=None):
        """Provide directory location to find frames."""
        self.waymo_converter = Waymo2Numpy()
        self.data_dir = 'data/train'

        # Set up logger if not given as arg
        if logger is not None:
            self.logger = logger
        else:
            logging.basicConfig(
                level=logging.INFO, stream=sys.stdout,
                format="%(asctime)s %(levelname)s %(message)s")
            self.logger = logging.getLogger('datasetCreator')
            self.logger.setLevel(verbosity) if verbosity is not None \
                else self.logger.setLevel(logging.INFO)

    def filterPcl(self, pcl):
        """Downsample and remove groundplane from pcl."""
        self.logger.info('Entr:filterPcl')
        pcl_out = remove_groundplane(np.array([list(pt) for pt in pcl]))
        self.logger.debug('Show:pts_removed=%i' % (len(pcl) - len(pcl_out)))
        self.logger.info('Exit:filterPcl')
        return pcl_out

    def clusterByBBox(self, pcl, bboxes, thresh=25):
        """Extract points from pcl within bboxes as clusters.

        Args:
            pcl: (n * 3) numpy array of xyz points
            bboxes: waymo pcl label output
            thresh: min int point num for cluster

        Returns: list of (n * 3) numpy arrays of xyz points

        """
        self.logger.info('Entr:clusterByBBox')
        obj_pcls = {}  # Hash map of bbox label : pcl

        # Convert from list of tuples to list of XYPairs
        pcl = [XYZPair(pt[0], pt[1], pt[2]) for pt in pcl]
        self.logger.info("Show:bbox_count: %i" % len(bboxes))

        for i, bbox in enumerate(bboxes):

            # Add pts in bounding box to cluster and remove from pcl
            try:
                cluster = [pt for pt in pcl if is_in_bbox(pt, bbox, self.logger)]
            except ValueError as e:
                self.logger.warning('Unable to use bbox due to non-parallel error')
                continue

            pcl = [pt for pt in pcl if pt not in cluster]
            self.logger.info(
                "Show:bbox=%i|%i, class=%i, id=%s, pt_count=%i"
                % (i, len(bboxes), bbox.type, bbox.id, len(cluster)))
            self.logger.debug("Show:pcl_len=%i" % len(pcl))

            # Threshold cluster size
            if len(cluster) >= thresh:
                obj_pcls[bbox.id] = cluster
            else:
                obj_pcls[bbox.id] = None
                self.logger.info(
                    "Note:cluster_size=%i < threshold=%i"\
                    % (len(cluster), thresh))

            #if i == 5: break  # Uncomment to use subset of bboxes for debug

        self.logger.info('Exit:clusterByBBox')
        return obj_pcls

    def computeClusterMetadata(self, cluster, bbox):
        """Compute key information from cluster to boil down pointcloud infoself.

        Args:
            cluster: list of xyz points within cluster
            bbox: waymo object label output

        Returns:
            features: Features object containing cluster features
        """

        self.logger.info('Entr:computeClusterMetadata')
        if cluster is None:
            raise TypeError(
                'None passed as cluster - ' \
                + 'possibly a too-small cluster passed from clusterByBBox?')
        np_cluster = np.array(cluster)
        features = extract_cluster_features(np_cluster, bbox)
        self.logger.info('Exit:computeClusterMetadata')
        return features

    def saveClusterMetadata(self, metadata, name):
        """Save cluster metadata from frame in a .json file. Uses frame name as
        .json filename

        Args:
            metadata: list of Features objects containing key information about
                each cluster in frame
            name: name of frame
        """
        self.logger.info('Entr:saveClusterMetadata')
        filename = '%s/%s.json' % (self.data_dir, str(name))
        self.logger.info('Show:save_loc=%s' % filename)

        # lambda function is used to serialize custom Features object
        with open(filename, 'w') as outfile:
            json.dump(metadata, outfile, default=lambda o: o.__dict__, indent=4)

        self.logger.info('Exit:saveClusterMetadata')

    def parseFrame(self, frame):
        """Extract and save data from a single given frame.

        Args:
            frame: waymo open dataset Frame with loaded data

        """
        self.logger.info('Entr:parseFrame')
        pcl, bboxes = self.waymo_converter.unpack_frame(frame)
        pcl = self.filterPcl(pcl)
        clusters = self.clusterByBBox(pcl, bboxes)
        metadata = [self.computeClusterMetadata(c, bboxes[i])
            for i, c in enumerate(clusters.values()) if c is not None]
        self.saveClusterMetadata(metadata, frame.context.name)
        self.logger.info('Exit:parseFrame')
        return

    def checkDataFile(self, frame):
        """Check if data file currently exists.

        Args:
            frame: waymo dataset Frame object whose context name is used to
                create filename for which to check

        """
        self.logger.info('Entr:checkDataFile')
        file = '%s/%s.json' % (self.data_dir, frame.context.name)
        self.logger.debug('Show:frame_file=%s' % file)
        self.logger.info('Exit:checkDataFile')
        return os.path.isfile(file)

    def run(self, overwrite=False):
        """Generate data for all scans in all .tfrecord files in dir.

        Args:
            overwrite: Bool for overwriting already existing data

        Todo:
            put glob + directory stuff here

        """
        self.logger.info('Entr:run')
        DIRECTORY = '/home/cnovak/Data/waymo-od/'
        #'/home/amy/test_ws/src/waymo-od/tutorial/'
        FILE = 'segment-15578655130939579324_620_000_640_000' \
             + '_with_camera_labels.tfrecord'
        #'frames'
        tfrecord = tf.data.TFRecordDataset(DIRECTORY+'/'+FILE,
         compression_type='')
        for i, scan in enumerate(tfrecord):
            frame = self.waymo_converter.create_frame(scan)
            frame.context.name = '%s-%i' % (frame.context.name, i)
            if self.checkDataFile(frame) and not overwrite:
                self.logger.info('Show:framedata_exists=True')
                continue
            self.logger.info(
                'Show:frame_num=%i, frame_id=%s' \
                % (i, str(frame.context.name)))
            self.parseFrame(frame)

        self.logger.info('Exit:run')
        return


class DatasetCreatorVis(DatasetCreator):
    """Class for visualizing DatasetCreator tasks with rviz."""

    def __init__(self):

        rp.init_node('dataset_creator_vis')
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
    creator = DatasetCreatorVis() if visualize \
        else DatasetCreator(verbosity=logging.DEBUG)
    creator.run()  # TODO Setup directory choosing
