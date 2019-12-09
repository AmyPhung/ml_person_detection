#!/usr/bin/env python

import sys # Needed for relative imports
sys.path.append('../') # Needed for relative imports

import datetime
import glob
import json
import logging
import os.path
import rospy as rp

import numpy as np
import tensorflow as tf

from collections import namedtuple
from modules.helperFunctions import *
from modules.waymo2ros import Waymo2Numpy, Waymo2Ros
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray


XYPair = namedtuple('XYPair', 'x y')
XYZPair = namedtuple('XYZPair', 'x y z')



def get_pts_in_bbox(pcl, bbox, logger=None):
    """Return ndarray of points from pcl within bbox."""
    # Get bbox limits in bbox coord frame
    x_lo = bbox.box.center_x - bbox.box.length/2
    x_hi = bbox.box.center_x + bbox.box.length/2
    y_lo = bbox.box.center_y - bbox.box.width/2
    y_hi = bbox.box.center_y + bbox.box.width/2
    if logger is not None:
        logger.debug('bbox limits: %0.2f-%0.2f, %0.2f-%0.2f'
            % (x_lo, x_hi, y_lo, y_hi))

    # Rotate points by bbox heading angle
    ang = np.radians(bbox.box.heading)
    r_mat = np.array(((np.cos(ang), np.sin(ang)), (-np.sin(ang), np.cos(ang))))
    r_pcl = np.matmul(pcl[:, 0:2], r_mat)

    # Add back in z and intensity data
    r_pcl = np.append(r_pcl, pcl[:, 2:], axis=1)

    # Sub-select pcl by bbox limits
    indxs = np.where(
        (x_lo < r_pcl[:, 0]) & (r_pcl[:, 0] < x_hi)
        & (y_lo < r_pcl[:, 1]) & (r_pcl[:, 1] < y_hi))[0]
    pcl_out = r_pcl[indxs]
    return pcl_out


class DatasetCreator(object):
    """Class for creating labeled cluster metadata from raw waymo data.

    Use this class to iterate through .tfrecord files, pull out frames,
    extract clusters from the frames using provided bounding boxes,
    calculate features of clusters, and save resulting features and
    metadata to create a cleaned dataset.

    """

    def __init__(self, dir_load, dir_save, logger=None, dir_log=None, verbosity=None):
        """Provide directory location to find frames."""
        self.waymo_converter = Waymo2Numpy()
        self.dir_load = dir_load
        self.dir_save = dir_save

        # Set up logger if not given as arg
        if logger is not None:
            self.logger = logger
        else:
            # Generate log file name
            d = datetime.datetime.now()
            filename = ('%s/%i-%i-%i-%i-%i.log'
                        % (dir_log, d.year, d.month, d.day, d.hour, d.minute))

            # Create logger with file and stream handlers
            self.logger = logging.getLogger('datasetCreator')
            sh = logging.StreamHandler(sys.stdout)
            fh = logging.FileHandler(filename)

            # Format file and stream logging
            formatter = logging.Formatter(
                '%(asctime)s %(levelname)s %(message)s')
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            sh.setFormatter(formatter)
            sh.setLevel(logging.INFO)
            self.logger.addHandler(fh)
            self.logger.addHandler(sh)
            self.logger.setLevel(logging.DEBUG)

        self.logger.info('Logging set up for createDataset object')
        self.logger.debug('Exit:__init__')

    def filterPcl(self, pcl):
        """Remove groundplane from pcl."""
        self.logger.debug('Entr:filterPcl')

        pcl_out = remove_groundplane(np.array([list(pt) for pt in pcl]))
        self.logger.debug('Show:pts_removed=%i' % (len(pcl) - len(pcl_out)))
        self.logger.debug('Exit:filterPcl')
        return pcl_out

    def clusterByBBox(self, pcl, bboxes, thresh=25):
        """Extract points from pcl within bboxes as clusters.

        Args:
            pcl: (n * 4) numpy array of xyz points and intensities
            bboxes: waymo pcl label output
            thresh: min int point num for cluster

        Returns:
            obj_pcls: list of (n * 4) numpy arrays of xyz points and intensities
        """
        
        self.logger.debug('Entr:clusterByBBox')

        obj_pcls = {}  # Hash map of bbox label : pcl
        self.logger.debug("bbox initial count: %i" % len(bboxes))

        for i, label in enumerate(bboxes):

            cluster = get_pts_in_bbox(pcl, label, self.logger)
            self.logger.debug(
                "bbox=%i * %i, class=%i, id=%s, pt_count=%i"
                % (i, len(bboxes), label.type, label.id, len(cluster)))

            # Threshold cluster size
            if len(cluster) >= thresh:
                obj_pcls[label.id] = cluster
            else:
                obj_pcls[label.id] = None
                self.logger.debug(
                    "cluster_size=%i under threshold=%i"\
                    % (len(cluster), thresh))

            #if i == 5: break  # Uncomment to use subset of bboxes for debug

        self.logger.debug('Exit:clusterByBBox')
        self.logger.debug("bbox final count: %i" % len([o for o in obj_pcls if o is not None]))
        return obj_pcls

    def computeClusterMetadata(self, cluster, bbox):
        """Compute key information from cluster to boil down pointcloud info.

        Args:
            cluster: list of xyz points and intensities within cluster
            bbox: waymo object label output

        Returns:
            features: Features object containing cluster features
        """

        self.logger.debug('Entr:computeClusterMetadata')
        if cluster is None:
            raise TypeError(
                'None passed as cluster - ' \
                + 'possibly a too-small cluster passed from clusterByBBox?')
        np_cluster = np.array(cluster)

        features = Features()
        features.cluster_id = bbox.id
        features.cls = bbox.type
        features.cnt = cluster.shape[0]
        features.parameters = extract_cluster_parameters(np_cluster, display=False)

        self.logger.debug('Exit:computeClusterMetadata')
        return features

    def saveClusterMetadata(self, metadata, name):
        """Save cluster metadata from frame in a .json file. Uses frame name as
        .json filename

        Args:
            metadata: list of Features objects containing key information about
                each cluster in frame
            name: name of frame
        """
        self.logger.debug('Entr:saveClusterMetadata')
        filename = '%s/%s.json' % (self.dir_save, str(name))
        self.logger.debug('save_loc=%s' % filename)

        # lambda function is used to serialize custom Features object
        with open(filename, 'w') as outfile:
            json.dump(metadata, outfile, default=lambda o: o.__dict__, indent=4)

        self.logger.debug('Exit:saveClusterMetadata')

    def parseFrame(self, frame):
        """Extract and save data from a single given frame.

        Args:
            frame: waymo open dataset Frame with loaded data

        """
        self.logger.debug('Entr:parseFrame')
        pcl, bboxes = self.waymo_converter.unpack_frame(frame)
        pcl = self.filterPcl(pcl)
        clusters = self.clusterByBBox(pcl, bboxes)
        metadata = [self.computeClusterMetadata(c, bboxes[i])
            for i, c in enumerate(clusters.values()) if c is not None]
        self.saveClusterMetadata(metadata, frame.context.name)
        self.logger.debug('Exit:parseFrame')
        return

    def checkDataFile(self, frame):
        """Check if data file currently exists.

        Args:
            frame: waymo dataset Frame object whose context name is used to
                create filename for which to check

        """
        self.logger.debug('Entr:checkDataFile')
        file = '%s/%s.json' % (self.dir_save, frame.context.name)
        self.logger.debug('Show:frame_file=%s' % file)
        self.logger.debug('Exit:checkDataFile')
        return os.path.isfile(file)

    def run(self, data_file, overwrite=False):
        """Generate data for all scans in all .tfrecord files in dir.

        Args:
            data_file: str .tfrecord file to parse
            overwrite: Bool for overwriting already existing data

        Todo:
            put glob + directory stuff here

        """
        self.logger.debug('Entr:run')
        tfrecord = tf.data.TFRecordDataset(data_file, compression_type='')
        record_len = sum(1 for _ in tf.python_io.tf_record_iterator(data_file))
        self.logger.debug('tfrecord has len %s' % record_len)

        progress = []  # Store progress shown to avoid rounding duplicates
        for i, scan in enumerate(tfrecord):

            # Print percent complete
            percent = int(100 * i / record_len)
            if percent % 10 == 0 and percent not in progress:
                progress.append(percent)
                self.logger.info(
                    'STATUS UPDATE: tfrecord parse is %i%% percent complete.'
                    % percent)

            frame = self.waymo_converter.create_frame(scan)
            frame.context.name = '%s-%i' % (frame.context.name, i)

            # Parse frame if relevant json file doesn't already exist
            if self.checkDataFile(frame) and not overwrite:
                self.logger.info(
                    'frame %s is already parsed.' % str(frame.context.name))
            else:
                self.logger.info(
                    'frame_num=%i, frame_id=%s' \
                    % (i, str(frame.context.name)))
                self.parseFrame(frame)

        self.logger.info(
            'STATUS UPDATE: tfrecord parse is 100% percent complete.')
        self.logger.debug('Exit:run')
        return


class DatasetCreatorVis(DatasetCreator):
    """Class for visualizing DatasetCreator tasks with rviz."""

    def __init__(
            self, dir_load, dir_save, logger=None, dir_log=None,
            verbosity=None, visualize=0):
        """Initialize Ros components, DatasetCreator, visualize setting."""

        self.visualize = visualize
        self.ros_converter = Waymo2Ros()
        rp.init_node('dataset_creator_vis', disable_signals=True)
        self.marker_pub = rp.Publisher('/bboxes', MarkerArray, queue_size=1)
        self.pcl_pub = rp.Publisher('/pcl', PointCloud2, queue_size=1)
        DatasetCreator.__init__(
            self, dir_load=dir_load, dir_save=dir_save, logger=logger,
            dir_log=dir_log, verbosity=verbosity)
        self.logger.debug('Exit:__init__')

    def parseFrame(self, frame):
        """Extract and save data from a single given frame, viz if specified.

        If self.visualize is 1, shows original data.
        If self.visualize is 2, shows filtered data.
        If self.visualize is 3, shows clustered data.
        
        Args:
            frame: waymo open dataset Frame with loaded data

        """
        self.logger.debug('Entr:parseFrame')
        pcl, bboxes = self.waymo_converter.unpack_frame(frame)
        # Update visualize param
        self.visualize = int(rp.get_param("/visualize", self.visualize)) 

        if self.visualize == 1:
            self.pcl_pub.publish(
                self.ros_converter.convert2pcl(pcl))
            self.marker_pub.publish(
                self.ros_converter.convert2markerarray(bboxes))

        pcl = self.filterPcl(pcl)

        if self.visualize == 2:
            self.pcl_pub.publish(
                self.ros_converter.convert2pcl(pcl))
            self.marker_pub.publish(
                self.ros_converter.convert2markerarray(bboxes))

        clusters = self.clusterByBBox(pcl, bboxes)
        if self.visualize == 3:
            try:
                good_clusters = {k:v for k, v in clusters.iteritems() if v is not None}
                self.pcl_pub.publish(self.ros_converter.convert2pcl(
                    np.concatenate(good_clusters.values())))
                self.marker_pub.publish(self.ros_converter.convert2markerarray(
                    [b for b in bboxes if str(b.id) in good_clusters.keys()]))
            except:
                self.logger.warning("No pcl with count > 25 pts")

        metadata = [self.computeClusterMetadata(c, bboxes[i])
            for i, c in enumerate(clusters.values()) if c is not None]
        self.saveClusterMetadata(metadata, frame.context.name)
        self.logger.debug('Exit:parseFrame')
        return


if __name__ == "__main__":
    """Set up directory locations and create dataset."""

    enable_rviz = rp.get_param("/enable_rviz", False)

    user = 'cnovak'
    loc_pkg = '/home/cnovak/Workspaces/catkin_ws/src/ml_person_detection'
    dataset = 'training_0000'

    args_default = {
        'dir_load' : '/home/%s/Data/waymo-od/%s' % (user, dataset),
        'dir_log' : '%s/logs' % loc_pkg,
        'dir_save' : '%s/data/%s' % (loc_pkg, dataset),
        'verbosity' : logging.DEBUG,
        'visualize' : 1
    }
    print(args_default)

    if enable_rviz:
        dir_load = rp.get_param("/dir_load", args_default['dir_load'])
        dir_log = rp.get_param("/dir_log", args_default['dir_log'])
        dir_save = rp.get_param("/dir_save", args_default['dir_save'])
        verbosity = int(rp.get_param("/verbosity", args_default['verbosity']))
        visualize = int(rp.get_param("/visualize", args_default['visualize']))

        creator = DatasetCreatorVis(
            dir_load=dir_load, dir_save=dir_save, dir_log=dir_log,
            verbosity=verbosity, visualize=visualize)

    else: 
        dir_load = args_default['dir_load']
        dir_log = args_default['dir_log']
        dir_save = args_default['dir_save']
        verbosity = args_default['verbosity']
        visualize = args_default['visualize']

        creator = DatasetCreator(
            dir_load=dir_load, dir_save=dir_save, dir_log=dir_log,
            verbosity=verbosity)
    
    creator.logger.info("enable_rviz = %s" % enable_rviz)
    file_list = glob.glob('%s/*.tfrecord' % dir_load)
    tfrecord_len = sum(1 for _ in file_list)
    creator.logger.info('Found %i tfrecord files' % tfrecord_len)

    for i, f in enumerate(file_list):

        # Print percent complete
        creator.logger.info(
            'STATUS UPDATE: dataset parse is %i%% percent complete.'
            % int(100 * i / tfrecord_len))

        creator.run(f)
