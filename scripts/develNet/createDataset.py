#!usr/bin/env python
import sys # Needed for relative imports
sys.path.append('../') # Needed for relative imports

import datetime
import glob
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
 
    # Rotate pcl by bbox heading angle
    ang = np.radians(bbox.box.heading)
    r_mat = np.array(((np.cos(ang), np.sin(ang)), (-np.sin(ang), np.cos(ang))))
    r_pcl = np.matmul(pcl[:, 0:2], r_mat)
    r_pcl = np.append(r_pcl, np.expand_dims(pcl[:, 2], 1), axis=1)

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

    def __init__(self, load_dir, save_dir, logger=None, log_dir=None, verbosity=None):
        """Provide directory location to find frames."""
        self.waymo_converter = Waymo2Numpy()
        self.load_dir = load_dir
        self.save_dir = save_dir

        # Set up logger if not given as arg
        if logger is not None:
            self.logger = logger
        else:
            # Generate log file name
            d = datetime.datetime.now()
            filename = ('%s/%i-%i-%i-%i-%i.log'
                        % (log_dir, d.year, d.month, d.day, d.hour, d.minute))

            # Create logger with file and stream handlers
            self.logger = logging.getLogger('datasetCreator')
            sh = logging.StreamHandler(sys.stdout)
            fh = logging.FileHandler(filename)

            # Format file and stream logging
            formatter = logging.Formatter(
                '%(asctime)s %(levelname)s %(message)s')
            fh.setFormatter(formatter)
            sh.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.addHandler(sh)

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
        self.logger.info("Show:bbox_count: %i" % len(bboxes))

        for i, label in enumerate(bboxes):

            cluster = get_pts_in_bbox(pcl, label, self.logger) 
            self.logger.info(
                "bbox=%i * %i, class=%i, id=%s, pt_count=%i"
                % (i, len(bboxes), label.type, label.id, len(cluster)))

            # Threshold cluster size
            if len(cluster) >= thresh:
                obj_pcls[label.id] = cluster
            else:
                obj_pcls[label.id] = None
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
        filename = '%s/%s.json' % (self.save_dir, str(name))
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
        file = '%s/%s.json' % (self.save_dir, frame.context.name)
        self.logger.debug('Show:frame_file=%s' % file)
        self.logger.info('Exit:checkDataFile')
        return os.path.isfile(file)

    def run(self, data_file, overwrite=False):
        """Generate data for all scans in all .tfrecord files in dir.

        Args:
            data_file: str .tfrecord file to parse
            overwrite: Bool for overwriting already existing data

        Todo:
            put glob + directory stuff here

        """
        self.logger.info('Entr:run')
        #'frames'
        tfrecord = tf.data.TFRecordDataset(data_file, compression_type='')
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
    user = 'cnovak'
    home_dir = '/home/%s' % user
    catkin_ws = 'catkin_ws/src/ml_person_detection'

    visualize = False
    load_dir = '%s/Data/waymo-od' % home_dir
    #load_dir = '/home/amy/test_ws/src/waymo-od/tutorial/'
    save_dir = '%s/Workspaces/%s/data/train' % (home_dir, catkin_ws)
    data_file = 'segment-15578655130939579324_620_000_640_000' \
         + '_with_camera_labels.tfrecord'
    log_dir = '%s/Workspaces/%s/logs' % (home_dir, catkin_ws)

    creator = DatasetCreatorVis() if visualize \
        else DatasetCreator(
            load_dir=load_dir, save_dir=save_dir, log_dir=log_dir,
            verbosity=logging.DEBUG)
    
    for f in glob.glob('%s/*.tfrecord' % load_dir):
        creator.run(f)  # TODO Setup directory choosing
