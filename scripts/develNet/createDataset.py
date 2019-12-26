#!/usr/bin/env python2
"""Module containing classes for creating and visualizing datasets.

Parses datasets from waymo-od and turns them into directories of json files,
where each json file contains lists of features representing point clusters
segmented from full point clouds based on waymo-od bounding boxes. More info
on the features used for cluster definition can be found on github.

Examples:
    To run without visualization:
        cd .../ml_person_detection/scripts
        python2 -m develNet.createDataset

    To run with rviz:
        roslaunch ml_person_detection createDataset.launch

"""
import datetime
import glob
import json
import logging
import os.path
import pdb
import sys

import rospy as rp
import numpy as np
import tensorflow as tf

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray

from modules.constants import *
from modules.helperFunctions import *
from modules.waymo2ros import Waymo2Numpy, Waymo2Ros


class DatasetCreator(object):
    """Class for creating labeled cluster metadata from raw waymo data.

    Use this class to iterate through .tfrecord files, pull out frames,
    extract clusters from the frames using provided bounding boxes,
    calculate features of clusters, and save resulting features and
    metadata to create a cleaned dataset.

    """

    def __init__(
            self, dir_load, dir_save=None, logger=None, dir_log=None,
            save_data=True, verbosity=None, density_thresh=0):
        """Provide directory location to find frames."""
        self.waymo_converter = Waymo2Numpy()
        self.dir_load = dir_load
        self.dir_save = dir_save
        self.density_thresh = density_thresh
        self.save_data = save_data

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
        self.logger.info("save_data: %s" % self.save_data)
        self.logger.debug('Exit:__init__')

    def filterPcl(self, pcl):
        """Remove groundplane from pcl."""
        self.logger.debug('Entr:filterPcl')

        pcl_out = remove_groundplane(
            np.array([list(pt) for pt in pcl]), z_thresh=GROUND_THRESHOLD)
        self.logger.debug('Show:pts_removed=%i' % (len(pcl) - len(pcl_out)))
        self.logger.debug('Exit:filterPcl')
        return pcl_out

    def clusterByBBox(self, pcl, bboxes, thresh=5):
        """Extract points from pcl within bboxes as clusters.

        Note:
            relatively small threshold kept to prevent math errors in
            subsequent feature/metadata computation

        Args:
            pcl: (n * 4) numpy array of xyz points and intensities
            bboxes: waymo pcl label output
            thresh: min point count for valid cluster

        Returns:
            valid_clusters: Hash map of bbox label : cluster as pcl
            valid_bboxes: List of bboxes with ids in valid_clusters.keys()

        """
        self.logger.debug('Entr:clusterByBBox')

        valid_clusters = {}  # Hash map of bbox label : pcl
        self.logger.debug("bbox initial count: %i" % len(bboxes))

        # Get thresholded cluster from each given bbox
        padding = (PADDING_X, PADDING_Y, PADDING_Z)
        for i, bbox in enumerate(bboxes):
            cluster = get_pts_in_bbox(pcl, bbox, padding=padding)
            self.logger.debug(
                "bbox=%i * %i, class=%i, id=%s, pt_count=%i"
                % (i, len(bboxes), bbox.type, bbox.id, len(cluster)))

            # Threshold cluster size
            if len(cluster) >= thresh:
                valid_clusters[bbox.id] = cluster
        
        valid_bboxes = [b for b in bboxes if b.id in valid_clusters.keys()]

        self.logger.debug(
            "bbox final count: %i"
            % len(valid_clusters))
        self.logger.debug('Exit:clusterByBBox')
        return valid_clusters, valid_bboxes

    def computeClusterMetadata(self, cluster, bbox, frame_id):
        """Compute key information from cluster to boil down pointcloud info.

        Args:
            cluster: list of xyz points and intensities within cluster
            bbox: waymo object label output
            frame_id: int of frame index into tfrecord

        Returns:
            features: Features object containing cluster features

        """
        self.logger.debug('Entr:computeClusterMetadata')

        if cluster is None:
            raise TypeError(
                'None passed as cluster - '
                + 'possibly a too-small cluster passed from clusterByBBox?')
        np_cluster = np.array(cluster)

        features = Features()
        features.cluster_id = bbox.id
        features.frame_id = frame_id
        features.cls = bbox.type
        features.cnt = cluster.shape[0]
        features.parameters = extract_cluster_parameters(np_cluster)

        self.logger.debug('Exit:computeClusterMetadata')
        return features

    def filterMetadata(self, metadata, clusters, thresh=20):
        """Remove clusters below density threshold from dataset.

        Args:
            metadata: list of Features objects containing key information about
                each cluster in frame
            clusters: Hash map of bbox label : pcl where the pcl contains
                (n * 4) numpy arrays of xyz points and intensities with None
                types removed
            thresh: minimum number of points per cubic meter

        Returns:
            filtered_metadata: list of Features objects which meet the minimum
                density requirement
            filtered_clusters: list of (n * 4) numpy arrays of xyz points and
                intensities corresponding with Features that meet the min
                density requirement

        """
        self.logger.debug('Entr:filterMetadata')
        filtered_metadata = []
        filtered_clusters = {}

        for i in range(len(metadata)):
            c = metadata[i]

            if c.parameters[7] > thresh:  # 7th parameter is density
                filtered_clusters[c.cluster_id] = clusters[c.cluster_id]
                filtered_metadata.append(c)
            # else:
            #     self.logger.warning("Sparse cluster detected in\
            #         filterMetadata")

        self.logger.debug('Exit:filterMetadata')
        return filtered_metadata, filtered_clusters

    def saveClusterMetadata(self, metadata, name):
        """Save cluster metadata from frame in a .json file.

        Note:
            Uses frame name as .json filename

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
                json.dump(
                    metadata, outfile, default=lambda o: o.as_dict(), indent=4)

        self.logger.debug('Exit:saveClusterMetadata')

    def parseFrame(self, frame, frame_id):
        """Extract and save data from a single given frame.

        Main function for filtering, clustering, and extracting features from
        pointclouds, follows process:
            1. Unpack pcl and bboxes from frame
            2. Filter frame (currently only removes groundplane)
            3. Cluster pcl by bboxes (
            4. Compute features for all bboxes that contain lidar points
            5. Downselect featuresets, filtering by certain parameters -
               (density, distance, raw pointcount, etc)
            6. Save sub-selected features in json files.

        Args:
            frame: waymo open dataset Frame with loaded data
            frame_id: index of waymo Frame in tfrecord

        """
        self.logger.debug('Entr:parseFrame')
        pcl, bboxes = self.waymo_converter.unpack_frame(frame)  # 1
        pcl = self.filterPcl(pcl)  # 2

        clusters, bboxes = self.clusterByBBox(pcl, bboxes)  # 3
        metadata = [self.computeClusterMetadata(clusters[b.id], b, frame_id)
                    for b in bboxes]  # 4

        metadata, clusters = self.filterMetadata(metadata, clusters)  # 5

        if self.save_data:
            self.saveClusterMetadata(metadata, frame.context.name)  # 6
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

    def run(self, data_file, file_number='', overwrite=False):
        """Generate data for all scans in all .tfrecord files in dir.

        Args:
            data_file: str .tfrecord file to parse
            file_number: optional str to print
            overwrite: Bool for overwriting already existing data

        Todo:
            put glob + directory stuff here

        """
        self.logger.debug('Entr:run')
        tfrecord = tf.data.TFRecordDataset(data_file, compression_type='')
        record_len = sum(1 for _ in tf.python_io.tf_record_iterator(data_file))
        self.logger.debug('Found %s frames in tfrecord' % record_len)

        progress = []  # Store progress shown to avoid rounding duplicates
        for i, scan in enumerate(tfrecord):

            # Transform raw waymo scan to numpy frame
            frame = self.waymo_converter.create_frame(scan)
            frame.context.name = '%s-%i' % (frame.context.name, i)

            # Print percent complete
            percent = int(100 * i / record_len)
            if percent % 10 == 0 and percent not in progress:
                progress.append(percent)
                self.logger.info(
                    'STATUS UPDATE: tfrecord %s parse is %i%% complete.'
                    % (file_number, percent))

            # Parse frame if relevant json file doesn't already exist
            if self.checkDataFile(frame) and not overwrite:
                self.logger.info(
                    'frame %i is already parsed.' % i)
            else:
                self.logger.info(
                    'frame #: %i, tfrecord id: %s'
                    % (i, str(frame.context.name)))
                self.parseFrame(frame, i)

        self.logger.info(
            'STATUS UPDATE: tfrecord parse is 100% percent complete.')
        self.logger.debug('Exit:run')
        return


class DatasetCreatorVis(DatasetCreator):
    """Class for visualizing DatasetCreator tasks with rviz."""

    def __init__(
            self, dir_load, dir_save, logger=None, dir_log=None,
            save_data=True, verbosity=None, visualize=0, density_thresh=0):
        """Initialize Ros components, DatasetCreator, visualize setting."""
        self.density_thresh = density_thresh
        self.visualize = visualize
        self.ros_converter = Waymo2Ros()
        rp.init_node('dataset_creator_vis', disable_signals=True)
        self.marker_pub = rp.Publisher('/bboxes', MarkerArray, queue_size=1)
        self.pcl_pub = rp.Publisher('/pcl', PointCloud2, queue_size=1)
        DatasetCreator.__init__(
            self, dir_load=dir_load, dir_save=dir_save, logger=logger,
            dir_log=dir_log, save_data=save_data, verbosity=verbosity,
            density_thresh=density_thresh)
        self.logger.debug('Exit:__init__')

    def pubData(self, pcl=None, bboxes=None):
        """Publish pointcloud and bounding boxes for rviz visualization."""
        self.pcl_pub.publish(self.ros_converter.convert2pcl(pcl))
        self.marker_pub.publish(self.ros_converter.convert2markerarray(bboxes))

    def parseFrame(self, frame, frame_id):
        """Extract and save data from a single given frame, viz if specified.

        Note:
            self.visualize attribute settings:
                1: shows original data.
                2: shows ground filtered data.
                3: shows clustered data.
                4: shows density filtered data.

        Args:
            frame: waymo open dataset Frame with loaded data
            frame_id: index of waymo Frame in tfrecord

        """
        self.logger.debug('Entr:parseFrame')
        self.visualize = int(
            rp.get_param("/visualize", self.visualize))
        self.density_thresh = int(
            rp.get_param("/density_thresh", self.density_thresh))

        pcl, bboxes = self.waymo_converter.unpack_frame(frame)  # 1

        if self.visualize == 1:
            self.pubData(pcl, bboxes)

        pcl = self.filterPcl(pcl)  # 2

        if self.visualize == 2:
            self.pubData(pcl, bboxes)

        clusters, bboxes = self.clusterByBBox(pcl, bboxes)  # 3

        if self.visualize == 3:
            if len(clusters) > 0:
                self.pubData(np.concatenate(clusters.values()), bboxes)
            else:
                self.logger.warning("No pcl with count > 10 pts")

        metadata = [self.computeClusterMetadata(
                    clusters[bbox.id], bbox, frame_id)
                    for bbox in bboxes]  # 4

        metadata, clusters = self.filterMetadata(
            metadata, clusters, self.density_thresh)  # 5

        # Show density filtered data if any clusters found
        if self.visualize == 4:
            if len(clusters) > 0:
                # Decided to plot full pointcloud since it makes it easier to
                # tell that nothing important is being removed. To only plot
                # points that are a part of the new sub-selected clusters,
                # uncomment this code
                # self.pcl_pub.publish(self.ros_converter.convert2pcl(
                #     np.concatenate(sub_clusters.values())))

                self.pubData(
                    pcl, [b for b in bboxes if str(b.id) in clusters.keys()])
            else:
                self.logger.warning("No pcl with density > 100 pts/m^3")

        if self.save_data:
            self.saveClusterMetadata(metadata, frame.context.name)  # 6
        self.logger.debug('Exit:parseFrame')
        return


if __name__ == "__main__":
    """Load settings from config file, enable rviz if necessary, etc."""

    # Load and unpack config parameters
    config_file = "/home/cnovak/Workspaces/catkin_ws/src/ml_person_detection/"\
                + "config/lapbot_config.json"
    with open(config_file, 'r') as file_in:
        config_params = json.load(file_in)

    enable_rviz = bool(config_params['enable_rviz'])
    enable_logging = bool(config_params['enable_logging'])
    save_data = bool(config_params['enable_datasave'])
    pkg_loc = config_params['pkg_loc']
    dataset = config_params['dataset']
    verbosity = int(config_params['verbosity'])
    dir_load = "%s/%s" % (config_params['data_loc'], dataset)

    # Initialize other parameters
    visualize = 2
    dir_log = "%s/logs" % pkg_loc
    dir_save = "%s/data/%s" % (pkg_loc, dataset)

    if enable_rviz:
        visualize = int(rp.get_param("/visualize", visualize))
        creator = DatasetCreatorVis(
            dir_load=dir_load, dir_save=dir_save, dir_log=dir_log,
            save_data=save_data, verbosity=verbosity, visualize=visualize)

    else:
        creator = DatasetCreator(
            dir_load=dir_load, dir_save=dir_save, dir_log=dir_log,
            save_data=save_data, verbosity=verbosity)

    creator.logger.info("enable_rviz = %s" % enable_rviz)

    # Get list of tfrecords in dataset
    file_list = glob.glob('%s/*.tfrecord' % dir_load)
    tfrecord_len = sum(1 for _ in file_list)
    creator.logger.info(
        'Found %i tfrecord files in dataset %s' % (tfrecord_len, dataset))

    # Process all tfrecords
    for i, f in enumerate(file_list):
        creator.logger.info(
            'STATUS UPDATE: dataset parse is %i%% percent complete.'
            % int(100 * i / tfrecord_len))
        creator.run(f, i)
