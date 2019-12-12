#!/usr/bin/env python

import sys # Needed for relative imports
sys.path.append('../') # Needed for relative imports

import datetime
import glob
import json
import logging
import os.path
import pdb

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

    def clusterByBBox(self, pcl, bboxes, thresh=5):
        """Extract points from pcl within bboxes as clusters.

        Args:
            pcl: (n * 4) numpy array of xyz points and intensities
            bboxes: waymo pcl label output
            thresh: min int point num for cluster (relatively small threshold -
                kept to prevent math errors in metadata computation)

        Returns:
            obj_pcls: Hash map of bbox label : pcl where the pcl contains
                (n * 4) numpy arrays of xyz points and intensities
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
                'None passed as cluster - ' \
                + 'possibly a too-small cluster passed from clusterByBBox?')
        np_cluster = np.array(cluster)

        features = Features()
        features.cluster_id = bbox.id
        features.frame_id = frame_id
        features.cls = bbox.type
        features.cnt = cluster.shape[0]
        features.parameters = extract_cluster_parameters(np_cluster, display=False)

        self.logger.debug('Exit:computeClusterMetadata')
        return features

    def filterMetadata(self, metadata, clusters, thresh=0):
        """Removes clusters with a density smaller than the specified threshold
        from the dataset

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
        filtered_metadata = []
        filtered_clusters = {}

        for i in range(len(metadata)):
            c = metadata[i]

            # print("HEREEEEE Current density: " + str(c.parameters[7]))
            # print(clusters.keys())
            # print(c)
            if c.parameters[7] > thresh: # 7th parameter is density
                try: # Make sure cluster is valid
                    filtered_clusters[c.cluster_id] = clusters[c.cluster_id]
                    filtered_metadata.append(c)
                except:
                    self.logger.warning("Invalid cluster detected in\
                        filterMetadata")

        return filtered_metadata, filtered_clusters

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
            json.dump(metadata, outfile, default=lambda o: o.as_dict(), indent=4)

        self.logger.debug('Exit:saveClusterMetadata')

    def parseFrame(self, frame, frame_id):
        """Extract and save data from a single given frame.

        Args:
            frame: waymo open dataset Frame with loaded data
            frame_id: index of waymo Frame in tfrecord

        """
        self.logger.debug('Entr:parseFrame')
        pcl, bboxes = self.waymo_converter.unpack_frame(frame)
        pcl = self.filterPcl(pcl)

        clusters = self.clusterByBBox(pcl, bboxes)
        metadata = [self.computeClusterMetadata(clusters[b.id], b, frame_id)
            for b in bboxes if clusters[b.id] is not None]

        sub_metadata = self.filterMetadata(metadata)

        # TODO: update this portion
        # clusters = self.clusterByBBox(pcl, bboxes) # remove threshold here
        # metadata = [self.computeClusterMetadata(c, bboxes[i])
        #     for i, c in enumerate(clusters.values()) if c is not None]
        # # subselect metatada here + remove  def subselect metadata

        self.saveClusterMetadata(sub_metadata, frame.context.name)
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
            print(frame.context.name)

            # Print percent complete
            percent = int(100 * i / record_len)
            if percent % 10 == 0 and percent not in progress:
                progress.append(percent)
                self.logger.info(
                    'STATUS UPDATE: tfrecord %s parse is %i%% percent complete.'
                    % (file_number, percent))

            # Parse frame if relevant json file doesn't already exist
            if self.checkDataFile(frame) and not overwrite:
                self.logger.info(
                    'frame %i is already parsed.' % i)
            else:
                self.logger.info(
                    'frame #: %i, tfrecord id: %s' \
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

    def parseFrame(self, frame, frame_id):
        """Extract and save data from a single given frame, viz if specified.

        If self.visualize is 1, shows original data.
        If self.visualize is 2, shows ground filtered data.
        If self.visualize is 3, shows clustered data.
        If self.visualize is 4, shows density filtered data.

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
        valid_clusters = {k:v for k, v in clusters.iteritems() if v is not None}


        if self.visualize == 3:
            try:
                self.pcl_pub.publish(self.ros_converter.convert2pcl(
                    np.concatenate(valid_clusters.values())))
                self.marker_pub.publish(self.ros_converter.convert2markerarray(
                    [b for b in bboxes if str(b.id) in valid_clusters.keys()]))
            except:
                self.logger.warning("No pcl with count > 10 pts")

        metadata = [self.computeClusterMetadata(c, bboxes[i], frame_id)
            for i, c in enumerate(clusters.values()) if c is not None]


        # TODO: Why do these two lists not contain the same IDs?
        print("valid cluster ids:" )
        print(valid_clusters.keys())
        print("metadata cluster ids:")
        for m in metadata:
            print(m.cluster_id)

        print("Total valid clusters: " + str(len(valid_clusters.keys())))
        print("Total metadata clusters: " + str(len(metadata)))

        """
        Sample output:

        valid cluster ids:
        [u'Ee_mzXJk_7iLoXOaubXynQ', u'lnsabxPnpxTwiSi6IZEC1A', u'mnbLrvlccsSxMbu3XZNcDg', u'MkgmvM5VY_Lp2EgxC2P_ZQ', u'2Xi-0d8Aw3n0zvFYT1yBxw', u'l89q4jEMsn39QEdf2qiTBg', u'Cn3AHZsEsTR5eFwUVgS7yw', u'HZA66xiTbbB_LZq-CdrDcw', u'qTmnVvSv7PNTxt-pK68UNA', u'AeRTj_4L-IdNkRHBoaj7xw', u'JjoUxXitmuYzQadOForLtQ', u'h-GOOiFizr4rHNMEq-nDkg', u'koXKDJzVChoS53XEwBPdfQ', u'D_yCrDP-M2h7r3RJPeJErA', u'tGbPZO2CJMGfHRSWqTq0yA', u'6MGT9Y1mbsZwOS80b4Nihg', u'nOjOHrs4psv_kOOB1A9wSA', u'qnHESxAiWHIZLW5nEcmMqQ', u'HVJHTVH1der6FWNfiEOUNg', u'5ekNqLg-YWWxZcE_5W4OSw', u'aBTbUSi037znY-hNa7J_ZQ', u'rpjJ0hqID5DalI0FBuE8lg', u'sVbb3pdkOuBU4CC4WUf-WQ', u'PrYtSvNA5qqMLZ1E4k7O8A', u'UndoHYW31MNWIg3056tqjg', u'Bg21XWEoxlKNz8_uhOuSbA', u'dNUPyq3wJfqiWEnVJsG2wg', u'UdhFOF90TBvuoplJ-yfCgA', u'T1mSZvmnInoXj4h2P6ugSg', u'bDFGJg0v-IaCSQmgaLzqfg', u'yNj-w6wKrkPE1niRGMymbQ', u'EEuGIMo89Q9MnS18HiKpmA', u'rKcEyPegJ1b1r2M9wDxJBA', u'IrV3-FLGr8Nl_7RKeAzigQ', u'HTJcmuLmTqvSr8kzyUio0g', u'w47T-KDS56vv_IlCmsDatQ', u'ClHSVxr_2Pj5JDTfahyyXw', u'NETnuoBC5IO1BeZZt_MkVQ', u'Gd2FkUNKD4yB5kJTYr3TNw', u'MeH1O9RQFGHyoC25Hi86gw', u'ixq5K8a5evhEqL9ATBolJw', u'OKdrreKQT9c_1txnTr1Tkg', u'qx8ZV6fDfH-qqkskex8Ogg', u'hkk3ahhL1r_CJ1xSUTmdGA', u'FioWhlJH7f5PuWLxMqrsnQ', u'pYZdovhB996HaQEUKm6kdA', u'OzZ8pBBiaKzvQysz7VifJw', u'mrkdGECF9WShrOBsvst4dg', u'ocVlLOdYh4aAh_zH3zycdw', u'25mllH8xnkbHMz8ypA6USw', u'NB_hUtXtt4VOirzjhqRjnQ', u'768k9LP92SMYvc7YTmiqYw', u'81yJQbo71X5IfiRekHFQZw', u'hTKc5ndZ02BmUAohcYn9Lw', u'VkkTxXmdNEeet6dgbBpGwg', u'WFEYwqHicAbq18aFP3P1rA']
        metadata cluster ids:
        25mllH8xnkbHMz8ypA6USw
        2Xi-0d8Aw3n0zvFYT1yBxw
        5b753dQapj6LPZu_2dJmJQ
        5ekNqLg-YWWxZcE_5W4OSw
        6eh1lZLIqOTWzjQtadJgmw
        768k9LP92SMYvc7YTmiqYw
        7dv07Yp5qurr3HirRzkUPQ
        81yJQbo71X5IfiRekHFQZw
        A-3WGJyCCPxK03vCp2zFUw
        Cn3AHZsEsTR5eFwUVgS7yw
        EEuGIMo89Q9MnS18HiKpmA
        FioWhlJH7f5PuWLxMqrsnQ
        FmwAL8gbB0kl5-izli8JzA
        Gd2FkUNKD4yB5kJTYr3TNw
        HTJcmuLmTqvSr8kzyUio0g
        HVJHTVH1der6FWNfiEOUNg
        HZA66xiTbbB_LZq-CdrDcw
        IrV3-FLGr8Nl_7RKeAzigQ
        JNrGOzbbSuTnFQeq_HquIw
        LWNa-ZgAWOY2o-jV_2TlQg
        MBmisKXDLINxWLTsgPZyCQ
        MkgmvM5VY_Lp2EgxC2P_ZQ
        OKdrreKQT9c_1txnTr1Tkg
        Oa3KHu0Ae10jLCA9BkUBew
        OzZ8pBBiaKzvQysz7VifJw
        PrYtSvNA5qqMLZ1E4k7O8A
        T1mSZvmnInoXj4h2P6ugSg
        TX4wi84fc2JlNuaAPr6F6Q
        TXn3ZnJHgTg4-H_ewLh8aw
        VaJtUuZ7fG9DVF36TEJvkQ
        VkkTxXmdNEeet6dgbBpGwg
        W2lA3Pu1vLlixgSTy-fyiA
        WNpDz5Y5bPqgFU-OezdIYw
        YtSf07TCKsmFN0O3S1pBzw
        Z5CfCVywESjYCEjN0LNmgQ
        bDFGJg0v-IaCSQmgaLzqfg
        c99aopGiqj8mkPhzBGV-pg
        fKeePMlAYqKN9N1rB2Dclw
        fL6kEwjIX99ZHD9gi1sHzA
        h-GOOiFizr4rHNMEq-nDkg
        hTKc5ndZ02BmUAohcYn9Lw
        hhwotDSyC2pfEoOLyL11Xw
        ixq5K8a5evhEqL9ATBolJw
        koXKDJzVChoS53XEwBPdfQ
        lnsabxPnpxTwiSi6IZEC1A
        nOjOHrs4psv_kOOB1A9wSA
        ocVlLOdYh4aAh_zH3zycdw
        pYZdovhB996HaQEUKm6kdA
        qTmnVvSv7PNTxt-pK68UNA
        qnHESxAiWHIZLW5nEcmMqQ
        qx8ZV6fDfH-qqkskex8Ogg
        rKcEyPegJ1b1r2M9wDxJBA
        vL92vZo_o1Njvmd8MbCYgw
        w47T-KDS56vv_IlCmsDatQ
        wUdoo5fYBB14sXPnyhr1MA
        yIOovB0l8xAA6AtitIrbpg

        """
        # sub_metadata, sub_clusters = \
        #     self.filterMetadata(metadata, valid_clusters)
        #
        # if self.visualize == 4:
        #     # try:
        #         # print(sub_clusters)
        #     self.pcl_pub.publish(self.ros_converter.convert2pcl(
        #         np.concatenate(sub_clusters.values())))
        #     self.marker_pub.publish(self.ros_converter.convert2markerarray(
        #         [b for b in bboxes if str(b.id) in sub_clusters.keys()]))
        #     # except:
        #         # self.logger.warning("No pcl with density > 100 pts/m^3")


        self.saveClusterMetadata(metadata, frame.context.name)
        self.logger.debug('Exit:parseFrame')
        return


if __name__ == "__main__":
    """Set up directory locations and create dataset."""

    enable_rviz = rp.get_param("/enable_rviz", False)

    user = 'amy'
    loc_pkg = '/home/amy/test_ws/src/ml_person_detection'
    dataset = 'training_0000'

    args_default = {
        'dir_load' : '/home/%s/test_ws/src/waymo-od/data' % (user),
        'dir_log' : '/home/%s/test_ws/src/waymo-od/logs' % (user),
        'dir_save' : '/home/%s/test_ws/src/waymo-od/save' % (user),
        'verbosity' : logging.DEBUG,
        'visualize' : 2
    }

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
    creator.logger.info(
        'Found %i tfrecord files in dataset %s' % (tfrecord_len, dataset))

    for i, f in enumerate(file_list):

        # Print percent complete
        creator.logger.info(
            'STATUS UPDATE: dataset parse is %i%% percent complete.'
            % int(100 * i / tfrecord_len))

        creator.run(f, i)
