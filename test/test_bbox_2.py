#!usr/bin/env python2

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from matplotlib.patches import Rectangle
from scripts.modules.waymo2ros import Waymo2Numpy


def unpack_one_frame(tfrecord_loc):
    """Returns unpackings of first frame of tfrecord."""

    converter = Waymo2Numpy()
    tfrecord = tf.data.TFRecordDataset(tfrecord_loc, compression_type='')

    for scan in tfrecord:
        frame = converter.create_frame(scan)
        break

    return converter.unpack_frame(frame)

if __name__ == "__main__":

    data_dir = '/home/cnovak/Data/waymo-od'
    data_file = 'segment-15578655130939579324_620_000_640_000' \
         + '_with_camera_labels.tfrecord'
    pcl, bboxes = unpack_one_frame('%s/%s' % (data_dir, data_file))

    label = bboxes[0]
    angle = label.box.heading
    angle = np.pi
    print('pcl shape init: %i rows, %i cols' % pcl.shape)
    rot_mat = np.array(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    print('rot_mat shape init: %i rows, %i cols' % rot_mat.shape)
    rot_pcl = np.matmul(pcl[:, 0:2], rot_mat)
    print('rot_pcl shape init: %i rows, %i cols' % rot_pcl.shape)
    pcl_sortx = rot_pcl[rot_pcl[:,0].argsort()]
    print('pcl_sortx shape init: %i rows, %i cols' % pcl_sortx.shape)
    pcl_sorty = pcl_sortx[pcl_sortx[:,1].argsort(kind='mergesort')]
    print('pcl_sorty shape init: %i rows, %i cols' % pcl_sorty.shape)
    indexes = np.where(
        (0 < pcl_sorty[:,0]) & (pcl_sorty[:,0] < 1)
        & (0 < pcl_sorty[:,1]) & (pcl_sorty[:,1] < 1))[0]
    print('number of points found: %i' % len(indexes))
    for i in indexes:
        print(pcl_sorty[i])
