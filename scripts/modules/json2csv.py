#!usr/bin/env python2

import csv
import glob
import json
import os
import re

def json2csv(dir, tfrecord):
    """Add .json files from 1+ tfrecords into one .csv.

    Args:
        dir: str directory within which json files are stored.
        tfrecord: str id of tfrecord (json files stored as <tfrecord>-#.json

    """

    dict_data = []
    fieldnames = [
        'frame_index', 'cluster_id', 'cnt', 'cls', 'density', 'vol', 'e_x', 'e_y', 'e_z']

    # Populate dict_data from all .json files
    for f in glob.glob('%s/*.json' % dir):
        frame_index = int(re.search('-([0-9]+).json$', f).group(1))

        with open(f, 'r') as read_file:
            data_read = json.load(read_file)
            for d in data_read:
                d['frame_index'] = frame_index
            dict_data = dict_data + [d for d in data_read]

    # Write dict_data to csv
    try:
        with open('%s.csv' % tfrecord, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error") 

if __name__ == "__main__":

    json2csv(
        '/home/cnovak/Workspaces/catkin_ws/src/ml_person_detection/data/train',
        '15578655130939579324_620_000_640_000')
