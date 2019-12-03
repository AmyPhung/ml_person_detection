#!/usr/bin/env python2

import csv
import glob
import json
import os
import re

def json2csv(json_dir, csv_dir, csv_name):
    """Add .json files from 1+ tfrecords into one .csv.

    Args:
        dir: str directory within which json files are stored.
        tfrecord: str id of tfrecord (json files stored as <tfrecord>-#.json

    """

    dict_data = []
    fieldnames = [
        'frame_index', 'cluster_id', 'cnt', 'cls',
        'density', 'vol', 'e_x', 'e_y', 'e_z',
        'max_intensity', 'avg_intensity', 'var_intensity']

    print('%s/*.json' % json_dir)
    # Populate dict_data from all .json files
    for f in glob.glob('%s/*.json' % json_dir):
        frame_index = int(re.search('-([0-9]+).json$', f).group(1))

        with open(f, 'r') as read_file:
            data_read = json.load(read_file)
            for d in data_read:
                d['frame_index'] = frame_index
            dict_data = dict_data + [d for d in data_read]

    # Write dict_data to csv
    try:
        print('%s/%s.csv' % (csv_dir, csv_name))
        with open('%s/%s.csv' % (csv_dir, csv_name), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in dict_data:
                data_row = {}
                data_row['frame_index'] = data['frame_index']
                data_row['cluster_id'] = data['cluster_id']
                data_row['cnt'] = data['cnt']
                data_row['cls'] = data['cls']
                data_row['density'] = data['parameters'][0]
                data_row['vol'] = data['parameters'][1]
                data_row['e_x'] = data['parameters'][2]
                data_row['e_y'] = data['parameters'][3]
                data_row['e_z'] = data['parameters'][4]
                data_row['max_intensity'] = data['parameters'][5]
                data_row['avg_intensity'] = data['parameters'][6]
                data_row['var_intensity'] = data['parameters'][7]
                # print(data_row)
                writer.writerow(data_row)

    except IOError:
        print("I/O error")

if __name__ == "__main__":

    csv_name = 'validation_0000'
    data_dir = '/home/cnovak/Workspaces/catkin_ws/src/ml_person_detection/data'
    json_dir = '%s/%s' % (data_dir, csv_name)
    json2csv(json_dir, json_dir, csv_name)
