#!/usr/bin/env python3
"""Function to concatenate multiple json data files into csv.

    Todo:
        Convert to argparse and pathlib

    Usage:
        1. navigate to .../m_person_detection/scripts
        2. `python2 -m modules.json2csv <json dir in ml_person_detection/data>
"""

try:
    import csv
    import glob
    import json
    import os
    import re
    import sys

    from modules.helperFunctions import Features

# Remind user to check their python virtual environment if import fails
except ImportError:
    print("REMINDER: Did you activate your python3 virtual environment?")
    raise


def json2csv(json_dir, csv_dir, csv_name):
    """Add .json files from 1+ tfrecords into one .csv.

    Args:
        json_dir: str directory within which json files are stored.
        csv_dir: str directory within which to store csv file.
        csv_name: str filename under which to save csv file.

    """

    # Generate list of field names to pull from json files
    dict_data = []
    feature_obj = Features()
    fieldnames = ['frame_index', 'cluster_id', 'cnt', 'cls']\
        + list(feature_obj.key.keys())

    # Populate dict_data from all .json files
    print('Compiling data from json files')
    for f in glob.glob('%s/*.json' % json_dir):
        frame_index = int(re.search('-([0-9]+).json$', f).group(1))

        with open(f, 'r') as read_file:
            data_read = json.load(read_file)
            for d in data_read:
                d['frame_index'] = frame_index
            dict_data = dict_data + [d for d in data_read]

    # Write dict_data to csv
    print('Beginning write operation')
    try:
        with open('%s/%s.csv' % (csv_dir, csv_name), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in dict_data:
                data_row = {}

                # 1-1 map all components that aren't parameters
                for key in data.keys():
                   if not key == 'parameters':
                       data_row[key] = data[key] 

                # 1-1 map all sub-components of parameters using Feature key
                for key, index in feature_obj.key.items():
                    data_row[key] = data['parameters'][index]

                data_row = {
                    k : v for k, v in data_row.items() if k in fieldnames}
                writer.writerow(data_row)

    except IOError:
        print("I/O error")

    print('Read json files from:\n %s' % json_dir)
    print('Saved csv to:\n %s/%s.csv' % (csv_dir, csv_name))

if __name__ == "__main__":

    # Only aspect that is computer-specific
    CATKIN_DIR_LOC = '/home/cnovak/Workspaces/catkin_ws'

    # Arg 1 is both name of csv and data directory within workspace
    if len(sys.argv) == 2:
        csv_name = sys.argv[1]
        data_dir = '%s/src/ml_person_detection/data' % CATKIN_DIR_LOC
        json_dir = '%s/%s' % (data_dir, csv_name)

        # Check existence of json_dir
        if os.path.isdir(json_dir):
            json2csv(json_dir, data_dir, csv_name)
        else:
            print('%s is not a valid directory.' % json_dir)

    else:
        print('Usage: python json2csv <json file dir (in workspace data dir)>')
        print('   ex. python json2csv training_0000')
