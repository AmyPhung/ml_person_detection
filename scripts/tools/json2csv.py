#!/usr/bin/env python3.6
"""Function to concatenate multiple json data files into csv.

    Todo:
        Setup logging, add verbosity arg

    Usage:
        1. navigate to .../ml_person_detection/scripts/tools
        2. run `./json2csv -h` for more information

"""
# Standard library imports
import argparse
import csv
import json
import pdb
import re  # Regex used to get frame index from .json filename

# Local imports
try:
    import modules.helperFunctions as hf
    from argparseActions import VerifyPathDirAction, CheckPathFileExistsAction
# Remind user to check their python virtual environment if import fails
except (ImportError, ModuleNotFoundError):
    print("REMINDER: Did you activate your python3 virtual environment?")
    raise


def json2csv(json_dir, csv_file, overwrite=False):
    """Add .json files from 1+ tfrecords into one .csv.

    Args:
        json_dir (pathlib.Path): folder of .json files
        csv_file (pathlib.Path): path to .csv file
        overwrite (bool): overwrite file @ csv_file

    """
    # Generate list of field names to pull from json files
    dict_data = []
    feature_obj = hf.Features()
    fieldnames = ['frame_index', 'cluster_id', 'cnt', 'cls']\
        + list(feature_obj.key.keys())

    # Populate dict_data from all .json files
    print('Compiling data from json files')
    for f in json_dir.glob("*.json"):
        frame_index = int(re.search('-([0-9]+).json$', f.name).group(1))

        with open(f, 'r') as read_file:
            data_read = json.load(read_file)
            for d in data_read:
                d['frame_index'] = frame_index
            dict_data = dict_data + [d for d in data_read]

    # Write dict_data to csv
    print('Beginning write operation')
    try:
        with csv_file.open('w') as io_obj:
            writer = csv.DictWriter(io_obj, fieldnames=fieldnames)
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

    print(f"Read .json files from:\n {json_dir}")
    print(f"Saved .csv file as:\n {csv_file}")
    return


if __name__ == "__main__":
    """Define cli interface with argparse; run json2csv()."""

    description = "Extract clusters from <data_dir>/*.json files "\
        "and concatenate into <csv_file>"

    cli = argparse.ArgumentParser(description=description)

    # Mandatory arguments
    cli.add_argument('data_dir', action=VerifyPathDirAction, type=str,
        help="folder containing .json files created by CreateDataset.py")
    cli.add_argument('csv_file', action=CheckPathFileExistsAction, type=str,
        help=".csv file to create.")

    # Optional arguments
    cli.add_argument('-o', '--overwrite', action='store_true', default=False,
        help="Overwrite <cs_file> if already exists")

    args = cli.parse_args()
    if args.csv_exists and not args.overwrite:
        raise FileExistsError(args.csv_file)

    json2csv(args.data_dir, args.csv_file, overwrite=args.overwrite)
