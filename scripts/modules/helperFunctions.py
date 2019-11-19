"""Module for shared code between waymo pipeline and panasonic pipeline.

This module holds functions that are used in both pipelines:
    remove_groundplane()
    extract_pcl_features()
"""

def remove_groundplane(pcl, ground_height):
    """Remove points in ground plane and return pcl."""
    pass

def extract_pcl_features(pcl):
    """Calculate features for net training from pcl, return in some format."""
