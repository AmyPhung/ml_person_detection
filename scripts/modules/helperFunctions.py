"""Module for shared code between waymo pipeline and panasonic pipeline.

This module holds functions that are used in both pipelines:
    remove_groundplane()
    extract_pcl_features()
"""

GROUND_THRESHOLD = 1  # meters

def remove_groundplane(pcl, z_thresh=GROUND_THRESHOLD):
    """Remove points below z-threshold and return pcl."""
    return pcl[pcl[:,2] > z_thresh]

def extract_cluster_features(cluster):
    """Calculate features for net training from pcl, return in some format.

    Args:
        cluster: ndarray (n * 3) of xyz points.

    Returns:
        ???

    """
    pass
