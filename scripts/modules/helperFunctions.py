"""Module for shared code between waymo pipeline and panasonic pipeline.

This module holds functions that are used in both pipelines:
    remove_groundplane()
    extract_pcl_features()
"""
import numpy as np

GROUND_THRESHOLD = 1  # meters

def remove_groundplane(pcl, z_thresh=GROUND_THRESHOLD):
    """Remove points below z-threshold and return pcl."""
    return pcl[pcl[:,2] > z_thresh]

def extract_cluster_features(cluster, bbox):
    """Calculate features for net training from pcl, return PclFeatures object
    containing all features.

    Args:
        cluster: ndarray (n * 3) of cluster points.
        bbox: waymo object label output

    Returns:
        features: Features object containing cluster features

    """

    id = bbox.id
    cls = bbox.type
    cnt = cluster.shape[0]

    # Compute eigenvalues along all three axes
    xyz_cov = np.cov(np.transpose(cluster))
    e_x, e_y, e_z = np.linalg.eigvals(xyz_cov)

    # Compute volume and point density
    vol = bbox.box.width * bbox.box.height
    density = cluster.shape[0]/vol

    output = Features(id, cls, cnt, e_x, e_y, e_z, vol, density)
    return output

class Features(object):
    """Class to store features of a cluster

    Attributes:
        cluster_id: unique unicode ID for cluster
        e_x = eigenvalue along x axis
        e_y = eigenvalue along y axis
        e_z = eigenvalue along z axis
        vol = volume of bounding box
        density = point density of cluster (num pts / volume)

    """
    def __init__(self, cluster_id, cls, cnt, e_x=0, e_y=0, e_z=0, vol=0, density=0):
        self.cluster_id = cluster_id
        self.cls = cls 
        self.cnt = cnt
        self.e_x = e_x
        self.e_y = e_y
        self.e_z = e_z
        self.vol = vol
        self.density = density

    def __str__(self):
        return "Cluster ID: "            + str(self.cluster_id) + "\n" \
               "Eigenvalue X: "          + str(self.e_x)        + "\n" \
               "Eigenvalue Y: "          + str(self.e_y)        + "\n" \
               "Eigenvalue Z: "          + str(self.e_z)        + "\n" \
               "Bounding box volume: "   + str(self.vol)        + "\n" \
               "Cluster point density: " + str(self.density)    + "\n" \
               "Count: "                 + str(self.cnt)        + "\n" \
               "Class: "                 + str(self.cls)        + "\n"
