"""Module for shared code between waymo pipeline and panasonic pipeline.

This module holds functions that are used in both pipelines:
    remove_groundplane()
    extract_pcl_features()
"""
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt

GROUND_THRESHOLD = 1  # meters

def remove_groundplane(pcl, z_thresh=GROUND_THRESHOLD):
    """Remove points below z-threshold and return pcl."""
    return pcl[pcl[:,2] > z_thresh]

def extract_cluster_parameters(cluster, display=False):
    """Calculate features for net training from pcl, return PclFeatures object
    containing all features.

    Args:
        cluster: ndarray (n * 3) of cluster points.

    Returns:
        parameters: list of parameters [e_x, e_y, e_z, vol, density]
            e_x = eigenvalue along x axis
            e_y = eigenvalue along y axis
            e_z = eigenvalue along z axis
            vol = volume of bounding box
            density = point density of cluster (num pts / volume)
    """

    # Compute eigenvalues along all three axes
    xyz_cov = np.cov(np.transpose(cluster))
    e_x, e_y, e_z = np.linalg.eigvals(xyz_cov)

    # Compute volume and point density
    vol = compute_volume(cluster, display)
    density = cluster.shape[0]/vol

    output = [e_x, e_y, e_z, vol, density]
    return output

def compute_volume(points, display=False):
    """Approximate 3D volume of bounding box by computing 2d convex hull in x-y
    plane then multiplying hull area by height
    """

    pts_xy = points[:,[0,1]] # Get x-y points
    pts_z = points[:,[2]] # Get z points
    hull = ConvexHull(pts_xy)
    area = hull.area
    height = pts_z.max() - pts_z.min()
    volume = area*height

    if display:
        plt.plot(points[:,0], points[:,1], 'o')
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

        plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
        plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
        plt.show()

    return volume


class Features(object):
    """Class to store features of a cluster

    Attributes:
        cluster_id: unique unicode ID for cluster
        cls: class ID of feature
        cnt: number of points
        parameters: list of parameters [e_x, e_y, e_z, vol, density]
            e_x = eigenvalue along x axis
            e_y = eigenvalue along y axis
            e_z = eigenvalue along z axis
            vol = volume of bounding box
            density = point density of cluster (num pts / volume)

    """
    def __init__(self, cluster_id=None, cls=None, cnt=None, e_x=0, e_y=0, e_z=0, vol=0, density=0):
        self.cluster_id = cluster_id
        self.cls = cls
        self.cnt = cnt
        self.parameters = [e_x, e_y, e_z, vol, density]

    def __str__(self):
        return "Cluster ID: "                + str(self.cluster_id)    + "\n" \
               "Point Count: "               + str(self.cnt)           + "\n" \
               "Class: "                     + str(self.cls)           + "\n" \
               "Paramters:"                                            + "\n" \
               "    Eigenvalue X: "          + str(self.parameters[0]) + "\n" \
               "    Eigenvalue Y: "          + str(self.parameters[1]) + "\n" \
               "    Eigenvalue Z: "          + str(self.parameters[2]) + "\n" \
               "    Bounding box volume: "   + str(self.parameters[3]) + "\n" \
               "    Cluster point density: " + str(self.parameters[4]) + "\n"
