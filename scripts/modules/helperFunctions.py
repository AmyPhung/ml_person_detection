#!/usr/bin/env python
"""Module for shared code between waymo pipeline and panasonic pipeline.

This module holds functions that are used in both pipelines:
    remove_groundplane()
    extract_pcl_features()
"""
import pdb
import waymo_open_dataset.label_pb2  # Imported for typechecking

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import resample
from scipy.spatial import ConvexHull, convex_hull_plot_2d

#TODO Move constants to constants.py file
GROUND_THRESHOLD = 0.1  # meters
MAX_CLUSTER_PTS = 200 # max number of points in a cluster

def remove_groundplane(pcl, z_thresh=GROUND_THRESHOLD):
    """Remove points below z-threshold inclusive and return pcl.

    Assumes pcl rows are [x, y, z, intensity]

    Args:
        pcl: (n * 4) numpy array of xyz points and intensities
        z_thresh: optional int|float threshold on z-axis

    Returns:
        pcl_out: pcl without points with z-values below or equal to threshold
    """
    if type(z_thresh) not in [int, float]:
        raise TypeError('received z_thresh arg of type %s' % type(z_thresh))
    if type(pcl) is not np.ndarray:
        raise TypeError('received pcl arg of type %s' % type(pcl))
    if pcl.shape[1] < 3:
        raise ValueError('pcl ndarray has too few rows')

    return pcl[pcl[:,2] > z_thresh]

def extract_cluster_parameters(cluster, display=False):
    """Calculate features for net training from pcl

    Args:
        cluster: ndarray (n * 4) of cluster points and intensities.

    Returns:
        parameters: list of parameters [e_x, e_y, e_z, vol, density]
            x = cluster COM along x axis
            y = cluster COM along y axis
            z = cluster COM along x axis
            e_x = eigenvalue along x axis
            e_y = eigenvalue along y axis
            e_z = eigenvalue along z axis
            vol = volume of bounding box
            density = point density of cluster (num pts / volume)
            max_intensity = maximum intensity in cluster
            mean_intensity = average intensity in cluster
            var_intensity = variance of intensity in cluster
    """
    if type(cluster) is not np.ndarray:
        raise TypeError('received cluster arg of type %s' % type(cluster))
    if cluster.shape[1] < 4:
        raise ValueError('cluster ndarray has too few rows')

    # Average x, y, and z to get COM of cluster
    x = np.mean(cluster[:,0])
    y = np.mean(cluster[:,1])
    z = np.mean(cluster[:,2])

    # Downsample to smaller number of points - objects close to lidar contain
    # significantly more points than ones further away
    if cluster.shape[0] > MAX_CLUSTER_PTS:
        cluster = resample(cluster, MAX_CLUSTER_PTS, t=None, axis=0, window=None)

    # Compute eigenvalues along all xyz axes
    xyz_cov = np.cov(np.transpose(cluster[:,:3]))
    e_x, e_y, e_z = np.linalg.eigvals(xyz_cov)

    # Compute volume and point density
    vol = compute_volume(cluster, display)
    density = cluster.shape[0]/vol

    # Compute maximum and mean object intensity
    max_intensity = np.max(cluster[:,3])
    mean_intensity = np.mean(cluster[:,3])
    var_intensity = np.var(cluster[:,3])

    output = [x, y, z, e_x, e_y, e_z, vol, density,
              max_intensity, mean_intensity, var_intensity]
    return output

def compute_volume(pcl, display=False):
    """Compute volume feature of pcl.
    
    Assumes pcl rows are [x, y, z, intensity]
    Uses volume attribute of ConvexHull as area due to stackoverflow post:
        https://stackoverflow.com/q/35664675
    Calculate approximate 3D volume of pcl with following steps:
        - Create convexhull encompassing pts in xy-plane
        - Multiply hull area by diff btw min-z and max-z points

    Args:
        pcl: (n * 3+) numpy array of xyz points and other information
        display: boolean for visualizing convexhull with pyplot

    """
    if type(pcl) is not np.ndarray:
        raise TypeError('received pcl arg of type %s' % type(pcl))
    if pcl.shape[1] < 3:
        raise ValueError('pcl ndarray has too few rows')

    pts_xy = pcl[:, [0,1]].astype(float)
    pts_z = pcl[:, [2]].astype(float)
    hull = ConvexHull(pts_xy)
    area = hull.volume  # See docstring notes 
    height = pts_z.max() - pts_z.min()
    volume = area * height

    if display:
        plt.plot(pcl[:,0], pcl[:, 1], 'bo')
        for simplex in hull.simplices:
            plt.plot(pcl[simplex, 0], pcl[simplex, 1], 'k-')

        plt.plot(pcl[hull.vertices, 0], pcl[hull.vertices,1], 'r--', lw=2)
        plt.plot(pcl[hull.vertices, 0], pcl[hull.vertices, 1], 'ro')
        plt.show()

    return volume

def get_pts_in_bbox(pcl, bbox, display=False):
    """Return ndarray of points from pcl within bbox.

    Given pointcloud and bounding box: 
        transforms pcl into bbox coordinate frame (translation & rotation)
        thresholds pcl by bbox dimensions, inclusive
        returns thresholded pcl in original coordinate system.

    Todo:
        add bbox padding arg and functionality.
        add bbox z-axis analysis functionality.

    Args:
        pcl: (n * 3+) ndarray 3d points with other information.
        bbox: tensorflow bounding box object to check for point collision.

    """
    if type(bbox) is not waymo_open_dataset.label_pb2.Label:
        raise TypeError('received bbox arg of type %s' % type(bbox))
    if type(pcl) is not np.ndarray:
        raise TypeError('received pcl arg of type %s' % type(pcl))
    if pcl.shape[1] < 3:
        raise ValueError('pcl ndarray has too few rows')

    # Unpack variables for readability
    x, y, z = bbox.box.center_x, bbox.box.center_y, bbox.box.center_z
    l, w, h = bbox.box.length, bbox.box.width, bbox.box.height

    # Get bbox limits in bbox coord frame
    x_lo, x_hi = x - l/2, x + l/2
    y_lo, y_hi = y - w/2, y + w/2

    ang = np.radians(bbox.box.heading)
    r_mat = np.array(((np.cos(ang), np.sin(ang)), (-np.sin(ang), np.cos(ang))))

    if display:  # Before translation - rotation
        points = np.matmul(np.asarray(
            [[-l/2, -w/2], [-l/2, w/2], [l/2, -w/2], [l/2, w/2],
            [0, 0]]), r_mat) + np.asarray([x, y])
        plt.plot(points[:, 0], points[:, 1], 'bo')
        plt.plot(pcl[:, 0], pcl[:, 1], 'ro')
        plt.show()

    t_mat = np.asarray([[x, y, z] for n in range(pcl.shape[0])])
    t_pcl = pcl[:, 0:3] - t_mat
    x, y, z = 0, 0, 0
    x_lo, x_hi, y_lo, y_hi = -l/2, l/2, -w/2, w/2
    r_pcl = np.matmul(t_pcl[:, 0:2], r_mat.T)

    if display:  # After translation - rotation
        points = np.asarray(
            [[-l/2, -w/2], [-l/2, w/2], [l/2, -w/2], [l/2, w/2],
            [x, y]])
        plt.plot(points[:, 0], points[:, 1], 'bo')
        plt.plot(r_pcl[:, 0], r_pcl[:, 1], 'ro')
        plt.show()

    # Sub-select pcl by bbox limits
    indxs = np.where(
        (x_lo <= r_pcl[:, 0]) & (r_pcl[:, 0] <= x_hi)
        & (y_lo <= r_pcl[:, 1]) & (r_pcl[:, 1] <= y_hi))[0]
    pcl_out = pcl[indxs].astype('float64')

    if display:  # After point selection
        points = np.matmul(np.asarray(
            [[-l/2, -w/2], [-l/2, w/2], [l/2, -w/2], [l/2, w/2],
            [0, 0]]), r_mat) + np.asarray([x, y])
        plt.plot(points[:, 0], points[:, 1], 'bo')
        plt.plot(pcl[:, 0], pcl[:, 1], 'ro')
        plt.plot(pcl_out[:, 0], pcl_out[:, 1], 'go')
        plt.show()

    return pcl_out


class Features(object):
    """Class to store features of a cluster

    Attributes:
        cluster_id: unique unicode ID for cluster
        cls: class ID of feature
        cnt: number of points
        key: dict of parameter variables : index in parameters
        parameters: list of parameters
            x = cluster COM along x axis
            y = cluster COM along y axis
            z = cluster COM along x axis
            e_x = eigenvalue along x axis
            e_y = eigenvalue along y axis
            e_z = eigenvalue along z axis
            vol = volume of bounding box
            density = point density of cluster (num pts / volume)
            max_intensity = maximum intensity in cluster
            mean_intensity = average intensity in cluster
            var_intensity = variance of intensity in cluster
    """
    def __init__(self, cluster_id=None, frame_id=None, cls=None, cnt=None,
                 x=0, y=0, z=0, e_x=0, e_y=0, e_z=0, vol=0, density=0,
                 max_intensity=0, mean_intensity=0, var_intensity=0):

        self.cluster_id = cluster_id
        self.frame_id = frame_id
        self.cls = cls
        self.cnt = cnt
        # Key to standardize self.parameters notation
        self.key = {
            'x' : 0, 'y' : 1, 'z' : 2, 'e_x' : 3, 'e_y' : 4, 'e_z' : 5,
            'vol' : 6, 'density' : 7,
            'max_intensity' : 8, 'mean_intensity' : 9, 'var_intensity' : 10}
        self.parameters = [x, y, z, e_x, e_y, e_z, vol, density,
                           max_intensity, mean_intensity, var_intensity]

    def __str__(self):
        return "Cluster ID: "                + str(self.cluster_id)    + "\n" \
               "Frame ID: "                  + str(self.frame_id)      + "\n" \
               "Point Count: "               + str(self.cnt)           + "\n" \
               "Class: "                     + str(self.cls)           + "\n" \
               "Parameters:"                                           + "\n" \
               "    COM x: "                 + str(self.parameters[0]) + "\n" \
               "    COM y: "                 + str(self.parameters[1]) + "\n" \
               "    COM z: "                 + str(self.parameters[2]) + "\n" \
               "    Eigenvalue X: "          + str(self.parameters[3]) + "\n" \
               "    Eigenvalue Y: "          + str(self.parameters[4]) + "\n" \
               "    Eigenvalue Z: "          + str(self.parameters[5]) + "\n" \
               "    Bounding box volume: "   + str(self.parameters[6]) + "\n" \
               "    Cluster point density: " + str(self.parameters[7]) + "\n" \
               "    Max Intensity: "         + str(self.parameters[8]) + "\n" \
               "    Average Intensity: "     + str(self.parameters[9]) + "\n" \
               "    Intensity Variance: "    + str(self.parameters[10]) + "\n"

    def as_dict(self):
        return {
            'cluster_id' : self.cluster_id,
            'frame_id' : self.frame_id,
            'cls' : self.cls,
            'cnt' : self.cnt,
            'parameters' : self.parameters
        }
