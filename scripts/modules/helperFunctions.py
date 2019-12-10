#!/usr/bin/env python
"""Module for shared code between waymo pipeline and panasonic pipeline.

This module holds functions that are used in both pipelines:
    remove_groundplane()
    extract_pcl_features()
"""
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.signal import resample
import matplotlib.pyplot as plt

GROUND_THRESHOLD = 0.1  # meters
MAX_CLUSTER_PTS = 200 # max number of points in a cluster

def remove_groundplane(pcl, z_thresh=GROUND_THRESHOLD):
    """Remove points below z-threshold and return pcl.

    Args:
        pcl: (n * 4) numpy array of xyz points and intensities

    Returns:
        pcl_out: (n * 4) numpy array of xyz points and intensities without
            points below a certain z value
    """
    return pcl[pcl[:,2] > z_thresh]

def extract_cluster_parameters(cluster, display=False):
    """Calculate features for net training from pcl, return PclFeatures object
    containing all features.

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

def get_pts_in_bbox(pcl, bbox, logger=None):
    """Return ndarray of points from pcl within bbox."""
    # Get bbox limits in bbox coord frame
    x_lo = bbox.box.center_x - bbox.box.length/2
    x_hi = bbox.box.center_x + bbox.box.length/2
    y_lo = bbox.box.center_y - bbox.box.width/2
    y_hi = bbox.box.center_y + bbox.box.width/2
    if logger is not None:
        logger.debug('bbox limits: %0.2f-%0.2f, %0.2f-%0.2f'
            % (x_lo, x_hi, y_lo, y_hi))
    else:
        print('bbox limits: %0.2f to %0.2f x, %0.2f to %0.2f y'
            % (x_lo, x_hi, y_lo, y_hi))

    # Rotate points by bbox heading angle
    ang = np.radians(bbox.box.heading)
    r_mat = np.array(((np.cos(ang), np.sin(ang)), (-np.sin(ang), np.cos(ang))))
    r_pcl = np.matmul(pcl[:, 0:2], r_mat)

    # Add back in z and intensity data
    r_pcl = np.append(r_pcl, pcl[:, 2:], axis=1)

    # Sub-select pcl by bbox limits
    indxs = np.where(
        (x_lo < r_pcl[:, 0]) & (r_pcl[:, 0] < x_hi)
        & (y_lo < r_pcl[:, 1]) & (r_pcl[:, 1] < y_hi))[0]
    pcl_out = r_pcl[indxs]
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
