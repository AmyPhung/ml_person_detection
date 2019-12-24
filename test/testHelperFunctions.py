import unittest
import json
import numpy as np
from scripts.modules.helperFunctions import *
from waymo_open_dataset.label_pb2 import Label

class testRemoveGroundplane(unittest.TestCase):
    """Define and test interface for remove_groundplane."""
    def setUp(self):
        """Create pcl for testing."""
        self.pointcount = 11  # Use odd numbers to center around 0
        self.pcl = np.zeros((self.pointcount, 4))
        for p in range(self.pcl.shape[0]):
            self.pcl[p][2] = p - (self.pcl.shape[0] - 1)/2

    def testBadArgs(self):
        """Test supplying function with bad arguments."""
        with self.assertRaises(
                TypeError, msg="function accepted string for z_thresh."):
            remove_groundplane(self.pcl, z_thresh='3')

        with self.assertRaises(
                TypeError, msg="function accepted nested list for pcl."):
            bad_pcl = [[0, 0, -1, 0], [0, 0, 0, 0], [0, 0, 1, 0]]
            remove_groundplane(bad_pcl)

        with self.assertRaises(
                TypeError, msg="function ran without args."):
            remove_groundplane()

        with self.assertRaises(
                TypeError, msg="function ran with swapped args."):
            remove_groundplane(3, z_thresh=self.pcl)

        with self.assertRaises(
                ValueError, msg="function accepted ndarray with 2 rows."):
            remove_groundplane(self.pcl[:, 0:1])

    def testNoThresh(self):
        """Test thresholding out no points."""
        pcl_out = remove_groundplane(self.pcl, -1 * self.pointcount)
        self.assertTrue(
            (pcl_out == self.pcl).all(),
            "returned pcl had points unexpectedly removed.")

    def testAllThresh(self):
        """Test thresholding out all points."""
        pcl_out = remove_groundplane(self.pcl, self.pointcount)
        self.assertTrue(
            pcl_out.shape == (0, 4),
            "returned pcl did not have all points removed as expected.")

    def testRegThresh(self):
        """Test threshold of 0 (middle of pcl)."""
        pcl_out = remove_groundplane(self.pcl, 0)
        self.assertTrue(
            pcl_out.shape == ((self.pointcount-1)/2, 4),
            "returning pointcloud is an unexpected length.")

        pt_check = [p == i + 1 for i, p in enumerate(pcl_out[:,2])]
        self.assertTrue(
            pt_check.count(True) == len(pt_check),
            "returned pcl points have unexpected z-values.")


class testComputeVolume(unittest.TestCase):
    """Define and test interface for compute_volume."""
    def setUp(self):
        """Create pcls for testing."""
        # Rectangular pcls
        self.pcl_list = [
            [-2, 4, -1], [2, 4, -1],
            [-2, -4, -1], [2, -4, -1],
            [-2, 4, 1], [2, 4, 1],
            [-2, -4, 1], [2, -4, 1]]
        self.pcl_fullrect = np.array(self.pcl_list)
        self.pcl_flatrect = np.array([
            [-2, 4, 0], [2, 4, 0],
            [-2, -4, 0], [2, -4, 0]])

        # Hexagonal pcls
        self.pcl_fullhex = np.array([
            [0, 1, 1], [1, 1, 1], [2, -0.5, 1],
            [0, -2, 1], [1, -2, 1], [-1, -0.5, 1],
            [0, 1, -1], [1, 1, -1], [2, -0.5, -1],
            [0, -2, -1], [1, -2, -1], [-1, -0.5, -1]])
        self.pcl_flathex = np.array([
            [0, 1, 0], [1, 1, 0], [2, -0.5, 0],
            [0, -2, 0], [1, -2, 0], [-1, -0.5, 0]])

    def testBadArgs(self):    
        """Test supplying function with bad arguments."""
        with self.assertRaises(
                TypeError, msg="function accepted nested list for pcl."):
            compute_volume(self.pcl_list)

        with self.assertRaises(
                TypeError, msg="function ran without args."):
            compute_volume()

        with self.assertRaises(
                ValueError, msg="function accepted ndarray with 2 rows."):
            compute_volume(self.pcl_fullrect[:, 0:1])

    def testZeroVolRect(self):
        """Test vol of rectangle."""
        act_vol = 0
        vol = compute_volume(self.pcl_flatrect)
        self.assertTrue(
            np.abs(act_vol - vol) < 0.0001,
            "function found %s pcl volume, not %s." % (vol, act_vol))

    def testZeroVolHex(self):
        """Test vol of hexagon."""
        act_vol = 0
        vol = compute_volume(self.pcl_flathex)
        self.assertTrue(
            np.abs(act_vol - vol) < 0.0001,
            "function found %s pcl volume, not %s." % (vol, act_vol))

    def testVolRect(self):
        """Test vol of rectangular prism straddling 0 in z."""
        act_vol = 64
        vol = compute_volume(self.pcl_fullrect)
        self.assertTrue(
            np.abs(act_vol - vol) < 0.0001,
            "function found %s pcl volume, not %s." % (vol, act_vol))

    def testVolHex(self):
        """Test vol of hexagonal prism straddling 0 in z."""
        act_vol = 12
        vol = compute_volume(self.pcl_fullhex)
        self.assertTrue(
            np.abs(act_vol - vol) < 0.0001,
            "function found %s pcl volume, not %s." % (vol, act_vol))


class testGetPtsInBBox(unittest.TestCase):
    """Define and test interface for get_pts_in_bbox."""
    def setUp(self):
        """Create pcls and bboxes for testing."""
        size = 5
        self.pcl_array = np.asarray([[i, j, k]
            for i in range(size) for j in range(size) for k in range(size)])

    def testBadArgs(self):    
        """Test supplying function with bad arguments."""

        label = Label()
        label.box.heading = 0
        label.box.center_x = label.box.center_y = label.box.center_z = 10
        label.box.height = label.box.length = label.box.width = 1
        pcl_list = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
        label_list = {
            'center_x': 0, 'center_y': 0, 'center_z': 0,
            'height': 1, 'length': 1, 'width': 1, 'heading': 0}

        with self.assertRaises(
                TypeError, msg="function accepted nested list for bbox."):
            get_pts_in_bbox(self.pcl_array, label_list)

        with self.assertRaises(
                TypeError, msg="function accepted nested list for pcl."):
            get_pts_in_bbox(pcl_list, label)

        with self.assertRaises(
                TypeError, msg="function ran without args."):
            get_pts_in_bbox()

        with self.assertRaises(
                TypeError, msg="function ran with args swapped."):
            get_pts_in_bbox(label, self.pcl_array)

        with self.assertRaises(
                ValueError, msg="function accepted ndarray with 2 rows."):
            get_pts_in_bbox(self.pcl_array[:, 0:1], label)

    def testBoxOutsidePcl(self):
        """Test bbox outside entire pcl."""
        label = Label()
        label.box.heading = 0
        label.box.center_x = label.box.center_y = label.box.center_z = 10
        label.box.height = label.box.length = label.box.width = 1
        self.assertTrue(
            len(get_pts_in_bbox(self.pcl_array, label)) == 0,
            "function found unexpected pts in bbox.")

    def testBoxAlignedWithPcl(self):
        """Test bbox within pcl, axes aligned."""
        label = Label()
        label.box.heading = 0
        label.box.center_x = label.box.center_y = label.box.center_z = 2 
        label.box.height = label.box.length = label.box.width = 1
        act_len = 5
        pts_len = len(get_pts_in_bbox(self.pcl_array, label))
        self.assertTrue(
            act_len == pts_len,
            "function found %i pts, not %i pts." % (pts_len, act_len))

    def testBox45DegreeWithPcl(self):
        """Test bbox within pcl, 45 degree axes angle."""
        label = Label()
        label.box.heading = 45
        label.box.center_x = label.box.center_y = label.box.center_z = 2
        label.box.width = np.sqrt(2)
        label.box.length = 2 * np.sqrt(2)
        label.box.height = 1
        act_len = 7 * 5
        pts_len = len(get_pts_in_bbox(self.pcl_array, label))
        self.assertTrue(
            act_len == pts_len,
            "function found %i pts, not %i pts." % (pts_len, act_len))

    def testBoxAbovePcl(self):
        """Test bbox above entire pcl.

        This test demonstrates that the function CURRENTLY does not threshold
        by z-height, and only does xy-bounding box.

        """
        label = Label()
        label.box.heading = 45
        label.box.center_x = label.box.center_y = 2
        label.box.center_z = 20
        label.box.width = np.sqrt(2)
        label.box.length = 2 * np.sqrt(2)
        label.box.height = 1
        act_len = 7 * 5
        pts_len = len(get_pts_in_bbox(self.pcl_array, label))
        self.assertTrue(
            act_len == pts_len,
            "function found %i pts, not %i pts." % (pts_len, act_len))


class testExtractClusterParameters(unittest.TestCase):
    """Define and test interface for extract_cluster_parameters."""
    def setUp(self):
        """Create cluster for testing."""
        size = 5

        self.cluster_list = [[i, j, k, 0]
            for i in range(size) for j in range(size) for k in range(size)]
        self.cluster_array = np.asarray(self.cluster_list)

        self.cluster2_array = np.asarray([[i, 2 * j, 3 * k, k]
            for i in range(size) for j in range(size) for k in range(size)])

    def testBadArgs(self):
        """Test supplying function with bad arguments."""
        with self.assertRaises(
                TypeError, msg="function accepted nested list for pcl."):
            extract_cluster_parameters(self.cluster_list)

        with self.assertRaises(
                TypeError, msg="function ran without args."):
            extract_cluster_parameters()

        with self.assertRaises(
                ValueError, msg="function accepted ndarray with 2 rows."):
            extract_cluster_parameters(self.cluster_array[:, 0:1])

    def testCubeNoIntensityCluster(self):
        """Test 4 x 4 x 4 cube of points with zero intensity as cluster."""
        param = extract_cluster_parameters(self.cluster_array)
        self.assertTrue(
            param[0] == 2, "Expected x 2, received %s" % param[0])
        self.assertTrue(
            param[1] == 2, "Expected y 2, received %s" % param[1])
        self.assertTrue(
            param[2] == 2, "Expected z 2, received %s" % param[2])
        self.assertTrue(
            param[3] == param[4] == param[5],
            "Expected e_x = e_y = e_z, found e_x %s, e_y %s, e_z %s"
            % (param[3], param[4], param[5]))
        self.assertTrue(
            param[6] == 64, "Expected vol-param 54, received %s" % param[6])
        self.assertTrue(
            np.abs(param[7] - 1.95312) < 0.00001,
            "Expected density-param 1.953, received %s" % param[7])
        self.assertTrue(
            param[8] == 0, "Expected max_intensity 0, received %s" % param[8])
        self.assertTrue(
            param[9] == 0, "Expected mean_intensity 0, received %s" % param[9])
        self.assertTrue(
            param[10] == 0, "Expected var_intensity 0, got %s" % param[10])

    def testRectangularPrismIntensityCluster(self):
        """Test 4 x 8 x 12 prism of points with intensity == z as cluster."""
        param = extract_cluster_parameters(self.cluster2_array)
        self.assertTrue(
            param[0] == 2, "Expected x 2, received %s" % param[0])
        self.assertTrue(
            param[1] == 4, "Expected y 4, received %s" % param[1])
        self.assertTrue(
            param[2] == 6, "Expected z 6, received %s" % param[2])
        self.assertTrue(
            param[3] == param[4]/2**2 == param[5]/3**2,
            "Expected e_x = e_y/4 = e_z/9, found e_x %s, e_y %s, e_z %s"
            % (param[3], param[4], param[5]))
        self.assertTrue(
            param[6] == 384, "Expected vol-param 384, received %s" % param[6])
        self.assertTrue(
            np.abs(param[7] - 0.32552) < 0.00001,
            "Expected density-param 0.32552, received %s" % param[7])
        self.assertTrue(
            param[8] == 4, "Expected max_intensity 4, received %s" % param[8])
        self.assertTrue(
            param[9] == 2, "Expected mean_intensity 2, received %s" % param[9])
        self.assertTrue(
            param[10] == 2, "Expected var_intensity 2, got %s" % param[10])
    
if __name__ == '__main__':
    nittest.main()
