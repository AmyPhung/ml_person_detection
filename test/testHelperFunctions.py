import unittest
import json
import numpy as np
from scripts.modules.helperFunctions import *

class testRemoveGroundplane(unittest.TestCase):
    """Define and test interface for remove_groundplane."""
    def setUp(self):
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


if __name__ == '__main__':
    unittest.main()
