#!usr/bin/env python
import rospy

class CreateDataset(object):
    def __init__(self):
        """Provide directory location to find frames."""
        pass

    def loadFrame(self):
        pass

    def filterFrame(self):
        pass

    def clusterByBBox(self):
        pass

    def computeClusterMetadata(self):
        pass

    def saveClusterMetadata(self):
        pass

    def parseFrame(self):
        """Main function."""
        # load frame
        # filter frame
        # cluster
        # compute data
        # save

    def run(self):
        """
        put glob + directory stuff here
        """
        pass



class DatasetCreatorVis(DatasetCreator):

    def __init__(self):

        # Do ros stuff here
        super(DatasetCreator, self).__init__()
        pass

    def parseFrame(self):
        """Overwrite CreateDataset run function, insert viz."""

        # load frame (from file)
        # publish to ros
        # publsih markers and such to ros
        # filter frame
        # cluster
        # compute
        # save
        pass

if __name__ == "__main__":
    visualize = False
    creator = DatasetCreatorVis() if visualize else DatasetCreator()
    directory = '/home/cnovak/Data/wayno-od'
    file = ' segment-15445436653637630344_3957_561_3977_561' \
        + '_with_camera_labels.tfrecord'
    creator.run(directory)
