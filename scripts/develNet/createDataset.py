class CreateDataset:
    def __init__(self):
        pass
        """
        provide directory location to find frames
        """
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
        """
        will overwrite with ROS in other object
        """
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



class CreateDatasetROS(CreateDataset):

    def __init__(self):
        # Do ros stuff here
        super.__init__()
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
