import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # Store the image processor
        self.preporcessors = preprocessors

        # If the preprocessors are None, initialize them as an empty list.
        if self.preporcessors is None:
            self.preporcessors = []

    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []

        # Loop over the input images.
        for (i, imagePath) in enumerate(imagePaths):
            # Load the image and extract the class label assuming that our path has the following
            # format: /path/to/dataset/{class}/{image}.jpg

            image=cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preporcessors is not None:
                # Loop over the prepocessors and apply each to the image.
                for p in self.preporcessors:
                    image = p.preprocess(image)

            # Treat or preprocessed image as a 'feature vector' by updating the data list followed by labels.
            data.append(image)
            labels.append(label)

            # Show an update every 'verbose' images.
            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print('[INFO] processed {}/{}'.format(i+1, len(imagePaths)))

        # Return a tuple of the data and labels.
        return (np.array(data), np.array(labels))
