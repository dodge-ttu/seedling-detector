from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from seedling_detector.preprocessing import simplepreprocessor as sp
from seedling_detector.datasets import simpledatasetloader as sd

from imutils import paths
import argparse

# Import the necessary packages.
import numpy as np
import cv2

# Initialize the class labels and set the seed of the pseudorandom number generator so we can reproduce our results
labels = ['dog', 'cat', 'panda']
np.random.seed(1)

# Randomly initialize our weight matrix and bias vector -- in a real world situation these values would be the product
# of training and learned with our model but for the sake of this example the values will be random.
W = np.random.randn(3, 3072)
b = np.random.randn(3)

# Load our example image, resize, then flatten into our "feature vector" representation.
orig = cv2.imread("/home/will/seedling-detection/beagle.png")
image = cv2.resize(orig, (32, 32)).flatten()






