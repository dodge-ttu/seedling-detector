from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from seedling_detector.preprocessing import simplepreprocessor as sp
from seedling_detector.datasets import simpledatasetloader as sd


from imutils import paths
import argparse

# Costruct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='The path to the input dataset')
ap.add_argument('-k', '--neighbors', type=int, default=1, help='Number of nearest neighbors for classification')
ap.add_argument('-j', '--jobs', type=int, default=-1, help='Number of jobs for k-NN distance (-1 uses all cores)')
args = vars(ap.parse_args())

print(args)

# Grab the lost of images that we'll be describing.
print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))

# Initialize the image preprocessor, load the dataset from disk, and reshape the data matrix.
sp = sp.SimplePreprocessor(32, 32)
sdl = sd.SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# Show some information on memory consumption of the images.
print('[INFO] features matrix: {0:.1f}MB'.format(data.nbytes / (1024 * 1000.0)))

# Encode the labels as integers.
le = LabelEncoder()
labels = le.fit_transform(labels)

# Partition the data into training and testing splits using 75%:25% training:testing.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Train and evaluate a k-NN classifier on the raw pixel intensities.
print('[INFO] evaluating k-NN classifier...')
model = KNeighborsClassifier(n_neighbors=args['neighbors'], n_jobs=args['jobs'])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))







