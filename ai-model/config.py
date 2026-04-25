# dataset folder locations saving process
DIR = "./data/mnist/"


TRAINING_IMAGE = DIR + "train-images.idx3-ubyte" # training images diretory 
TESTING_IMAGE = DIR + "t10k-images.idx3-ubyte" # testing images diretory 

TRAINING_LABELS = DIR + "train-labels.idx1-ubyte" # training labels directory
TESTING_LABELS = DIR + "t10k-labels.idx1-ubyte" # testing labels directory

# image normalization settings
MAX_PIXEL = 255.0
NUMBER_OF_CLASSES = 10