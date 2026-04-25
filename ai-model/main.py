from src import MNISTLoader
from src import Preprocessor

loader = MNISTLoader()
encoder = Preprocessor()

train_images, train_labels = loader.train_data()
train_images = encoder.normalize(train_images)
train_labels = encoder.one_hot_encoding(train_labels)
print(train_labels[0])
print(train_images[0][:300])
