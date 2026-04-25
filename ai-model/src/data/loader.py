import numpy as np
from config import TRAINING_IMAGE, TESTING_IMAGE, TRAINING_LABELS, TESTING_LABELS
import struct

class MNISTLoader:
	
	def _load_images(self, filepath):
		with open(filepath, 'rb') as f:
			magic, num, rows, columns =  struct.unpack('>IIII', f.read(16))
			images = np.frombuffer(f.read(), dtype = np.uint8)
			images = images.reshape(num, rows * columns)
			return images
		
	def _load_labels(self, filepath):
		with open(filepath, 'rb') as f:
			magic, num = struct.unpack('>II', f.read(8))
			labels = np.frombuffer(f.read(), dtype = np.uint8)
			return labels
		
	def train_data(self):
		images = self._load_images(TRAINING_IMAGE)
		labels = self._load_labels(TRAINING_LABELS)
		return images, labels
		
	def test_data(self):
		images = self._load_images(TESTING_IMAGE)
		labels = self._load_labels(TESTING_LABELS)
		return images, labels