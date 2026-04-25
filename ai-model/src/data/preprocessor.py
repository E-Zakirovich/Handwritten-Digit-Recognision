import numpy as np
from config import MAX_PIXEL, NUMBER_OF_CLASSES


class Preprocessor:
	
	def normalize(self, IMAGES):
		return IMAGES / MAX_PIXEL
		
	def one_hot_encoding(self, LABELS):
		one_hot = np.zeros((LABELS.shape[0], NUMBER_OF_CLASSES))
		one_hot[np.arange(LABELS.shape[0]), LABELS] = 1
		return one_hot