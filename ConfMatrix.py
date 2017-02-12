import matplotlib.pyplot as plt
import itertools
import numpy as np

EPS = 1e-7

def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()

	if normalize:
		cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis]+EPS)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, "",
				horizontalalignment="center",
				color="white" if cm[i, j] > thresh else "black"
		)

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
