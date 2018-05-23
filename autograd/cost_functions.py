import autograd.numpy as np

def logloss(probas, outputs):
	"""
	Return the log loss
	"""
	return - (outputs*np.log(probas) + (1-outputs)*np.log(1-probas))

