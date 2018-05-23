import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector

from MLP import MLP 
from goodfellow_backprop import goodfellow_backprop	
from helpers import simpleTiming, profiling, check_correctness

import pdb

###############################################################################
# Making up some data and model(s)

SEED = 1
np.random.seed(seed=SEED)

# N: Number of samples
# D: Dimension of input and of each Layer
# L: Number of hidden layers
N, D = 17, 3
hidden_sizes = [5]

X_t = torch.Tensor(np.random.randn(N, D))
y_t = torch.Tensor(np.random.rand(N))
dataset = torch.utils.data.TensorDataset(X_t, y_t)
fullbatch = torch.utils.data.DataLoader(dataset, batch_size=N)
individual = torch.utils.data.DataLoader(dataset, batch_size=1)

model = MLP(input_size = D, hidden_sizes = hidden_sizes)
model.train(True)

###############################################################################
# Options suggested in https://github.com/pytorch/pytorch/issues/7786

def suggestion1():
	"""
	https://github.com/pytorch/pytorch/issues/7786#issuecomment-391269793
	"""
	grads = []
	for i, (x, y) in enumerate(fullbatch):
		logits = model.forward(x)
		losses = list()
		for j in range(logits.shape[0]):
			model.zero_grad()
			losses.append(F.binary_cross_entropy_with_logits(logits[j], y[j]))
			
	result = torch.autograd.grad(tuple(losses), list(model.parameters()))

	return parameters_to_vector(result)

###############################################################################
# Running things

res1 = suggestion1()
print(res1.shape)