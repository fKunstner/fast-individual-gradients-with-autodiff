import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector

from MLP import MLP 
from goodfellow_backprop import goodfellow_backprop	
from helpers import simpleTiming, profiling, check_correctness

###############################################################################
# Making up some data and model(s)

SEED = 1
np.random.seed(seed=SEED)

# N: Number of samples
# D: Dimension of input and of each Layer
# L: Number of hidden layers
N, D, L = 16, 700, 5
hidden_sizes = list(D for l in range(L))

X_t = torch.Tensor(np.random.randn(N, D))
y_t = torch.Tensor(np.random.rand(N))
dataset = torch.utils.data.TensorDataset(X_t, y_t)
fullbatch = torch.utils.data.DataLoader(dataset, batch_size=N)
individual = torch.utils.data.DataLoader(dataset, batch_size=1)

model = MLP(input_size = D, hidden_sizes = hidden_sizes)
model.train(True)

# Hack for multimod function (see below)
models = list()
for n in range(N):
	torch.manual_seed(SEED)
	model = MLP(input_size = D, hidden_sizes = hidden_sizes)
	model.train(True)
	models.append(model)

###############################################################################
# Different ways of computing the gradient 

def full():
	"""
	Computes the gradient of the complete objective function
	"""
	for i, (x, y) in enumerate(fullbatch):
		model.zero_grad()

		logits = model.forward(x)
		loss = F.binary_cross_entropy_with_logits(logits.view((-1,)), y)
		loss.backward()

	grads = np.array(parameters_to_vector(list([p.grad for p in model.parameters()])))
	return grads

def naive():
	"""
	Computes the predictions in a full-batch fasion,
	then call backward on the individual losses
	"""
	grads = []
	for i, (x, y) in enumerate(fullbatch):
		logits = model.forward(x)

		for j in range(logits.shape[0]):
			model.zero_grad()
			loss = F.binary_cross_entropy_with_logits(logits[j], y[j])
			loss.backward(retain_graph=True)
			
			vectorized = parameters_to_vector(list([p.grad for p in model.parameters()]))
			grads.append(np.array(vectorized))
			
	return np.mean(np.vstack(grads), axis=0)

def goodfellow():
	"""
	Use Goodfellow's trick to compute individual gradients.
	Ref: Efficient per-example gradient computations
	at: https://arxiv.org/abs/1510.01799
	"""
	for i, (x, y) in enumerate(fullbatch):
		model.zero_grad()

		logits, H_list, Z_list = model.forward_goodfellow(x)
		loss = F.binary_cross_entropy_with_logits(logits.view((-1,)), y)
			
		gradients = goodfellow_backprop(N, H_list, torch.autograd.grad(loss, Z_list))
	return np.mean(np.array(gradients), axis=1)
	
def multimod():
	"""
	Define multiple copies of the parameters, such that the loss associated with
	each sample is associated with a different set of parameters, forcing them to
	not be aggregated.
	
	Inspired by 
	https://github.com/tensorflow/tensorflow/issues/4897#issuecomment-290997283
	
	To do this, I have defined N copies of the model initialized at the same values.
	"""
	losses = list()
	params = list()
	
	N_PARAMS = len(list(models[0].parameters()))
	for i, (x, y) in enumerate(individual):
		models[i].zero_grad()
		logits = models[i].forward(x)
		loss = F.binary_cross_entropy_with_logits(logits.view((-1,)), y.float())
		
		losses.append(loss)
		for p in models[i].parameters():
			params.append(p)
		
	allgrads = torch.autograd.grad(losses, params)
	
	def list_of_grads_to_matrix(list_of_grads):
		grads = list()
		for grad_idx in range(int(len(allgrads)/N_PARAMS)):
			grad = list()
			for param_idx in range(N_PARAMS):
				grad.append(allgrads[grad_idx*N_PARAMS + param_idx])
			grads.append(parameters_to_vector(grad))
		return np.vstack(grads)

	return np.mean(list_of_grads_to_matrix(allgrads), axis=0)

###############################################################################
# Running the whole thing
	
names = ["naive", "goodf", "multi"]
methods = [naive, goodfellow, multimod]

check_correctness(full, names, methods)
simpleTiming(full, names, methods)
profiling(full, names, methods)

