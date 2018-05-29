import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
import pdb 

from goodfellow_backprop import goodfellow_backprop	
from helpers import simpleTiming, profiling, check_correctness

###############################################################################
# Making up some data and model(s)
if True:
	SEED = 1
	np.random.seed(seed=SEED)

	# N: Number of samples
	# D: Dimension of input and of each Layer
	# L: Number of hidden layers
	N, D, L = 5, 3, 0
	hidden_sizes = list(D for l in range(L))

	X = torch.tensor(np.random.randn(N, D)).float()
	y = torch.tensor(np.random.rand(N), requires_grad=True).float()
	ww = torch.tensor(np.random.rand(D), requires_grad=True).view(-1,1).float()

import torch.nn.functional as F

def protect(someFunc):
	class Protect(torch.autograd.Function):
		@staticmethod
		def forward(ctx, *args):
			ctx.inputs = args
			
			pdb.set_trace()
			
			ctx.rr = torch.mm(args[0], args[1])
			pdb.set_trace()
			zzz = torch.autograd.grad(ctx.rr, args[1], torch.eye(ctx.rr.shape[0]))
			
			pdb.set_trace()
			
			
			ctx.rr = someFunc(*args)
			ctx.dargs = []
			for input in ctx.inputs:
				ctx.dargs.append(torch.autograd.grad(ctx.rr, input) if input.requires_grad else None)
				pdb.set_trace()
			return ctx.rr

		@staticmethod
		def backward(ctx, gg):
			print("back")
			dargs = list()
			return dargs
	
	return Protect.apply

###########################3

#pdb.set_trace()

#zzz = torch.autograd.grad(torch.mm(X, ww), ww, torch.eye(X.shape[0]).float())

pdb.set_trace()


#pdb.set_trace()
###		
mmp = protect(torch.mm)

z = torch.sum(mmp(X, ww))

g = torch.autograd.grad(z, ww)

pdb.set_trace()
		
