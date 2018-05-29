import torch
import pdb
from time import time

VERBOSE = True
EXTRA_MM_INFO = True

# These custom ops break the usual assumption that gradients given to
# backward have the same shapes as outputs. They are expected to have
# an extra leading dimension, which batches independent reverse mode
# passes.
class BatchedReverseMM(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, y):
		ctx.save_for_backward(x, y)
		ctx.name = "------------------- MM ID(%f)" % torch.rand(1)
		return torch.matmul(x, y)

	@staticmethod
	def backward(ctx, batched_grad):
		x, y = ctx.saved_variables
		print(ctx.name)
		start = time()
		
		#print(x.shape, y.shape, batched_grad.shape)
		dx_tens = torch.zeros(batched_grad.shape[0], batched_grad.shape[1], y.shape[0])
		dy_tens = torch.zeros(batched_grad.shape[0], x.shape[1], batched_grad.shape[2])
		print("placeholders done in ", time() - start)
		start = time()
		#print(dx.shape, dx_tens.shape)
		#print(dy.shape, dy_tens.shape)

		print("d",dx_tens.shape)
		print("g",batched_grad.shape)
		print("x",x.shape)
		for n in range(batched_grad.shape[0]):
			#if dx.shape[2] == 2:
			#	pdb.set_trace()
			dy_tens[n,:,:] = torch.ger(x[n,:], batched_grad[n,n,:])
		print("dy", time() - start)
		#print(batched_grad.shape,"\n", batched_grad)
		#print(x.shape,"\n", x)
		#print(dy.shape,"\n", dy)
		
		#pdb.set_trace()
		start = time()
		dx_mat = torch.mm(batched_grad.sum(dim=0), y.t())
		for n in range(dx_tens.shape[0]):
			dx_tens[n,n,:] = dx_mat[n,:]
		print("dx", time() - start)
		
		#pdb.set_trace()
			
			
		#return dx, dy
		return dx_tens, dy_tens
		
class BatchedReLU(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		ctx.mask = x >= 0
		ctx.name = "------------------- Relu ID(%f)" % torch.rand(1)
		return torch.relu(x)

	@staticmethod
	def backward(ctx, batched_grad):
		if VERBOSE:
			print(time(), ctx.name)
		return batched_grad * ctx.mask.type_as(batched_grad).expand_as(batched_grad)

class BatchedSigmoid(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		ctx.save_for_backward(x)
		ctx.name = "------------------- Sigmoid ID(%f)" % torch.rand(1)
		return torch.sigmoid(x)

	@staticmethod
	def backward(ctx, batched_grad):
		x, = ctx.saved_variables
		if VERBOSE:
			print(ctx.name)
		sig = torch.sigmoid(x)
		dx = sig * (1-sig)
		
		return batched_grad * dx.type_as(batched_grad).expand_as(batched_grad)

class BatchedLogLoss(torch.autograd.Function):
	@staticmethod
	def forward(ctx, probabilities, targets):
		ctx.save_for_backward(probabilities, targets)
		ctx.name = "------------------- LogLoss ID(%f)" % torch.rand(1)
		return torch.sum(-(targets*torch.log(probabilities) + (1-targets)*torch.log(1-probabilities)))

	@staticmethod
	def backward(ctx, batched_grad):
		probabilities, targets, = ctx.saved_variables
		if VERBOSE:
			print(ctx.name)
		dx = (targets-probabilities)/((probabilities - 1)*probabilities)
		#pdb.set_trace()
		return batched_grad * dx.type_as(batched_grad).expand_as(batched_grad), None

matmul = BatchedReverseMM.apply
relu = BatchedSigmoid.apply
sigmoid = BatchedSigmoid.apply
logloss = BatchedLogLoss.apply

