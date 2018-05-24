import torch
import pdb
from time import time

VERBOSE = False
EXTRA_MM_INFO = False

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
		if VERBOSE:
			print(time(), ctx.name, batched_grad.shape)
		dx = torch.matmul(batched_grad, y.t())
		if VERBOSE:
			print(time(), " (computed d/dx)")
		dy = torch.matmul(x.t(), batched_grad)
		if VERBOSE:
			print(time(), " (computed d/dy)")
		if VERBOSE and EXTRA_MM_INFO:
			print(batched_grad.shape)
			print(batched_grad)
			print(a.shape)
			print(a)
			print(b.shape)
			print(b)
		return dx, dy

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
			print(time(), ctx.name)
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
			print(time(), ctx.name)
		dx = (targets-probabilities)/((probabilities - 1)*probabilities)
		return batched_grad * dx.type_as(batched_grad).expand_as(batched_grad), None

matmul = BatchedReverseMM.apply
relu = BatchedSigmoid.apply
sigmoid = BatchedSigmoid.apply
logloss = BatchedLogLoss.apply
