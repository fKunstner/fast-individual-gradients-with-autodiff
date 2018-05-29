import torch
import pdb
from time import time

VERBOSE = True
EXTRA_MM_INFO = False

# These custom ops break the usual assumption that gradients given to
# backward have the same shapes as outputs. They are expected to have
# an extra leading dimension, which batches independent reverse mode
# passes.
class ReverseMM(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, y):
		ctx.save_for_backward(x, y)
		ctx.name = "------------------- MM ID(%f)" % torch.rand(1)
		return torch.matmul(x, y)

	@staticmethod
	def backward(ctx, batched_grad):
		print("MM", batched_grad.shape)
		x, y = ctx.saved_variables
		dx = torch.matmul(batched_grad, y.t())
		dy = torch.matmul(x.t(), batched_grad)
		return dx, dy

class ReLU(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		ctx.mask = x >= 0
		ctx.name = "------------------- Relu ID(%f)" % torch.rand(1)
		return torch.relu(x)

	@staticmethod
	def backward(ctx, batched_grad):
		print("relu", batched_grad.shape)
		return batched_grad * ctx.mask.type_as(batched_grad).expand_as(batched_grad)

class Sigmoid(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		ctx.save_for_backward(x)
		ctx.name = "------------------- Sigmoid ID(%f)" % torch.rand(1)
		return torch.sigmoid(x)

	@staticmethod
	def backward(ctx, batched_grad):
		print("sig", batched_grad.shape)
		x, = ctx.saved_variables
		sig = torch.sigmoid(x)
		dx = sig * (1-sig)
		return batched_grad * dx.type_as(batched_grad).expand_as(batched_grad)

class LogLoss(torch.autograd.Function):
	@staticmethod
	def forward(ctx, probabilities, targets):
		ctx.save_for_backward(probabilities, targets)
		ctx.name = "------------------- LogLoss ID(%f)" % torch.rand(1)
		return torch.sum(-(targets*torch.log(probabilities) + (1-targets)*torch.log(1-probabilities)))

	@staticmethod
	def backward(ctx, batched_grad):
		print("LLoss", batched_grad.shape)
		probabilities, targets, = ctx.saved_variables
		dx = (targets-probabilities)/((probabilities - 1)*probabilities)
		return batched_grad * dx.type_as(batched_grad).expand_as(batched_grad), None

matmul = ReverseMM.apply
relu = Sigmoid.apply
sigmoid = Sigmoid.apply
logloss = LogLoss.apply
