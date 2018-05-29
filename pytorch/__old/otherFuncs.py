import torch
import pdb
from time import time

VERBOSE = False
EXTRA_MM_INFO = True

class BatchedReverseMM(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, w):
		ctx.save_for_backward(x, w)
		ctx.name = ("-"*32)+ " MM"
		return torch.matmul(x, w)

	@staticmethod
	def backward(ctx, g):
		print(ctx.name) if VERBOSE else None
		x,ww, = ctx.saved_variables
		
		if VERBOSE:
			print("g",g.shape)
			print("x",x.shape)
			print("w",ww.shape)

		#pdb.set_trace()
		dx, dy = None, None
		if x.requires_grad:
			dx = torch.mm(g, ww.t())
		if ww.requires_grad:
			dy = x.unsqueeze(1)*g.unsqueeze(2)
		
		#pdb.set_trace()
		return dx, dy
		
class BatchedReLU(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		ctx.save_for_backward(x)
		ctx.mask = x >= 0
		ctx.name = ("-"*32)+ " Relu"
		return torch.relu(x)

	@staticmethod
	def backward(ctx, g):
		print(ctx.name) if VERBOSE else None
		x, = ctx.saved_variables
		if VERBOSE:
			print(g.shape)
			print(x.shape)
			
		dx = None
		if x.requires_grad:
			dx = g * ctx.mask.type_as(g).expand_as(g)
			if VERBOSE:
				print(dx.shape)
			
		return dx
		
class BatchedSigmoid(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		ctx.save_for_backward(x)
		ctx.name = ("-"*32)+ " Sigmoid"
		return torch.sigmoid(x)

	@staticmethod
	def backward(ctx, g):
		print(ctx.name) if VERBOSE else None
	
		x, = ctx.saved_variables

		if VERBOSE:
			print("g", g.shape)
			print("x", x.shape)
			
		dx = None
		if x.requires_grad:
			sig = torch.sigmoid(x)
			dx = sig * (1-sig)
			#pdb.set_trace()
			dx = dx * g
			
			if VERBOSE:
				print("d", dx.shape)
			
		return dx

class BatchedLogLoss(torch.autograd.Function):
	@staticmethod
	def forward(ctx, p, t):
		ctx.save_for_backward(p, t)
		ctx.name = ("-"*32)+ " LogLoss"
		return torch.sum(-(t*torch.log(p) + (1-t)*torch.log(1-p)))

	@staticmethod
	def backward(ctx, g):
		print(ctx.name) if VERBOSE else None
		
		x, y, = ctx.saved_variables
		
		if VERBOSE:
			print("g ", g.shape)
			print("x ", x.shape)
			print("y ", y.shape)
			
		dx, dy = None, None
		if x.requires_grad:
			dx = g * ((y-x)/((x - 1)*x))
			if VERBOSE:
				print("dx", getattr(dx, "shape", None))
		if y.requires_grad:
			dy = g * ((x-y)/((y - 1)*y))
			if VERBOSE:
				print("dy", getattr(dy, "shape", None))
			
		return dx, dy

class Protect(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, name):
		ctx.name = name
		return x

	@staticmethod
	def backward(ctx, g):
		if VERBOSE:
			print(ctx.name)
			print(g.shape)	
		#pdb.set_trace()
		return g, None
		
		
protect = Protect.apply
matmul = BatchedReverseMM.apply
relu = BatchedReLU.apply
sigmoid = BatchedSigmoid.apply
logloss = BatchedLogLoss.apply
