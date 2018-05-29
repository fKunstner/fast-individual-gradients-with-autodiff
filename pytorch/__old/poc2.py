import torch
import pdb
import batchFuncs
import ops

batch_size, D_in, D_hid, D_out = 2, 3, 5, 1
W1 = torch.randn(D_in, D_hid, requires_grad=True)
W2 = torch.randn(D_hid, D_out, requires_grad=True)
X = torch.randn(batch_size, D_in, requires_grad=True)

class ProtectAggregate(torch.autograd.Function):
	@staticmethod
	def forward(ctx, X):
		ctx.save_for_backward(X)
		return X

	@staticmethod
	def backward(ctx, batched_grad):
		X, = ctx.saved_variables
		print("protect", batched_grad.shape)
		return batched_grad


p = ProtectAggregate.apply

output = torch.sum(ops.matmul(ops.matmul(p(X), p(W1)), p(W2)), dim=0)

jacobian = torch.autograd.grad(output, [W1, W2], retain_graph=True)

print(jacobian)
pdb.set_trace()




