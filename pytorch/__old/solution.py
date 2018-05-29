import torch
import pdb
import batchFuncs

matmul = batchFuncs.BatchedReverseMM.apply
relu = batchFuncs.BatchedSigmoid.apply
sigmoid = batchFuncs.BatchedSigmoid.apply


__SHOULD_TEST = False
if __SHOULD_TEST:

	batch_size, input_size, output_size = 3, 2, 1
	x = torch.randn(batch_size, input_size)
	W = torch.randn(input_size, output_size, requires_grad=True)

	# NOTE: need to use the custom ops here
	output = relu(matmul(x, W))
	jac_elems = output_size * batch_size # it's really the size of one dim of the jacobian
	batch_grad_output = torch.eye(jac_elems, jac_elems).view(jac_elems, *output.shape)

	jacobian, = torch.autograd.grad(output, W, batch_grad_output)
	print(jacobian.shape)
	pdb.set_trace()
	
__SHOULD_TEST_2 = True
if __SHOULD_TEST_2:

	batch_size, D_in, D_hid, D_out = 2, 3, 5, 1
	x = torch.randn(batch_size, D_in)
	W = torch.randn(D_in*D_hid + D_hid*D_out, requires_grad=True)
	W1 = torch.randn(D_in, D_hid, requires_grad=True)
	W2 = torch.randn(D_hid, D_out, requires_grad=True)
	
	#pdb.set_trace()
	#output = matmul(x, W[0:D_in*D_hid].view(D_in, D_hid))
	output = sigmoid(matmul(sigmoid(matmul(x, W1)), W2))
	
	#jac_elems = D_out * batch_size # it's really the size of one dim of the jacobian
	jac_elems = batch_size # it's really the size of one dim of the jacobian
	#batch_grad_output = torch.eye(jac_elems, jac_elems).view(jac_elems, *output.shape)
	
	batch_grad_output = torch.eye(jac_elems, jac_elems).view(jac_elems, jac_elems, 1)
	
	jac = torch.autograd.grad(output, [W1, W2], batch_grad_output)
	pdb.set_trace()
		
	
	
__SHOULD_TEST_3 = False
if __SHOULD_TEST_3:
	class LinearWithBatchGradFn(torch.autograd.Function):
		@staticmethod
		def forward(ctx, inp, weight, bias=None):
			ctx.save_for_backward(inp, weight, bias)
			return torch.nn.functional.linear(inp, weight, bias)
		@staticmethod
		def backward(ctx, grad_out):
			inp, weight, bias = ctx.saved_tensors
			grad_bias = grad_out if bias is not None else None
			return grad_out @ weight, (inp.unsqueeze(1)*grad_out.unsqueeze(2)), grad_bias
	linear_with_batch_grad = LinearWithBatchGradFn.apply

	
	
	
	batch_size, D_in, D_hid, D_out = 2, 3, 5, 1
	w1 = torch.randn(D_in, D_hid, requires_grad=True)
	w2 = torch.randn(D_hid, D_out, requires_grad=True)
	inp = torch.randn(batch_size, D_in, requires_grad=True)

	
	#pdb.set_trace()
	#output = matmul(x, W[0:D_in*D_hid].view(D_in, D_hid))
	output = matmul(matmul(x, W1), W2)
	
	#jac_elems = D_out * batch_size # it's really the size of one dim of the jacobian
	jac_elems = batch_size # it's really the size of one dim of the jacobian
	#batch_grad_output = torch.eye(jac_elems, jac_elems).view(jac_elems, *output.shape)
	
	batch_grad_output = torch.eye(jac_elems, jac_elems).view(jac_elems, jac_elems, 1)
	
	jacobian = torch.autograd.grad(output, [W1, W2], batch_grad_output)
	pdb.set_trace()
	print(jacobian.shape)
	pdb.set_trace()
	
	
	
	