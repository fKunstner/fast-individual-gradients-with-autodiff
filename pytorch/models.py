import torch
import batchFuncs as batch
import pdb
		
class MLP2():
	def __init__(self, Ds=[]):
		self.act = batch.sigmoid
		self.Ds = Ds
		
		self.params_per_layers_idx = []
		D_curr = Ds[0]
		param_idx = 0
		for D_next in Ds[1:]:
			n_params = D_curr*D_next

			self.params_per_layers_idx.append(list(range(param_idx, param_idx + n_params)))
			
			param_idx += n_params
			D_curr = D_next
			
		self.params = torch.tensor(torch.randn(param_idx), requires_grad=True)
		
	def forward(self, x):
		inputs = x.reshape(-1, self.Ds[0])
		D_curr = self.Ds[0]

		for layer_id in range(1, len(self.Ds)):
			D_out = self.Ds[layer_id]
			param_idx = self.params_per_layers_idx[layer_id-1]
			W = self.params[param_idx].reshape((D_curr, D_out))
			outputs = batch.matmul(inputs, W)
			act = self.act(outputs)
			
			D_curr = D_out
			inputs = act
		return inputs
		
class MLP2():
	def __init__(self, Ds=[]):
		self.act = batch.sigmoid
		self.Ds = Ds
		
		self.params = []
		D_curr = Ds[0]
		for D_next in Ds[1:]:
			self.params.append(torch.randn(D_curr, D_next, requires_grad=True))
			D_curr = D_next

	def forward(self, x):
		inputs = x
		
		for layer_id in range(len(self.params)):
			linearComb = batch.matmul(inputs, self.params[layer_id])
			activations = batch.sigmoid(linearComb)
			inputs = activations
			
		return inputs
		
	