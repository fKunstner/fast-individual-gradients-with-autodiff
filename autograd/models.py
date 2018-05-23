import autograd.numpy as np
from autograd.scipy.misc import logsumexp
import pdb

def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)

class MLP():
	def __init__(self, Ds=[]):
		self.act = sigmoid
		self.Ds = Ds
		
		self.params_per_layers_idx = []
		D_curr = Ds[0]
		param_idx = 0
		for D_next in Ds[1:]:
			n_params = D_curr*D_next

			self.params_per_layers_idx.append(np.array(list(range(param_idx, param_idx + n_params))))
			
			param_idx += n_params
			D_curr = D_next
			
			
		self.params = np.random.randn(param_idx, 1)
		
	def forward(self, x):
		inputs = x.reshape(-1, self.Ds[0])
		D_curr = self.Ds[0]

		for layer_id in range(1, len(self.Ds)):
			D_out = self.Ds[layer_id]
			param_idx = self.params_per_layers_idx[layer_id-1]
			W = self.params[param_idx].reshape((D_curr, D_out))
			outputs = np.dot(inputs, W)
			act = self.act(outputs)
			
			D_curr = D_out
			inputs = act
		return inputs
