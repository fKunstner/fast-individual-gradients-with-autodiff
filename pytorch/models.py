import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb #debugging

class MLP(nn.Module):
	"""
	"Standard" MLP with support with goodfellow's backprop trick
	"""
	def __init__(self, input_size, hidden_sizes):
		super(type(self), self).__init__()
		
		self.input_size = input_size
		self.act = F.relu
		
		if len(hidden_sizes) == 0:
			self.hidden_layers = []
			self.output_layer = nn.Linear(self.input_size, 1)
		else:
			self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
			self.output_layer = nn.Linear(hidden_sizes[-1], 1)

	def forward(self, x):
		"""
		Forward pass that returns also returns
		* the activations (H) and
		* the linear combinations (Z)
		of each layer, to be able to use the trick from [1].
		
		Args:
			- x : The inputs of the network
		Returns:
			- logits
			- activations at each layer (including the inputs) 
			- linear combinations at each layer 

		> [1] EFFICIENT PER-EXAMPLE GRADIENT COMPUTATIONS
		> by Ian Goodfellow
		> https://arxiv.org/pdf/1510.01799.pdf
		"""
		x = x.view(-1, self.input_size)
		out = x
		
		# Save the model inputs, which are considered the activations of the 0'th layer.
		activations = [out]
		linearCombs = []

		for layer in self.hidden_layers:
			linearComb = layer(out)
			out = self.act(linearComb)

			# Save the activations and linear combinations from this layer.
			activations.append(out)
			linearComb.retain_grad()
			linearComb.requires_grad_(True)
			linearCombs.append(linearComb)

		logits = self.output_layer(out)
			
		logits.retain_grad()
		logits.requires_grad_(True)
		linearCombs.append(logits)
		
		return (logits, activations, linearCombs)
