import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
	def __init__(self, input_size, hidden_sizes, output_size=None):
		super(type(self), self).__init__()
		
		self.input_size = input_size
		if output_size:
			self.output_size = output_size
			self.squeeze_output = False
		else :
			self.output_size = 1
			self.squeeze_output = True
		self.act = F.relu
		
		if len(hidden_sizes) == 0:
			self.hidden_layers = []
			self.output_layer = nn.Linear(self.input_size, self.output_size)
		else:
			self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
			self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size)

	def forward(self, x):
		x = x.view(-1, self.input_size)
		out = x

		for layer in self.hidden_layers:
			Z = layer(out)
			out = self.act(Z)

		logits = self.output_layer(out)
		if self.squeeze_output:
			logits = torch.squeeze(logits)
			
		return logits

	def forward_goodfellow(self, x):
		x = x.view(-1, self.input_size)
		out = x
		
		# Save the model inputs, which are considered the activations of the 0'th layer.
		H_list = [out]
		Z_list = []

		for layer in self.hidden_layers:
			Z = layer(out)
			out = self.act(Z)

			# Save the activations and linear combinations from this layer.
			H_list.append(out)
			Z.retain_grad()
			Z.requires_grad_(True)
			Z_list.append(Z)

		logits = self.output_layer(out)
		if self.squeeze_output:
			logits = torch.squeeze(logits)
			
		# Save the final model ouputs, which are the linear combinations from the final layer.
		logits.retain_grad()
		logits.requires_grad_(True)
		Z_list.append(logits)
		
		return (logits, H_list, Z_list)
