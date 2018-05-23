import torch
from torch.nn.utils import parameters_to_vector

def goodfellow_backprop(M, H_list, Z_list):
	"""
		Computes the gradients using the reparameterization proposed in "Efficient per-example gradient computations." by Ian Goodfellow. 
		This method assumes that every layer in the network has bias parameters.
		
		Note that the tensors in Z_list must have requires_grad=True and should retain gradients (see Tensor.retain_grad()).
		You must call loss.backward() before using this function.
		
		Arguments:
		M (int): The size of the minibatch.
		H_list (list[torch.Tensor]): A list of activations from each layer of the neural network.
			The shape of the tensors should be (M x l), where l is the size of each layer's output.
			The first element of H_list should be the network inputs.
		Z_list (list[torch.Tensor]): A list of the linear combinations produced at every
			layer of the network. These linear combinations must be **pre-activation** function. The
			shape of each tensor should be (M x l), where l is the size of each layer's output. The
			last element of Z_list should be the network's output (logits). Each tensor in Z_list must
			have a gradient w.r.t to the final loss.
	"""

	G = None

	for j in range(M):
		grads = []
		for i, z in enumerate(Z_list):

			# Gradients of layer parameters:
			g = z[j].mul_(M)
			if not any(g.size()):
				g = g.unsqueeze(0)

			grads.append(torch.ger(g, H_list[i].detach()[j, :]))

			# Gradients of bias parameters:
			grads.append(g)

		grad_vector = parameters_to_vector(grads)

		if G is None:
			G = torch.zeros(grad_vector.size()[0], M)
		G[:, j] = grad_vector

	return G	
