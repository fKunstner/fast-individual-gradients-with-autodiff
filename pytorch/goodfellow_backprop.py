import torch
import pdb #debugging

def goodfellow_backprop(activations, linearGrads):
	grads = []
	for i in range(len(linearGrads)):
		G, X = linearGrads[i], activations[i]
		if len(G.shape) < 2:
			G = G.unsqueeze(1)
		
		G *= G.shape[0] # if the function is an average
		
		grads.append(torch.bmm(G.unsqueeze(2), X.unsqueeze(1)))
		grads.append(G)

	return grads
