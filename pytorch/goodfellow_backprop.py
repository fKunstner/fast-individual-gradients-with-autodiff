import torch
import pdb #debugging

def goodfellow_backprop(H_list, Z_list):
	grads = []
	for i in range(len(Z_list)):
		Z, H = Z_list[i], H_list[i]
		if len(Z.shape) < 2:
			Z = Z.unsqueeze(1)
		Z *= Z.shape[0]
		grads.append(torch.bmm(Z.unsqueeze(2), H.unsqueeze(1)))
		grads.append(Z)

	return grads
