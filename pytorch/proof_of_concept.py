import torch


###############################################################################
# Proof of concept - "Working", somewhat

x1 = torch.tensor([1,1,1], requires_grad=False).float()
x2 = torch.tensor([2,2,2], requires_grad=False).float()
w = torch.tensor([0,0,0], requires_grad=True).float()

f1 = torch.dot(x1, w)
f2 = torch.dot(x2, w)
stacked = torch.stack((f1, f2))

print("\"Working\" proof of concept:")
print("  x1:", x1, x1.shape)
print("  x2:", x2, x2.shape)
print("  w:", w, w.shape)
print("  f1:", f1, f1.shape)
print("  f2:", f2, f2.shape)
print("  stacked:", stacked, stacked.shape)

grads, = torch.autograd.grad(stacked, w, torch.eye(2).view(2,2,1), retain_graph=True)

print("  grads:", grads, grads.shape)

###############################################################################
# Proof of not-concept - not-"Working", somewhat

import torch.nn.functional as F

f1 = F.sigmoid(torch.dot(x1, w))
f2 = F.sigmoid(torch.dot(x2, w))
stacked = torch.stack((f1, f2))

print("\"Not-Working\" proof of concept:")

try:
	grads, = torch.autograd.grad(stacked, w, torch.eye(2).view(2,2,1), retain_graph=True)
except Exception as inst:
	print("  Error during backward pass:")
	print(" ",type(inst))
	#print(" ",inst.args)
	print(" ",inst)
