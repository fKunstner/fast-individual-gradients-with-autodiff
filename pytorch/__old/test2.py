import numpy as np
from time import time
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
import pdb 
import otherFuncs as batch

from goodfellow_backprop import goodfellow_backprop	
from helpers import simpleTiming, profiling, check_correctness

SEED = 1
np.random.seed(seed=SEED)

# N: Number of samples
# D: Dimension of input and of each Layer
# L: Number of hidden layers
N = 1000
Do = 1000
D1 = 100

X = torch.tensor(np.random.randn(N, D1)).float()
y = torch.tensor(np.random.rand(N), requires_grad=True).float()
ww_hid = torch.tensor(np.random.rand(D1, Do), requires_grad=True).float()
ww = torch.tensor(np.random.rand(Do), requires_grad=True).view(-1,1).float()

print(X.shape, ww_hid.shape, ww.shape)

###

class MyMM(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, w, gradHolder=None):
		ctx.save_for_backward(x, w, gradHolder)
		return torch.mm(x, w)

	@staticmethod
	def backward(ctx, g):
		x,ww,gradHolder = ctx.saved_variables

		dx, dy = None, None
		if x.requires_grad:
			dx = torch.mm(g, ww.t())
		if ww.requires_grad:
			#pdb.set_trace()
			#dy = x.unsqueeze(1)*g.unsqueeze(2)
			#dy = x.t() @ g
			#dy = (x * g).unsqueeze(1)
			if gradHolder is None:
				dy = torch.bmm(x.unsqueeze(2), g.unsqueeze(1), out=gradHolder)
			else:
				dy = torch.bmm(x.unsqueeze(2), g.unsqueeze(1))
		
		return dx, dy, None 


###

Ws = [ww_hid, ww]

gradHolders = list()
gradHolders.append(torch.zeros(N, Do, D1))	
gradHolders.append(torch.zeros(N,D1,1))

start=time()
inp = X
for w in Ws:
	inp = torch.mm(inp, w)
res1 = torch.sum(inp)

grad1 = torch.autograd.grad(res1, Ws)
print("%.4f" % (time()-start))

start=time()
inp = X
for l in range(len(Ws)):
	inp = MyMM.apply(inp, Ws[l])
	#inp = MyMM.apply(inp, Ws[l], gradHolders[l])
res2 = torch.sum(inp)

grad2 = torch.autograd.grad(res2, Ws)
print("%.4f" % (time()-start))


print(list(grad1[l].shape for l in range(len(grad1))))
print(list(grad2[l].shape for l in range(len(grad2))))
