import autograd.numpy as np
import autograd as ag
import models
import cost_functions
import pdb
from helpers import simpleTiming
SEED = 1
N = 1000
D = 500
Ds = [D, D, 1]
np.random.seed(seed=SEED)

net = models.MLP(Ds)

x = np.random.randn(N, D)
y = np.random.rand(N,1)


###############################################################################
# 

w = net.params

def obj(w, x, y):
	net.params = w
	logits = net.forward(x)
	costs = cost_functions.logloss(logits, y)
	return np.mean(costs)

def obj_(w, x, y):
	net.params = w
	logits = net.forward(x)
	costs = cost_functions.logloss(logits, y)
	return costs.reshape((-1,))

def full():
	grad = ag.grad(obj)
	grad(w, x, y)
	
def jac():
	jac = ag.jacobian(obj_)
	jac(w, x, y)


names = ["jac"]
methods = [jac]

simpleTiming(full, names, methods)
		
pdb.set_trace()

