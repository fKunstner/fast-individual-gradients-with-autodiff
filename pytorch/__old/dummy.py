import torch
import pdb
import batchFuncs
import ops

batch_size, D_in, D_hid, D_out = 2, 3, 5, 1
G = torch.tensor(
[[[1,2,3],
[0,0,0]],
[[0,0,0],
[5,7,11]]]
)
X = torch.tensor(
[[13,17],
[19,23],
[29,31]]
)

print(G)
print(X)
print(torch.matmul(X, G))

output = torch.zeros(G.shape[0], X.shape[0], G.shape[2])

for n in range(G.shape[1]):
	output[n,:,:] = torch.ger(X[:,n], G[n,n,:])

print(output)
pdb.set_trace()