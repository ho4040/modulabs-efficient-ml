import torch
from torch import nn
from torch.sparse import to_sparse_semi_structured

input = torch.rand(64, 64).half().cuda()
print(input)
mask = torch.Tensor([0, 0, 1, 1]).tile((64, 16)).cuda().bool()
print(mask)
linear = nn.Linear(64, 64).half().cuda()
linear.weight = nn.Parameter(to_sparse_semi_structured(linear.weight.masked_fill(~mask, 0)))