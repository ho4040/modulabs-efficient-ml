import torch
from torch.sparse import to_sparse_semi_structured

a = torch.Tensor([0, 0, 1, 1]).tile((64, 16)).half().cuda()
b = torch.rand(64, 64).half().cuda()
c = torch.mm(a, b)
a_sparse = to_sparse_semi_structured(a)

print("a", a)
print("a_sparse", a_sparse)
print("b", b)
print("c", c)

is_close = torch.allclose(c, torch.mm(a_sparse, b))
print("is_close", is_close)