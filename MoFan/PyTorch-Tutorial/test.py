import torch

x = torch.Tensor([1, 2, 3, 4])
torch.unsqueeze(x, 0)
print(x)
print(x.shape)
torch.unsqueeze(x, 0)
print(x)
print(x.shape)