import torch
import torchvision
import numpy as np

np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)
torch2numpy = torch_data.numpy()

print(
    '\nnumpy', '\n', np_data,
    '\ntorch', '\n', torch_data,
    '\ntorch to numpy', '\n', torch2numpy
    )
