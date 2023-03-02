from turtle import backward
import torch
import torchvision
import numpy as np
import random
from torch import tensor

# x = torch.empty(4 , 4 , dtype = torch.long)
# print(x)

# 创建一个4行4列的填充了随机元素的矩阵
# x = torch.rand(4 , 4)
# print(x)

# 初始化一个全零矩阵
# x = torch.zeros(5 , 3 , dtype = torch.long)
# print(x)

# 直接传入矩阵数据
# x = torch.tensor([[4 , 4.5],[1 , 3]])
# print(x)
# x = torch.Tensor([[1 , 2 , 3],[4 , 5 , 6]])
# print(x)
# print(x.size())

# y = torch.Tensor([[1 , 2 , 3],[4 , 5 , 6]])
# print(x + y)
# print(torch.add(x , y))
# print(x * y)

#view操作，修改矩阵的维数
# x = torch.tensor([[1 , 2 , 3],[4 , 5 , 6]])
# y = x.view(6)
# print(y)
# y = x.view(3,2)
# print(y)

# 与numpy协作
# x = torch.ones(5)
# y = x.numpy()
# print(y)
# x = np.ones(5)
# y = torch.from_numpy(x)
# print(y)

#允许对矩阵求导
# x = torch.rand(3 , 4 , requires_grad = True)
# print(x)
# y = torch.rand(3 , 4 , requires_grad = True)
# t = x + y
# b = t.sum()
# b.backward()
# print(y.grad)
#例子
# x = torch.rand(1 )
# w = torch.rand(1 ,  requires_grad = True)
# b = torch.rand(1 ,  requires_grad = True)
# y = x * w
# z = y + b
# z.backward(retain_graph = True)#不对梯度清零会进行累加
# print(x)
# print(w)
# print(b)
# print(y)
# print(b.grad)
# print(w.grad)

# x_values =  []
# for i in range(0 ,999):
#     x = random.uniform(0,100)
#     x_values.append(x)
# print(x_values)


# x = tensor([[1 , 2],[3 , 4]])
# #矩阵乘法
# y = x.matmul(x)
# print(y)

print(dir(torch.utils.data.DataLoader))
help(torch.utils.data.DataLoader)